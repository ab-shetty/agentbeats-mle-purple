# MLE-Bench Purple Agent

An autonomous ML-engineering agent built for the [AgentBeats](https://agentbeats.dev)
**MLE-Bench** benchmark. Given a Kaggle-style competition (description, train/test
data, sample submission), the agent plans a solution, writes code, runs it, debugs
failures, optionally refines or ensembles, and returns a `submission.csv` that the
benchmark grades against the held-out leaderboard.

The agent runs entirely on the AgentBeats Quick-Submit runner — **4 vCPU, 16 GB RAM,
no GPU** — and handles tabular, text, and image competitions from a single binary.

## How it works

The agent implements an **AIDE-style plan / code / execute / debug loop** driven by
an OpenAI reasoning model (`gpt-5-mini` by default; configurable per submission):

```
   ┌──────────────────────────────────────────────────────────────┐
   │                       Inspect dataset                        │
   │  description.md · sample_submission.csv · file tree · EDA    │
   │  dtypes, nulls, target distribution, mutual-information,     │
   │  group-ID detection (for GroupKFold decisions)               │
   └──────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Plan  (LLM → JSON)                                          │
   │  modality · task type · metric · CV strategy · ranked        │
   │  model_plan · iteration_strategy {n_drafts, refine, ensemble}│
   └──────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Code  (LLM → solution.py)                                   │
   │  Self-contained Python: reads DATA_DIR, writes OUTPUT_PATH   │
   └──────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Execute  (sandboxed subprocess, per-iter wall-clock budget) │
   │  capture stdout/stderr, validate submission against schema   │
   └──────────────────────────────────────────────────────────────┘
                                 │
                  pass ──────────┴────────── fail / invalid CSV
                    │                                │
                    ▼                                ▼
          ┌──────────────────┐            ┌──────────────────────┐
          │ Optional Refine  │            │ Debug: feed error    │
          │ (Optuna · stack ·│            │ back, regenerate the │
          │  feature eng. ·  │            │ script, try again    │
          │  TTA · larger    │            │ (≤ MAX_DEBUG_ITERS)  │
          │  backbone)       │            └──────────────────────┘
          └──────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Optional Self-   │
          │ Consistency      │
          │ Ensemble: blend  │
          │ N drafts via OOF │
          │ meta-learner     │
          └──────────────────┘
                    │
                    ▼
              submission.csv
```

### Key design choices

- **Planner-first.** Before any code is written, the planner inspects the dataset
  and emits a structured JSON plan. The plan ranks candidate model families by
  *expected leaderboard score within the execution budget* — not by safety. The
  same plan also picks the iteration strategy (single shot vs. refine vs.
  multi-draft ensemble) so the agent doesn't burn budget over-engineering an
  easy task or under-spending on a hard one.
- **Modality-specialized prompts.** The coder system prompt branches on
  `tabular | text | image | other` with concrete CPU recipes (LightGBM /
  CatBoost / XGBoost; TF-IDF + OvR LogReg; timm pretrained backbones; small
  U-Net / DnCNN for image-to-image). This avoids wasted iterations on
  obviously-wrong approaches.
- **EDA passed to the LLM.** A target-aware data profile (dtypes, null counts,
  mutual information vs. target, group-ID detection, target homogeneity per
  group) is included in the planner prompt so the LLM picks GroupKFold when it
  matters and prioritizes feature engineering on high-MI columns.
- **Hard wall-clock budget.** Every `solution.py` runs under a per-iteration
  subprocess timeout. The coder prompt enforces a "≤ 80% of budget" rule and
  instructs the LLM to pick a leaner version of the same approach rather than
  hope a slow CPU finishes in time. A finished mediocre submission beats a
  timed-out one.
- **Real debug, not paper-overs.** When a script crashes or produces an invalid
  submission, the full traceback / schema error is fed back to the coder with an
  explicit instruction to fix the root cause — no `try/except` swallowing.
- **Self-consistency ensemble (optional).** When the planner requests multiple
  drafts, each subsequent draft is told to *swap the model family* (rotation:
  LightGBM → CatBoost → XGBoost; LogReg → LinearSVC → SGD; resnet18 →
  efficientnet_b0) while *keeping* the working feature pipeline. Drafts emit
  out-of-fold predictions to `OOF_PATH` and a meta-learner stacks them.
- **Refinement pass (optional).** A separate refine prompt picks exactly one
  tactic — Optuna search, stacking with a complementary base model, targeted
  feature engineering, or TTA / larger backbone — instead of a free-form
  rewrite that tends to regress.

## Repository layout

```
src/
  server.py          A2A Starlette server (port 8080)
  executor.py        A2A executor: extracts tarball, runs agent, emits artifact
  agent.py           Plan / code / execute / debug / refine / ensemble loop
  openai_client.py   OpenAI Responses API wrapper (reasoning models)
scripts/
  local_test.py                Mocked-green driver for local dry-runs
  fetch_spaceship_titanic.sh   Pull a sample Kaggle dataset
  fetch_dogs_vs_cats.sh
Dockerfile           CPU-only image: torch CPU + sklearn / lightgbm / xgboost /
                     catboost / timm / transformers + opencv-headless
amber-manifest.json5 AgentBeats deployment manifest (image, env, config schema)
requirements.txt
roadmap.md           Internal handoff doc (per-competition status & tunings)
```

## A2A protocol

The agent speaks the AgentBeats A2A protocol over HTTP on port 8080.

**Inbound** (from the green grader):
- `Message.parts[0]` — `TextPart` of `instructions.txt`.
- `Message.parts[1]` — `FilePart(FileWithBytes)` of `competition.tar.gz`,
  extracted to `home/data/{description.md, train.csv, test.csv,
  sample_submission.csv, ...}`.

**Outbound**:
- `TaskStatusUpdateEvent` heartbeats while running.
- Final `TaskArtifactUpdateEvent` carrying `submission.csv` as a base64
  `FilePart`. Final task state: `TaskState.completed`.

## Configuration

The manifest exposes everything as Quick-Submit config so the same image can be
reused across competitions with per-task overrides:

| Key | Default | Purpose |
|---|---|---|
| `openai_api_key` | required | OpenAI credential (secret) |
| `openai_model` | `gpt-5-mini` | Planner / coder / refiner model |
| `reasoning_effort` | `medium` | `low` / `medium` / `high` |
| `max_debug_iters` | `5` | Plan / code / run cycles per draft |
| `subprocess_timeout_sec` | `1500` | Per-iter `solution.py` wall clock |
| `refine_timeout_sec` | — | Override for the refinement pass |
| `total_budget_sec` | — | Hard cap across the full run |

## Local development

```bash
python3.13 -m pip install -r requirements.txt
./scripts/fetch_spaceship_titanic.sh ./data/spaceship-titanic
export OPENAI_API_KEY=sk-...
python -m src.server --host 127.0.0.1 --port 8080
# In another shell:
python scripts/local_test.py --data-dir ./data/spaceship-titanic
```

`local_test.py` plays the role of the green grader: it tarballs the dataset,
sends it via A2A, saves the returned CSV as `submission.csv`, and schema-checks
it against `sample_submission.csv`.
