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
an OpenAI reasoning model (configurable per submission):

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
   │  Self-review  (LLM pass, gated by SELF_REVIEW)               │
   │  10 known bug categories → NO_ISSUES or repaired script      │
   └──────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Execute  (sandboxed subprocess, dynamic wall-clock budget)  │
   │  capture stdout/stderr, validate submission against schema   │
   └──────────────────────────────────────────────────────────────┘
                                 │
                  ok ────────────┴──────────── execution_bug / schema_bug
                    │                                │
                    ▼                                ▼
          ┌──────────────────┐            ┌──────────────────────┐
          │ Optional Refine  │            │ Repair (min-diff):   │
          │ (Optuna · stack ·│            │ patch failing lines, │
          │  feature eng. ·  │            │ keep model & pipeline│
          │  TTA · larger    │            │ (≤ MAX_DEBUG_ITERS)  │
          │  backbone) OR    │            └──────────────────────┘
          │ next draft with  │
          │ rotated model    │
          │ family           │
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
   ┌──────────────────────────────────────────────────────────────┐
   │  Pick winner: ensemble → best-CV single draft →              │
   │  sample_submission.csv fallback (always ship a valid CSV)    │
   └──────────────────────────────────────────────────────────────┘
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
- **Dynamic wall-clock budget.** The agent tracks a single total wall-clock
  budget (default 4h, matching the Quick-Submit cap) and derives each iter's
  timeout from the remaining budget — the first draft can spend most of the
  budget if needed, later iters get whatever's left, and a fixed reserve at
  the end is held back for ensembling, schema validation, and CSV write. The
  coder prompt enforces a "≤ 80% of budget" rule so a slow CPU doesn't push
  an iter over; a finished mediocre submission beats a timed-out one.
- **Pre-execution self-review.** Between code-gen and subprocess execution, a
  separate LLM pass reviews the script for ten known bug categories (CatBoost
  NaN in `cat_features`, LightGBM 4.x API misuse, missing
  `enable_categorical=True`, CV leakage from non-OOF target encodings,
  submission schema mismatch, bool-vs-string label confusion, image dataloader
  `None` collate, hard-coded paths, missing `CV score:` print, writing to the
  wrong path) and emits either `NO_ISSUES` or a repaired script. A subprocess
  timeout costs 15–30 min, so this LLM call is cheap insurance.
- **Repair vs. redraft branching.** After each iter the loop classifies the
  result (`execution_bug` / `schema_bug` / `ok`) and dispatches to a different
  prompt: a **minimum-diff repair** prompt patches the precise failing line(s)
  while keeping the model family and feature pipeline intact, vs. a fresh
  **draft** prompt only when starting over. No `try/except` band-aids — repairs
  must address the root cause.
- **Best-of-N with safe fallback.** When the ensemble is disabled or the
  averaged submission fails schema validation, the agent picks the best
  individual valid draft by CV score (respecting the metric's
  lower-is-better/higher-is-better direction inferred by the planner). If no
  draft produced a valid submission at all, the agent ships a copy of
  `sample_submission.csv` so the grader always receives a well-formed CSV
  rather than a hard failure.
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
  local_test.py      Mocked-green driver for local dry-runs
  planner_ping.py    Standalone smoke test for the planner JSON contract
Dockerfile           CPU-only image: torch CPU + sklearn / lightgbm / xgboost /
                     catboost / timm / transformers + opencv-headless
amber-manifest.json5 AgentBeats deployment manifest (image, env, config schema)
requirements.txt
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

The manifest exposes the primary knobs as Quick-Submit config so the same image
can be reused across competitions with per-task overrides:

| Key | Default | Purpose |
|---|---|---|
| `openai_api_key` | required | OpenAI credential (secret) |
| `openai_model` | `gpt-5.4` | Planner / coder / refiner model |
| `reasoning_effort` | `medium` | `low` / `medium` / `high` |
| `max_debug_iters` | `5` | Plan / code / run cycles per draft |
| `total_budget_sec` | `14400` (4h) | Hard cap across the full run; matches the empirical Quick-Submit cap |
| `subprocess_timeout_sec` | = `total_budget_sec` | Upper bound on a single `solution.py` iter; per-iter timeouts are derived dynamically from remaining budget |
| `refine_timeout_sec` | `1.5 × subprocess_timeout_sec` | Upper bound on the refinement pass |

Additional env-var knobs read by the agent (not in the manifest schema; set via
the container env if you need to override):

| Env var | Default | Purpose |
|---|---|---|
| `FINALIZE_RESERVE_SEC` | `120` | End-of-run reserve held back for ensembling, schema validation, CSV write |
| `SELF_REVIEW` | `1` | Pre-execution LLM self-review pass between code-gen and subprocess; set to `0` to disable |
| `SELF_CONSISTENCY_N` | planner-decided | Force a specific number of self-consistency drafts, overriding the planner's `iteration_strategy.n_drafts` |
| `OPENAI_RETRY_ATTEMPTS` | `4` | Retries on transient OpenAI errors (429 / 5xx / connection) |
| `OPENAI_RETRY_BASE_SEC` | `2.0` | Initial backoff delay; exponential with jitter |
| `OPENAI_RETRY_MAX_SEC` | `20.0` | Cap on backoff delay |
| `WORKSPACE_DIR` | `/tmp/purple_workspace` | Where the executor extracts the competition tarball |

## Local development

```bash
python3.13 -m pip install -r requirements.txt
# Drop a Kaggle competition's files into ./data/<competition>/
# (description.md, train.csv, test.csv, sample_submission.csv, ...)
export OPENAI_API_KEY=sk-...
python -m src.server --host 127.0.0.1 --port 8080
# In another shell:
python scripts/local_test.py --data-dir ./data/<competition>
```

`local_test.py` plays the role of the green grader: it tarballs the dataset,
sends it via A2A, saves the returned CSV as `submission.csv`, and schema-checks
it against `sample_submission.csv`.
