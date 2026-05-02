# Roadmap — MLE-Bench Purple Agent

This file is the handoff doc. If a fresh agent picks up this work, read this
top-to-bottom before touching code.

## Goal

Build a competitive **purple agent** for the AgentBeats Phase 2 MLE-Bench
benchmark (research agent track). User signed up at agentbeats.dev as
`ashetty21@berkeley.edu`. Target a top-3 finish on at least 4
per-competition sub-leaderboards.

## Background — important things to know

### 1. There is no aggregate MLE-Bench leaderboard
Each Kaggle competition has its **own** sub-leaderboard at
`github.com/RDI-Foundation/MLE-bench-agentbeats-leaderboard`. Active comps:
spaceship-titanic, aerial-cactus-identification, dogs-vs-cats-redux,
right-whale-redux, jigsaw-toxic-comment-classification, denoising-dirty-documents.

Spaceship-titanic has 157/172 logged runs — it's the busy/canonical lane.
Each result JSON includes `score`, medal thresholds (gold/silver/bronze),
`is_lower_better`, and medal flags. **Top-3 is per competition**, ranked by
raw score.

### Sub-leaderboard snapshot (user-provided, 2026-05-01)

Use this for prioritization and difficulty calibration only; do not encode
hard-coded competition recipes from it into the agent.

| Sub-leaderboard | Current top / status | Score |
|---|---:|---:|
| Aerial Cactus Identification |<redacted-name>, 1st | 0.99995 |
| Denoising Dirty Documents | <redacted-name>, 1st | 0.01262 |
| Dogs vs Cats Redux | No results | — |
| ICML 2013 Whale Challenge | No results; user cannot accept rules | — |
| Jigsaw Toxic Comment Classification | <redacted-name>, 1st | 0.98113 |
| Spaceship Titanic | <redacted-name>, 1st | 0.83218 |

### 2. Wire protocol (A2A over HTTP)
The purple agent serves an A2A endpoint on **port 8080** (hardcoded in the
official manifest). Source of truth for the contract:
`https://github.com/RDI-Foundation/mle-bench-green/blob/main/src/agent.py`.

Inbound from green:
- `Message.parts[0]` = `TextPart` containing instructions.txt
- `Message.parts[1]` = `FilePart(FileWithBytes)` of `competition.tar.gz`
  (extracts to `home/data/{description.md, train.csv, test.csv, sample_submission.csv, ...}`)

Outbound to green:
- Optional handshake: `TaskStatusUpdateEvent` whose status message text
  contains `"validate"` AND a `FilePart` of a candidate `submission.csv`.
  Green replies with a TextPart `"Submission is valid"` or `"Error: ..."`.
- Final: `TaskArtifactUpdateEvent` whose artifact has a `FilePart` of
  `submission.csv`. Green base64-decodes `file.bytes` and grades it.
- Final task state: `TaskState.completed`.

The green's httpx client has `timeout=3600` (1 hour). The Quick Submit GH
Actions runner additionally enforces `RESULTS_TIMEOUT_MINUTES=30` by default.

### 3. Two submission paths into the leaderboard repo (different time caps)

Confirmed 2026-05-01 by reading the workflows in
`RDI-Foundation/MLE-bench-agentbeats-leaderboard/.github/workflows/`:

- **Quick Submit (PR-based, via agentbeats.dev)** — `quick-submit-runner.yml`
  polls with `deadline=$((SECONDS + 60 * RESULTS_TIMEOUT_MINUTES))`, default
  **30 min** (overridable per-leaderboard via `vars.QUICK_SUBMIT_TIMEOUT_MINUTES`,
  but submitters cannot change it). This is the path we've been targeting.
- **Run Scenario (push `scenario.toml` to a fork)** — `run-scenario.yml` runs
  `docker compose up --abort-on-container-exit` with **no explicit timeout**,
  so it inherits GitHub Actions' default ~6 h job limit. Some leaderboard
  entries (e.g.
  `https://github.com/RDI-Foundation/MLE-bench-agentbeats-leaderboard/actions/runs/25197385191`)
  show 3 h+ durations — those are using this path, not Quick Submit.

Implication: there's a "short mode" tuned for the 30-min Quick Submit cap and a
potential "long mode" for the un-capped fork-push path. Today we ship short
mode only.

### 4. Grading runner is CPU-only with a 30-min walltime cap (short mode)

For the Quick Submit path:
- `runs-on: ubuntu-latest` → standard GitHub-hosted runner (4 vCPU, 16 GB RAM,
  ~14 GB disk, **no GPU**).
- The 30 minutes covers everything: image pulls, container start, green
  agent shipping the tarball over A2A, our agent's full plan/code/run loop,
  schema validation, and results capture. Realistic agent-only budget after
  pulls + transfer of an 800 MB-class dataset is closer to **20–25 min**.
- Total agent runtime budget is
  `MAX_DEBUG_ITERS × SUBPROCESS_TIMEOUT_SEC + OpenAI calls + tarball IO`.
  Bumping per-iter timeout without lowering iter count busts the cap.
  Sensible per-modality defaults:
  - Tabular: `MAX_DEBUG_ITERS=3, SUBPROCESS_TIMEOUT_SEC=300`.
  - Image (e.g. dogs-vs-cats, aerial-cactus): `MAX_DEBUG_ITERS=2,
    SUBPROCESS_TIMEOUT_SEC=700`. One full draft + one repair pass.
  - Set these per-competition in the AgentBeats config block of the manifest.

### 5. Image / manifest contract
Image must be `linux/amd64`, pushed to GHCR, package made public.
Manifest expects port 8080 and a single A2A endpoint. The user's
`OPENAI_API_KEY` is passed via `${config.openai_api_key}` (mark `secret: true`).
`OPENAI_MODEL` can now be overridden at submission time via
`${config.openai_model}`; when omitted, runtime default remains `gpt-5-mini`.
Pin to digest at submission time (`@sha256:...`).

## Submitted results so far (Quick Submit, agentbeats.dev)

As of 2026-05-01:

| Competition | Rank | Score | Total duration |
|---|---:|---:|---:|
| spaceship-titanic | 50th | 0.81839 | 6m 31s |
| aerial-cactus-identification | 8th | 0.99592 | 10m 41s |
| dogs-vs-cats-redux-kernels-edition | **1st** | 0.03321 | 29m 9s |
| icml-2013-whale-challenge | — | — | green-side error (see below) |

Notes:
- Dogs-vs-cats placed 1st on an empty sub-leaderboard. The 29m 9s total
  duration ran right against the 30-min Quick Submit cap.
- Whale-challenge run aborted on the green with
  `Failed to prepare competition data: EOF when reading a line`. This is a
  green-side bug — almost certainly `input()` (likely the Kaggle
  accept-rules prompt) being called against a closed stdin. Nothing the
  purple can do; waiting on green-side fix or pre-accepted rules. Matches
  the snapshot note that the user cannot accept rules for this competition.

## What is built

| Path | Status | Notes |
|---|---|---|
| `src/server.py` | ✅ | A2A Starlette app, port 8080, agent card. Honors `A2A_MAX_CONTENT_LENGTH` (default 512 MiB) for large tarballs. |
| `src/executor.py` | ✅ | Extracts tarball to a workspace, calls agent, emits FilePart artifact. |
| `src/agent.py` | ✅ | Plan → Code → Execute → Debug loop. Universal: tabular / text / image / other, modality-branched coder prompt. Includes structured dataset profiling (dtypes/nulls/example values), stronger tabular leakage/dtype guidance, repair flow for execution / schema bugs, and self-consistency with ensembling for cheap modalities. |
| `src/openai_client.py` | ✅ | Responses API wrapper, model from `OPENAI_MODEL` env. |
| `scripts/local_test.py` | ✅ | Mocked-green driver. Sends instructions+tar, captures artifact. |
| `scripts/fetch_spaceship_titanic.sh` | ✅ | Kaggle CLI wrapper + writes `description.md`. |
| `scripts/fetch_dogs_vs_cats.sh` | ✅ | Kaggle CLI wrapper for `dogs-vs-cats-redux-kernels-edition`; supports new `KAGGLE_API_TOKEN` env var auth. |
| `Dockerfile` | ✅ | python:3.12-slim + CPU torch wheels (separate layer), exposes 8080. |
| `amber-manifest.json5` | ✅ | Image ref placeholder — update before submit. |
| `requirements.txt` | ✅ | a2a-sdk pinned `>=0.3.20,<1.0`; openai, pandas, numpy, sklearn, lightgbm, xgboost, timm, transformers, pillow, opencv-headless, kaggle. |

## What is NOT done yet

1. **Validation handshake unused.** The agent does NOT currently negotiate
   schema with the green via the `"validate"` message. Adding that would
   catch malformed submissions before final grading. Path: in `executor.py`,
   between iterations, send a status update with text `"validate"` + a
   `FilePart` of the candidate CSV. The current driver in `scripts/local_test.py`
   does not implement validation either. **Lift in difficulty: medium.**
2. **Quality tuning of prompts.** The current planner/coder system prompts
   are reasonable defaults but were not iterated against actual leaderboard
   scores. Keep prompt improvements generic and task-derived: the agent should
   infer feature engineering and model choices from `description.md`, file
   layout, data previews, and `sample_submission.csv`, rather than receiving
   hard-coded competition recipes.
3. **No "long mode" for the fork-push Run Scenario path.** Today we tune for
   the 30-min Quick Submit cap. A long-mode profile (bigger iter budget,
   EfficientNet-B3+, 5-fold + TTA) gated by an env var would let the same
   image compete on the un-capped path for heavier comps. See "Where to push
   next" #4.
4. **Self-consistency quality is not yet leaderboard-proven.** The mechanism
   now works end-to-end locally, but the latest spaceship-titanic smoke
   (`cv=0.813758`, `cv=0.810192`, valid ensemble artifact) did NOT exceed the
   user's existing leaderboard result of `0.81839`. Reliability improved;
   competitive lift is still unproven.

## Verified locally

- All five Python files compile under Python 3.12.
- Requirements installed under the local Python 3.13 environment
  (`pip` is bound to Python 3.13; `/usr/bin/python` is Python 3.8 and cannot run
  this code).
- Server boots and serves `/.well-known/agent-card.json` on port 8080.
- A2A large-payload cap was raised from the SDK default 10 MiB to 512 MiB
  (`A2A_MAX_CONTENT_LENGTH`); for the 840 MB dogs-vs-cats tarball we run
  local with 2 GiB.
- `a2a-sdk` version pin is critical: pip's resolver picks `1.0.x` by default,
  which has a breaking API change (no `a2a.server.apps`). `requirements.txt`
  pins `>=0.3.20,<1.0` — keep it that way unless you also rewrite the imports.
- Smoke / full-data runs (chronological):
  - `jigsaw-toxic-comment-classification-challenge-smoke`: valid 3000-row
    submission in 164.7s; `cv=0.918476`.
  - `denoising-dirty-documents-smoke`: valid 278640-row submission in 104.3s;
    `cv=0.142267`.
  - `aerial-cactus-identification-smoke`: 85.4s pre-2026-05-01; 42.6s after
    the structured dataset profile + budget-aware image guidance.
  - `aerial-cactus-identification` full data: planner correctly inferred
    nested image roots. With `MAX_DEBUG_ITERS=2`, agent self-corrected past
    an OpenCV dtype bug and produced a valid 4000-row submission in 455.0s
    (`cv=0.9576`, then `cv=0.959771`). Control loop subsequently fixed to
    stop after the first clean submission.
  - `dogs-vs-cats-redux-kernels-edition` full data (post-repair): with
    `MAX_DEBUG_ITERS=2`, `SUBPROCESS_TIMEOUT_SEC=700`, iter 1 produced a
    valid 12,500-row submission in 532.8s (`cv=0.076047` log-loss).
  - `spaceship-titanic` post-repair regression: iter 1 succeeded with
    `cv=0.757` in 144s; first-draft path is unchanged from pre-refactor, so
    the lower CV vs prior 0.811 baseline is model variance + GroupKFold-by-
    family being more rigorous than the prior StratifiedKFold.
  - `spaceship-titanic` self-consistency smoke (2026-05-02): with
    `SELF_CONSISTENCY_N=2`, `MAX_DEBUG_ITERS=5`, `SUBPROCESS_TIMEOUT_SEC=300`,
    the planner correctly recognized `Transported` as boolean and called for
    group-aware validation from `PassengerId`. Draft 1 succeeded with
    `cv=0.813758`; draft 2 initially failed to create a submission, then the
    repair flow recovered it to `cv=0.810192`. Agent returned a valid ensemble
    artifact in 188.2s with correct submission schema. This validates the
    self-consistency control path mechanically, but not as a leaderboard gain
    versus the existing `0.81839` submission.

## Open questions for the user

1. **Whether to implement the validation handshake** (see "What is NOT done
   yet" #1). Costs ~half a day; reduces the chance of a 0-score submission
   from a schema bug.
2. **Whether to invest in a "long mode" profile** for the fork-push Run
   Scenario path. Pays off most on heavier image / multi-modal comps.

## How to test (10-minute smoke test)

Once dataset is in `./data/spaceship-titanic/`:

```bash
# Terminal 1
export OPENAI_API_KEY=sk-...
export MAX_DEBUG_ITERS=3
export SUBPROCESS_TIMEOUT_SEC=300
python -m src.server --host 127.0.0.1 --port 8080

# Terminal 2
python scripts/local_test.py --data-dir ./data/spaceship-titanic
```

This is a smoke-test profile, not the code defaults. Current code defaults are
`MAX_DEBUG_ITERS=5` and `SUBPROCESS_TIMEOUT_SEC=1500`; for a 10-minute local
check, override them as above.

Expectation with the smoke-test profile (3 iterations, 300s timeout each):
- Total run time 4–8 minutes (most spent in OpenAI calls).
- Final `submission.csv` matches `sample_submission.csv` schema.
- LightGBM CV accuracy ~0.75–0.81 on 5-fold (varies with seed and CV strategy).
- Public score lands around bronze threshold (~0.810) on spaceship-titanic.

If it fails:
- Check `agent.py` logs for which iteration crashed.
- The generated `solution_*.py` files live in `WORKSPACE_DIR/run_<id>/solutions/`.
- The generated CSVs live in `WORKSPACE_DIR/run_<id>/submissions/`.

## Submission flow (user does these)

GitHub user is `ab-shetty` (Kaggle username is `abhishek1shetty`). Repo name:
`agentbeats-mle-purple` → image `ghcr.io/ab-shetty/agentbeats-mle-purple`.

Already done once (the four submitted results above came through this flow).
To re-submit after agent changes:

1. **Push to GitHub.** The publish workflow auto-builds + pushes the image
   to `ghcr.io/ab-shetty/agentbeats-mle-purple` on every push to main
   (linux/amd64).
2. **Make the package public** (one-time): open
   `https://github.com/ab-shetty?tab=packages`, switch visibility to
   **Public** so the AgentBeats runner can pull without auth.
3. **Update `amber-manifest.json5`** with the pinned digest from the
   workflow's job summary
   (`ghcr.io/ab-shetty/agentbeats-mle-purple@sha256:<DIGEST>`). Commit + push.
4. **Submit on agentbeats.dev** via Quick Submit:
   - Manifest URL:
     `https://raw.githubusercontent.com/ab-shetty/agentbeats-mle-purple/main/amber-manifest.json5`
   - Pick the MLE-Bench leaderboard + a competition.
   - Paste the encrypted `openai_api_key` secret.
   - Optional Config JSON, e.g.
     `{"openai_model":"gpt-5.2","reasoning_effort":"high"}`
   - The runner forks the leaderboard repo, opens a PR, and writes the
     result JSON.

## Where to push next for top-3

1. **Improve spaceship-titanic ranking (currently 50th, 0.81839).** Tabular
   lane has the most-tuned competition. Likely lifts: better CV strategy,
   richer feature engineering from `description.md`, ensembling LightGBM +
   XGBoost + small MLP. Keep recipes generic / task-derived.
2. **Improve aerial-cactus ranking (currently 8th, 0.99592).** Score is
   close to ceiling (top is 0.99995); marginal gains from TTA, larger
   backbone within the 700s per-iter budget, threshold calibration.
3. **Implement the validate handshake** (see § "What is NOT done yet" #1).
4. **Add a "long mode" profile for the fork-push path.** Env-var-gated:
   bigger `MAX_DEBUG_ITERS`, larger backbones, 5-fold + TTA. Same image,
   different config, lets us compete on heavier comps without busting the
   30-min Quick Submit cap.
5. **Tune self-consistency / ensembling, do not just add it.** The mechanism
   is now implemented and mechanically works, but next work is to improve its
   draft success rate and only enable it where it is score-positive within the
   30-minute budget.
