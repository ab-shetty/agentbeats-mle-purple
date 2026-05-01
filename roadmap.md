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

### Current AgentBeats MLE-Bench sub-leaderboard snapshot

User-provided reference snapshot as of 2026-05-01. Use this for prioritization
and difficulty calibration only; do not encode hard-coded competition recipes
from it into the agent.

| Sub-leaderboard | Current top / status | Score |
|---|---:|---:|
| Aerial Cactus Identification |<redacted-name>, 1st | 0.99995 |
| Denoising Dirty Documents | <redacted-name>, 1st | 0.01262 |
| Dogs vs Cats Redux | No results | — |
| ICML 2013 Whale Challenge | No results; user cannot accept rules | — |
| Jigsaw Toxic Comment Classification | <redacted-name> — Claude Sonnet 4.6, 1st | 0.98113 |
| Spaceship Titanic | <redacted-name> — Claude Sonnet 4.6, 1st | 0.83218 |

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

### 2.5. Grading runner is CPU-only with a 30-min walltime cap
Confirmed 2026-05-01 by reading
`RDI-Foundation/MLE-bench-agentbeats-leaderboard/.github/workflows/quick-submit-runner.yml`:
- `runs-on: ubuntu-latest` → standard GitHub-hosted runner (4 vCPU, 16 GB RAM,
  ~14 GB disk, **no GPU**).
- `RESULTS_TIMEOUT_MINUTES: 30` (overridable per-leaderboard via
  `vars.QUICK_SUBMIT_TIMEOUT_MINUTES`, but **submitters cannot change it**).
- The 30 minutes covers everything: image pulls, container start, green
  agent shipping the tarball over A2A, our agent's full plan/code/run loop,
  schema validation, and results capture. Realistic agent-only budget after
  pulls + transfer of an 800 MB-class dataset is closer to **20–25 min**.
- Implication: total agent runtime budget is
  `MAX_DEBUG_ITERS × SUBPROCESS_TIMEOUT_SEC + OpenAI calls + tarball IO`.
  Bumping per-iter timeout without lowering iter count busts the cap.
  Sensible per-modality defaults:
  - Tabular: `MAX_DEBUG_ITERS=3, SUBPROCESS_TIMEOUT_SEC=300`.
  - Image (e.g. dogs-vs-cats, aerial-cactus): `MAX_DEBUG_ITERS=2,
    SUBPROCESS_TIMEOUT_SEC=700`. One full draft + one repair pass.
  - Set these per-competition in the AgentBeats config block of the manifest.

### 3. Image / manifest contract
Image must be `linux/amd64`, pushed to GHCR, package made public.
Manifest expects port 8080 and a single A2A endpoint. The user's
`OPENAI_API_KEY` is passed via `${config.openai_api_key}` (mark `secret: true`).
Pin to digest at submission time (`@sha256:...`).

## What is built

| Path | Status | Notes |
|---|---|---|
| `src/server.py` | ✅ | A2A Starlette app, port 8080, agent card. |
| `src/executor.py` | ✅ | Extracts tarball to a workspace, calls agent, emits FilePart artifact. |
| `src/agent.py` | ✅ | Plan → Code → Execute → Debug loop. Universal: tabular / text / image / other, modality-branched coder prompt. |
| `src/openai_client.py` | ✅ | Responses API wrapper, model from `OPENAI_MODEL` env. |
| `scripts/local_test.py` | ✅ | Mocked-green driver. Sends instructions+tar, captures artifact. |
| `scripts/fetch_spaceship_titanic.sh` | ✅ | Kaggle CLI wrapper + writes `description.md`. |
| `scripts/fetch_dogs_vs_cats.sh` | ✅ | Kaggle CLI wrapper for `dogs-vs-cats-redux-kernels-edition`; supports new `KAGGLE_API_TOKEN` env var auth. |
| `Dockerfile` | ✅ | python:3.12-slim + CPU torch wheels (separate layer), exposes 8080. |
| `amber-manifest.json5` | ✅ | Image ref placeholder — update before submit. |
| `requirements.txt` | ✅ | a2a-sdk, openai, pandas, numpy, sklearn, lightgbm, xgboost, timm, transformers, pillow, opencv-headless. |

## What is NOT done yet

1. ~~End-to-end smoke test~~ ✅ Done. Spaceship-titanic CV ~0.811 (bronze
   range) on the universal image.  Jigsaw smoke test attempted
   (2026-05-01) — initially blocked by an A2A JSON-RPC `Payload too large`
   error on the 53 MB `competition.tar.gz`. **Fixed 2026-05-01:** `src/server.py`
   now passes `max_content_length` to `A2AStarletteApplication`, defaulting to
   512 MiB via `A2A_MAX_CONTENT_LENGTH`. In-process regression check confirmed
   an 11 MiB JSON-RPC body returns `Payload too large` at the SDK's 10 MiB
   limit but reaches normal JSON-RPC routing (`Method not found`) with the new
   512 MiB limit. Next step is to retry the real jigsaw smoke test.
2. **Image not built / pushed**. Need GHCR push to `ghcr.io/<gh-user>/mle-bench-purple:v1`.
3. **Validation handshake unused**. The agent does NOT currently negotiate
   schema with the green via the `"validate"` message. Adding that would
   catch malformed submissions before final grading. Path: in `executor.py`,
   between iterations, send a status update with text `"validate"` + a
   `FilePart` of the candidate CSV. The current driver in `scripts/local_test.py`
   does not implement validation either. **Lift in difficulty: medium.**
4. **No CV-track support** (no torch/transformers in the slim image). For
   competitions like aerial-cactus-identification or dogs-vs-cats, build a
   `Dockerfile.cv` variant with `torch`/`torchvision`/`timm` and either ship
   two images or one fat image (~3 GB).
5. **Quality tuning of prompts.** The current planner/coder system prompts
   are reasonable defaults but were not iterated against actual leaderboard
   scores. Keep prompt improvements generic and task-derived: the agent should
   infer feature engineering and model choices from `description.md`, file
   layout, data previews, and `sample_submission.csv`, rather than receiving
   hard-coded competition recipes.
6. ~~Debug loop is still too rewrite-heavy.~~ ✅ Repair flow landed 2026-05-01.
   Earlier hardening (structured dataset profile, fast-CPU image guidance, stop
   after first clean submission) is still in place. New on 2026-05-01:
   - `_classify_failure()` routes the next iter:
     - `rc=0 + submission_ok` → `ok` (loop exits early as before).
     - `rc=-2` → `timeout` → full rewrite via `_draft_code` with a
       reduce-scope hint (smaller image size, fewer folds, lighter backbone).
     - `rc=0 + bad submission` → `schema_bug` → minimal-diff repair.
     - any other non-zero rc → `execution_bug` → minimal-diff repair.
   - `_repair_code()` uses a separate `SYSTEM_PROMPT_REPAIR` system prompt that
     emphasizes "fix the bug, don't redesign". Sends the full prior code plus
     stdout/stderr/validation error as context.
   - The rewrite-on-timeout path now also tells the model the previous run
     timed out and to reduce scope, instead of being indistinguishable from a
     bug rewrite.

## Verified locally

- All five Python files compile under Python 3.12.
- Requirements installed under the local Python 3.13 environment
  (`pip` is bound to Python 3.13; `/usr/bin/python` is Python 3.8 and cannot run
  this code).
- Server boots and serves `/.well-known/agent-card.json` on port 8080.
- A2A large-payload cap was raised from the SDK default 10 MiB to 512 MiB
  (`A2A_MAX_CONTENT_LENGTH`).
- `jigsaw-toxic-comment-classification-challenge-smoke`: valid `3000`-row
  submission in `164.7s`; generated script completed with `cv=0.918476`.
- `denoising-dirty-documents-smoke`: valid `278640`-row submission in `104.3s`;
  generated script completed with `cv=0.142267`.
- `aerial-cactus-identification-smoke`: valid `500`-row submission in `85.4s`
  before the 2026-05-01 agent changes. After adding the structured dataset
  profile + budget-aware image guidance, the same smoke run finished in `42.6s`.
- `aerial-cactus-identification` full data: planner correctly inferred nested
  image roots (`train/train/*.jpg`, `test/test/*.jpg`). With
  `MAX_DEBUG_ITERS=1`, the first generated script failed on an OpenCV dtype bug
  during feature extraction and fell back to `sample_submission.csv`. With
  `MAX_DEBUG_ITERS=2`, the agent self-corrected and produced a valid `4000`-row
  submission in `455.0s`; the two successful iterations logged `cv=0.9576` and
  `cv=0.959771`. After that run, the control loop was fixed to stop after the
  first clean submission.
- `dogs-vs-cats-redux-kernels-edition` full data (2026-05-01, post-repair):
  with `MAX_DEBUG_ITERS=2`, `SUBPROCESS_TIMEOUT_SEC=700`, the first iteration
  produced a valid 12,500-row submission in `532.8s` (`cv=0.076047` log-loss
  on internal holdout). Tarball was 840 MB — required bumping
  `A2A_MAX_CONTENT_LENGTH` past the 512 MiB default; we run local with 2 GiB.
  Repair path was not exercised because iter 1 succeeded.
- **a2a-sdk version pin is critical**: pip's resolver picks `1.0.x` by default,
  which has a breaking API change (no `a2a.server.apps`). `requirements.txt`
  pins `>=0.3.20,<1.0` — keep it that way unless you also rewrite the imports.

## Not yet verified

- Whether any of the locally validated submissions (Aerial Cactus, Jigsaw,
  Denoising, Spaceship Titanic, Dogs vs Cats) are competitive on the AgentBeats
  sub-leaderboards or merely valid. Local CV is encouraging across the board
  but no external leaderboard score has been checked from this repo state.
- End-to-end run inside the actual AgentBeats Quick Submit runner (next step
  is the GHCR push + manifest update + Quick Submit).
- Whether the repair path actually recovers a real failed run end-to-end on
  data we hold (the dogs-vs-cats run that exercised it would have helped, but
  iter 1 succeeded).

## Open questions for the user

1. **Whether to implement the validation handshake** before first submission.
   It costs ~half a day to add but reduces the chance of a 0-score submission
   from a schema bug.

## How to test (10-minute smoke test)

Once dataset is in `./data/spaceship-titanic/`:

```bash
# Terminal 1
export OPENAI_API_KEY=sk-...
python -m src.server --host 127.0.0.1 --port 8080

# Terminal 2
python scripts/local_test.py --data-dir ./data/spaceship-titanic
```

Expectation with defaults (3 iterations, 180s timeout each):
- Total run time 4–8 minutes (most spent in OpenAI calls).
- Final `submission.csv` matches `sample_submission.csv` schema.
- LightGBM CV accuracy ~0.80–0.81 on 5-fold.
- Public score should land around bronze threshold (~0.810).

If it fails:
- Check `agent.py` logs for which iteration crashed.
- The generated `solution_*.py` files live in `WORKSPACE_DIR/run_<id>/solutions/`.
- The generated CSVs live in `WORKSPACE_DIR/run_<id>/submissions/`.

## Submission steps (user does these)

GitHub user is `ab-shetty` (Kaggle username is `abhishek1shetty`). Repo name:
`agentbeats-mle-purple` → image `ghcr.io/ab-shetty/agentbeats-mle-purple`.

1. **Push this repo to GitHub** (create a new public repo on github.com):
   ```bash
   cd /root/agentbeats-mle-purple
   git init && git add . && git commit -m "Initial purple agent"
   git branch -M main
   git remote add origin https://github.com/ab-shetty/agentbeats-mle-purple.git
   git push -u origin main
   ```
2. **The publish workflow auto-builds + pushes the image** to
   `ghcr.io/ab-shetty/agentbeats-mle-purple` on every push to main
   (linux/amd64). After the first run, open the new package on
   `https://github.com/ab-shetty?tab=packages` and switch its
   visibility to **Public** so the AgentBeats runner can pull without auth.
3. **Update `amber-manifest.json5`**: copy the pinned digest from the
   workflow's job summary (it prints
   `ghcr.io/ab-shetty/agentbeats-mle-purple@sha256:<DIGEST>`) and
   replace the `image:` field. Commit + push.
4. **Manifest URL** for Quick Submit:
   `https://raw.githubusercontent.com/ab-shetty/agentbeats-mle-purple/main/amber-manifest.json5`
5. **Submit on agentbeats.dev** via Quick Submit:
   - Pick the MLE-Bench leaderboard.
   - Pick `spaceship-titanic` as the competition.
   - Paste manifest URL.
   - Paste the encrypted `openai_api_key` secret.
   - The runner forks the leaderboard repo, opens a PR, and writes the result JSON.

## Where to push next for top-3

1. **Use `dogs-vs-cats-redux-kernels-edition` as the next ranking lane.**
   Aerial Cactus is now a decent local image testbed, but the current
   AgentBeats Dogs vs Cats sub-leaderboard is empty, so even a solid generic
   image agent may place immediately.
2. **Keep iterating the coder/debug loop generically.** The dataset-profile
   work improved Aerial materially, but the next lift is a true repair prompt
   instead of full script rewrites after ordinary runtime bugs.
3. **Implement the validate handshake** (see § "What is NOT done yet" #3).
4. **Add CV-image variant** for image competitions.
5. ~~Use higher iteration budgets selectively.~~ Reframed by §2.5: the 30-min
   outer cap means total budget is fixed, so picking `MAX_DEBUG_ITERS` and
   `SUBPROCESS_TIMEOUT_SEC` is a per-modality split of that envelope, not a
   knob to push higher. Defaults already in §2.5.
6. **Self-consistency / ensembling.** Run 3 distinct solution drafts in
   parallel (different seeds / model families) and majority-vote / average
   probabilities for the final submission.
