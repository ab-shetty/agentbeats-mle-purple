# Roadmap — MLE-Bench Purple Agent

This file is the handoff doc. If a fresh agent picks up this work, read this
top-to-bottom before touching code.

## Goal

Build a competitive **purple agent** for the AgentBeats Phase 2 MLE-Bench
benchmark (research agent track). User signed up at agentbeats.dev as
`ashetty21@berkeley.edu`. Target a top-3 finish on at least one
per-competition sub-leaderboard, starting with **`spaceship-titanic`**.

## Background — important things to know

### 1. There is no aggregate MLE-Bench leaderboard
Each Kaggle competition has its **own** sub-leaderboard at
`github.com/RDI-Foundation/MLE-bench-agentbeats-leaderboard`. Active comps:
spaceship-titanic, aerial-cactus-identification, dogs-vs-cats-redux,
right-whale-redux, jigsaw-toxic-comment-classification, denoising-dirty-documents,
mlsp-2013-birds, dog-breed-identification, text-normalization-english.

Spaceship-titanic has 157/172 logged runs — it's the busy/canonical lane.
Each result JSON includes `score`, medal thresholds (gold/silver/bronze),
`is_lower_better`, and medal flags. **Top-3 is per competition**, ranked by
raw score.

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
| `Dockerfile` | ✅ | python:3.12-slim + CPU torch wheels (separate layer), exposes 8080. |
| `amber-manifest.json5` | ✅ | Image ref placeholder — update before submit. |
| `requirements.txt` | ✅ | a2a-sdk, openai, pandas, numpy, sklearn, lightgbm, xgboost, timm, transformers, pillow, opencv-headless. |

## What is NOT done yet

1. ~~End-to-end smoke test~~ ✅ Done. Spaceship-titanic CV ~0.811 (bronze
   range) on the universal image.  Jigsaw smoke test attempted
   (2026-05-01) — **blocked by an A2A JSON-RPC `Payload too large` error**
   on the 53 MB `competition.tar.gz`. The server returned `Code=-32600
   Payload too large` before reaching the executor. Two implications:
     - The `scripts/local_test.py` driver hits the SDK's default JSON-RPC
       size cap. Need to either raise that cap on the server (look for a
       max-payload kwarg on `A2AStarletteApplication` / its router), or
       have the driver chunk via multiple parts. The real green agent
       likely uses the same default — so this is **a real risk for any
       MLE-Bench competition with a >~10 MB tarball** (which is most of
       them; spaceship-titanic was an outlier at <1 MB).
     - Until that is resolved, only small-data comps (spaceship-titanic,
       aerial-cactus 32×32, denoising-dirty-documents) have been
       de-risked. **Investigate first thing next session.**
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
   scores. The biggest quality lever is the **coder** prompt — encourage
   feature engineering specific to spaceship-titanic (split `Cabin` into
   `deck/num/side`, parse `PassengerId` group, group-aware imputation,
   stack LightGBM + CatBoost) to push from bronze (~0.810) toward silver
   (~0.814) and gold (~0.821).

## Verified locally

- All five Python files compile under Python 3.12.
- Server boots and serves `/.well-known/agent-card.json` on port 8080.
- **a2a-sdk version pin is critical**: pip's resolver picks `1.0.x` by default,
  which has a breaking API change (no `a2a.server.apps`). `requirements.txt`
  pins `>=0.3.20,<1.0` — keep it that way unless you also rewrite the imports.

## Not yet verified

- End-to-end run with real `OPENAI_API_KEY` + spaceship-titanic data. The
  server starts and the protocol shape is right, but a full plan/code/exec
  cycle has not been exercised. The user said the API key is in env; running
  one full cycle will cost a few cents in tokens.

## Open questions for the user

1. **Kaggle credentials.** They sent `KGAT_6c248b1d98a39a0e233a8ee54557d2cc`.
   That looks like a single token, but Kaggle's CLI needs `username` + `key`
   in `~/.kaggle/kaggle.json`. Ask them to confirm or to provide the
   `kaggle.json` contents. Workaround: they can manually download the dataset
   from kaggle.com and unzip it into `./data/spaceship-titanic/`.
2. **Whether to implement the validation handshake** before first submission.
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

1. **Iterate the coder prompt on spaceship-titanic.** The model hint that
   moves the needle most is "engineer features from Cabin (deck/num/side),
   parse PassengerId group, and stack LightGBM + CatBoost with stratified
   5-fold CV". Add competition-specific hints conditional on
   `competition_id` if you can detect it from `description.md`.
2. **Bump `MAX_DEBUG_ITERS` to 5+** when wall-clock budget allows; pair with
   `REASONING_EFFORT=high`.
3. **Implement the validate handshake** (see § "What is NOT done yet" #3).
4. **Add CV-image variant** for image competitions.
5. **Self-consistency / ensembling.** Run 3 distinct solution drafts in
   parallel (different seeds / model families) and majority-vote / average
   probabilities for the final submission.
