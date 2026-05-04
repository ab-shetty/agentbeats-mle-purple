# Roadmap — MLE-Bench Purple Agent

This file is the handoff doc. If a fresh agent picks up this work, read this
top-to-bottom before touching code.

## Goal

Build a competitive **purple agent** for the AgentBeats Phase 2 MLE-Bench
benchmark (research agent track). User signed up at agentbeats.dev as
`ashetty21@berkeley.edu`. Target a top-3 finish on at least 4
per-competition sub-leaderboards.

There is no aggregate MLE-Bench leaderboard. Each Kaggle competition has its
**own** sub-leaderboard at
`github.com/RDI-Foundation/MLE-bench-agentbeats-leaderboard`. Top-3 is per
competition, ranked by raw score. Active comps: spaceship-titanic,
aerial-cactus-identification, dogs-vs-cats-redux, right-whale-redux,
jigsaw-toxic-comment-classification, denoising-dirty-documents.

## Current scores

Sub-leaderboard tops (snapshot 2026-05-01) and our latest submitted /
best-local-CV per competition.

| Competition | Top score | Lower better? | Submitted (rank) | Best local CV |
|---|---|---|---|---|
| aerial-cactus-identification | 0.99995 | no | 0.99592 (8th) | 0.998859 ensemble; 0.999383 single iter |
| denoising-dirty-documents | 0.01262 | yes | — | 0.180302 (RMSE; ~14× off top) |
| dogs-vs-cats-redux | — (empty) | yes | 0.03321 (**1st**) | 0.076047 |
| jigsaw-toxic-comment-classification | 0.98113 | no | — | 0.918476 (smoke only) |
| spaceship-titanic | 0.83218 | no | 0.81839 (50th) | 0.788105 |
| icml-2013-whale-challenge | — | — | green-side error: `EOF when reading a line` (likely Kaggle accept-rules `input()` against closed stdin). User cannot accept rules. Blocked. |

## Environment & contracts

### A2A wire protocol
Source of truth: `https://github.com/RDI-Foundation/mle-bench-green/blob/main/src/agent.py`.

Inbound from green:
- `Message.parts[0]` = `TextPart` containing instructions.txt.
- `Message.parts[1]` = `FilePart(FileWithBytes)` of `competition.tar.gz`
  (extracts to `home/data/{description.md, train.csv, test.csv, sample_submission.csv, ...}`).

Outbound to green:
- Optional handshake: `TaskStatusUpdateEvent` whose status message text contains
  `"validate"` AND a `FilePart` of a candidate `submission.csv`. Green replies
  `"Submission is valid"` or `"Error: ..."`. **We do not use this today.**
- Final: `TaskArtifactUpdateEvent` whose artifact has a `FilePart` of
  `submission.csv`. Green base64-decodes `file.bytes` and grades it.
- Final task state: `TaskState.completed`.

Green's httpx client `timeout=3600` (1h). Quick Submit GH Actions runner has
a configurable `RESULTS_TIMEOUT_MINUTES` (defaults to 30 in the workflow but
each leaderboard repo overrides via `vars.QUICK_SUBMIT_TIMEOUT_MINUTES`).
**For MLE-bench the cap is far higher than 30 min** — see "Walltime cap"
below.

### Submission paths
- **Quick Submit (PR-based via agentbeats.dev)** — `quick-submit-runner.yml`.
  This is the path we target.
- **Run Scenario (push `scenario.toml` to a fork)** — `run-scenario.yml`,
  inherits GitHub Actions' ~6 h job default. With Quick Submit's effective
  cap also in the multi-hour range on MLE-bench, there's little reason to
  ship a separate "long mode" today.

### Walltime cap (empirical)
Workflow source has `RESULTS_TIMEOUT_MINUTES` defaulting to `30` but with
`vars.QUICK_SUBMIT_TIMEOUT_MINUTES` override. Repo variables aren't readable
via API by non-collaborators, so the *actual* cap on MLE-bench has to be
inferred from successful run durations. Snapshot 2026-05-03 — most recent 20
Quick Submit runs on `MLE-bench-agentbeats-leaderboard`:

| Duration | Conclusion |
|---:|---|
| 4–13 min | mix of pass / fail (small tasks or early failure) |
| 29 min | success |
| 84, 86 min | success |
| 202–213 min | success (3 separate runs, ~3.5h each) |
| 263 min | success (~4.4h) |

Working assumption: **MLE-bench Quick Submit accepts at least 4 hours, likely
6 hours.** Our prior "30-min cap → short-mode" framing was based on the
workflow default, not the leaderboard's actual override. We were leaving
~90% of the available budget on the table.

### Grading runner (Quick Submit)
- `runs-on: ubuntu-latest`: 4 vCPU, 16 GB RAM, ~14 GB disk, **no GPU**.
- Total agent runtime is roughly
  `MAX_DEBUG_ITERS × SUBPROCESS_TIMEOUT_SEC + OpenAI calls + tarball IO +
  setup overhead (image pull, container start, ~3-5 min)`. With a ~4h cap
  there is plenty of room for proper training; the limiting factor is now
  per-iter wall time, not total iterations.
- Sensible per-modality defaults (set in the AgentBeats config block of the
  manifest if you want to override the runtime defaults below):
  - Tabular: `MAX_DEBUG_ITERS=5, SUBPROCESS_TIMEOUT_SEC=900` (15 min/iter).
    LightGBM + XGBoost + MLP fit comfortably in 5–10 min.
  - Image: `MAX_DEBUG_ITERS=4, SUBPROCESS_TIMEOUT_SEC=1800,
    REFINE_TIMEOUT_SEC=2700` (30 min initial, 45 min refinement). Enough to
    train a real CNN for a few epochs at 224 px.
  - Text: `MAX_DEBUG_ITERS=4, SUBPROCESS_TIMEOUT_SEC=1500`. TF-IDF + LR is
    minutes; a transformer fine-tune fits if the dataset is small.

### Image / manifest
- Image must be `linux/amd64`, pushed to GHCR, package set Public.
- Manifest expects port 8080 and a single A2A endpoint.
- `OPENAI_API_KEY` passed via `${config.openai_api_key}` (`secret: true`).
- `OPENAI_MODEL`, `MAX_DEBUG_ITERS`, `SUBPROCESS_TIMEOUT_SEC`,
  `REFINE_TIMEOUT_SEC`, `REASONING_EFFORT` are config knobs.
- Pin to digest at submission time (`@sha256:...`).
- `a2a-sdk` pinned `>=0.3.20,<1.0` — pip's resolver picks `1.0.x` by default
  which has a breaking API change (no `a2a.server.apps`). Don't bump unless
  you also rewrite the imports.
- `A2A_MAX_CONTENT_LENGTH` raised from 10 MiB SDK default to 512 MiB; for
  840 MB dogs-vs-cats tarball use 2 GiB locally.

## What is built

| Path | Notes |
|---|---|
| `src/server.py` | A2A Starlette app, port 8080, agent card. Honors `A2A_MAX_CONTENT_LENGTH`. |
| `src/executor.py` | Extracts tarball to workspace, calls agent, emits FilePart artifact. |
| `src/agent.py` | Plan → Code → Execute → Debug loop. Universal modality branching (tabular / text / image / other). Includes: structured dataset profiling, target-aware EDA (MI scores, missing-value patterns, group-target purity for GroupKFold detection, image class balance), tabular leakage/dtype guidance, repair flow, self-consistency with diversity directive, refinement loop (after `target_n` valid drafts, remaining iters refine the best in-place), CV-based ensemble filter, separate `REFINE_TIMEOUT_SEC` for refinement iters. |
| `src/openai_client.py` | Responses API wrapper, `OPENAI_MODEL` env. |
| `scripts/local_test.py` | Mocked-green driver. |
| `scripts/fetch_*.sh` | Kaggle CLI wrappers for spaceship-titanic, dogs-vs-cats. |
| `scripts/run_eval_quick.sh`, `run_eval_sweep.sh` | Local eval drivers. |
| `Dockerfile` | python:3.12-slim + CPU torch wheels (separate layer), exposes 8080. |
| `amber-manifest.json5` | Image ref placeholder — update before submit. |
| `requirements.txt` | a2a-sdk pinned `>=0.3.20,<1.0`; openai, pandas, numpy, sklearn, lightgbm, xgboost, timm, transformers, pillow, opencv-headless, kaggle. |

## Open questions for the user

1. **Implement the validation handshake?** (~half day.) In `executor.py`,
   between iterations send `TaskStatusUpdateEvent` with text `"validate"` +
   `FilePart` of candidate CSV; wait for green's `"Submission is valid"` /
   `"Error: ..."`. Reduces 0-score risk from schema bugs.
2. ~~Invest in a "long mode" profile for the fork-push Run Scenario path?~~
   **Moot as of 2026-05-03** — Quick Submit accepts ~4h runs already (see
   "Walltime cap" above). Same code can do "long mode" through Quick Submit
   by raising the config-level budgets.

## Next priorities (ranked)

1. **Tabular: improve spaceship-titanic.** Currently 50th @ 0.81839; top
   0.83218. Likely lifts: better CV strategy (the new EDA already produces
   GroupKFold from `pct_groups_with_constant_target`), richer feature
   engineering from `description.md`, ensembling LightGBM + XGBoost + small
   MLP. Keep recipes generic / task-derived.
2. **Image: re-submit aerial-cactus.** Local CV 0.998859 ensemble of 4
   refined drafts beats current submitted 0.99592 (8th). Free climb if a
   leaderboard re-submission goes through.
3. **Refinement prompt: allow approach switching.** Today `_refine_code`
   biases toward tweaking the current solution. When the current approach
   plateaus or regresses (see denoising 0.180 → 0.21 → 0.26 → 0.29), the right
   move is often to switch to a different `model_plan` entry. Pass the full
   `model_plan` into the refine prompt; permit swapping when the last
   refinement regressed. Generic / task-derived — no competition recipes.
4. **Image: re-test denoising at high budget.** With ~4h cap (not 30 min), a
   real supervised CNN trained on the 144 (noisy, clean) pairs is feasible.
   Aim for the leaderboard top of 0.01262 RMSE.
5. **Implement the validation handshake** (see open question 1).
6. **Stacking meta-learner in `_ensemble`.** Currently averages numeric
   columns, majority-votes the rest. Have refined drafts emit OOF predictions
   to a known path so a Ridge/LogReg meta-learner can stack them.
7. **Tune self-consistency / draft success rate.** The filter + refinement
   mechanisms work; the limiting factor on tabular is now that often only 1
   valid draft survives the iter budget, so the filter and ensemble never
   really engage.

## Submission flow

GitHub user `ab-shetty` (Kaggle: `abhishek1shetty`).
Repo: `agentbeats-mle-purple` → image `ghcr.io/ab-shetty/agentbeats-mle-purple`.

1. **Push to GitHub.** Publish workflow auto-builds + pushes to GHCR on every
   push to main (linux/amd64).
2. **Make the package public** (one-time) at
   `https://github.com/ab-shetty?tab=packages`.
3. **Update `amber-manifest.json5`** with the pinned digest from the workflow
   job summary (`ghcr.io/ab-shetty/agentbeats-mle-purple@sha256:<DIGEST>`).
   Commit + push.
4. **Submit on agentbeats.dev** via Quick Submit:
   - Manifest URL:
     `https://raw.githubusercontent.com/ab-shetty/agentbeats-mle-purple/main/amber-manifest.json5`.
   - Pick the MLE-Bench leaderboard + a competition.
   - Paste the encrypted `openai_api_key` secret.
   - Optional Config JSON. With the empirical ~4h cap, sensible per-modality
     starting points (override what the runtime defaults set):
     - Image: `{"max_debug_iters":"4","subprocess_timeout_sec":"1800","refine_timeout_sec":"2700","reasoning_effort":"high"}`
     - Tabular: `{"max_debug_iters":"5","subprocess_timeout_sec":"900","refine_timeout_sec":"1200","reasoning_effort":"high"}`
     - Text: `{"max_debug_iters":"4","subprocess_timeout_sec":"1500","refine_timeout_sec":"2100","reasoning_effort":"high"}`
   - Runner forks the leaderboard repo, opens a PR, writes the result JSON.

## How to test (10-min local smoke)

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

Code defaults are `MAX_DEBUG_ITERS=5` and `SUBPROCESS_TIMEOUT_SEC=1500` — for
a 10-min check, override as above. Expect 4–8 min total, LightGBM CV
~0.75–0.81, public score around bronze (~0.810).

If it fails:
- Check `agent.py` logs for which iteration crashed.
- Generated `solution_*.py` live in `WORKSPACE_DIR/run_<id>/solutions/`.
- Generated CSVs live in `WORKSPACE_DIR/run_<id>/submissions/`.

For multi-comp eval: `scripts/run_eval_quick.sh` (spaceship + cactus, ~25 min).

## Changelog

### 2026-05-04 — silent-failure submission + hardening fixes

Two real submissions failed catastrophically. Root cause and fixes shipped.

**What happened:**
- Jigsaw run: scored **0.5 AUC** (chance-level — fell back to
  `sample_submission.csv` after iter 0 produced nothing).
- Denoising run: **`Status: failed — Agent did not submit a valid
  submission.csv`** (no submission produced after 3 iters).

**Root cause (forensics from run logs):**
1. Quick Submit config did not pass `openai_model`. Manifest's
   `OPENAI_MODEL: { when: "config.openai_model", value: "..." }` only sets
   the env IF the user supplied it; otherwise `openai_client.complete`
   defaulted to `gpt-5-mini`. Both real runs ran the wrong model.
2. Across all iters of both runs, the subprocess "completed" in
   **11–13 ms** with `rc=0` and no submission CSV. Impossible for any real
   Python startup (cold start is ~30 ms minimum). Diagnosis: gpt-5-mini
   under our prompt was emitting prose / partial output without a
   ```python``` code block. `_extract_code` fell through to its raw-text
   return path, wrote markdown to `solution_*.py`, Python evaluated it
   trivially and exited clean.
3. Self-review pass never fired (no `Self-review (...)` log lines). The
   silent branch in `_review_code` is `if not code or not code.strip():
   return code` — an empty `code` from `_extract_code` skipped review with
   no log, so the silent failure had no in-process check.
4. With every iter producing nothing, the loop fell through to either the
   `sample_submission.csv` fallback (jigsaw, scored 0.5) or no submission
   at all (denoising, ran out of iters before a fallback path was
   reachable, since iter pool was empty).

**Fixes shipped (this commit):**
- `src/agent.py` `_extract_code` — never returns raw markdown. Returns
  fenced code if present (```python``` or bare ```), otherwise empty
  string. Also recognizes bare ``` fences whose contents contain
  python-shape tokens (`import`, `def`, `class`, `from`).
- `src/agent.py` `_execute` — short-circuits on empty/whitespace code with
  `rc=-4` and a clear stderr message. `_classify_failure` already routes
  any `rc != 0 / != -2` to `execution_bug`, so the next iter runs the
  repair flow instead of silently re-executing nothing.
- `src/openai_client.py` — default model `gpt-5-mini` → `gpt-5.4` so an
  unset `openai_model` config doesn't silently downgrade.
- `src/server.py` — agent-card name uses `os.environ.get('OPENAI_MODEL',
  'gpt-5.4')` instead of hardcoded "gpt-5-mini" string. Logs now reflect
  what's actually running.

**Submission steps:**
1. Push → publish workflow rebuilds image to GHCR.
2. Update `amber-manifest.json5` with new `@sha256:` digest from the
   workflow job summary.
3. Resubmit. With these defaults, even an empty config JSON (just
   `openai_api_key`) ships gpt-5.4 + total_budget=14400.

### 2026-05-03 — planner-ping diagnostic + adaptive iteration plan

Investigation triggered by gpt-5 high-effort regression on spaceship-titanic
(0.81839 → 0.80115, 50th → 117th). Built diagnostic improvements and tested
v2 on a smoke run, then attempted long-runs on denoising + jigsaw.

**v2 changes shipped (`src/agent.py`, `requirements.txt`):**
- Added `catboost` + `optuna` to deps + AVAILABLE LIBRARIES.
- TABULAR section: explicit CatBoost NaN-in-cat-features footgun, XGBoost
  `enable_categorical`, OOF-save directive, Optuna budget cap.
- `SYSTEM_PROMPT_DIVERSITY_DIRECTIVE`: explicit per-modality rotation order
  (Tabular: LGBM → CatBoost → XGBoost → HistGB. Text: TF-IDF+LR → +LinearSVC →
  +SGD. Image: timm backbone → different same-size backbone). Inline CatBoost
  snippet with NaN handling.
- `SYSTEM_PROMPT_REFINE`: replaced "WHERE TO ACTUALLY GET LIFT" with 5
  concrete TACTICS (Optuna HP, OOF stacking, FE, Image TTA, threshold tuning).
- New `SYSTEM_PROMPT_REVIEW` + `_review_code` method: bounded LLM call before
  every execute (draft / refine / repair). 10 critical bug categories
  (CatBoost NaN, LGBM 4.x API, XGB enable_categorical, CV leakage, schema
  mismatch, bool dict, dataloader None, OUTPUT_PATH, CV print, hardcoded
  paths). Output contract: `NO_ISSUES` or full repaired Python block. Gated
  by `SELF_REVIEW=1` env (default on).
- `OOF_PATH` env var wired in `_execute` (one CSV per iteration in
  `oofs_dir`). Stacking meta-learner not yet implemented.

**v2 smoke validation (spaceship-titanic, gpt-5-mini):**
| Iter | CV | Notes |
|---|---|---|
| 0 (LGBM) | 0.807 | self-review patched draft (14965→15392 chars) |
| 1 (CatBoost/XGB) | 0.814 | refine no longer crashes on NaN-in-cat |
| 2 (refine) | 0.812 | self-review patched (15938→16122) |

Ensemble of 3 vs baseline ensemble of 2. End-to-end loop now resilient.

**Long-run attempt 1 (denoising + jigsaw, gpt-5-mini, parallel):**
- denoising: `MAX_DEBUG_ITERS=4 SUBPROCESS_TIMEOUT_SEC=1800 REFINE_TIMEOUT_SEC=2700`
- jigsaw:    `MAX_DEBUG_ITERS=4 SUBPROCESS_TIMEOUT_SEC=1500 REFINE_TIMEOUT_SEC=2100`

Both iter 0 timed out: `rc=-2 ok=False cv=None err=submission CSV was not created`.
- Denoising: U-Net training + sliding-window inference on 144 train + 72 test
  exceeded 1800s on CPU.
- Jigsaw: TF-IDF (word 1-3 + char 3-6) + 6×OvR LogReg with 5-fold CV on 159k
  rows exceeded 1500s.

Killed both, archived logs as `*.log.run1`. Lesson: subprocess timeouts must
match what the planner actually estimates (planner said 20-45 min for jigsaw,
we gave 25). Planner-ping diagnostic confirmed gpt-5-mini picks the right
families (TF-IDF + per-label LR for jigsaw; small U-Net for denoising) — the
planner is not the bottleneck. The bottleneck is fixed iteration budget +
no adaptive strategy.

**Next changes (in flight):**
- (A) Add `iteration_strategy` to planner JSON: `n_drafts` (1-3), `do_refine`,
  `do_ensemble`, `rationale`. Loop honors planner's call instead of always
  burning `MAX_DEBUG_ITERS`. When `model_plan[0]` is a strong well-known
  recipe likely correct first try, planner sets `n_drafts=1` and saves
  iters / API spend.
- (C) Coder prompt tightened: must NOT exceed `EXECUTION BUDGET`. If
  `model_plan[0]`'s estimate exceeds budget, pick a leaner version of the
  same approach.

`scripts/planner_ping.py` added: stub-input diagnostic that runs only
`_make_plan` against a comp dir, no execution. Used to validate planner
output without burning subprocess time.

### 2026-05-03 — high-budget denoising re-test + bug fixes
Two runs on `denoising-dirty-documents` with the new high-budget defaults
(`MAX_DEBUG_ITERS=4`, `SUBPROCESS_TIMEOUT_SEC=1800`,
`REFINE_TIMEOUT_SEC=2700`, `REASONING_EFFORT=high`).

Run 1 (`eval_runs/20260503T051203Z_denoising_high/`, 1035s): cv=**0.085307**
on iter 1. Big jump over the 0.180 from the small-budget run. But two bugs
found:
- Planner returned **empty output** under `reasoning_effort=high` —
  `max_output_tokens=2000` was eaten by reasoning. Without a plan,
  `is_lower_better` stayed `None`.
- `_filter_for_ensemble` and `_cv_score_for_sort` treated
  `is_lower_better=None` as falsy → silent higher-better assumption. Filter
  kept the worse 0.116 draft alongside the 0.085 best, and the refinement
  loop kept telling iter 2 it was the "best" because sort negation never
  fired.

Bug fixes shipped in same commit:
- Planner `max_output_tokens` 2000 → 6000.
- `_filter_for_ensemble`: returns just the last valid draft and warns when
  `is_lower_better is None`.
- `_cv_score_for_sort`: requires `is_lower_better is True` before negating.

Run 2 with bug fixes (`eval_runs/20260503T053122Z_denoising_high/`, 3793s):
- Planner correctly emitted JSON; `model_plan[0]` was a small U-Net trained
  on (noisy, clean) pairs (the planner reframe is working).
- Coder implemented the U-Net.
- Iter 0: U-Net at 25 min CPU → cv=**0.288932** (worse than the prior tuned
  classical run's 0.085).
- Iters 1, 3 failed; iter 2 reproduced 0.288. Ensemble of 2 (both 0.288).

**Lesson — planner can't accurately predict CPU performance.** "Best by
expected score" ranked U-Net over classical, but on this hardware × dataset
(144 train pairs, ~25 min CPU) a tuned classical baseline (NL-Means with
per-image h-search + linear regressor on noise-stat features) was actually
better. Mitigation in same commit: bumped image `target_n` from 1 → 2 in
`_self_consistency_target` so a diverse-family second draft can catch this.

### 2026-05-03 — discovered actual Quick Submit cap is ~4h, not 30 min
Empirical: the most recent 20 Quick Submit runs on
`MLE-bench-agentbeats-leaderboard` include multiple successes at 84, 86,
202, 207, 212, 213, and 263 minutes. The workflow's `RESULTS_TIMEOUT_MINUTES`
defaults to `30` but the leaderboard repo has overridden
`vars.QUICK_SUBMIT_TIMEOUT_MINUTES` to a much higher value (probably ~360 min,
matching GitHub Actions' default 6h job cap).

Implications:
- Our previous "short-mode-only, tune for 30 min" framing was wrong; we were
  using ~10% of available wall-clock.
- The "long mode for fork-push" plan is moot — Quick Submit *is* long mode.
- Image and text modalities can now train real models (CNN multi-epoch,
  transformer fine-tune on small data), not just thin baselines.
- Per-modality default budgets bumped (see "Grading runner" above and
  manifest config examples in the submission flow).

Code defaults raised in this commit (`src/agent.py`):
- `SUBPROCESS_TIMEOUT_SEC` default: 1500 → 1800.
- `MAX_DEBUG_ITERS` default: 5 → 5 (unchanged; per-iter timeout is the lever).
- `REFINE_TIMEOUT_SEC`: stays `min(1.5 × SUBPROCESS_TIMEOUT_SEC, 1800)`
  formula, but the `1800` ceiling lifted to `3600`.

### 2026-05-03 — denoising-dirty-documents end-to-end
`eval_runs/20260503T043817Z_denoising/`. `MAX_DEBUG_ITERS=4`,
`SUBPROCESS_TIMEOUT_SEC=600`, `REFINE_TIMEOUT_SEC=900`.

| Iter | CV (RMSE) | Path |
|---|---|---|
| 0 | **0.180302** | initial draft |
| 1 | 0.20778 | refinement regressed |
| 2 | 0.256561 | refinement regressed |
| 3 | 0.286462 | refinement regressed |

Filter kept 1/4 (only iter 0 within `best * 1.10 = 0.198` for lower-better) →
fell through to best_single → returned iter 0's 14.23M-row submission. Total
1001s. **First live engagement of `_filter_for_ensemble`** — without it,
naive averaging of all 4 would have produced ~0.23 RMSE.

CV 0.180 is ~14× off the leaderboard top of 0.01262. Iter 0 picked the
cheapest `model_plan` option (classical Gaussian background subtract +
fastNlMeans + median + morphological + blend); refinements stayed inside that
classical pipeline instead of climbing to plan option 2 (supervised CNN on
`(noisy, clean)` pairs). Motivates next-priority #3 (refinement prompt should
allow approach switching).

### 2026-05-03 — refinement budget knob shipped
New env var `REFINE_TIMEOUT_SEC` (default `min(1.5 × SUBPROCESS_TIMEOUT_SEC,
1800)`) threaded into `_execute` only on the refinement branch via optional
`timeout` arg. Initial-draft and repair iters still use `SUBPROCESS_TIMEOUT_SEC`,
so refinement gets headroom (TTA / larger backbones) without spending more on
exploration. Refine prompt's EXECUTION BUDGET hint now reads `refine_timeout`.
Exposed in `amber-manifest.json5` as `config.refine_timeout_sec`.

Live test on aerial-cactus (`eval_runs/20260503T042202Z_refine/`,
`MAX_DEBUG_ITERS=4`, `SUBPROCESS_TIMEOUT_SEC=600`, `REFINE_TIMEOUT_SEC=900`):

| Iter | CV | Path |
|---|---|---|
| 0 | 0.997475 | initial |
| 1 | 0.998859 | refinement (+0.0014) |
| 2 | 0.998618 | refinement |
| 3 | 0.998842 | refinement |

Total 522s. All 4 drafts within filter epsilon → ensemble of 4. No timeouts.

### 2026-05-03 — ensemble filter shipped
`MLEBenchAgent._filter_for_ensemble` (`src/agent.py`) drops drafts whose CV is
meaningfully worse than the best draft before they reach `_ensemble`. If
filtering leaves <2 drafts, falls through to the existing best-single path.

Thresholds:
- Higher-better: keep `cv >= best - max(0.005, 0.10 * (1 - best))`. At
  best=0.999 the gap tightens to 0.005 (drops a 0.991 draft); at best=0.79
  the gap is ~0.021 (keeps a 0.78 draft, drops a 0.50 draft).
- Lower-better: keep `cv <= best * 1.10` (10% relative slack; tolerates
  noise on small log-loss values).
- `cv_score=None` drafts are dropped.

Motivated by 2026-05-03 quick validation
(`eval_runs/20260503T004906Z_quick/`): spaceship had `cv=0.78/0.79` averaged
with `cv=0.496` (near majority baseline 0.50); cactus had `cv=0.999` averaged
with `cv=0.991`.

### 2026-05-03 — tabular tuning + richer EDA profile
1. **Tabular tuning.** `_self_consistency_target` returns 2 for tabular (was
   4) and 2 for text (was 3). `SYSTEM_PROMPT_DIVERSITY_DIRECTIVE` rewritten:
   pick exactly ONE axis to vary (model family); explicitly KEEP the previous
   draft's feature pipeline, encoders, CV split, target encoding, file
   handling. Removed the "substantially different" framing that caused
   gpt-5-mini to emit ambitious, brittle rewrites.
2. **Richer static EDA in `_dataset_profile`.** New `_target_aware_eda`
   section (no LLM in the loop), added before the planner runs:
   - inferred target column (train ∖ test, last-column tiebreak),
   - target distribution + majority-class baseline,
   - top features by mutual information (sklearn
     `mutual_info_classif`/`_regression` on a 2000-row sample, after pruning
     text blobs and likely-id columns),
   - top missing-value co-occurrence patterns,
   - compound-ID group statistics including
     `pct_groups_with_constant_target` (directly tells the planner whether
     to pick GroupKFold over StratifiedKFold),
   - image class-balance from the labels CSV when applicable.

Quick validation (`eval_runs/20260503T004906Z_quick/`): spaceship planner
quoted `pct_groups_with_constant_target=0.876` and picked GroupKFold; cactus
iter 0 hit cv=0.999383 (single-iter score above prior leaderboard top
0.99995).

### 2026-05-02 — refinement loop added
After `target_n` diverse drafts succeed, remaining iters call `_refine_code`
on the current best valid draft. Refined drafts join `valid` and participate
in the ensemble. Image runs benefit most since `target_n=1` for image — every
iter after the first is a refinement pass.

### 2026-05-01 — initial submitted results
Submitted via Quick Submit on agentbeats.dev:

| Competition | Rank | Score | Total duration |
|---|---:|---:|---:|
| spaceship-titanic | 50th | 0.81839 | 6m 31s |
| aerial-cactus-identification | 8th | 0.99592 | 10m 41s |
| dogs-vs-cats-redux-kernels-edition | **1st** | 0.03321 | 29m 9s |
| icml-2013-whale-challenge | — | — | green-side error |

Dogs-vs-cats placed 1st on an empty sub-leaderboard, ran right against the
30-min Quick Submit cap.
