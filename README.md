# MLE-Bench Purple Agent (gpt-5-mini)

An A2A-protocol purple agent for the [AgentBeats](https://agentbeats.dev) MLE-Bench
benchmark. Receives a Kaggle-style competition tarball from the green agent,
runs an AIDE-style plan/code/execute/debug loop using `gpt-5-mini`, and returns
`submission.csv` as a task artifact.

## Layout

```
src/
  server.py          # A2A Starlette server (port 8080)
  executor.py        # A2A executor — extracts tarball, runs agent, emits artifact
  agent.py           # AIDE-style plan/code/execute/debug loop
  openai_client.py   # OpenAI Responses API wrapper
scripts/
  local_test.py                # Mocked-green driver
  fetch_spaceship_titanic.sh   # Pull dataset from Kaggle
Dockerfile
amber-manifest.json5
requirements.txt
roadmap.md           # Handoff doc — read this first if continuing the work
```

## Quick start (local 10-minute test)

```bash
# 0. Install
python3.13 -m pip install -r requirements.txt

# 1. Download dataset (needs ~/.kaggle/kaggle.json)
./scripts/fetch_spaceship_titanic.sh ./data/spaceship-titanic

# 2. Boot purple
export OPENAI_API_KEY=sk-...
python -m src.server --host 127.0.0.1 --port 8080

# 3. In another shell, drive it
python scripts/local_test.py --data-dir ./data/spaceship-titanic
```

`local_test.py` saves the returned CSV as `submission.csv` and prints a schema
sanity check against `sample_submission.csv`.

## Tunables (env vars)

- `OPENAI_API_KEY` — required.
- `OPENAI_MODEL` — default `gpt-5-mini`.
- `REASONING_EFFORT` — `low|medium|high`, default `medium`.
- `MAX_DEBUG_ITERS` — number of plan/code/run cycles, default `5`.
- `SUBPROCESS_TIMEOUT_SEC` — per-iteration `solution.py` timeout, default `1500`.
- `A2A_MAX_CONTENT_LENGTH` — inbound JSON-RPC request cap in bytes, default
  `536870912` (512 MiB).

## Submission to AgentBeats

See `roadmap.md` § "Submission steps".
