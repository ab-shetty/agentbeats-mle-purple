#!/usr/bin/env bash
# Download the spaceship-titanic dataset for local testing.
#
# Auth: either ~/.kaggle/kaggle.json OR KAGGLE_API_TOKEN env var (new
# Kaggle CLI 2.x format — single opaque token starting with `KGAT_`).
#
# After downloading we add a description.md (Kaggle's official task page text)
# so the layout matches what the real green agent serves to the purple.

set -euo pipefail
DATA_DIR="${1:-./data/spaceship-titanic}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 1
fi

kaggle competitions download -c spaceship-titanic -p .
unzip -o spaceship-titanic.zip
rm -f spaceship-titanic.zip

cat > description.md <<'MD'
# Spaceship Titanic

In this competition your task is to predict whether a passenger was transported
to an alternate dimension during the Spaceship Titanic's collision with a
spacetime anomaly.

## Files

- `train.csv` — Personal records for ~8700 passengers, with the binary target
  `Transported` (True/False).
- `test.csv` — Personal records for ~4300 passengers; predict `Transported`.
- `sample_submission.csv` — Submission format with columns `PassengerId` and
  `Transported`.

## Evaluation

Submissions are evaluated on classification accuracy. Higher is better.

## Submission format

A CSV with header `PassengerId,Transported` containing one row per test
passenger. `Transported` must be `True` or `False`.
MD

echo "Done. Data dir: $DATA_DIR"
ls -la "$DATA_DIR"
