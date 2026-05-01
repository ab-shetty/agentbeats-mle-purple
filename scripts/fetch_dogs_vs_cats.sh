#!/usr/bin/env bash
# Download the dogs-vs-cats-redux-kernels-edition dataset for local testing.
#
# Auth: either ~/.kaggle/kaggle.json OR KAGGLE_API_TOKEN env var (new
# Kaggle CLI 2.x format — single opaque token starting with `KGAT_`).
#
# You must have accepted the competition rules at
#   https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/rules
# before this download will succeed.

set -euo pipefail
DATA_DIR="${1:-./data/dogs-vs-cats-redux}"
COMP="dogs-vs-cats-redux-kernels-edition"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 1
fi

kaggle competitions download -c "$COMP" -p .
unzip -o "${COMP}.zip"
rm -f "${COMP}.zip"
# The competition ships train.zip / test.zip nested archives.
[[ -f train.zip ]] && unzip -oq train.zip && rm -f train.zip
[[ -f test.zip  ]] && unzip -oq test.zip  && rm -f test.zip

cat > description.md <<'MD'
# Dogs vs. Cats Redux: Kernels Edition

Distinguish images of dogs from cats. This is the binary image classification
companion to the original Dogs vs. Cats competition.

## Files

- `train/` — 25,000 labeled JPGs named `cat.<id>.jpg` or `dog.<id>.jpg`.
- `test/`  — 12,500 unlabeled JPGs named `<id>.jpg`.
- `sample_submission.csv` — Submission format with columns `id,label`, where
  `label` is the predicted probability that the image is a **dog**.

## Evaluation

Submissions are scored on log loss between the predicted probability and the
true class label (0 = cat, 1 = dog). Lower is better. Predictions are clipped
to `[1e-15, 1 - 1e-15]` server-side.

## Submission format

A CSV with header `id,label` and one row per test image. `id` matches the
filename stem (e.g. `1` for `1.jpg`).
MD

echo "Done. Data dir: $DATA_DIR"
ls -la "$DATA_DIR" | head -20
