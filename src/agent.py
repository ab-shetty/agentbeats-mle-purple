"""AIDE-style ML engineering agent for MLE-Bench.

Loop:
  1. Inspect: read description.md, sample_submission.csv, list data dir,
     peek at train/test if tabular.
  2. Plan: gpt-5-mini drafts a JSON plan (modality, task type, target, metric, models).
  3. Code: gpt-5-mini writes a complete solution.py, branching on modality
     (tabular | text | image | other).
  4. Execute: run solution.py in a subprocess, capture stdout/stderr.
  5. Debug: if execution failed or submission.csv is malformed, feed the
     error back and ask for a revised solution.py. Repeat up to N times.
  6. Return path to the best submission.csv.

Universal image: pandas/sklearn/lightgbm/xgboost (tabular) +
torch/torchvision/timm (image) + transformers (text). CPU-only execution.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import openai_client

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_PLANNER = """You are a senior Kaggle Grandmaster planning a solution for an MLE-bench competition.

You will receive:
  - The competition description (Kaggle-style)
  - A listing of the data directory (files / subdirectories)
  - The header + first rows of train.csv / test.csv (if tabular)
  - The header of sample_submission.csv

Output a JSON object (and ONLY a JSON object, no prose) with these keys:
  modality:         one of ["tabular","text","image","audio","other"]
  task_type:        one of ["binary_classification","multiclass_classification","multilabel_classification","regression","segmentation","other"]
  target_column:    the name of the target column in train.csv (or null if the
                    target is implicit, e.g. image folder structure)
  id_column:        the column from sample_submission.csv that identifies test rows
  eval_metric:      the official metric (e.g. "accuracy","logloss","rmse","auc","map@5","f1","logloss_multiclass")
  is_lower_better:  true|false
  data_layout:      one short sentence describing where the train/test inputs live
                    (e.g. "train.csv with text column 'comment_text'", or
                     "train/<class>/*.jpg + test/*.jpg, label encoded by folder")
  feature_notes:    short bullet list of feature-engineering ideas specific to this dataset
  model_plan:       short bullet list of model approaches in priority order, with
                    rough wall-clock guesses (CPU only, total budget 30 min)

Be concise. Be correct about column names, metric direction, and modality."""


SYSTEM_PROMPT_CODER = """You are a senior Kaggle Grandmaster writing a single self-contained Python script that solves an MLE-bench competition.

GENERAL CONSTRAINTS:
  - Output ONLY a Python code block delimited by ```python ... ``` and nothing else.
  - Read data from os.environ['DATA_DIR']. Write predictions to os.environ['OUTPUT_PATH'].
  - Output CSV MUST exactly match the schema (column names + ordering) of sample_submission.csv
    and contain a row for every test row, in the same order as sample_submission.csv when possible.
  - Print a short summary line "CV score: <number>" if you compute cross-validation.
  - Wrap the script in `if __name__ == "__main__": main()` style.
  - Execution is CPU-only. Total wall-clock budget: ~30 minutes (subprocess timeout enforced).
    Plan training to finish in ~20 min and leave time for inference + writing the CSV.
  - If the previous attempt produced an error or invalid submission, fix the root cause —
    do not just paper over with try/except.

AVAILABLE LIBRARIES:
  - Always: pandas, numpy, scikit-learn, lightgbm, xgboost, scipy.
  - Image / CV: torch (CPU), torchvision, timm, PIL, cv2 (opencv-python-headless).
  - Text / NLP: transformers, scikit-learn TfidfVectorizer.

MODALITY-SPECIFIC GUIDANCE:

TABULAR:
  - Default: LightGBM with stratified 5-fold CV.
  - LightGBM 4.x API: do NOT pass `early_stopping_rounds` or `verbose` as fit() kwargs.
    Use callbacks: `clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])`.
  - For categoricals, prefer pandas `category` dtype + native LightGBM handling
    (`categorical_feature='auto'`) over one-hot.
  - Engineer features specific to the dataset (parse compound IDs, group keys,
    aggregate stats, etc.) before falling back to generic encoders.

TEXT (e.g. jigsaw-toxic-comment-classification):
  - Strong CPU baseline: TF-IDF (word 1-2grams + char 3-5grams, sublinear_tf) +
    one-vs-rest logistic regression. Fast, often medal-zone.
  - For multi-label problems, predict probabilities per class independently and
    output them in the column order of sample_submission.csv.
  - Optional escalation: distilbert-base-uncased via transformers — only if you
    can fit one epoch in <15 min on CPU (usually you cannot for >50k rows; stick
    with TF-IDF then).

IMAGE (e.g. aerial-cactus, dogs-vs-cats, denoising-dirty-documents, whale):
  - Use timm with a pretrained small backbone (e.g. `resnet18`, `mobilenetv3_small_100`,
    `efficientnet_b0`). Replace the head for the target classes.
  - Train on CPU with a SMALL image size (e.g. 96–160 px) and 1–3 epochs to stay
    within budget. Use a DataLoader with num_workers=2 and a sensible batch size (32–128).
  - For binary cactus / dogs-vs-cats: BCEWithLogitsLoss + sigmoid → threshold 0.5
    or output the probability if sample_submission.csv expects floats.
  - For denoising-dirty-documents (image-to-image): a tiny U-Net trained for a few
    epochs at 128–256 px works; output PNGs / pixel arrays per the submission spec.
  - Use torch.set_num_threads(4) to bound BLAS.
  - If timm is too slow, fall back to a from-scratch tiny CNN (3 conv blocks)
    trained briefly — better a finished mediocre submission than a timed-out one.

OTHER:
  - Read the description carefully and synthesize a sensible CPU-friendly approach."""


@dataclass
class StepResult:
    code: str
    stdout: str
    stderr: str
    returncode: int
    submission_ok: bool
    submission_error: str | None
    cv_score: float | None


class MLEBenchAgent:
    def __init__(self, *, workspace: Path, data_dir: Path, instructions: str) -> None:
        self.workspace = workspace
        self.data_dir = data_dir
        self.instructions = instructions

        self.solutions_dir = workspace / "solutions"
        self.solutions_dir.mkdir(exist_ok=True)
        self.submissions_dir = workspace / "submissions"
        self.submissions_dir.mkdir(exist_ok=True)

        self.max_iters = int(os.environ.get("MAX_DEBUG_ITERS", "5"))
        self.subprocess_timeout = int(os.environ.get("SUBPROCESS_TIMEOUT_SEC", "1500"))

        self.description = self._read_text(data_dir / "description.md")
        self.sample_submission = self._read_text(data_dir / "sample_submission.csv", limit_lines=5)
        self.train_preview = self._dataset_preview(data_dir / "train.csv")
        self.test_preview = self._dataset_preview(data_dir / "test.csv")
        self.data_listing = self._list_data_dir(data_dir)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _read_text(path: Path, limit_lines: int | None = None) -> str:
        if not path.exists():
            return ""
        try:
            text = path.read_text(errors="replace")
        except Exception:
            return ""
        if limit_lines is not None:
            text = "\n".join(text.splitlines()[:limit_lines])
        return text

    @staticmethod
    def _dataset_preview(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            with path.open("r", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 6:
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines)
        except Exception:
            return ""

    @staticmethod
    def _list_data_dir(data_dir: Path, max_entries: int = 60) -> str:
        """One-line-per-entry listing of the data dir, recursing one level into
        subdirectories so image-based competitions (train/<class>/*.jpg) are
        legible to the model. Caps to max_entries to keep prompt small."""
        if not data_dir.exists():
            return ""
        lines: list[str] = []
        try:
            top = sorted(data_dir.iterdir())
        except Exception:
            return ""
        for entry in top:
            if len(lines) >= max_entries:
                lines.append("...")
                break
            try:
                if entry.is_dir():
                    children = list(entry.iterdir())
                    sample_names = ", ".join(c.name for c in children[:5])
                    suffix = ", ..." if len(children) > 5 else ""
                    lines.append(
                        f"{entry.name}/  ({len(children)} entries: {sample_names}{suffix})"
                    )
                else:
                    size = entry.stat().st_size
                    lines.append(f"{entry.name}  ({size} bytes)")
            except Exception:
                lines.append(entry.name)
        return "\n".join(lines)

    @staticmethod
    def _extract_code(text: str) -> str:
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1)
        # If model forgot the fence, return as-is.
        return text

    # ------------------------------------------------------------------ steps

    def _make_plan(self) -> str:
        user = (
            "COMPETITION DESCRIPTION:\n"
            f"{self.description[:8000]}\n\n"
            "DATA DIR LISTING:\n"
            f"{self.data_listing}\n\n"
            "SAMPLE SUBMISSION (first lines):\n"
            f"{self.sample_submission}\n\n"
            "TRAIN PREVIEW (first lines, may be empty for non-tabular comps):\n"
            f"{self.train_preview}\n\n"
            "TEST PREVIEW (first lines, may be empty):\n"
            f"{self.test_preview}\n"
        )
        plan_text = openai_client.complete(SYSTEM_PROMPT_PLANNER, user, max_output_tokens=2000)
        # Try to validate JSON; fall back to raw text if it doesn't parse.
        try:
            json.loads(plan_text)
        except Exception:
            # Strip code fences if present.
            stripped = re.sub(r"```(?:json)?\s*|```", "", plan_text).strip()
            try:
                json.loads(stripped)
                plan_text = stripped
            except Exception:
                logger.warning("Planner output is not valid JSON; keeping raw text.")
        logger.info("PLAN:\n%s", plan_text)
        return plan_text

    def _draft_code(self, plan: str, history: list[StepResult]) -> str:
        history_block = ""
        if history:
            last = history[-1]
            history_block = (
                "PREVIOUS ATTEMPT FAILED OR PRODUCED INVALID SUBMISSION.\n"
                f"--- previous code ---\n{last.code[:6000]}\n"
                f"--- stdout (tail) ---\n{last.stdout[-2000:]}\n"
                f"--- stderr (tail) ---\n{last.stderr[-2000:]}\n"
                f"--- submission_error ---\n{last.submission_error or '(none)'}\n"
                "Rewrite the script from scratch, fixing the root cause.\n\n"
            )

        user = (
            f"{history_block}"
            "PLAN (JSON):\n"
            f"{plan}\n\n"
            "COMPETITION DESCRIPTION (truncated):\n"
            f"{self.description[:6000]}\n\n"
            "DATA DIR LISTING:\n"
            f"{self.data_listing}\n\n"
            "SAMPLE SUBMISSION (first lines):\n"
            f"{self.sample_submission}\n\n"
            "TRAIN PREVIEW (may be empty):\n"
            f"{self.train_preview}\n\n"
            "TEST PREVIEW (may be empty):\n"
            f"{self.test_preview}\n\n"
            "Write a complete solution.py. Use os.environ['DATA_DIR'] for the data dir "
            "and os.environ['OUTPUT_PATH'] for the submission CSV path."
        )
        text = openai_client.complete(SYSTEM_PROMPT_CODER, user, max_output_tokens=12000)
        return self._extract_code(text)

    def _execute(self, code: str, iter_idx: int) -> StepResult:
        script_path = self.solutions_dir / f"solution_{iter_idx}.py"
        script_path.write_text(code)
        output_path = self.submissions_dir / f"submission_{iter_idx}.csv"
        env = os.environ.copy()
        env["DATA_DIR"] = str(self.data_dir)
        env["OUTPUT_PATH"] = str(output_path)
        # Avoid the script trying to use a GPU it doesn't have.
        env.setdefault("CUDA_VISIBLE_DEVICES", "")

        stdout, stderr, rc = "", "", -1
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            stderr = f"TimeoutExpired after {self.subprocess_timeout}s: {e}"
            rc = -2
        except Exception as e:
            stderr = f"Subprocess error: {e}"
            rc = -3

        ok, err = self._validate_submission(output_path)
        cv = self._extract_cv(stdout)
        logger.info(
            "Iter %d: rc=%d ok=%s cv=%s err=%s",
            iter_idx, rc, ok, cv, err if not ok else "",
        )
        return StepResult(
            code=code, stdout=stdout, stderr=stderr, returncode=rc,
            submission_ok=ok, submission_error=err, cv_score=cv,
        )

    def _validate_submission(self, output_path: Path) -> tuple[bool, str | None]:
        if not output_path.exists():
            return False, "submission CSV was not created"
        try:
            import pandas as pd

            sample_path = self.data_dir / "sample_submission.csv"
            sub = pd.read_csv(output_path)
            if sample_path.exists():
                sample = pd.read_csv(sample_path)
                if list(sub.columns) != list(sample.columns):
                    return False, (
                        f"column mismatch: got {list(sub.columns)} expected {list(sample.columns)}"
                    )
                if len(sub) != len(sample):
                    return False, f"row count mismatch: got {len(sub)} expected {len(sample)}"
            return True, None
        except Exception as e:
            return False, f"could not parse submission: {e}"

    @staticmethod
    def _extract_cv(stdout: str) -> float | None:
        m = re.search(r"CV score:\s*([-+]?\d*\.?\d+)", stdout)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    # -------------------------------------------------------------------- run

    def run(self) -> Path:
        plan = self._make_plan()
        history: list[StepResult] = []
        best: StepResult | None = None
        best_path: Path | None = None

        for i in range(self.max_iters):
            logger.info("=== iteration %d/%d ===", i + 1, self.max_iters)
            code = self._draft_code(plan, history)
            result = self._execute(code, i)
            history.append(result)

            sub_path = self.submissions_dir / f"submission_{i}.csv"
            if result.submission_ok:
                if best is None or self._is_better(result, best):
                    best = result
                    best_path = sub_path

        if best_path is None:
            # No valid submission at all. As a last-ditch effort, copy
            # sample_submission.csv (if any) so we at least have *something*.
            sample = self.data_dir / "sample_submission.csv"
            fallback = self.submissions_dir / "submission_fallback.csv"
            if sample.exists():
                fallback.write_bytes(sample.read_bytes())
                logger.warning("Falling back to sample_submission.csv")
                return fallback
            # Genuinely nothing — return a path that doesn't exist; the
            # executor will report failure to the green agent.
            return self.submissions_dir / "submission_missing.csv"

        return best_path

    @staticmethod
    def _is_better(a: StepResult, b: StepResult) -> bool:
        # Without metric direction we can't compare CV scores reliably; prefer
        # any valid submission over none, then prefer higher CV (heuristic).
        if a.cv_score is None and b.cv_score is None:
            return False
        if a.cv_score is None:
            return False
        if b.cv_score is None:
            return True
        return a.cv_score > b.cv_score
