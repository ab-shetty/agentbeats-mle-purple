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

import csv
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
                    rough wall-clock guesses (CPU only, using the provided execution budget)

Be concise. Be correct about column names, metric direction, modality, and
the actual observed file layout. If image roots are nested, use the detected
paths instead of assuming files live directly under train/ or test/."""


SYSTEM_PROMPT_CODER = """You are a senior Kaggle Grandmaster writing a single self-contained Python script that solves an MLE-bench competition.

GENERAL CONSTRAINTS:
  - Output ONLY a Python code block delimited by ```python ... ``` and nothing else.
  - Read data from os.environ['DATA_DIR']. Write predictions to os.environ['OUTPUT_PATH'].
  - Use the provided dataset profile and actual observed paths. Do not hard-code
    competition names or assume generic train/test roots if the files are nested.
  - Output CSV MUST exactly match the schema (column names + ordering) of sample_submission.csv
    and contain a row for every test row, in the same order as sample_submission.csv when possible.
  - Print a short summary line "CV score: <number>" if you compute cross-validation.
  - Wrap the script in `if __name__ == "__main__": main()` style.
  - Execution is CPU-only. Respect the provided subprocess timeout budget.
    Plan training to finish well inside that budget and leave time for inference + writing the CSV.
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
  - Resolve train/test image roots dynamically from the observed dataset profile
    or by searching under DATA_DIR for filenames from train.csv / sample_submission.csv.
  - Use timm with a pretrained small backbone (e.g. `resnet18`, `mobilenetv3_small_100`,
    `efficientnet_b0`). Replace the head for the target classes.
  - Train on CPU with a SMALL image size (e.g. 96–160 px) and 1–3 epochs to stay
    within budget. Use a DataLoader with num_workers=2 and a sensible batch size (32–128).
  - If images are tiny (for example 64x64 or smaller), strongly prefer a fast
    baseline first: raw pixels / HOG / color statistics or a tiny CNN. Only add
    heavier pretrained feature extraction if the budget comfortably allows it.
  - For larger CPU datasets, prefer a single holdout split or 3-fold CV over 5-fold
    unless the dataset is small enough that 5-fold clearly fits the budget.
  - For binary cactus / dogs-vs-cats: BCEWithLogitsLoss + sigmoid → threshold 0.5
    or output the probability if sample_submission.csv expects floats.
  - For denoising-dirty-documents (image-to-image): a tiny U-Net trained for a few
    epochs at 128–256 px works; output PNGs / pixel arrays per the submission spec.
  - Use torch.set_num_threads(4) to bound BLAS.
  - If timm is too slow, fall back to a from-scratch tiny CNN (3 conv blocks)
    trained briefly — better a finished mediocre submission than a timed-out one.

OTHER:
  - Read the description carefully and synthesize a sensible CPU-friendly approach."""


SYSTEM_PROMPT_REPAIR = """You are a senior Python engineer fixing a bug in an existing ML script.

You will receive:
  - The previous solution.py
  - The captured stdout / stderr from running it (stderr is the source of truth)
  - The validation error if the produced submission CSV was malformed

Your job is to output a REPAIRED full solution.py that fixes the root cause with
the minimum possible diff. Constraints:

  - Output ONLY a Python code block delimited by ```python ... ``` and nothing else.
  - Keep the overall approach (model family, feature pipeline, CV strategy) identical
    unless the bug *is* the approach. This is a repair, not a redesign.
  - Identify the precise line(s) that caused the failure and patch them. Do not
    refactor unrelated code, do not rename functions, do not swap models.
  - Common bug categories to repair correctly:
      * AttributeError / TypeError / NameError → fix the misuse, keep the call site.
      * FileNotFoundError / wrong path → use the actual paths from the dataset profile.
      * DataLoader collate errors when __getitem__ returns None → filter Nones in the
        Dataset, or use a custom collate_fn that drops None, or guarantee a tensor.
      * LightGBM 4.x API errors → use callbacks, not fit() kwargs.
      * Submission schema mismatch → match sample_submission.csv columns and order.
      * NaN / non-finite predictions → clip or fillna before writing CSV.
  - Read data from os.environ['DATA_DIR']. Write predictions to os.environ['OUTPUT_PATH'].
  - Must produce a submission CSV that matches sample_submission.csv exactly.
  - Print "CV score: <number>" if you compute CV (preserve the existing print)."""


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
        self.dataset_profile = self._dataset_profile()
        self.is_lower_better: bool | None = None

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

    def _dataset_profile(self) -> str:
        profile = {
            "train_csv": self._csv_profile(self.data_dir / "train.csv"),
            "test_csv": self._csv_profile(self.data_dir / "test.csv"),
            "sample_submission_csv": self._csv_profile(self.data_dir / "sample_submission.csv"),
            "image_layout": self._image_layout_profile(),
        }
        return json.dumps(profile, indent=2, sort_keys=True)

    @staticmethod
    def _csv_profile(path: Path, max_rows: int = 3) -> dict[str, object]:
        if not path.exists():
            return {"exists": False}

        try:
            with path.open("r", newline="", errors="replace") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                rows = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(row)
        except Exception as e:
            return {"exists": True, "error": str(e)}

        return {
            "exists": True,
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "columns": columns,
            "preview_rows": rows,
        }

    def _image_layout_profile(self) -> dict[str, object]:
        train_ids = self._csv_column_values(self.data_dir / "train.csv", "id")
        test_ids = self._csv_column_values(self.data_dir / "sample_submission.csv", "id")
        image_files = self._sample_image_files(limit=6)

        return {
            "train_root_candidates": self._match_image_roots(train_ids),
            "test_root_candidates": self._match_image_roots(test_ids),
            "sample_images": image_files,
        }

    @staticmethod
    def _csv_column_values(path: Path, column: str, limit: int = 8) -> list[str]:
        if not path.exists():
            return []

        try:
            with path.open("r", newline="", errors="replace") as f:
                reader = csv.DictReader(f)
                values = []
                for row in reader:
                    value = row.get(column)
                    if value:
                        values.append(str(value))
                    if len(values) >= limit:
                        break
                return values
        except Exception:
            return []

    def _match_image_roots(self, filenames: list[str], max_roots: int = 4) -> list[dict[str, object]]:
        if not filenames:
            return []

        counts: Counter[str] = Counter()
        for name in filenames:
            try:
                matches = list(self.data_dir.rglob(name))
            except Exception:
                matches = []
            for match in matches[:3]:
                try:
                    rel_parent = str(match.parent.relative_to(self.data_dir))
                except Exception:
                    rel_parent = str(match.parent)
                counts[rel_parent] += 1

        roots = []
        for path, count in counts.most_common(max_roots):
            roots.append({"path": path, "matched_filenames": count})
        return roots

    def _sample_image_files(self, limit: int = 6) -> list[dict[str, object]]:
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
        items: list[dict[str, object]] = []
        try:
            candidates = (
                p for p in self.data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in image_exts
            )
            for path in candidates:
                rel_path = str(path.relative_to(self.data_dir))
                item: dict[str, object] = {"path": rel_path}
                try:
                    from PIL import Image

                    with Image.open(path) as img:
                        item["size"] = list(img.size)
                except Exception:
                    pass
                items.append(item)
                if len(items) >= limit:
                    break
        except Exception:
            return items
        return items

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
            f"EXECUTION BUDGET (seconds): {self.subprocess_timeout}\n\n"
            "COMPETITION DESCRIPTION:\n"
            f"{self.description[:8000]}\n\n"
            "DATA DIR LISTING:\n"
            f"{self.data_listing}\n\n"
            "STRUCTURED DATASET PROFILE:\n"
            f"{self.dataset_profile}\n\n"
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
        if history and (history[-1].returncode != 0 or not history[-1].submission_ok):
            last = history[-1]
            reason = "TIMED OUT" if last.returncode == -2 else "FAILED OR PRODUCED INVALID SUBMISSION"
            scope_hint = (
                "Previous attempt exceeded the execution budget. Reduce scope:"
                " smaller image size, fewer folds, smaller backbone, or skip TTA/calibration.\n\n"
                if last.returncode == -2
                else "Rewrite the script from scratch, fixing the root cause.\n\n"
            )
            history_block = (
                f"PREVIOUS ATTEMPT {reason}.\n"
                f"--- previous code ---\n{last.code[:6000]}\n"
                f"--- stdout (tail) ---\n{last.stdout[-2000:]}\n"
                f"--- stderr (tail) ---\n{last.stderr[-2000:]}\n"
                f"--- submission_error ---\n{last.submission_error or '(none)'}\n"
                + scope_hint
            )

        user = (
            f"{history_block}"
            f"EXECUTION BUDGET (seconds): {self.subprocess_timeout}\n\n"
            "PLAN (JSON):\n"
            f"{plan}\n\n"
            "COMPETITION DESCRIPTION (truncated):\n"
            f"{self.description[:6000]}\n\n"
            "DATA DIR LISTING:\n"
            f"{self.data_listing}\n\n"
            "STRUCTURED DATASET PROFILE:\n"
            f"{self.dataset_profile}\n\n"
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

    def _repair_code(self, last: StepResult) -> str:
        user = (
            f"EXECUTION BUDGET (seconds): {self.subprocess_timeout}\n\n"
            "DATA DIR LISTING:\n"
            f"{self.data_listing}\n\n"
            "STRUCTURED DATASET PROFILE:\n"
            f"{self.dataset_profile}\n\n"
            "SAMPLE SUBMISSION (first lines):\n"
            f"{self.sample_submission}\n\n"
            "--- previous solution.py (full) ---\n"
            f"{last.code}\n"
            "--- stdout (tail) ---\n"
            f"{last.stdout[-3000:]}\n"
            "--- stderr (tail) ---\n"
            f"{last.stderr[-3000:]}\n"
            "--- submission_error ---\n"
            f"{last.submission_error or '(none)'}\n\n"
            "Output the repaired full solution.py. Minimum diff. Keep the overall approach."
        )
        text = openai_client.complete(SYSTEM_PROMPT_REPAIR, user, max_output_tokens=12000)
        return self._extract_code(text)

    @staticmethod
    def _classify_failure(result: StepResult) -> str:
        """Classify why an iteration failed.

        Returns one of: ok | timeout | schema_bug | execution_bug.
        Routes the next iteration to repair (execution_bug, schema_bug) vs
        full rewrite with reduced scope (timeout) vs nothing (ok).
        """
        if result.returncode == 0 and result.submission_ok:
            return "ok"
        if result.returncode == -2:
            return "timeout"
        if result.returncode == 0 and not result.submission_ok:
            return "schema_bug"
        return "execution_bug"

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
                id_col = sample.columns[0] if len(sample.columns) else None
                if id_col and id_col in sub.columns:
                    if not sub[id_col].astype(str).equals(sample[id_col].astype(str)):
                        return False, f"id/order mismatch in column {id_col}"
                pred_cols = [c for c in sub.columns if c != id_col]
                for col in pred_cols:
                    if sub[col].isna().any():
                        return False, f"column {col} contains NaN values"
                    if pd.api.types.is_numeric_dtype(sub[col]) and not np.isfinite(sub[col].to_numpy()).all():
                        return False, f"column {col} contains non-finite numeric values"
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
        self.is_lower_better = self._plan_is_lower_better(plan)
        history: list[StepResult] = []
        best: StepResult | None = None
        best_path: Path | None = None

        for i in range(self.max_iters):
            logger.info("=== iteration %d/%d ===", i + 1, self.max_iters)
            if history:
                category = self._classify_failure(history[-1])
                logger.info("Previous iter classified as: %s", category)
                if category in {"execution_bug", "schema_bug"}:
                    code = self._repair_code(history[-1])
                else:
                    # timeout or first run after ok (shouldn't happen — ok exits)
                    code = self._draft_code(plan, history)
            else:
                code = self._draft_code(plan, history)
            result = self._execute(code, i)
            history.append(result)

            sub_path = self.submissions_dir / f"submission_{i}.csv"
            if result.submission_ok:
                if best is None or self._is_better(result, best):
                    best = result
                    best_path = sub_path
                if result.returncode == 0:
                    logger.info("Stopping early after successful iteration %d", i + 1)
                    break

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
    def _plan_is_lower_better(plan: str) -> bool | None:
        try:
            parsed = json.loads(plan)
        except Exception:
            return None
        value = parsed.get("is_lower_better")
        return value if isinstance(value, bool) else None

    def _is_better(self, a: StepResult, b: StepResult) -> bool:
        # Without metric direction we can't compare CV scores reliably; prefer
        # any valid submission over none, then prefer higher CV (heuristic).
        if a.cv_score is None and b.cv_score is None:
            return False
        if a.cv_score is None:
            return False
        if b.cv_score is None:
            return True
        if self.is_lower_better is True:
            return a.cv_score < b.cv_score
        return a.cv_score > b.cv_score
