"""Генерация и строгая проверка submission.csv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.pipeline import AnalysisPipeline


@dataclass(frozen=True)
class SubmissionResult:
    path: Path
    model_name: str
    rows: int
    min_prediction: float
    max_prediction: float


def validate_submission(
    submission: pd.DataFrame,
    expected: pd.DataFrame,
    target_column: str = "primary_ndvi_true",
) -> None:
    expected_columns = ["anon_polygon_id", "date", target_column]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Неверные колонки submission: {list(submission.columns)}")
    if len(submission) != len(expected):
        raise ValueError(f"Ожидалось {len(expected)} строк, получено {len(submission)}")
    if submission[["anon_polygon_id", "date"]].duplicated().any():
        raise ValueError("В submission есть повторяющиеся anon_polygon_id + date")
    if submission[target_column].isna().any():
        raise ValueError("В submission есть NaN")
    if not np.isfinite(submission[target_column]).all():
        raise ValueError("В submission есть бесконечные значения")

    expected_keys = set(map(tuple, expected[["anon_polygon_id", "date"]].to_numpy()))
    actual_keys = set(map(tuple, submission[["anon_polygon_id", "date"]].to_numpy()))
    if actual_keys != expected_keys:
        raise ValueError("Ключи submission не совпадают с synthetic gaps")


def create_submission(
    input_path: Path,
    train_path: Path | None,
    output_path: Path,
    method: str | None = None,
    target_column: str = "primary_ndvi_true",
    config: AppConfig | None = None,
) -> SubmissionResult:
    pipeline = AnalysisPipeline.from_csv(input_path, train_path, config=config)
    if "is_synthetic_gap" not in pipeline.data.columns:
        raise ValueError("В тестовом датасете отсутствует is_synthetic_gap")

    target_mask = pipeline.data["is_synthetic_gap"].fillna(False).astype(bool)
    target_rows = pipeline.data.loc[target_mask, ["anon_polygon_id", "date"]].copy()
    predictions = pipeline.predict_targets(target_mask, method=method)

    submission = target_rows.copy()
    submission["date"] = submission["date"].dt.strftime("%Y-%m-%d")
    submission[target_column] = predictions.loc[target_rows.index, "prediction"]
    submission = submission.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)

    expected = target_rows.copy()
    expected["date"] = expected["date"].dt.strftime("%Y-%m-%d")
    validate_submission(submission, expected, target_column=target_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    return SubmissionResult(
        path=output_path,
        model_name=str(predictions["model"].iloc[0]),
        rows=len(submission),
        min_prediction=float(submission[target_column].min()),
        max_prediction=float(submission[target_column].max()),
    )
