"""Загрузка и нормализация конкурсных датасетов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COMMON_COLUMNS = {
    "anon_polygon_id",
    "date",
    "primary_ndvi",
    "crop_type",
}


def load_dataset(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}")

    frame = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COMMON_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"В {path} отсутствуют колонки: {sorted(missing)}")

    frame = frame.copy()
    frame["anon_polygon_id"] = frame["anon_polygon_id"].astype(str)
    frame["crop_type"] = frame["crop_type"].astype(str)
    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["doy"] = frame["date"].dt.dayofyear.astype("int16")

    if frame.duplicated(["anon_polygon_id", "date"]).any():
        raise ValueError(f"В {path} есть дубликаты anon_polygon_id + date")

    return frame.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)


def clean_primary_series(series: pd.Series) -> pd.Series:
    """Удаляет только заведомо нечисловые значения, сохраняя исходный масштаб."""

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.replace([np.inf, -np.inf], np.nan)


def combine_context(current: pd.DataFrame, reference: pd.DataFrame | None) -> pd.DataFrame:
    """Объединяет текущий ряд с историческим reference без дубликатов."""

    current_part = current.copy()
    current_part["_dataset"] = "current"
    current_part["_current_row"] = np.arange(len(current_part), dtype=int)

    if reference is None:
        return current_part

    reference_part = reference.copy()
    reference_part["_dataset"] = "reference"
    reference_part["_current_row"] = -1

    columns = sorted(set(current_part.columns) | set(reference_part.columns))
    combined = pd.concat(
        [reference_part.reindex(columns=columns), current_part.reindex(columns=columns)],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        ["anon_polygon_id", "date"], keep="last"
    )
    return combined.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
