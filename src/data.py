"""Загрузка и нормализация конкурсных датасетов."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config import ExternalDataConfig


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


def load_external_training_data(
    config: "ExternalDataConfig", *, include_when_disabled: bool = False
) -> pd.DataFrame | None:
    """Загружает проверенные external-файлы, не смешивая их с context."""

    if not config.enabled and not include_when_disabled:
        return None
    if not config.paths:
        raise ValueError("Для external data не задан ни один путь")

    frames: list[pd.DataFrame] = []
    for path in config.paths:
        frame = load_dataset(path)
        invalid_ids = ~frame["anon_polygon_id"].str.startswith(
            config.polygon_id_prefix
        )
        if invalid_ids.any():
            example = frame.loc[invalid_ids, "anon_polygon_id"].iloc[0]
            raise ValueError(
                f"External polygon ID {example!r} не начинается с "
                f"{config.polygon_id_prefix!r}"
            )
        missing_crop = frame["crop_type"].isin(["", "nan", "None"])
        frame.loc[missing_crop, "crop_type"] = config.crop_type_fallback
        frame["_data_source"] = f"external:{path.name}"
        frame["_sample_weight"] = config.sample_weight
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.duplicated(["anon_polygon_id", "date"]).any():
        raise ValueError("External-файлы содержат повторяющиеся polygon ID + date")
    return combined.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)


def combine_training_sources(
    competition: pd.DataFrame, external: pd.DataFrame | None
) -> pd.DataFrame:
    """Объединяет источники только для обучения и назначает веса строк."""

    primary = competition.copy()
    primary["_data_source"] = "competition"
    primary["_sample_weight"] = 1.0
    if external is None or external.empty:
        return primary

    overlap = set(primary["anon_polygon_id"]) & set(external["anon_polygon_id"])
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(f"External polygon ID пересекается с train: {example}")
    columns = sorted(set(primary.columns) | set(external.columns))
    return pd.concat(
        [primary.reindex(columns=columns), external.reindex(columns=columns)],
        ignore_index=True,
    ).sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
