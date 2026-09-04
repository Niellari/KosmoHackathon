"""Обучаемая метамодель восстановления NDVI по временному контексту."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.interpolation import GapInterpolator


CROP_COLUMNS = {
    "озимая пшеница": "crop_winter_wheat",
    "подсолнечник": "crop_sunflower",
    "пастбища/зерновые": "crop_pasture_grain",
    "зерновые": "crop_grain",
}

FEATURE_COLUMNS = [
    "previous_1",
    "next_1",
    "previous_2",
    "next_2",
    "previous_days",
    "next_days",
    "baseline",
    "linear",
    "historical",
    "historical_std",
    "n_reference_years_calc",
    "crop_curve",
    "doy",
    "doy_sin",
    "doy_cos",
    "year_offset",
    *CROP_COLUMNS.values(),
]


def _safe_average(left: float, right: float) -> float:
    values = [value for value in (left, right) if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def _linear_value(
    previous: float, following: float, previous_days: float, next_days: float
) -> float:
    if all(np.isfinite(v) for v in (previous, following, previous_days, next_days)):
        total = previous_days + next_days
        if total > 0:
            return float(previous + (following - previous) * previous_days / total)
    return _safe_average(previous, following)


def _calendar_features(record: dict, crop_type: str, doy: int, year: int) -> None:
    record["doy"] = float(doy)
    record["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
    record["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
    record["year_offset"] = float(year - 2010)
    for label, column in CROP_COLUMNS.items():
        record[column] = float(crop_type == label)


def build_training_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Строит leave-one-out признаки для всех известных наблюдений."""

    known = frame[frame["primary_ndvi"].notna()].copy()
    crop_curves = GapInterpolator(frame)._crop_curves
    records: list[dict] = []
    targets: list[float] = []
    indices: list[int] = []

    for (_, _), group in known.groupby(["anon_polygon_id", "year"], sort=False):
        group = group.sort_values("date")
        values = group["primary_ndvi"].to_numpy(float)
        days = group["doy"].to_numpy(float)
        rows = list(group.iterrows())
        for position, (index, row) in enumerate(rows):
            previous_1 = values[position - 1] if position >= 1 else np.nan
            next_1 = values[position + 1] if position + 1 < len(values) else np.nan
            previous_2 = values[position - 2] if position >= 2 else np.nan
            next_2 = values[position + 2] if position + 2 < len(values) else np.nan
            previous_days = days[position] - days[position - 1] if position >= 1 else np.nan
            next_days = days[position + 1] - days[position] if position + 1 < len(days) else np.nan
            baseline = _safe_average(previous_1, next_1)
            record = {
                "previous_1": previous_1,
                "next_1": next_1,
                "previous_2": previous_2,
                "next_2": next_2,
                "previous_days": previous_days,
                "next_days": next_days,
                "baseline": baseline,
                "linear": _linear_value(
                    previous_1, next_1, previous_days, next_days
                ),
                "historical": row.get("ndvi_climatology_mean", np.nan),
                "historical_std": row.get("ndvi_climatology_std", np.nan),
                "n_reference_years_calc": row.get("n_reference_years", np.nan),
                "crop_curve": crop_curves.get(
                    (str(row["crop_type"]), int(row["doy"])), np.nan
                ),
            }
            _calendar_features(
                record, str(row["crop_type"]), int(row["doy"]), int(row["year"])
            )
            records.append(record)
            targets.append(float(row["primary_ndvi"]))
            indices.append(index)

    features = pd.DataFrame(records, index=indices).reindex(columns=FEATURE_COLUMNS)
    return features, pd.Series(targets, index=indices, name="primary_ndvi")


def build_target_features(
    context: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит признаки без использования всех целевых synthetic gaps."""

    interpolator = GapInterpolator(context)
    base = interpolator.predict(targets, method="ensemble", exclude_all_targets=True)
    excluded_by_group = {
        key: set(group["date"])
        for key, group in targets.groupby(["anon_polygon_id", "year"])
    }
    records: list[dict] = []

    for index, row in targets.iterrows():
        key = (str(row["anon_polygon_id"]), int(row["year"]))
        group = interpolator._groups.get(key)
        excluded = excluded_by_group.get(key, {row["date"]})
        if group is None:
            observed = pd.DataFrame(columns=context.columns)
        else:
            observed = group[
                group["primary_ndvi"].notna() & ~group["date"].isin(excluded)
            ].sort_values("date")

        dates = observed["date"].to_numpy(dtype="datetime64[ns]")
        values = observed["primary_ndvi"].to_numpy(float)
        target_date = np.datetime64(row["date"])
        position = int(np.searchsorted(dates, target_date))

        previous_1 = values[position - 1] if position >= 1 else np.nan
        previous_2 = values[position - 2] if position >= 2 else np.nan
        next_1 = values[position] if position < len(values) else np.nan
        next_2 = values[position + 1] if position + 1 < len(values) else np.nan
        previous_days = (
            float((target_date - dates[position - 1]) / np.timedelta64(1, "D"))
            if position >= 1
            else np.nan
        )
        next_days = (
            float((dates[position] - target_date) / np.timedelta64(1, "D"))
            if position < len(dates)
            else np.nan
        )

        polygon_history = interpolator._polygon_groups.get(str(row["anon_polygon_id"]))
        historical_std = np.nan
        n_reference_years = 0.0
        if polygon_history is not None:
            candidates = polygon_history[
                (polygon_history["year"] != int(row["year"]))
                & ((polygon_history["doy"] - int(row["doy"])).abs() <= 21)
            ]
            if len(candidates):
                historical_std = float(candidates["primary_ndvi"].std(ddof=0))
                n_reference_years = float(candidates["year"].nunique())

        record = {
            "previous_1": previous_1,
            "next_1": next_1,
            "previous_2": previous_2,
            "next_2": next_2,
            "previous_days": previous_days,
            "next_days": next_days,
            "baseline": float(base.at[index, "baseline"]),
            "linear": _linear_value(
                previous_1, next_1, previous_days, next_days
            ),
            "historical": base.at[index, "historical"],
            "historical_std": historical_std,
            "n_reference_years_calc": n_reference_years,
            "crop_curve": base.at[index, "crop_curve"],
        }
        _calendar_features(
            record, str(row["crop_type"]), int(row["doy"]), int(row["year"])
        )
        records.append(record)

    return pd.DataFrame(records, index=targets.index).reindex(columns=FEATURE_COLUMNS), base


class TrainedGapModel:
    """LightGBM-регрессор с безопасным откатом к детерминированному ансамблю."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None

    def fit(self, frame: pd.DataFrame) -> "TrainedGapModel":
        matplotlib_cache = Path(tempfile.gettempdir()) / "agropulse-matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            self.model = None
            return self

        features, target = build_training_features(frame)
        self.model = LGBMRegressor(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=24,
            min_child_samples=40,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            verbosity=-1,
            random_state=self.random_state,
        )
        self.model.fit(features, target)
        return self

    def predict(self, context: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        features, base = build_target_features(context, targets)
        result = base.copy()
        if self.model is None:
            return result
        prediction = self.model.predict(features)
        invalid = ~np.isfinite(prediction)
        prediction[invalid] = result.loc[invalid, "prediction"]
        result["prediction"] = prediction
        result["model"] = "lightgbm"
        return result
