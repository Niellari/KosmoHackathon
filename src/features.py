"""Единое построение признаков для обучения и инференса всех ML-моделей."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FeaturesConfig
from src.interpolation import GapInterpolator


CROP_COLUMNS = {
    "озимая пшеница": "crop_winter_wheat",
    "подсолнечник": "crop_sunflower",
    "пастбища/зерновые": "crop_pasture_grain",
    "зерновые": "crop_grain",
}


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


class FeatureBuilder:
    """Строит одинаковую схему признаков на train и целевых пропусках."""

    def __init__(self, config: FeaturesConfig):
        self.config = config

    @property
    def feature_names(self) -> list[str]:
        result: list[str] = []
        if self.config.neighbors.enabled:
            result.extend(["previous_1", "next_1"])
            if self.config.neighbors.count >= 2:
                result.extend(["previous_2", "next_2"])
            if self.config.neighbors.include_distances:
                result.extend(["previous_days", "next_days"])
        if self.config.interpolation.baseline:
            result.append("baseline")
        if self.config.interpolation.linear:
            result.append("linear")
        if self.config.polygon_history.enabled:
            result.append("historical")
            if self.config.polygon_history.include_std:
                result.append("historical_std")
            if self.config.polygon_history.include_reference_years:
                result.append("n_reference_years_calc")
        if self.config.crop_curve.enabled:
            result.append("crop_curve")
        if self.config.calendar.enabled:
            if self.config.calendar.include_doy:
                result.append("doy")
            if self.config.calendar.cyclic_encoding:
                result.extend(["doy_sin", "doy_cos"])
            if self.config.calendar.include_year:
                result.append("year_offset")
        if self.config.crop_type.enabled:
            result.extend(CROP_COLUMNS.values())
        if not result:
            raise ValueError("В конфигурации отключены все признаки")
        return result

    def make_interpolator(
        self,
        context: pd.DataFrame,
        ensemble_params: dict | None = None,
    ) -> GapInterpolator:
        history = self.config.polygon_history
        crop = self.config.crop_curve
        return GapInterpolator(
            context,
            neighbors_enabled=self.config.neighbors.enabled,
            history_enabled=history.enabled,
            history_window=history.doy_window,
            history_scale=history.weighting_scale,
            crop_curve_enabled=crop.enabled,
            crop_window=crop.doy_window,
            crop_scale=crop.weighting_scale,
            crop_aggregation=crop.aggregation,
            ensemble_params=ensemble_params,
        )

    def _calendar_features(
        self, record: dict, crop_type: str, doy: int, year: int
    ) -> None:
        record["doy"] = float(doy)
        record["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
        record["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
        record["year_offset"] = float(year - 2010)
        for label, column in CROP_COLUMNS.items():
            record[column] = float(crop_type == label)

    def build_training_set(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Строит leave-one-out признаки для известных наблюдений."""

        known = frame[frame["primary_ndvi"].notna()].copy()
        crop_curves = self.make_interpolator(frame)._crop_curves
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
                previous_days = (
                    days[position] - days[position - 1] if position >= 1 else np.nan
                )
                next_days = (
                    days[position + 1] - days[position]
                    if position + 1 < len(days)
                    else np.nan
                )
                record = {
                    "previous_1": previous_1,
                    "next_1": next_1,
                    "previous_2": previous_2,
                    "next_2": next_2,
                    "previous_days": previous_days,
                    "next_days": next_days,
                    "baseline": _safe_average(previous_1, next_1),
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
                self._calendar_features(
                    record, str(row["crop_type"]), int(row["doy"]), int(row["year"])
                )
                records.append(record)
                targets.append(float(row["primary_ndvi"]))
                indices.append(index)

        features = pd.DataFrame(records, index=indices).reindex(
            columns=self.feature_names
        )
        return features, pd.Series(targets, index=indices, name="primary_ndvi")

    def build_prediction_set(
        self, context: pd.DataFrame, targets: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Строит признаки без использования всех целевых synthetic gaps."""

        interpolator = self.make_interpolator(context)
        base = interpolator.predict(
            targets, method="ensemble", exclude_all_targets=True
        )
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
                    group["primary_ndvi"].notna()
                    & ~group["date"].isin(excluded)
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

            polygon_history = interpolator._polygon_groups.get(
                str(row["anon_polygon_id"])
            )
            historical_std = np.nan
            n_reference_years = 0.0
            if polygon_history is not None and self.config.polygon_history.enabled:
                candidates = polygon_history[
                    (polygon_history["year"] != int(row["year"]))
                    & (
                        (polygon_history["doy"] - int(row["doy"])).abs()
                        <= self.config.polygon_history.doy_window
                    )
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
            self._calendar_features(
                record, str(row["crop_type"]), int(row["doy"]), int(row["year"])
            )
            records.append(record)

        features = pd.DataFrame(records, index=targets.index).reindex(
            columns=self.feature_names
        )
        return features, base
