"""Единое построение признаков для обучения и инференса всех ML-моделей."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FeaturesConfig
from src.interpolation import GapInterpolator, local_quadratic_value, pchip_value


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
        if self.config.interpolation.pchip:
            result.append("pchip_prediction")
        if self.config.interpolation.local_quadratic:
            result.append("local_quadratic_prediction")
        if self.config.interpolation.differences:
            if self.config.interpolation.pchip and self.config.interpolation.linear:
                result.append("pchip_minus_linear")
            if (
                self.config.interpolation.local_quadratic
                and self.config.interpolation.linear
            ):
                result.append("quadratic_minus_linear")
            if self.config.interpolation.linear and self.config.interpolation.baseline:
                result.append("linear_minus_neighbor_mean")
        if self.config.interpolation.agreement:
            result.extend(
                [
                    "interpolation_mean",
                    "interpolation_std",
                    "interpolation_range",
                ]
            )
        if self.config.polygon_history.enabled:
            result.append("historical")
            if self.config.polygon_history.include_std:
                result.append("historical_std")
            if self.config.polygon_history.include_reference_years:
                result.append("n_reference_years_calc")
            if self.config.polygon_history.expanded_statistics:
                for window in self.config.polygon_history.doy_windows:
                    result.extend(
                        [
                            f"historical_mean_w{window}",
                            f"historical_median_w{window}",
                            f"historical_std_w{window}",
                            f"historical_q25_w{window}",
                            f"historical_q75_w{window}",
                            f"historical_iqr_w{window}",
                            f"historical_years_w{window}",
                        ]
                    )
                result.extend(
                    ["historical_recent_weighted", "historical_year_trend"]
                )
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
        dynamics = self.config.temporal_dynamics
        if dynamics.enabled:
            if dynamics.gap_geometry:
                result.extend(
                    ["gap_span_days", "gap_position", "neighbor_asymmetry"]
                )
            if dynamics.slopes:
                result.extend(
                    [
                        "previous_interval_days",
                        "next_interval_days",
                        "slope_before",
                        "slope_after",
                        "slope_between",
                        "slope_change",
                    ]
                )
            if dynamics.acceleration:
                result.append("local_acceleration")
            if dynamics.local_statistics:
                result.extend(
                    [
                        "neighbor_mean",
                        "neighbor_std",
                        "neighbor_min",
                        "neighbor_max",
                        "neighbor_range",
                    ]
                )
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

    def _calculated_history_features(
        self, interpolator: GapInterpolator, row: pd.Series
    ) -> dict[str, float]:
        """Считает историю одинаково для train и inference без текущего сезона."""

        history = self.config.polygon_history
        result: dict[str, float] = {
            "historical": np.nan,
            "historical_std": np.nan,
            "n_reference_years_calc": 0.0,
        }
        arrays = interpolator._history_arrays.get(str(row["anon_polygon_id"]))
        if arrays is None or not history.enabled:
            return result

        target_year = int(row["year"])
        target_doy = int(row["doy"])
        years, days, all_values = arrays
        other_seasons = years != target_year
        distances = np.abs(days - target_doy)

        def selected(window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            mask = other_seasons & (distances <= window)
            return all_values[mask], years[mask], distances[mask]

        values, selected_years, selected_distances = selected(history.doy_window)
        if len(values):
            result["historical"] = interpolator._weighted_stat(
                values, selected_distances, scale=history.weighting_scale
            )
            result["historical_std"] = float(np.std(values))
            result["n_reference_years_calc"] = float(
                np.unique(selected_years).size
            )

        if not history.expanded_statistics:
            return result

        largest_values, largest_years, largest_distances = selected(
            max(history.doy_windows)
        )
        for window in history.doy_windows:
            window_values, window_years, _ = selected(window)
            prefix = f"historical_{{}}_w{window}"
            if not len(window_values):
                result[prefix.format("years")] = 0.0
            else:
                result[prefix.format("years")] = float(
                    np.unique(window_years).size
                )
            result[prefix.format("mean")] = (
                float(np.mean(window_values)) if len(window_values) else np.nan
            )
            result[prefix.format("median")] = (
                float(np.median(window_values)) if len(window_values) else np.nan
            )
            result[prefix.format("std")] = (
                float(np.std(window_values)) if len(window_values) else np.nan
            )
            q25 = (
                float(np.quantile(window_values, 0.25))
                if len(window_values)
                else np.nan
            )
            q75 = (
                float(np.quantile(window_values, 0.75))
                if len(window_values)
                else np.nan
            )
            result[prefix.format("q25")] = q25
            result[prefix.format("q75")] = q75
            result[prefix.format("iqr")] = (
                q75 - q25 if np.isfinite(q25) and np.isfinite(q75) else np.nan
            )

        result["historical_recent_weighted"] = np.nan
        result["historical_year_trend"] = np.nan
        if len(largest_values):
            year_distances = np.abs(largest_years - target_year).astype(float)
            weights = np.exp(
                -0.5 * np.square(largest_distances / history.weighting_scale)
                -0.5 * np.square(year_distances / history.recent_year_scale)
            )
            if np.isfinite(weights).any() and weights.sum() > 0:
                result["historical_recent_weighted"] = float(
                    np.average(largest_values, weights=weights)
                )
            unique_years = np.unique(largest_years)
            if len(unique_years) >= 2:
                annual_values = np.asarray(
                    [
                        np.median(largest_values[largest_years == year])
                        for year in unique_years
                    ],
                    dtype=float,
                )
                result["historical_year_trend"] = float(
                    np.polyfit(unique_years.astype(float), annual_values, 1)[0]
                )
        return result

    def _calendar_features(
        self, record: dict, crop_type: str, doy: int, year: int
    ) -> None:
        record["doy"] = float(doy)
        record["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
        record["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
        record["year_offset"] = float(year - 2010)
        for label, column in CROP_COLUMNS.items():
            record[column] = float(crop_type == label)

    def _temporal_features(
        self,
        record: dict,
        previous_1: float,
        next_1: float,
        previous_2: float,
        next_2: float,
        previous_days: float,
        next_days: float,
        previous_2_days: float,
        next_2_days: float,
    ) -> None:
        gap_span = (
            previous_days + next_days
            if np.isfinite(previous_days) and np.isfinite(next_days)
            else np.nan
        )
        record["gap_span_days"] = gap_span
        record["gap_position"] = (
            previous_days / gap_span
            if np.isfinite(gap_span) and gap_span > 0
            else np.nan
        )
        record["neighbor_asymmetry"] = (
            (previous_days - next_days) / gap_span
            if np.isfinite(gap_span) and gap_span > 0
            else np.nan
        )

        previous_interval = (
            previous_2_days - previous_days
            if np.isfinite(previous_2_days) and np.isfinite(previous_days)
            else np.nan
        )
        next_interval = (
            next_2_days - next_days
            if np.isfinite(next_2_days) and np.isfinite(next_days)
            else np.nan
        )
        record["previous_interval_days"] = previous_interval
        record["next_interval_days"] = next_interval
        slope_before = (
            (previous_1 - previous_2) / previous_interval
            if np.isfinite(previous_1)
            and np.isfinite(previous_2)
            and np.isfinite(previous_interval)
            and previous_interval > 0
            else np.nan
        )
        slope_after = (
            (next_2 - next_1) / next_interval
            if np.isfinite(next_1)
            and np.isfinite(next_2)
            and np.isfinite(next_interval)
            and next_interval > 0
            else np.nan
        )
        slope_between = (
            (next_1 - previous_1) / gap_span
            if np.isfinite(previous_1)
            and np.isfinite(next_1)
            and np.isfinite(gap_span)
            and gap_span > 0
            else np.nan
        )
        record["slope_before"] = slope_before
        record["slope_after"] = slope_after
        record["slope_between"] = slope_between
        record["slope_change"] = (
            slope_after - slope_before
            if np.isfinite(slope_before) and np.isfinite(slope_after)
            else np.nan
        )
        local_span = previous_interval + next_interval
        record["local_acceleration"] = (
            2.0 * (slope_after - slope_before) / local_span
            if np.isfinite(slope_before)
            and np.isfinite(slope_after)
            and np.isfinite(local_span)
            and local_span > 0
            else np.nan
        )

        neighbors = np.array(
            [previous_2, previous_1, next_1, next_2], dtype=float
        )
        neighbors = neighbors[np.isfinite(neighbors)]
        record["neighbor_mean"] = (
            float(neighbors.mean()) if len(neighbors) else np.nan
        )
        record["neighbor_std"] = (
            float(neighbors.std()) if len(neighbors) else np.nan
        )
        record["neighbor_min"] = (
            float(neighbors.min()) if len(neighbors) else np.nan
        )
        record["neighbor_max"] = (
            float(neighbors.max()) if len(neighbors) else np.nan
        )
        record["neighbor_range"] = (
            float(neighbors.max() - neighbors.min()) if len(neighbors) else np.nan
        )

    def _interpolation_features(
        self,
        record: dict,
        previous_1: float,
        next_1: float,
        previous_2: float,
        next_2: float,
        previous_days: float,
        next_days: float,
        previous_2_days: float,
        next_2_days: float,
    ) -> None:
        points = [
            (-previous_2_days, previous_2),
            (-previous_days, previous_1),
            (next_days, next_1),
            (next_2_days, next_2),
        ]
        pchip = pchip_value(points)
        quadratic = local_quadratic_value(points)
        linear = float(record.get("linear", np.nan))
        neighbor_mean = float(record.get("baseline", np.nan))
        record["pchip_prediction"] = pchip
        record["local_quadratic_prediction"] = quadratic
        record["pchip_minus_linear"] = (
            pchip - linear
            if np.isfinite(pchip) and np.isfinite(linear)
            else np.nan
        )
        record["quadratic_minus_linear"] = (
            quadratic - linear
            if np.isfinite(quadratic) and np.isfinite(linear)
            else np.nan
        )
        record["linear_minus_neighbor_mean"] = (
            linear - neighbor_mean
            if np.isfinite(linear) and np.isfinite(neighbor_mean)
            else np.nan
        )
        configured_candidates: list[float] = []
        if self.config.interpolation.baseline:
            configured_candidates.append(neighbor_mean)
        if self.config.interpolation.linear:
            configured_candidates.append(linear)
        if self.config.interpolation.pchip:
            configured_candidates.append(pchip)
        if self.config.interpolation.local_quadratic:
            configured_candidates.append(quadratic)
        candidates = np.asarray(configured_candidates, dtype=float)
        candidates = candidates[np.isfinite(candidates)]
        record["interpolation_mean"] = (
            float(candidates.mean()) if len(candidates) else np.nan
        )
        record["interpolation_std"] = (
            float(candidates.std()) if len(candidates) else np.nan
        )
        record["interpolation_range"] = (
            float(candidates.max() - candidates.min())
            if len(candidates)
            else np.nan
        )

    def build_training_set(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Строит leave-one-out признаки для известных наблюдений."""

        known = frame[frame["primary_ndvi"].notna()].copy()
        interpolator = self.make_interpolator(frame)
        crop_curves = interpolator._crop_curves
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
                previous_2_days = (
                    days[position] - days[position - 2]
                    if position >= 2
                    else np.nan
                )
                next_2_days = (
                    days[position + 2] - days[position]
                    if position + 2 < len(days)
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
                if (
                    self.config.polygon_history.calculation
                    == "leave_one_season_out"
                ):
                    record.update(
                        self._calculated_history_features(interpolator, row)
                    )
                self._calendar_features(
                    record, str(row["crop_type"]), int(row["doy"]), int(row["year"])
                )
                self._interpolation_features(
                    record,
                    previous_1,
                    next_1,
                    previous_2,
                    next_2,
                    previous_days,
                    next_days,
                    previous_2_days,
                    next_2_days,
                )
                self._temporal_features(
                    record,
                    previous_1,
                    next_1,
                    previous_2,
                    next_2,
                    previous_days,
                    next_days,
                    previous_2_days,
                    next_2_days,
                )
                records.append(record)
                targets.append(float(row["primary_ndvi"]))
                indices.append(index)

        features = pd.DataFrame(records, index=indices).reindex(
            columns=self.feature_names
        )
        features = features.apply(pd.to_numeric, errors="coerce").astype(float)
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
            previous_2_days = (
                float((target_date - dates[position - 2]) / np.timedelta64(1, "D"))
                if position >= 2
                else np.nan
            )
            next_2_days = (
                float((dates[position + 1] - target_date) / np.timedelta64(1, "D"))
                if position + 1 < len(dates)
                else np.nan
            )

            calculated_history = self._calculated_history_features(
                interpolator, row
            )

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
                "historical": calculated_history["historical"],
                "historical_std": calculated_history["historical_std"],
                "n_reference_years_calc": calculated_history[
                    "n_reference_years_calc"
                ],
                "crop_curve": base.at[index, "crop_curve"],
            }
            if self.config.polygon_history.expanded_statistics:
                record.update(calculated_history)
            self._calendar_features(
                record, str(row["crop_type"]), int(row["doy"]), int(row["year"])
            )
            self._interpolation_features(
                record,
                previous_1,
                next_1,
                previous_2,
                next_2,
                previous_days,
                next_days,
                previous_2_days,
                next_2_days,
            )
            self._temporal_features(
                record,
                previous_1,
                next_1,
                previous_2,
                next_2,
                previous_days,
                next_days,
                previous_2_days,
                next_2_days,
            )
            records.append(record)

        features = pd.DataFrame(records, index=targets.index).reindex(
            columns=self.feature_names
        )
        features = features.apply(pd.to_numeric, errors="coerce").astype(float)
        return features, base
