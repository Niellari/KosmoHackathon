"""Восстановление пропусков временного ряда primary_ndvi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


NDVI_MIN = -1.0
NDVI_MAX = 1.0


def _valid_local_points(
    points: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Очищает и сортирует локальные точки вокруг даты прогноза."""

    valid = [
        (float(day), float(value))
        for day, value in points
        if np.isfinite(day)
        and np.isfinite(value)
        and NDVI_MIN <= value <= NDVI_MAX
    ]
    if not valid:
        return np.array([], dtype=float), np.array([], dtype=float)
    valid.sort(key=lambda item: item[0])
    x = np.asarray([item[0] for item in valid], dtype=float)
    y = np.asarray([item[1] for item in valid], dtype=float)
    unique = np.r_[True, np.diff(x) > 0]
    return x[unique], y[unique]


def pchip_value(points: list[tuple[float, float]]) -> float:
    """Интерполирует значение в дате x=0 монотонным кубическим PCHIP."""

    x, y = _valid_local_points(points)
    if len(x) < 2 or x[0] >= 0 or x[-1] <= 0:
        return np.nan
    try:
        value = float(PchipInterpolator(x, y, extrapolate=False)(0.0))
    except ValueError:
        return np.nan
    return float(np.clip(value, NDVI_MIN, NDVI_MAX)) if np.isfinite(value) else np.nan


def local_quadratic_value(points: list[tuple[float, float]]) -> float:
    """Аппроксимирует локальный изгиб параболой с защитой от overshoot."""

    x, y = _valid_local_points(points)
    if len(x) < 3 or x[0] >= 0 or x[-1] <= 0:
        return np.nan
    scale = float(np.max(np.abs(x)))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    coefficients = np.polyfit(x / scale, y, deg=2)
    value = float(np.polyval(coefficients, 0.0))
    local_range = float(np.max(y) - np.min(y))
    margin = max(0.10, 0.50 * local_range)
    lower = max(NDVI_MIN, float(np.min(y)) - margin)
    upper = min(NDVI_MAX, float(np.max(y)) + margin)
    return float(np.clip(value, lower, upper)) if np.isfinite(value) else np.nan


@dataclass(frozen=True)
class PredictionDetails:
    prediction: float
    baseline: float
    historical: float | None
    crop_curve: float | None
    previous_value: float | None
    next_value: float | None
    previous_days: int | None
    next_days: int | None
    confidence: float


class GapInterpolator:
    """Интерполятор, устойчивый к новым полигонам и отсутствующим источникам."""

    def __init__(
        self,
        context: pd.DataFrame,
        *,
        neighbors_enabled: bool = True,
        history_enabled: bool = True,
        history_window: int = 21,
        history_scale: float = 7.0,
        crop_curve_enabled: bool = True,
        crop_window: int = 7,
        crop_scale: float = 3.0,
        crop_aggregation: str = "median",
        ensemble_params: dict[str, float] | None = None,
    ):
        self.context = context.copy()
        self.neighbors_enabled = neighbors_enabled
        self.history_enabled = history_enabled
        self.history_window = history_window
        self.history_scale = history_scale
        self.crop_curve_enabled = crop_curve_enabled
        self.crop_window = crop_window
        self.crop_scale = crop_scale
        self.crop_aggregation = crop_aggregation
        weights = ensemble_params or {}
        self.baseline_weight_near = float(weights.get("baseline_weight_near", 0.86))
        self.baseline_weight_far = float(weights.get("baseline_weight_far", 0.72))
        self.historical_weight = float(weights.get("historical_weight", 0.22))
        self.crop_curve_weight = float(weights.get("crop_curve_weight", 0.06))
        self.context["primary_ndvi"] = pd.to_numeric(
            self.context["primary_ndvi"], errors="coerce"
        )
        self._groups = {
            key: group.sort_values("date")
            for key, group in self.context.groupby(
                ["anon_polygon_id", "year"], sort=False
            )
        }
        self._polygon_groups = {
            key: group[group["primary_ndvi"].notna()].copy()
            for key, group in self.context.groupby("anon_polygon_id", sort=False)
        }
        self._crop_groups = {
            key: group[group["primary_ndvi"].notna()].copy()
            for key, group in self.context.groupby("crop_type", sort=False)
        }
        self._crop_curves = self._build_crop_curves()

    def _build_crop_curves(self) -> dict[tuple[str, int], float]:
        """Предварительно рассчитывает сглаженную сезонную кривую культуры."""

        curves: dict[tuple[str, int], float] = {}
        if not self.crop_curve_enabled:
            return curves
        observed = self.context[self.context["primary_ndvi"].notna()]
        grouped = observed.groupby(["crop_type", "doy"], as_index=False)["primary_ndvi"]
        daily = (
            grouped.mean() if self.crop_aggregation == "mean" else grouped.median()
        ).sort_values(["crop_type", "doy"])
        for crop_type, group in daily.groupby("crop_type", sort=False):
            days = group["doy"].to_numpy(float)
            values = group["primary_ndvi"].to_numpy(float)
            for doy in range(1, 367):
                selected = np.abs(days - doy) <= self.crop_window
                if not selected.any():
                    continue
                curves[(str(crop_type), doy)] = self._weighted_stat(
                    values[selected],
                    np.abs(days[selected] - doy),
                    scale=self.crop_scale,
                )
        return curves

    @staticmethod
    def _weighted_stat(values: np.ndarray, distances: np.ndarray, scale: float) -> float:
        weights = np.exp(-0.5 * np.square(distances / scale))
        if not np.isfinite(weights).any() or weights.sum() == 0:
            return float(np.nanmedian(values))
        return float(np.average(values, weights=weights))

    def _neighbors(
        self,
        polygon_id: str,
        year: int,
        date: pd.Timestamp,
        excluded_dates: set[pd.Timestamp],
    ) -> tuple[float | None, float | None, int | None, int | None]:
        group = self._groups.get((polygon_id, year))
        if group is None or not self.neighbors_enabled:
            return None, None, None, None

        observed = group[
            group["primary_ndvi"].notna() & ~group["date"].isin(excluded_dates)
        ]
        previous = observed[observed["date"] < date].tail(1)
        following = observed[observed["date"] > date].head(1)

        previous_value = None if previous.empty else float(previous.iloc[0]["primary_ndvi"])
        next_value = None if following.empty else float(following.iloc[0]["primary_ndvi"])
        previous_days = (
            None if previous.empty else int((date - previous.iloc[0]["date"]).days)
        )
        next_days = (
            None if following.empty else int((following.iloc[0]["date"] - date).days)
        )
        return previous_value, next_value, previous_days, next_days

    def _historical_value(
        self, polygon_id: str, year: int, doy: int, excluded_dates: set[pd.Timestamp]
    ) -> float | None:
        group = self._polygon_groups.get(polygon_id)
        if group is None or not self.history_enabled:
            return None
        candidates = group[
            (group["year"] != year)
            & ~group["date"].isin(excluded_dates)
            & ((group["doy"] - doy).abs() <= self.history_window)
        ]
        if candidates.empty:
            return None
        distances = (candidates["doy"] - doy).abs().to_numpy(float)
        return self._weighted_stat(
            candidates["primary_ndvi"].to_numpy(float),
            distances,
            scale=self.history_scale,
        )

    def _crop_value(
        self, crop_type: str, doy: int, excluded_dates: set[pd.Timestamp]
    ) -> float | None:
        # В агрегированной кривой вклад одной скрываемой точки пренебрежимо мал.
        if not self.crop_curve_enabled:
            return None
        return self._crop_curves.get((crop_type, doy))

    def predict_row(
        self,
        row: pd.Series,
        excluded_dates: set[pd.Timestamp] | None = None,
        method: str = "ensemble",
    ) -> PredictionDetails:
        excluded_dates = excluded_dates or {row["date"]}
        prev, nxt, prev_days, next_days = self._neighbors(
            str(row["anon_polygon_id"]),
            int(row["year"]),
            row["date"],
            excluded_dates,
        )

        if prev is not None and nxt is not None:
            # Baseline из условия: среднее двух ближайших известных значений.
            baseline = (prev + nxt) / 2.0
        elif prev is not None:
            baseline = prev
        elif nxt is not None:
            baseline = nxt
        else:
            baseline = np.nan

        historical = self._historical_value(
            str(row["anon_polygon_id"]),
            int(row["year"]),
            int(row["doy"]),
            excluded_dates,
        )
        crop_curve = self._crop_value(str(row["crop_type"]), int(row["doy"]), excluded_dates)

        candidates: list[tuple[float, float]] = []
        if np.isfinite(baseline):
            # Соседние значения наиболее информативны для короткого пропуска.
            nearest_distance = min(
                value for value in (prev_days, next_days) if value is not None
            )
            baseline_weight = (
                self.baseline_weight_near
                if nearest_distance <= 16
                else self.baseline_weight_far
            )
            candidates.append((float(baseline), baseline_weight))
        if historical is not None and np.isfinite(historical):
            candidates.append((historical, self.historical_weight))
        if crop_curve is not None and np.isfinite(crop_curve):
            candidates.append((crop_curve, self.crop_curve_weight))

        if not candidates:
            prediction = 0.35
        elif method == "baseline" and np.isfinite(baseline):
            prediction = float(baseline)
        else:
            values = np.array([item[0] for item in candidates], dtype=float)
            weights = np.array([item[1] for item in candidates], dtype=float)
            prediction = float(np.average(values, weights=weights))

        neighbor_count = int(prev is not None) + int(nxt is not None)
        confidence = 0.35 + 0.2 * neighbor_count
        if historical is not None:
            confidence += 0.15
        if prev_days is not None and next_days is not None:
            confidence -= min(0.2, max(prev_days, next_days) / 150.0)
        confidence = float(np.clip(confidence, 0.1, 0.95))

        return PredictionDetails(
            prediction=prediction,
            baseline=float(baseline) if np.isfinite(baseline) else prediction,
            historical=historical,
            crop_curve=crop_curve,
            previous_value=prev,
            next_value=nxt,
            previous_days=prev_days,
            next_days=next_days,
            confidence=confidence,
        )

    def predict(
        self,
        targets: pd.DataFrame,
        method: str = "ensemble",
        exclude_all_targets: bool = True,
    ) -> pd.DataFrame:
        excluded_by_group: dict[tuple[str, int], set[pd.Timestamp]] = {}
        if exclude_all_targets:
            for key, group in targets.groupby(["anon_polygon_id", "year"]):
                excluded_by_group[key] = set(group["date"])

        records = []
        for index, row in targets.iterrows():
            excluded = excluded_by_group.get(
                (str(row["anon_polygon_id"]), int(row["year"])),
                {row["date"]},
            )
            details = self.predict_row(row, excluded_dates=excluded, method=method)
            records.append(
                {
                    "index": index,
                    "prediction": details.prediction,
                    "baseline": details.baseline,
                    "historical": details.historical,
                    "crop_curve": details.crop_curve,
                    "confidence": details.confidence,
                }
            )
        return pd.DataFrame.from_records(records).set_index("index")
