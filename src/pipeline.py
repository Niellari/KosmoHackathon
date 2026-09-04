"""Общий аналитический pipeline для CLI и веб-интерфейса."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.anomalies import detect_anomalies
from src.data import clean_primary_series, combine_context, load_dataset
from src.interpolation import GapInterpolator
from src.model import TrainedGapModel


class AnalysisPipeline:
    def __init__(self, data: pd.DataFrame, reference: pd.DataFrame | None = None):
        self.data = data.copy()
        self.data["primary_ndvi"] = clean_primary_series(self.data["primary_ndvi"])
        self.reference = None if reference is None else reference.copy()

    @classmethod
    def from_csv(
        cls, data_path: Path | str, reference_path: Path | str | None = None
    ) -> "AnalysisPipeline":
        data = load_dataset(data_path)
        reference = load_dataset(reference_path) if reference_path else None
        return cls(data, reference)

    def predict_targets(
        self, target_mask: pd.Series, method: str = "ensemble"
    ) -> pd.DataFrame:
        if len(target_mask) != len(self.data):
            raise ValueError("Размер target_mask не совпадает с датасетом")
        targets = self.data.loc[target_mask].copy()
        if targets.empty:
            raise ValueError("Не найдено строк для предсказания")

        current = self.data.copy()
        current.loc[target_mask, "primary_ndvi"] = np.nan
        context = combine_context(current, self.reference)
        if method == "ml":
            training_source = self.reference if self.reference is not None else current
            return TrainedGapModel().fit(training_source).predict(context, targets)
        interpolator = GapInterpolator(context)
        return interpolator.predict(targets, method=method, exclude_all_targets=True)

    def analyze_polygon(self, polygon_id: str, year: int | None = None) -> pd.DataFrame:
        polygon_all = self.data[self.data["anon_polygon_id"] == polygon_id].copy()
        if polygon_all.empty:
            raise KeyError(f"Полигон не найден: {polygon_id}")

        if year is None:
            year = int(polygon_all["year"].max())
        polygon = polygon_all[polygon_all["year"] == year].copy()
        if polygon.empty:
            raise KeyError(f"Для полигона {polygon_id} нет данных за {year} год")

        polygon["primary_ndvi_filled"] = polygon["primary_ndvi"]
        missing = polygon["primary_ndvi"].isna()
        if missing.any():
            context = combine_context(self.data, self.reference)
            predictions = GapInterpolator(context).predict(
                polygon.loc[missing], method="ensemble", exclude_all_targets=True
            )
            polygon.loc[missing, "primary_ndvi_filled"] = predictions["prediction"]
            # Для восстановленных точек недоступная норма оценивается по другим годам.
            polygon.loc[missing, "ndvi_climatology_mean"] = predictions["historical"]
            std_fallback = polygon_all["ndvi_climatology_std"].median()
            if not np.isfinite(std_fallback):
                std_fallback = 0.1
            polygon.loc[missing, "ndvi_climatology_std"] = std_fallback
        return detect_anomalies(polygon.reset_index(drop=True))

    def validate(self, sample_size: int = 3000, seed: int = 42) -> dict[str, float]:
        known = self.data.index[self.data["primary_ndvi"].notna()].to_numpy()
        if not len(known):
            raise ValueError("В датасете нет известных primary_ndvi")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(known, size=min(sample_size, len(known)), replace=False)
        mask = self.data.index.isin(chosen)
        truth = self.data.loc[mask, "primary_ndvi"].to_numpy(float)

        baseline = self.predict_targets(pd.Series(mask), method="baseline")["prediction"]
        ensemble = self.predict_targets(pd.Series(mask), method="ml")["prediction"]
        return {
            "baseline_rmse": float(np.sqrt(np.mean(np.square(truth - baseline.to_numpy())))),
            "ml_rmse": float(np.sqrt(np.mean(np.square(truth - ensemble.to_numpy())))),
        }
