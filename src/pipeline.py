"""Общий аналитический pipeline для CLI и веб-интерфейса."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.anomalies import detect_anomalies
from src.config import AppConfig, load_config
from src.data import (
    clean_primary_series,
    combine_context,
    combine_training_sources,
    load_dataset,
    load_external_training_data,
)
from src.predictor import PredictorService


class AnalysisPipeline:
    def __init__(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        training_extra: pd.DataFrame | None = None,
        config: AppConfig | None = None,
    ):
        self.data = data.copy()
        self.data["primary_ndvi"] = clean_primary_series(self.data["primary_ndvi"])
        self.reference = None if reference is None else reference.copy()
        self.training_extra = (
            None if training_extra is None else training_extra.copy()
        )
        self.config = config or load_config()
        self._predictors: dict[str, PredictorService] = {}

    @classmethod
    def from_csv(
        cls,
        data_path: Path | str,
        reference_path: Path | str | None = None,
        config: AppConfig | None = None,
    ) -> "AnalysisPipeline":
        data = load_dataset(data_path)
        reference = load_dataset(reference_path) if reference_path else None
        active_config = config or load_config()
        training_extra = load_external_training_data(active_config.data.external)
        return cls(
            data,
            reference,
            training_extra=training_extra,
            config=active_config,
        )

    def _training_source(
        self,
        primary: pd.DataFrame,
        context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        competition = (
            context
            if self.config.training.use_context_labels and context is not None
            else primary
        )
        return combine_training_sources(competition, self.training_extra)

    def _model_name(self, requested: str | None) -> str:
        if requested in (None, "ml"):
            return self.config.models.selected
        return requested

    def _create_predictor(
        self, model_name: str, training_source: pd.DataFrame, cache: bool
    ) -> PredictorService:
        if model_name not in self.config.models.available:
            choices = ", ".join(sorted(self.config.models.available))
            raise ValueError(f"Неизвестная модель {model_name!r}. Доступны: {choices}")
        if cache and model_name in self._predictors:
            return self._predictors[model_name]
        models_config = self.config.models.model_copy(update={"selected": model_name})
        predictor = PredictorService(
            models_config, self.config.features, self.config.training
        )
        predictor.prepare(training_source)
        if cache:
            self._predictors[model_name] = predictor
        return predictor

    def prepare_model(self, model_name: str | None = None) -> str:
        """Загружает или обучает выбранную модель один раз при старте приложения."""

        selected = self._model_name(model_name)
        primary = self.reference if self.reference is not None else self.data
        context = combine_context(self.data, self.reference)
        training_source = self._training_source(primary, context)
        predictor = self._create_predictor(
            selected, training_source, cache=self.reference is not None
        )
        return predictor.selected_name

    def predict_targets(
        self, target_mask: pd.Series, method: str | None = None
    ) -> pd.DataFrame:
        if len(target_mask) != len(self.data):
            raise ValueError("Размер target_mask не совпадает с датасетом")
        targets = self.data.loc[target_mask].copy()
        if targets.empty:
            raise ValueError("Не найдено строк для предсказания")

        current = self.data.copy()
        current.loc[target_mask, "primary_ndvi"] = np.nan
        context = combine_context(current, self.reference)
        primary = self.reference if self.reference is not None else current
        training_source = self._training_source(primary, context)
        model_name = self._model_name(method)
        predictor = self._create_predictor(
            model_name,
            training_source,
            cache=self.reference is not None,
        )
        return predictor.predict(context, targets)

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
            target_mask = self.data.index.isin(polygon.loc[missing].index)
            predictions = self.predict_targets(pd.Series(target_mask), method=None)
            polygon.loc[missing, "primary_ndvi_filled"] = predictions["prediction"]
            # Для восстановленных точек недоступная норма оценивается по другим годам.
            polygon.loc[missing, "ndvi_climatology_mean"] = predictions["historical"]
            std_fallback = polygon_all["ndvi_climatology_std"].median()
            if not np.isfinite(std_fallback):
                std_fallback = 0.1
            polygon.loc[missing, "ndvi_climatology_std"] = std_fallback
        return detect_anomalies(polygon.reset_index(drop=True))

    def validate(
        self, sample_size: int = 3000, seed: int = 42, model_name: str | None = None
    ) -> dict[str, float | str]:
        known = self.data.index[self.data["primary_ndvi"].notna()].to_numpy()
        if not len(known):
            raise ValueError("В датасете нет известных primary_ndvi")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(known, size=min(sample_size, len(known)), replace=False)
        mask = self.data.index.isin(chosen)
        truth = self.data.loc[mask, "primary_ndvi"].to_numpy(float)

        baseline = self.predict_targets(pd.Series(mask), method="baseline")["prediction"]
        selected_name = self._model_name(model_name)
        selected = self.predict_targets(pd.Series(mask), method=selected_name)["prediction"]
        return {
            "baseline_rmse": float(np.sqrt(np.mean(np.square(truth - baseline.to_numpy())))),
            "model": selected_name,
            "model_rmse": float(np.sqrt(np.mean(np.square(truth - selected.to_numpy())))),
        }
