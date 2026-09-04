"""Обучаемые табличные модели с общим интерфейсом."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureBuilder
from src.models.base import GapModel, ModelUnavailableError


class TabularGapModel(GapModel):
    estimator = None

    def _create_estimator(self):
        raise NotImplementedError

    def _residual_column(self) -> str:
        return {
            "neighbor_mean": "baseline",
            "linear": "linear",
        }[self.training.residual_baseline]

    def fit(self, train: pd.DataFrame, features: FeatureBuilder) -> "TabularGapModel":
        self.estimator = self._create_estimator()
        matrix, target = features.build_training_set(train)
        if self.training.target_mode == "residual":
            baseline = pd.to_numeric(
                matrix[self._residual_column()], errors="coerce"
            )
            usable = baseline.notna() & np.isfinite(baseline)
            if not usable.any():
                raise ValueError(
                    "Не удалось рассчитать baseline ни для одной обучающей строки"
                )
            matrix = matrix.loc[usable]
            target = target.loc[usable] - baseline.loc[usable]
        self.estimator.fit(matrix, target)
        return self

    def predict(
        self,
        context: pd.DataFrame,
        targets: pd.DataFrame,
        features: FeatureBuilder,
    ) -> pd.DataFrame:
        if self.estimator is None:
            raise RuntimeError(f"Модель {self.name!r} не обучена")
        matrix, base = features.build_prediction_set(context, targets)
        result = base.copy()
        model_output = np.asarray(self.estimator.predict(matrix), dtype=float)
        if self.training.target_mode == "residual":
            baseline = pd.to_numeric(
                matrix[self._residual_column()], errors="coerce"
            ).to_numpy(float)
            prediction = baseline + model_output
            prediction[~np.isfinite(baseline)] = np.nan
        else:
            prediction = model_output
        invalid = ~np.isfinite(prediction)
        prediction[invalid] = result.loc[invalid, "prediction"]
        result["prediction"] = prediction
        result["model"] = self.name
        result["target_mode"] = self.training.target_mode
        result["residual_baseline"] = (
            self.training.residual_baseline
            if self.training.target_mode == "residual"
            else None
        )
        return result


class LightGBMModel(TabularGapModel):
    def _create_estimator(self):
        matplotlib_cache = Path(tempfile.gettempdir()) / "agropulse-matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ModelUnavailableError(
                "Для модели lightgbm установите пакет lightgbm"
            ) from error
        return LGBMRegressor(**self.params)


class CatBoostModel(TabularGapModel):
    def _create_estimator(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise ModelUnavailableError(
                "Для модели catboost установите пакет: pip install catboost"
            ) from error
        return CatBoostRegressor(**self.params)


class RandomForestModel(TabularGapModel):
    def _create_estimator(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import make_pipeline
        except ImportError as error:
            raise ModelUnavailableError(
                "Для модели random_forest установите пакет scikit-learn"
            ) from error
        return make_pipeline(
            SimpleImputer(strategy="median"), RandomForestRegressor(**self.params)
        )
