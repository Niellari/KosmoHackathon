"""Обучаемые табличные модели с общим интерфейсом."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureBuilder
from src.models.base import GapModel, ModelUnavailableError
from src.training import TestLikeGapGenerator


class TabularGapModel(GapModel):
    estimator = None

    def _create_estimator(self):
        raise NotImplementedError

    def _residual_column(self) -> str:
        return {
            "neighbor_mean": "baseline",
            "linear": "linear",
        }[self.training.residual_baseline]

    def fit(
        self, train: pd.DataFrame, features: FeatureBuilder
    ) -> "TabularGapModel":
        self.estimator = self._create_estimator()
        matrix, target, sample_weight = self._build_training_data(train, features)
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
            sample_weight = sample_weight.loc[usable]
        if np.allclose(sample_weight.to_numpy(float), 1.0):
            self.estimator.fit(matrix, target)
        elif self.name == "random_forest":
            step_name = self.estimator.steps[-1][0]
            self.estimator.fit(
                matrix,
                target,
                **{f"{step_name}__sample_weight": sample_weight},
            )
        else:
            self.estimator.fit(matrix, target, sample_weight=sample_weight)
        return self

    @staticmethod
    def _sample_weights(train: pd.DataFrame, indices: pd.Index) -> pd.Series:
        if "_sample_weight" not in train.columns:
            return pd.Series(1.0, index=indices, name="sample_weight")
        return pd.to_numeric(
            train.loc[indices, "_sample_weight"], errors="coerce"
        ).fillna(1.0).astype(float).rename("sample_weight")

    def _build_training_data(
        self, train: pd.DataFrame, features: FeatureBuilder
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        masking = self.training.gap_masking
        if masking.strategy == "leave_one_out":
            matrix, target = features.build_training_set(train)
            return matrix, target, self._sample_weights(train, matrix.index)

        matrices: list[pd.DataFrame] = []
        targets: list[pd.Series] = []
        weights: list[pd.Series] = []
        for batch in TestLikeGapGenerator(masking).generate(train):
            matrix, _ = features.build_prediction_set(batch.context, batch.targets)
            weights.append(
                self._sample_weights(train, matrix.index).reset_index(drop=True)
            )
            matrices.append(matrix.reset_index(drop=True))
            targets.append(batch.truth.reindex(matrix.index).reset_index(drop=True))
        if not matrices:
            raise ValueError("Генератор test-like пропусков не создал обучающих строк")
        return (
            pd.concat(matrices, ignore_index=True),
            pd.concat(targets, ignore_index=True).rename("primary_ndvi"),
            pd.concat(weights, ignore_index=True).rename("sample_weight"),
        )

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


class HistoryRoutedLightGBMModel(TabularGapModel):
    """Два LightGBM-эксперта для полей с историей и без неё."""

    routing_feature_names = {
        "historical",
        "historical_std",
        "n_reference_years_calc",
    }

    @classmethod
    def _is_history_feature(cls, column: str) -> bool:
        return column == "historical" or column.startswith("historical_") or (
            column == "n_reference_years_calc"
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_estimator = None
        self.cold_start_estimator = None
        self.history_columns: list[str] = []
        self.cold_start_columns: list[str] = []
        self.min_reference_years = int(self.params.get("min_reference_years", 1))
        if self.min_reference_years < 1:
            raise ValueError("min_reference_years должен быть не меньше 1")

    def _expert_params(self, expert: str) -> dict:
        common = self.params.get("common_params", {})
        overrides = self.params.get(f"{expert}_params", {})
        if not isinstance(common, dict) or not isinstance(overrides, dict):
            raise ValueError("Параметры routed LightGBM должны быть YAML-объектами")
        return {**common, **overrides}

    def _create_lightgbm(self, params: dict):
        matplotlib_cache = Path(tempfile.gettempdir()) / "agropulse-matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ModelUnavailableError(
                "Для routed-модели установите пакет lightgbm"
            ) from error
        return LGBMRegressor(**params)

    def _history_available(self, matrix: pd.DataFrame) -> pd.Series:
        missing = self.routing_feature_names - set(matrix.columns)
        if missing:
            raise ValueError(
                "history_routed_lightgbm требует признаки: "
                + ", ".join(sorted(missing))
            )
        historical = pd.to_numeric(matrix["historical"], errors="coerce")
        years = pd.to_numeric(
            matrix["n_reference_years_calc"], errors="coerce"
        )
        return historical.notna() & np.isfinite(historical) & (
            years >= self.min_reference_years
        )

    @staticmethod
    def _fit_lightgbm(estimator, matrix, target, sample_weight) -> None:
        if np.allclose(sample_weight.to_numpy(float), 1.0):
            estimator.fit(matrix, target)
        else:
            estimator.fit(matrix, target, sample_weight=sample_weight)

    def fit(
        self, train: pd.DataFrame, features: FeatureBuilder
    ) -> "HistoryRoutedLightGBMModel":
        matrix, target, sample_weight = self._build_training_data(train, features)
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
            sample_weight = sample_weight.loc[usable]

        history_rows = self._history_available(matrix)
        if not history_rows.any():
            raise ValueError("В train нет строк с доступной историей полигона")
        self.history_columns = list(matrix.columns)
        self.cold_start_columns = [
            column
            for column in matrix.columns
            if not self._is_history_feature(column)
        ]
        if not self.cold_start_columns:
            raise ValueError("После исключения истории не осталось признаков")
        self.history_estimator = self._create_lightgbm(
            self._expert_params("history_rich")
        )
        self.cold_start_estimator = self._create_lightgbm(
            self._expert_params("cold_start")
        )
        self._fit_lightgbm(
            self.history_estimator,
            matrix.loc[history_rows, self.history_columns],
            target.loc[history_rows],
            sample_weight.loc[history_rows],
        )
        self._fit_lightgbm(
            self.cold_start_estimator,
            matrix[self.cold_start_columns],
            target,
            sample_weight,
        )
        return self

    def predict(
        self,
        context: pd.DataFrame,
        targets: pd.DataFrame,
        features: FeatureBuilder,
    ) -> pd.DataFrame:
        if self.history_estimator is None or self.cold_start_estimator is None:
            raise RuntimeError(f"Модель {self.name!r} не обучена")
        matrix, base = features.build_prediction_set(context, targets)
        history_rows = self._history_available(matrix)
        model_output = np.full(len(matrix), np.nan, dtype=float)
        if history_rows.any():
            model_output[history_rows.to_numpy()] = self.history_estimator.predict(
                matrix.loc[history_rows, self.history_columns]
            )
        cold_rows = ~history_rows
        if cold_rows.any():
            model_output[cold_rows.to_numpy()] = self.cold_start_estimator.predict(
                matrix.loc[cold_rows, self.cold_start_columns]
            )

        if self.training.target_mode == "residual":
            baseline = pd.to_numeric(
                matrix[self._residual_column()], errors="coerce"
            ).to_numpy(float)
            prediction = baseline + model_output
            prediction[~np.isfinite(baseline)] = np.nan
        else:
            prediction = model_output
        invalid = ~np.isfinite(prediction)
        prediction[invalid] = base.loc[invalid, "prediction"]
        result = base.copy()
        result["prediction"] = prediction
        result["model"] = self.name
        result["route"] = np.where(history_rows, "history_rich", "cold_start")
        result["target_mode"] = self.training.target_mode
        result["residual_baseline"] = (
            self.training.residual_baseline
            if self.training.target_mode == "residual"
            else None
        )
        return result


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
