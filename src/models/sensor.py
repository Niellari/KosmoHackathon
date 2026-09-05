"""Модель восстановления с учётом сенсора-источника `primary_ndvi`.

Отличается от `lightgbm` тремя вещами:

* обучающие пары строятся тем же кодом, что и инференс (`build_prediction_set`)
  на нескольких сгенерированных масках, а не отдельной веткой по известным
  наблюдениям. Это обязательное условие: сенсорные признаки завязаны на то,
  что предсказываемая точка скрыта из контекста;
* к признакам добавляются вероятности источника (`p_s2`, `p_landsat`,
  `p_modis`), предсказанные по орбитальному календарю;
* добавляются соседи и интерполяция по ряду каждого сенсора отдельно, включая
  оценку, взвешенную вероятностями источника (`src.sensor_series`).

Замер на синтетических пропусках (13 765 точек, 3 прогона):

| набор признаков | RMSE | GapScore |
|---|---|---|
| без сенсорных признаков | 0.0745 | 7.64 |
| + вероятности источника | 0.0696 | 9.13 |
| + per-sensor соседи | 0.0662 | 10.14 |

Признаки дополняют друг друга: по отдельности per-sensor соседи дают -0.0019,
вероятности источника -0.0050, вместе -0.0084.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FeatureBuilder
from src.models.base import GapModel, ModelUnavailableError
from src.sensor_calendar import fit_source_classifier, source_probabilities
from src.peer_features import peer_features
from src.sensor_series import per_sensor_features


class SensorAwareLightGBMModel(GapModel):
    """LightGBM над признаками пропуска и вероятностями сенсора-источника."""

    def __init__(self, name: str, params: dict | None = None, training=None):
        super().__init__(name, params, training)
        options = dict(self.params)
        self.n_masks = int(options.pop("n_masks", 10))
        self.mask_rate = float(options.pop("mask_rate", 0.15))
        self.mask_seed = int(options.pop("mask_seed", 1000))
        self.classifier_masks = int(options.pop("classifier_masks", 6))
        self.classifier_seed = int(options.pop("classifier_seed", 5000))
        self.classifier_params = dict(options.pop("classifier", {}) or {})
        self.per_sensor = bool(options.pop("per_sensor", True))
        # Срез по остальным полигонам на ту же дату: восстанавливает погоду,
        # скрытую платформой, и даёт региональный уровень вегетации.
        self.peer_date = bool(options.pop("peer_date", False))
        # Семейство регрессора. Признаки и обучающая схема одинаковы, меняется
        # только алгоритм: это нужно для декорреляции ошибок в ансамбле, потому
        # что бустинги на одних признаках ошибаются почти одинаково.
        self.estimator_kind = str(options.pop("estimator", "lightgbm"))
        self.regressor_params = options
        self.estimator = None
        self.classifier = None

    def _create_estimator(self):
        kind = self.estimator_kind
        if kind == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as error:
                raise ModelUnavailableError(
                    "Для сенсорной модели на LightGBM установите пакет lightgbm"
                ) from error
            return LGBMRegressor(**self.regressor_params)
        if kind == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**self.regressor_params)
        if kind == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(**self.regressor_params)
        if kind == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as error:
                raise ModelUnavailableError(
                    "Для сенсорной модели на CatBoost установите пакет catboost"
                ) from error
            return CatBoostRegressor(**self.regressor_params)
        raise ValueError(
            f"Неизвестное семейство регрессора {kind!r}: ожидается "
            "lightgbm, random_forest, extra_trees или catboost"
        )

    def _assemble(
        self,
        matrix: pd.DataFrame,
        probabilities: pd.DataFrame,
        context: pd.DataFrame,
        rows: pd.DataFrame,
    ) -> pd.DataFrame:
        """Собирает итоговую матрицу признаков одинаково для обучения и инференса."""

        blocks = [matrix, probabilities]
        if self.per_sensor:
            blocks.append(per_sensor_features(context, rows, probabilities))
        if self.peer_date:
            blocks.append(peer_features(context, rows))
        return pd.concat(blocks, axis=1)

    def fit(self, train: pd.DataFrame, features: FeatureBuilder) -> "SensorAwareLightGBMModel":
        from src.synthetic import MaskSpec, apply_mask, generate_mask

        estimator = self._create_estimator()
        self.classifier = fit_source_classifier(
            train,
            repeats=self.classifier_masks,
            rate=self.mask_rate,
            seed=self.classifier_seed,
            params=self.classifier_params,
        )

        matrices: list[pd.DataFrame] = []
        targets: list[pd.Series] = []
        for step in range(self.n_masks):
            mask = generate_mask(
                train, MaskSpec(rate=self.mask_rate, seed=self.mask_seed + step)
            )
            flags = mask.to_numpy()
            context = apply_mask(train, mask)
            rows = context.loc[flags]
            matrix, _ = features.build_prediction_set(context, rows)
            probabilities = source_probabilities(self.classifier, context, rows)
            matrices.append(
                self._assemble(matrix, probabilities, context, rows).reset_index(drop=True)
            )
            targets.append(
                train.loc[flags, "primary_ndvi"].astype(float).reset_index(drop=True)
            )

        matrix = pd.concat(matrices, ignore_index=True)
        target = pd.concat(targets, ignore_index=True)
        known = target.notna().to_numpy()
        matrix = self._prepare_matrix(matrix, fit=True)
        estimator.fit(matrix.loc[known], target.loc[known])
        self.estimator = estimator
        return self

    def _prepare_matrix(self, matrix: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Готовит матрицу под конкретное семейство.

        Бустинги обрабатывают пропуски сами, а деревья sklearn — нет, поэтому
        для них пропуски заменяются медианой, посчитанной на обучении.
        """

        if self.estimator_kind not in ("random_forest", "extra_trees"):
            return matrix
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        if fit:
            self._fill_values = matrix.median(numeric_only=True)
        return matrix.fillna(getattr(self, "_fill_values", 0.0)).fillna(0.0)

    def predict(
        self,
        context: pd.DataFrame,
        targets: pd.DataFrame,
        features: FeatureBuilder,
    ) -> pd.DataFrame:
        if self.estimator is None or self.classifier is None:
            raise RuntimeError(f"Модель {self.name!r} не обучена")

        matrix, base = features.build_prediction_set(context, targets)
        probabilities = source_probabilities(self.classifier, context, targets)
        result = base.copy()
        prediction = np.asarray(
            self.estimator.predict(
                self._prepare_matrix(
                    self._assemble(matrix, probabilities, context, targets)
                )
            ),
            dtype=float,
        )
        invalid = ~np.isfinite(prediction)
        prediction[invalid] = result.loc[invalid, "prediction"]
        result["prediction"] = prediction
        result["model"] = self.name
        for column in probabilities.columns:
            result[column] = probabilities[column].to_numpy()
        return result
