"""LightGBM над общими и sensor-aware признаками."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FeatureBuilder
from src.models.base import GapModel, ModelUnavailableError
from src.sensor_features import (
    fit_source_classifier,
    sensor_series_features,
    source_labels,
    source_probabilities,
)


PROTECTED_COLUMNS = {
    "anon_polygon_id",
    "date",
    "crop_type",
    "is_synthetic_gap",
    "year",
    "doy",
    "_data_source",
    "_sample_weight",
}


class SensorAwareLightGBMModel(GapModel):
    """Модель, устраняющая межсенсорное смещение primary_ndvi."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = dict(self.params)
        self.feature_version = int(options.pop("feature_version", 1))
        self.polygon_identity = bool(
            options.pop("polygon_identity", self.feature_version >= 7)
        )
        self.season_identity = bool(
            options.pop("season_identity", self.feature_version >= 9)
        )
        self.n_masks = int(options.pop("n_masks", 8))
        self.mask_rate = float(options.pop("mask_rate", 0.15))
        self.mask_seed = int(options.pop("mask_seed", 1000))
        self.classifier_masks = int(options.pop("classifier_masks", 5))
        self.classifier_seed = int(options.pop("classifier_seed", 5000))
        self.classifier_params = dict(options.pop("classifier", {}) or {})
        self.source_experts = bool(options.pop("source_experts", False))
        self.expert_weight = float(options.pop("expert_weight", 0.75))
        self.expert_params = dict(options.pop("expert", {}) or {})
        if not 0.0 <= self.expert_weight <= 1.0:
            raise ValueError("expert_weight должен находиться в диапазоне 0..1")
        self.regressor_params = options
        self.estimator = None
        self.expert_estimators: dict[int, object] = {}
        self.classifier = None
        self.feature_columns: list[str] = []
        self.polygon_ids: list[str] = []
        self.season_ids: list[str] = []

    @staticmethod
    def _mask(frame: pd.DataFrame, rate: float, seed: int):
        known = frame["primary_ndvi"].notna().to_numpy()
        selected = known & (np.random.default_rng(seed).random(len(frame)) < rate)
        indices = frame.index[selected]
        context = frame.copy()
        if "is_synthetic_gap" in context:
            context["is_synthetic_gap"] = (
                context["is_synthetic_gap"].fillna(False).astype(bool)
            )
        else:
            context["is_synthetic_gap"] = False
        hidden = [column for column in context.columns if column not in PROTECTED_COLUMNS]
        context.loc[indices, hidden] = np.nan
        context.loc[indices, "is_synthetic_gap"] = True
        targets = context.loc[indices].copy()
        truth = pd.to_numeric(frame.loc[indices, "primary_ndvi"], errors="coerce")
        labels = source_labels(frame.loc[indices])
        weights = (
            pd.to_numeric(frame.loc[indices, "_sample_weight"], errors="coerce")
            .fillna(1.0)
            if "_sample_weight" in frame
            else pd.Series(1.0, index=indices)
        )
        return context, targets, truth.astype(float), labels, weights.astype(float)

    def _assemble(
        self,
        base: pd.DataFrame,
        probabilities: pd.DataFrame,
        context: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> pd.DataFrame:
        sensors = sensor_series_features(
            context, targets, probabilities, self.feature_version
        )
        parts = [base, probabilities, sensors]
        if self.polygon_identity:
            parts.append(self._polygon_features(targets))
        if self.season_identity:
            parts.append(self._season_identity_features(targets))
        return pd.concat(parts, axis=1)

    def _polygon_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        values = rows["anon_polygon_id"].astype(str)
        return pd.DataFrame(
            {
                f"polygon_{polygon}": values.eq(polygon).to_numpy(float)
                for polygon in self.polygon_ids
            },
            index=rows.index,
        )

    @staticmethod
    def _season_keys(rows: pd.DataFrame) -> pd.Series:
        return rows["anon_polygon_id"].astype(str) + "__" + rows["year"].astype(str)

    def _season_identity_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        values = self._season_keys(rows)
        return pd.DataFrame(
            {
                f"season_{season}": values.eq(season).to_numpy(float)
                for season in self.season_ids
            },
            index=rows.index,
        )

    def fit(self, train: pd.DataFrame, features: FeatureBuilder):
        if self.training.target_mode != "direct":
            raise ValueError("sensor_lightgbm пока поддерживает только target_mode=direct")
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ModelUnavailableError("Для sensor_lightgbm установите lightgbm") from error

        self.polygon_ids = sorted(train["anon_polygon_id"].astype(str).unique())
        self.season_ids = sorted(self._season_keys(train).unique())

        classifier_batches = []
        for step in range(self.classifier_masks):
            context, targets, _, labels, _ = self._mask(
                train, self.mask_rate, self.classifier_seed + step
            )
            classifier_batches.append((context, targets, labels))
        self.classifier = fit_source_classifier(
            classifier_batches, self.classifier_params, self.feature_version
        )

        matrices: list[pd.DataFrame] = []
        targets_list: list[pd.Series] = []
        weights_list: list[pd.Series] = []
        labels_list: list[pd.Series] = []
        for step in range(self.n_masks):
            context, target_rows, truth, labels, weights = self._mask(
                train, self.mask_rate, self.mask_seed + step
            )
            base, _ = features.build_prediction_set(context, target_rows)
            probabilities = source_probabilities(
                self.classifier, context, target_rows, self.feature_version
            )
            matrices.append(
                self._assemble(base, probabilities, context, target_rows).reset_index(drop=True)
            )
            targets_list.append(truth.reset_index(drop=True))
            weights_list.append(weights.reset_index(drop=True))
            labels_list.append(pd.Series(labels).reset_index(drop=True))

        matrix = pd.concat(matrices, ignore_index=True)
        target = pd.concat(targets_list, ignore_index=True)
        sample_weight = pd.concat(weights_list, ignore_index=True)
        source = pd.concat(labels_list, ignore_index=True)
        usable = target.notna() & np.isfinite(target)
        self.feature_columns = list(matrix.columns)
        self.estimator = LGBMRegressor(**self.regressor_params)
        self.estimator.fit(
            matrix.loc[usable, self.feature_columns],
            target.loc[usable],
            sample_weight=sample_weight.loc[usable],
        )
        self.expert_estimators = {}
        if self.source_experts:
            expert_params = {**self.regressor_params, **self.expert_params}
            for label in range(3):
                selected = usable & source.eq(label)
                if selected.sum() < 100:
                    continue
                expert = LGBMRegressor(**expert_params)
                expert.fit(
                    matrix.loc[selected, self.feature_columns],
                    target.loc[selected],
                    sample_weight=sample_weight.loc[selected],
                )
                self.expert_estimators[label] = expert
        return self

    def predict(
        self,
        context: pd.DataFrame,
        targets: pd.DataFrame,
        features: FeatureBuilder,
    ) -> pd.DataFrame:
        if self.estimator is None or self.classifier is None:
            raise RuntimeError(f"Модель {self.name!r} не обучена")
        base_matrix, fallback = features.build_prediction_set(context, targets)
        probabilities = source_probabilities(
            self.classifier, context, targets, self.feature_version
        )
        matrix = self._assemble(base_matrix, probabilities, context, targets)
        global_prediction = np.asarray(
            self.estimator.predict(matrix.loc[:, self.feature_columns]), dtype=float
        )
        prediction = global_prediction.copy()
        expert_prediction = global_prediction.copy()
        if self.source_experts and self.expert_estimators:
            expert_outputs = np.column_stack(
                [
                    np.asarray(
                        self.expert_estimators[label].predict(
                            matrix.loc[:, self.feature_columns]
                        ),
                        dtype=float,
                    )
                    if label in self.expert_estimators
                    else global_prediction
                    for label in range(3)
                ]
            )
            expert_prediction = np.sum(
                expert_outputs * probabilities.to_numpy(float), axis=1
            )
            prediction = (
                (1.0 - self.expert_weight) * global_prediction
                + self.expert_weight * expert_prediction
            )
        invalid = ~np.isfinite(prediction)
        prediction[invalid] = fallback.loc[invalid, "prediction"]
        result = fallback.copy()
        result["prediction"] = prediction
        result["global_prediction"] = global_prediction
        result["expert_prediction"] = expert_prediction
        result["model"] = self.name
        for column in probabilities:
            result[column] = probabilities[column]
        return result
