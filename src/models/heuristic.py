"""Детерминированные baseline-модели."""

from __future__ import annotations

import pandas as pd

from src.features import FeatureBuilder
from src.models.base import GapModel


class BaselineModel(GapModel):
    def fit(self, train: pd.DataFrame, features: FeatureBuilder) -> "BaselineModel":
        return self

    def predict(
        self,
        context: pd.DataFrame,
        targets: pd.DataFrame,
        features: FeatureBuilder,
    ) -> pd.DataFrame:
        result = features.make_interpolator(context).predict(
            targets, method="baseline", exclude_all_targets=True
        )
        result["model"] = self.name
        return result


class HeuristicEnsembleModel(GapModel):
    def fit(
        self, train: pd.DataFrame, features: FeatureBuilder
    ) -> "HeuristicEnsembleModel":
        return self

    def predict(
        self,
        context: pd.DataFrame,
        targets: pd.DataFrame,
        features: FeatureBuilder,
    ) -> pd.DataFrame:
        result = features.make_interpolator(
            context, ensemble_params=self.params
        ).predict(targets, method="ensemble", exclude_all_targets=True)
        result["model"] = self.name
        return result
