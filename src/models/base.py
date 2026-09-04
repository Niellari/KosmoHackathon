"""Общий контракт моделей восстановления пропусков."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.config import TrainingConfig
from src.features import FeatureBuilder


class ModelUnavailableError(RuntimeError):
    """Выбранная модель недоступна из-за отсутствующей зависимости."""


class GapModel(ABC):
    def __init__(
        self,
        name: str,
        params: dict | None = None,
        training: TrainingConfig | None = None,
    ):
        self.name = name
        self.params = dict(params or {})
        self.training = training or TrainingConfig()

    @abstractmethod
    def fit(self, train: pd.DataFrame, features: FeatureBuilder) -> "GapModel":
        ...

    @abstractmethod
    def predict(
        self,
        context: pd.DataFrame,
        targets: pd.DataFrame,
        features: FeatureBuilder,
    ) -> pd.DataFrame:
        ...
