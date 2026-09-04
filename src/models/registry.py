"""Явный безопасный реестр поддерживаемых моделей."""

from __future__ import annotations

from src.config import ModelDefinition
from src.models.base import GapModel
from src.models.boosting import CatBoostModel, LightGBMModel, RandomForestModel
from src.models.heuristic import BaselineModel, HeuristicEnsembleModel


MODEL_REGISTRY: dict[str, type[GapModel]] = {
    "baseline": BaselineModel,
    "heuristic_ensemble": HeuristicEnsembleModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "random_forest": RandomForestModel,
}


def create_model(name: str, definition: ModelDefinition) -> GapModel:
    try:
        model_class = MODEL_REGISTRY[definition.type]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Неизвестный тип модели {definition.type!r}. Доступны: {choices}"
        ) from error
    return model_class(name=name, params=definition.params)
