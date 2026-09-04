"""Честное A/B-сравнение external data на целях конкурсного train."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import AppConfig
from src.data import load_dataset, load_external_training_data
from src.pipeline import AnalysisPipeline


def with_external_weight(config: AppConfig, weight: float) -> AppConfig:
    payload = config.data.external.model_dump()
    for source in payload["sources"]:
        if source["enabled"]:
            source["sample_weight"] = weight
    external = config.data.external.__class__.model_validate(payload)
    data = config.data.model_copy(update={"external": external})
    return config.model_copy(update={"data": data})


def with_external_source_weights(
    config: AppConfig, weights: dict[str, float]
) -> AppConfig:
    """Переопределяет веса только указанных именованных источников."""

    payload = config.data.external.model_dump()
    known_names = {source["name"] for source in payload["sources"]}
    unknown = sorted(set(weights) - known_names)
    if unknown:
        raise ValueError(f"Неизвестные external sources: {', '.join(unknown)}")
    for source in payload["sources"]:
        if source["name"] in weights:
            source["sample_weight"] = weights[source["name"]]
    external = config.data.external.__class__.model_validate(payload)
    data = config.data.model_copy(update={"external": external})
    return config.model_copy(update={"data": data})


def validate_external_ab(
    train_path: Path | str,
    config: AppConfig,
    model_name: str,
    sample_size: int,
    seeds: list[int],
) -> dict:
    competition = load_dataset(train_path)
    external = load_external_training_data(
        config.data.external, include_when_disabled=True
    )
    if external is None or external.empty:
        raise ValueError("External data для A/B-валидации не найдены")

    experiment_config = external_experiment_config(config)
    known = competition.index[competition["primary_ndvi"].notna()].to_numpy()
    if not len(known):
        raise ValueError("В конкурсном train нет известных primary_ndvi")

    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            known, size=min(sample_size, len(known)), replace=False
        )
        target_mask = competition.index.isin(chosen)
        truth = competition.loc[target_mask, "primary_ndvi"].to_numpy(float)

        base_pipeline = AnalysisPipeline(competition, config=experiment_config)
        external_pipeline = AnalysisPipeline(
            competition,
            training_extra=external,
            config=experiment_config,
        )
        base_prediction = base_pipeline.predict_targets(
            target_mask, method=model_name
        )["prediction"].to_numpy(float)
        external_prediction = external_pipeline.predict_targets(
            target_mask, method=model_name
        )["prediction"].to_numpy(float)
        base_rmse = _rmse(truth, base_prediction)
        external_rmse = _rmse(truth, external_prediction)
        runs.append(
            {
                "seed": int(seed),
                "base_rmse": base_rmse,
                "external_rmse": external_rmse,
                "change_percent": (external_rmse / base_rmse - 1.0) * 100.0,
            }
        )

    base_mean = float(np.mean([run["base_rmse"] for run in runs]))
    external_mean = float(np.mean([run["external_rmse"] for run in runs]))
    return {
        "model": model_name,
        "sample_size": min(sample_size, len(known)),
        "external_rows": len(external),
        "external_known_targets": int(external["primary_ndvi"].notna().sum()),
        "runs": runs,
        "base_rmse_mean": base_mean,
        "external_rmse_mean": external_mean,
        "change_percent": (external_mean / base_mean - 1.0) * 100.0,
    }


def external_experiment_config(config: AppConfig) -> AppConfig:
    """Включает external и изолирует эксперимент от production-кэша."""

    external = config.data.external.model_copy(update={"enabled": True})
    data = config.data.model_copy(update={"external": external})
    features = config.features
    if config.validation.external_ab.disable_crop_features:
        features = features.model_copy(
            update={
                "crop_type": features.crop_type.model_copy(
                    update={"enabled": False}
                ),
                "crop_curve": features.crop_curve.model_copy(
                    update={"enabled": False}
                ),
            }
        )
    definitions = {
        name: definition.model_copy(
            update={"artifact_path": None, "load_if_exists": False}
        )
        for name, definition in config.models.available.items()
    }
    models = config.models.model_copy(update={"available": definitions})
    return config.model_copy(
        update={"data": data, "features": features, "models": models}
    )


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))
