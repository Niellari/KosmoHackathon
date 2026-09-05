"""Честная оценка моделей на синтетических пропусках.

Отличия от `main.py validate`:

* маскируется 15% наблюдений одновременно, как на платформе, поэтому у части
  пропусков соседи тоже скрыты и задача не оказывается легче реальной;
* половина полигонов целиком исключается из обучения — это воспроизводит те
  39 из 78 тестовых полигонов, которых нет в train;
* признаки на обучении строятся тем же кодом, что и на инференсе, без доступа
  к скрытым точкам и к нецензурированной климатологии организаторов;
* результат усредняется по нескольким маскам, а не по одной случайной выборке.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.data import clean_primary_series
from src.predictor import PredictorService
from src.synthetic import MaskSpec, MaskedSplit, iter_splits


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    repeats: int
    n_targets: int
    rmse: float
    rmse_seen: float
    rmse_unseen: float
    baseline_rmse: float

    def render(self) -> str:
        gap_score = round(30 * max(0.0, 1 - self.rmse / 0.10), 2)
        return "\n".join(
            [
                f"Модель:            {self.model}",
                f"Прогонов:          {self.repeats}",
                f"Целей суммарно:    {self.n_targets}",
                f"Baseline RMSE:     {self.baseline_rmse:.6f}",
                f"RMSE:              {self.rmse:.6f}",
                f"  видимые полигоны:   {self.rmse_seen:.6f}",
                f"  невидимые полигоны: {self.rmse_unseen:.6f}",
                f"Ожидаемый GapScore:{gap_score:>7.2f}",
            ]
        )


def _disable_artifacts(config: AppConfig) -> AppConfig:
    """Отключает кэш моделей: бенчмарк не должен перезаписывать боевые артефакты."""

    available = {
        name: definition.model_copy(
            update={"artifact_path": None, "load_if_exists": False}
        )
        for name, definition in config.models.available.items()
    }
    return config.model_copy(
        update={"models": config.models.model_copy(update={"available": available})}
    )


def _squared_errors(
    config: AppConfig, split: MaskedSplit, model_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    models_config = config.models.model_copy(update={"selected": model_name})
    predictor = PredictorService(models_config, config.features, config.training)
    predictor.prepare(split.train_source)
    predicted = predictor.predict(split.context, split.targets)

    truth = split.truth.to_numpy(float)
    errors = np.square(predicted["prediction"].to_numpy(float) - truth)
    baseline_errors = np.square(predicted["baseline"].to_numpy(float) - truth)
    return errors, baseline_errors, split.unseen_targets.to_numpy()


def _rmse(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.sqrt(values.mean())) if len(values) else float("nan")


def run_benchmark(
    frame: pd.DataFrame,
    config: AppConfig,
    model_name: str,
    repeats: int = 3,
    mask_rate: float = 0.15,
    seed: int = 42,
    holdout_fraction: float = 0.5,
) -> BenchmarkResult:
    frame = frame.copy()
    frame["primary_ndvi"] = clean_primary_series(frame["primary_ndvi"])
    config = _disable_artifacts(config)

    errors: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    unseen: list[np.ndarray] = []
    spec = MaskSpec(rate=mask_rate, seed=seed)
    for split in iter_splits(frame, repeats, spec, holdout_fraction):
        model_errors, baseline_errors, unseen_flags = _squared_errors(
            config, split, model_name
        )
        errors.append(model_errors)
        baseline.append(baseline_errors)
        unseen.append(unseen_flags)

    all_errors = np.concatenate(errors)
    all_baseline = np.concatenate(baseline)
    all_unseen = np.concatenate(unseen)
    return BenchmarkResult(
        model=model_name,
        repeats=repeats,
        n_targets=int(len(all_errors)),
        rmse=_rmse(all_errors),
        rmse_seen=_rmse(all_errors[~all_unseen]),
        rmse_unseen=_rmse(all_errors[all_unseen]),
        baseline_rmse=_rmse(all_baseline),
    )
