"""Сравнение источников обучения на скрытой части известных строк test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data import combine_context, combine_training_sources, load_dataset
from src.features import FeatureBuilder
from src.models.sensor import PROTECTED_COLUMNS, SensorAwareLightGBMModel


def mask_known(frame, seed: int, rate: float = 0.15):
    known = frame["primary_ndvi"].notna().to_numpy()
    selected = known & (np.random.default_rng(seed).random(len(frame)) < rate)
    context = frame.copy()
    hidden_columns = [
        column for column in context.columns if column not in PROTECTED_COLUMNS
    ]
    context.loc[selected, hidden_columns] = np.nan
    context.loc[selected, "is_synthetic_gap"] = True
    return context, context.loc[selected].copy(), frame.loc[selected, "primary_ndvi"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", nargs="+", choices=("reference", "context", "combined"),
        default=("context", "combined"),
    )
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    config = load_config()
    reference = load_dataset(config.data.train_path)
    current = load_dataset(config.data.test_path)
    current, targets, truth = mask_known(current, args.seed)
    prediction_context = combine_context(current, reference)
    sources = {
        "reference": reference,
        "context": current,
        "combined": prediction_context,
    }
    params = {
        **config.models.available["sensor_lightgbm"].params,
        "n_masks": 4,
        "classifier_masks": 3,
        "n_estimators": 500,
        "expert": {"n_estimators": 400, "num_leaves": 40},
    }
    builder = FeatureBuilder(config.features)
    truth_values = truth.to_numpy(float)
    predictions: dict[str, np.ndarray] = {}
    output_dir = Path("artifacts/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        train = combine_training_sources(sources[mode], None)
        model = SensorAwareLightGBMModel(
            f"sensor_{mode}", params, config.training
        ).fit(train, builder)
        prediction = model.predict(prediction_context, targets, builder)[
            "prediction"
        ].to_numpy(float)
        predictions[mode] = prediction
        np.savez_compressed(
            output_dir / f"training_source_seed{args.seed}_{mode}.npz",
            truth=truth_values,
            prediction=prediction,
        )
        score = np.sqrt(np.mean(np.square(prediction - truth_values)))
        print(f"{mode}: targets={len(targets)} rmse={score:.6f}")

    if len(predictions) == 2:
        left, right = args.modes
        for weight in np.linspace(0, 1, 21):
            prediction = (1 - weight) * predictions[left] + weight * predictions[right]
            score = np.sqrt(np.mean(np.square(prediction - truth_values)))
            print(f"blend_{right}={weight:.2f}: rmse={score:.6f}")


if __name__ == "__main__":
    main()
