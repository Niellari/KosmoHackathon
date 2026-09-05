"""Диагностика интерполяций при известном истинном сенсоре цели."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import combine_context, load_dataset
from src.features import FeatureBuilder
from src.models.sensor import PROTECTED_COLUMNS
from src.sensor_features import PROBABILITY_COLUMNS, sensor_series_features, source_labels


def score(truth: np.ndarray, prediction: np.ndarray, mask=None) -> float:
    usable = np.isfinite(prediction)
    if mask is not None:
        usable &= mask
    return float(np.sqrt(np.mean(np.square(prediction[usable] - truth[usable]))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    config = load_config()
    reference = load_dataset(config.data.train_path)
    current = load_dataset(config.data.test_path)
    known = current["primary_ndvi"].notna().to_numpy()
    selected = known & (np.random.default_rng(args.seed).random(len(current)) < 0.15)
    truth = current.loc[selected, "primary_ndvi"].to_numpy(float)
    labels = source_labels(current.loc[selected])
    targets = current.loc[selected].copy()
    hidden = [column for column in current if column not in PROTECTED_COLUMNS]
    current.loc[selected, hidden] = np.nan
    current.loc[selected, "is_synthetic_gap"] = True
    context = combine_context(current, reference)

    probabilities = pd.DataFrame(
        np.eye(3)[labels], index=targets.index, columns=PROBABILITY_COLUMNS
    )
    sensors = sensor_series_features(context, targets, probabilities, feature_version=4)
    base, _ = FeatureBuilder(config.features).build_prediction_set(context, targets)
    seen = targets["anon_polygon_id"].isin(set(reference["anon_polygon_id"])).to_numpy()
    candidates = {
        "primary_linear": base["linear"].to_numpy(float),
        "primary_pchip": base["pchip_prediction"].to_numpy(float),
        "oracle_sensor": sensors["source_interpolation"].to_numpy(float),
        "oracle_harmonized": sensors["harmonized_source_interpolation"].to_numpy(float),
    }
    for name, prediction in candidates.items():
        coverage = float(np.isfinite(prediction).mean())
        print(
            f"{name}: coverage={coverage:.3f} rmse={score(truth, prediction):.6f} "
            f"seen={score(truth, prediction, seen):.6f} "
            f"unseen={score(truth, prediction, ~seen):.6f}"
        )
    for label, name in enumerate(("s2", "landsat", "modis")):
        selected_source = labels == label
        prediction = candidates["oracle_sensor"]
        print(
            f"oracle_{name}: n={selected_source.sum()} "
            f"rmse={score(truth, prediction, selected_source):.6f}"
        )


if __name__ == "__main__":
    main()
