"""Честный A/B sensor_lightgbm на одной общей платформоподобной маске."""

from __future__ import annotations

import argparse

import numpy as np

from src.config import load_config
from src.data import load_dataset
from src.features import FeatureBuilder
from src.models.sensor import PROTECTED_COLUMNS, SensorAwareLightGBMModel


def make_split(frame, seed: int, rate: float = 0.15, transductive: bool = False):
    known = frame["primary_ndvi"].notna().to_numpy()
    hidden = known & (np.random.default_rng(seed).random(len(frame)) < rate)
    polygons = np.sort(frame["anon_polygon_id"].unique())
    unseen = set(
        np.random.default_rng(seed).choice(
            polygons, size=round(len(polygons) / 2), replace=False
        )
    )
    context = frame.copy()
    columns = [column for column in context if column not in PROTECTED_COLUMNS]
    context.loc[hidden, columns] = np.nan
    context.loc[hidden, "is_synthetic_gap"] = True
    targets = context.loc[hidden].copy()
    truth = frame.loc[hidden, "primary_ndvi"].to_numpy(float)
    train = (
        context.copy()
        if transductive
        else context[~context["anon_polygon_id"].isin(unseen)].copy()
    )
    unseen_targets = targets["anon_polygon_id"].isin(unseen).to_numpy()
    return train, context, targets, truth, unseen_targets


def rmse(truth, prediction, selected=None) -> float:
    if selected is None:
        selected = np.ones(len(truth), dtype=bool)
    return float(np.sqrt(np.mean(np.square(truth[selected] - prediction[selected]))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transductive", action="store_true")
    args = parser.parse_args()

    config = load_config()
    frame = load_dataset(config.data.train_path)
    train, context, targets, truth, unseen = make_split(
        frame, args.seed, transductive=args.transductive
    )
    feature_builder = FeatureBuilder(config.features)
    base_params = dict(config.models.available["sensor_lightgbm"].params)

    predictions = {}
    for version in args.versions:
        params = {**base_params, "feature_version": version}
        model = SensorAwareLightGBMModel(
            f"sensor_v{version}", params, config.training
        ).fit(train, feature_builder)
        prediction = model.predict(context, targets, feature_builder)[
            "prediction"
        ].to_numpy(float)
        predictions[version] = prediction
        print(
            f"v{version}: rmse={rmse(truth, prediction):.6f} "
            f"seen={rmse(truth, prediction, ~unseen):.6f} "
            f"unseen={rmse(truth, prediction, unseen):.6f}"
        )

    if len(predictions) == 2:
        left, right = args.versions
        for weight in np.linspace(0, 1, 21):
            blended = (1 - weight) * predictions[left] + weight * predictions[right]
            print(f"blend_right={weight:.2f}: rmse={rmse(truth, blended):.6f}")


if __name__ == "__main__":
    main()
