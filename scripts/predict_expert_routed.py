"""Создаёт submission с разными global/source-expert весами seen/unseen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import combine_context, load_dataset
from src.pipeline import AnalysisPipeline
from src.submission import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seen-weight", type=float, required=True)
    parser.add_argument("--unseen-weight", type=float, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    if not 0 <= args.seen_weight <= 1 or not 0 <= args.unseen_weight <= 1:
        raise ValueError("Веса должны находиться в диапазоне 0..1")

    config = load_config(args.config)
    current = load_dataset(config.data.test_path)
    reference = load_dataset(config.data.train_path)
    target_mask = current["is_synthetic_gap"].fillna(False).astype(bool)
    targets = current.loc[target_mask].copy()
    context = combine_context(current, reference)
    pipeline = AnalysisPipeline(current, reference=reference, config=config)
    training = pipeline._training_source(reference, context)
    predictor = pipeline._create_predictor(
        config.models.selected, training, cache=True
    )
    prediction = predictor.predict(context, targets)

    seen = targets["anon_polygon_id"].isin(set(reference["anon_polygon_id"]))
    weights = np.where(seen, args.seen_weight, args.unseen_weight)
    values = (
        (1 - weights) * prediction["global_prediction"].to_numpy(float)
        + weights * prediction["expert_prediction"].to_numpy(float)
    )
    result = targets[["anon_polygon_id", "date"]].copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result[config.predict.prediction_column] = values
    result = result.sort_values(["anon_polygon_id", "date"]).reset_index(drop=True)
    validate_submission(
        result,
        result[["anon_polygon_id", "date"]],
        target_column=config.predict.prediction_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Создано: {args.output} | строк: {len(result)}")


if __name__ == "__main__":
    main()
