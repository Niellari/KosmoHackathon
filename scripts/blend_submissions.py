"""Смешивает два submission с разными весами для seen/unseen полигонов."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.submission import validate_submission


KEYS = ["anon_polygon_id", "date"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/train_dataset.csv"))
    parser.add_argument("--seen-weight", type=float, required=True)
    parser.add_argument("--unseen-weight", type=float, required=True)
    args = parser.parse_args()

    if not 0 <= args.seen_weight <= 1 or not 0 <= args.unseen_weight <= 1:
        raise ValueError("Веса должны находиться в диапазоне 0..1")
    base = pd.read_csv(args.base)
    candidate = pd.read_csv(args.candidate)
    target = next(column for column in base.columns if column not in KEYS)
    if list(candidate.columns) != list(base.columns):
        raise ValueError("Схемы submission-файлов не совпадают")
    aligned = base.merge(candidate, on=KEYS, suffixes=("_base", "_candidate"))
    if len(aligned) != len(base) or len(candidate) != len(base):
        raise ValueError("Ключи submission-файлов не совпадают")

    seen_polygons = set(pd.read_csv(args.train, usecols=["anon_polygon_id"])["anon_polygon_id"])
    weights = aligned["anon_polygon_id"].isin(seen_polygons).map(
        {True: args.seen_weight, False: args.unseen_weight}
    )
    result = aligned[KEYS].copy()
    result[target] = (
        (1 - weights) * aligned[f"{target}_base"]
        + weights * aligned[f"{target}_candidate"]
    )
    validate_submission(result, base[KEYS], target_column=target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Создано: {args.output} | строк: {len(result)}")


if __name__ == "__main__":
    main()
