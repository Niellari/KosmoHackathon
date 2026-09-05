"""Строит submission продолжением направления между двумя прогнозами."""

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
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--clip-min", type=float)
    parser.add_argument("--clip-max", type=float)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    candidate = pd.read_csv(args.candidate)
    if list(candidate.columns) != list(base.columns):
        raise ValueError("Схемы submission-файлов не совпадают")

    target_columns = [column for column in base.columns if column not in KEYS]
    if len(target_columns) != 1:
        raise ValueError("Ожидалась ровно одна колонка предсказания")
    target = target_columns[0]

    aligned = base.merge(candidate, on=KEYS, suffixes=("_base", "_candidate"))
    if len(aligned) != len(base) or len(candidate) != len(base):
        raise ValueError("Ключи submission-файлов не совпадают")

    result = aligned[KEYS].copy()
    result[target] = aligned[f"{target}_base"] + args.factor * (
        aligned[f"{target}_candidate"] - aligned[f"{target}_base"]
    )
    if args.clip_min is not None or args.clip_max is not None:
        result[target] = result[target].clip(args.clip_min, args.clip_max)

    validate_submission(result, base[KEYS], target_column=target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"Создано: {args.output} | строк: {len(result)} | "
        f"min: {result[target].min():.6f} | max: {result[target].max():.6f}"
    )


if __name__ == "__main__":
    main()
