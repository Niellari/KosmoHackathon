"""Генерация обучающих пропусков, похожих на конкурсный test."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import GapMaskingConfig


@dataclass(frozen=True)
class MaskedTrainingBatch:
    """Одна реплика контекста, целей и скрытых истинных значений."""

    context: pd.DataFrame
    targets: pd.DataFrame
    truth: pd.Series
    replicate: int


class TestLikeGapGenerator:
    """Скрывает непересекающиеся последовательные блоки известных наблюдений."""

    __test__ = False

    def __init__(self, config: GapMaskingConfig):
        self.config = config

    def generate(self, frame: pd.DataFrame) -> Iterator[MaskedTrainingBatch]:
        """Возвращает детерминированные реплики с разными блочными масками."""

        known = frame[frame["primary_ndvi"].notna()]
        if known.empty:
            raise ValueError("Нельзя создать пропуски: в primary_ndvi нет значений")

        for replicate in range(self.config.replicas):
            rng = np.random.default_rng(self.config.random_state + replicate)
            selected: list[int] = []
            for _, group in known.groupby(
                ["anon_polygon_id", "year"], sort=False
            ):
                ordered = group.sort_values("date")
                positions = self._sample_positions(len(ordered), rng)
                selected.extend(ordered.index.to_numpy()[positions].tolist())

            selected = sorted(set(selected))
            if not selected:
                raise ValueError(
                    "Нельзя создать пропуски: в каждом сезоне меньше двух наблюдений"
                )
            context = frame.copy()
            if "is_synthetic_gap" not in context.columns:
                context["is_synthetic_gap"] = False
            truth = pd.to_numeric(
                context.loc[selected, "primary_ndvi"], errors="coerce"
            ).astype(float)
            protected = {
                "anon_polygon_id",
                "date",
                "crop_type",
                "is_synthetic_gap",
                "year",
                "doy",
            }
            hidden_columns = [
                column for column in context.columns if column not in protected
            ]
            context.loc[selected, hidden_columns] = np.nan
            dates = pd.to_datetime(context.loc[selected, "date"])
            context.loc[selected, "year"] = dates.dt.year.astype(
                context["year"].dtype
            ).to_numpy()
            context.loc[selected, "doy"] = dates.dt.dayofyear.astype(
                context["doy"].dtype
            ).to_numpy()
            context.loc[selected, "is_synthetic_gap"] = True
            targets = context.loc[selected].copy()
            yield MaskedTrainingBatch(
                context=context,
                targets=targets,
                truth=truth.rename("primary_ndvi"),
                replicate=replicate,
            )

    def _sample_positions(
        self, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        if size == 0:
            return np.array([], dtype=int)

        target_count = max(
            1,
            min(size - 1, int(round(size * self.config.target_fraction))),
        )
        if size == 1:
            return np.array([], dtype=int)

        lengths = np.array(sorted(self.config.block_length_weights), dtype=int)
        weights = np.array(
            [self.config.block_length_weights[int(length)] for length in lengths],
            dtype=float,
        )
        weights /= weights.sum()
        occupied = np.zeros(size, dtype=bool)

        while int(occupied.sum()) < target_count:
            remaining = target_count - int(occupied.sum())
            candidates: dict[int, np.ndarray] = {}
            for candidate_length in lengths[
                (lengths <= size) & (lengths <= remaining)
            ]:
                starts = [
                    start
                    for start in range(size - int(candidate_length) + 1)
                    if not occupied[
                        max(0, start - 1) : min(
                            size, start + int(candidate_length) + 1
                        )
                    ].any()
                ]
                if starts:
                    candidates[int(candidate_length)] = np.asarray(
                        starts, dtype=int
                    )
            if not candidates:
                break
            feasible_lengths = np.asarray(sorted(candidates), dtype=int)
            feasible_weights = np.array(
                [
                    weights[np.where(lengths == length)[0][0]]
                    for length in feasible_lengths
                ]
            )
            feasible_weights /= feasible_weights.sum()
            length = int(rng.choice(feasible_lengths, p=feasible_weights))
            start = int(rng.choice(candidates[length]))
            occupied[start : start + length] = True

        return np.flatnonzero(occupied)
