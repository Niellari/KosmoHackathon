"""Проверки генератора синтетических пропусков."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.synthetic import (
    MASKED_COLUMNS,
    MaskSpec,
    apply_mask,
    gap_profile,
    generate_mask,
    holdout_polygons,
    iter_splits,
    make_split,
)


def build_frame(polygons: int = 6, days: int = 200, observed_every: int = 5):
    """Синтетический ряд: наблюдения стоят на регулярной орбитальной сетке."""

    records = []
    for polygon in range(polygons):
        for offset in range(days):
            date = pd.Timestamp("2020-04-01") + pd.Timedelta(days=offset)
            observed = offset % observed_every == polygon % observed_every
            records.append(
                {
                    "anon_polygon_id": f"AOI-{polygon:04d}",
                    "date": date,
                    "primary_ndvi": 0.4 + 0.001 * offset if observed else np.nan,
                    "s2_ndvi": 0.4 if observed else np.nan,
                    "era5_temp_c": 20.0,
                    "ndvi_climatology_mean": 0.5 if observed else np.nan,
                    "ndvi_climatology_std": 0.1 if observed else np.nan,
                    "n_reference_years": 15,
                    "crop_type": "зерновые",
                    "year": date.year,
                    "doy": date.dayofyear,
                }
            )
    return pd.DataFrame(records)


class GenerateMaskTests(unittest.TestCase):
    def test_only_observed_points_are_masked(self):
        frame = build_frame()
        mask = generate_mask(frame, MaskSpec(rate=0.5, seed=1))

        self.assertTrue(frame.loc[mask, "primary_ndvi"].notna().all())

    def test_mask_rate_matches_specification(self):
        frame = build_frame(polygons=12, days=400)
        mask = generate_mask(frame, MaskSpec(rate=0.15, seed=7))
        observed = int(frame["primary_ndvi"].notna().sum())

        self.assertAlmostEqual(mask.sum() / observed, 0.15, delta=0.02)

    def test_same_seed_is_reproducible(self):
        frame = build_frame()
        first = generate_mask(frame, MaskSpec(seed=3))
        second = generate_mask(frame, MaskSpec(seed=3))
        other = generate_mask(frame, MaskSpec(seed=4))

        self.assertTrue(first.equals(second))
        self.assertFalse(first.equals(other))

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            MaskSpec(rate=0.0)
        with self.assertRaises(ValueError):
            MaskSpec(rate=1.0)


class ApplyMaskTests(unittest.TestCase):
    def test_hides_every_platform_column(self):
        frame = build_frame()
        mask = generate_mask(frame, MaskSpec(rate=0.5, seed=2))
        masked = apply_mask(frame, mask)

        hidden = [c for c in MASKED_COLUMNS if c in frame.columns]
        hidden = [c for c in hidden if c not in ("year", "doy")]
        self.assertTrue(masked.loc[mask, hidden].isna().all().all())

    def test_calendar_columns_are_restored_from_date(self):
        frame = build_frame()
        mask = generate_mask(frame, MaskSpec(rate=0.5, seed=2))
        masked = apply_mask(frame, mask)

        self.assertTrue(masked["year"].notna().all())
        self.assertTrue((masked["doy"] == masked["date"].dt.dayofyear).all())

    def test_unmasked_rows_are_untouched(self):
        frame = build_frame()
        mask = generate_mask(frame, MaskSpec(rate=0.5, seed=2))
        masked = apply_mask(frame, mask)

        kept = ~mask.to_numpy()
        pd.testing.assert_series_equal(
            frame.loc[kept, "primary_ndvi"], masked.loc[kept, "primary_ndvi"]
        )

    def test_size_mismatch_is_rejected(self):
        frame = build_frame()
        with self.assertRaises(ValueError):
            apply_mask(frame, pd.Series([True, False]))


class SplitTests(unittest.TestCase):
    def test_holdout_polygons_are_absent_from_training(self):
        frame = build_frame()
        split = make_split(frame, MaskSpec(seed=5), holdout_fraction=0.5)

        training = set(split.train_source["anon_polygon_id"])
        self.assertTrue(split.holdout_polygons)
        self.assertFalse(training & set(split.holdout_polygons))

    def test_truth_is_removed_from_context(self):
        frame = build_frame()
        split = make_split(frame, MaskSpec(seed=5))

        self.assertEqual(len(split.truth), len(split.targets))
        self.assertTrue(split.truth.notna().all())
        self.assertTrue(
            split.context.loc[split.targets.index, "primary_ndvi"].isna().all()
        )

    def test_repeats_produce_different_masks(self):
        frame = build_frame()
        splits = list(iter_splits(frame, repeats=3, spec=MaskSpec(seed=11)))
        signatures = {tuple(sorted(split.targets.index)) for split in splits}

        self.assertEqual(len(splits), 3)
        self.assertEqual(len(signatures), 3)

    def test_zero_holdout_keeps_every_polygon(self):
        frame = build_frame()
        split = make_split(frame, MaskSpec(seed=5), holdout_fraction=0.0)

        self.assertEqual(split.holdout_polygons, frozenset())
        self.assertFalse(split.unseen_targets.any())


class ProfileTests(unittest.TestCase):
    def test_profile_reports_platform_shaped_metrics(self):
        # Выборка должна быть достаточно большой, чтобы биномиальный шум доли
        # масок был заметно меньше допуска проверки.
        frame = build_frame(polygons=20, days=800)
        masked = apply_mask(frame, generate_mask(frame, MaskSpec(rate=0.15, seed=9)))
        profile = gap_profile(masked)

        self.assertAlmostEqual(profile["mask_rate"], 0.15, delta=0.02)
        # Независимый Бернулли даёт геометрические длины серий: доля одиночных ~ 1-p.
        self.assertAlmostEqual(profile["run_share_1"], 0.85, delta=0.05)
        self.assertGreater(profile["n_gaps"], 0)

    def test_profile_requires_gap_column(self):
        with self.assertRaises(ValueError):
            gap_profile(build_frame())


if __name__ == "__main__":
    unittest.main()
