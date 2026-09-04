"""Проверки test-like маскирования и локальной временной динамики."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import (
    FeaturesConfig,
    GapMaskingConfig,
    InterpolationConfig,
)
from src.features import FeatureBuilder
from src.training import TestLikeGapGenerator


def temporal_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "anon_polygon_id": ["AOI-X"] * 5,
            "date": pd.date_range("2025-04-01", periods=5, freq="D"),
            "primary_ndvi": [0.1, 0.2, np.nan, 0.6, 0.8],
            "crop_type": ["зерновые"] * 5,
        }
    )
    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["doy"] = frame["date"].dt.dayofyear.astype("int16")
    frame["s2_ndvi"] = [0.11, 0.21, np.nan, 0.61, 0.81]
    return frame


class TestLikeGapGeneratorTests(unittest.TestCase):
    def test_mask_is_deterministic_and_target_is_hidden(self):
        frame = temporal_frame()
        extra = frame.copy()
        extra["anon_polygon_id"] = "AOI-Y"
        frame = pd.concat([frame, extra], ignore_index=True)
        config = GapMaskingConfig(
            target_fraction=0.4,
            replicas=2,
            block_length_weights={1: 1.0},
            random_state=7,
        )

        first = list(TestLikeGapGenerator(config).generate(frame))
        second = list(TestLikeGapGenerator(config).generate(frame))

        self.assertEqual(len(first), 2)
        self.assertEqual(
            first[0].targets.index.tolist(), second[0].targets.index.tolist()
        )
        self.assertTrue(first[0].targets["primary_ndvi"].isna().all())
        self.assertTrue(first[0].targets["s2_ndvi"].isna().all())
        self.assertTrue(first[0].targets["is_synthetic_gap"].all())
        masked_primary = first[0].context.loc[
            first[0].targets.index, "primary_ndvi"
        ]
        self.assertTrue(masked_primary.isna().all())
        self.assertTrue(first[0].truth.notna().all())
        self.assertTrue(first[0].truth.index.equals(first[0].targets.index))
        self.assertTrue(first[0].targets["year"].notna().all())
        self.assertTrue(first[0].targets["doy"].notna().all())

    def test_generated_blocks_keep_requested_length(self):
        config = GapMaskingConfig(
            target_fraction=0.12,
            replicas=1,
            block_length_weights={3: 1.0},
            random_state=11,
        )
        selected = TestLikeGapGenerator(config)._sample_positions(
            100, np.random.default_rng(11)
        )
        runs = np.split(selected, np.where(np.diff(selected) != 1)[0] + 1)

        self.assertEqual([len(run) for run in runs], [3, 3, 3, 3])


class TemporalDynamicsTests(unittest.TestCase):
    def test_prediction_features_describe_local_shape(self):
        frame = temporal_frame()
        targets = frame.loc[[2]]

        feature_config = FeaturesConfig(
            interpolation=InterpolationConfig(
                pchip=True,
                local_quadratic=True,
                differences=True,
                agreement=True,
            )
        )
        matrix, _ = FeatureBuilder(feature_config).build_prediction_set(frame, targets)
        row = matrix.loc[2]

        self.assertAlmostEqual(row["gap_span_days"], 2.0)
        self.assertAlmostEqual(row["gap_position"], 0.5)
        self.assertAlmostEqual(row["neighbor_asymmetry"], 0.0)
        self.assertAlmostEqual(row["slope_before"], 0.1)
        self.assertAlmostEqual(row["slope_after"], 0.2)
        self.assertAlmostEqual(row["slope_between"], 0.2)
        self.assertAlmostEqual(row["slope_change"], 0.1)
        self.assertAlmostEqual(row["local_acceleration"], 0.1)
        self.assertAlmostEqual(row["neighbor_mean"], 0.425)
        self.assertAlmostEqual(row["neighbor_range"], 0.7)
        self.assertTrue(np.isfinite(row["pchip_prediction"]))
        self.assertTrue(np.isfinite(row["local_quadratic_prediction"]))
        self.assertAlmostEqual(
            row["pchip_minus_linear"],
            row["pchip_prediction"] - row["linear"],
        )
        self.assertGreaterEqual(row["interpolation_range"], 0.0)


if __name__ == "__main__":
    unittest.main()
