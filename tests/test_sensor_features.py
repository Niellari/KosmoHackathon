"""Проверки sensor-aware признаков без обучения тяжёлой модели."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.sensor_features import sensor_series_features, source_labels
from src.models.sensor import SensorAwareLightGBMModel


class SensorFeatureTests(unittest.TestCase):
    def setUp(self):
        dates = pd.to_datetime(
            ["2024-04-01", "2024-04-03", "2024-04-05", "2024-04-07"]
        )
        self.frame = pd.DataFrame(
            {
                "anon_polygon_id": ["A"] * 4,
                "date": dates,
                "crop_type": ["зерновые"] * 4,
                "s2_ndvi": [0.2, np.nan, 0.6, np.nan],
                "landsat_ndvi": [np.nan, 0.3, np.nan, 0.7],
                "modis_ndvi": [np.nan] * 4,
                "s2_evi": [0.1, np.nan, 0.5, np.nan],
                "s2_ndwi": [-0.2, np.nan, 0.2, np.nan],
                "landsat_evi": [np.nan, 0.2, np.nan, 0.6],
                "landsat_ndwi": [np.nan, -0.1, np.nan, 0.3],
                "modis_evi": [np.nan] * 4,
                "era5_temp_c": [10.0, np.nan, 14.0, 16.0],
                "era5_precip_mm": [0.0, np.nan, 4.0, 2.0],
                "primary_ndvi": [0.2, 0.3, 0.6, 0.7],
            }
        )

    def test_source_priority_matches_primary_ndvi_contract(self):
        labels = source_labels(self.frame)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])

    def test_sensor_interpolation_uses_same_sensor_neighbors(self):
        target = self.frame.iloc[[1]].copy()
        context = self.frame.copy()
        context.loc[target.index, "landsat_ndvi"] = np.nan
        probabilities = pd.DataFrame(
            {"p_s2": [1.0], "p_landsat": [0.0], "p_modis": [0.0]},
            index=target.index,
        )
        features = sensor_series_features(context, target, probabilities)

        self.assertAlmostEqual(features.iloc[0]["s2_interpolation"], 0.4)
        self.assertAlmostEqual(features.iloc[0]["source_interpolation"], 0.4)

    def test_harmonized_interpolation_corrects_neighbor_sensor_bias(self):
        context = self.frame.copy()
        context["s2_ndvi"] = [0.2, np.nan, np.nan, 0.6]
        context["landsat_ndvi"] = [0.3, 0.5, np.nan, 0.7]
        context["primary_ndvi"] = [0.2, 0.5, np.nan, 0.6]
        target = context.iloc[[2]].copy()
        probabilities = pd.DataFrame(
            {"p_s2": [1.0], "p_landsat": [0.0], "p_modis": [0.0]},
            index=target.index,
        )

        features = sensor_series_features(
            context, target, probabilities, feature_version=4
        )

        self.assertAlmostEqual(
            features.iloc[0]["harmonized_source_interpolation"], 0.5
        )

    def test_mask_normalizes_nullable_gap_flag(self):
        frame = self.frame.copy()
        frame["is_synthetic_gap"] = np.nan

        context, *_ = SensorAwareLightGBMModel._mask(frame, 1.0, 42)

        self.assertEqual(context["is_synthetic_gap"].dtype, bool)
        self.assertTrue(context["is_synthetic_gap"].all())

    def test_auxiliary_indices_are_interpolated_per_sensor(self):
        target = self.frame.iloc[[1]].copy()
        context = self.frame.copy()
        probabilities = pd.DataFrame(
            {"p_s2": [1.0], "p_landsat": [0.0], "p_modis": [0.0]},
            index=target.index,
        )

        features = sensor_series_features(
            context, target, probabilities, feature_version=5
        )

        self.assertAlmostEqual(features.iloc[0]["s2_evi_interpolation"], 0.3)
        self.assertAlmostEqual(features.iloc[0]["source_evi_interpolation"], 0.3)

    def test_weather_is_interpolated_within_polygon(self):
        target = self.frame.iloc[[1]].copy()
        probabilities = pd.DataFrame(
            {"p_s2": [1.0], "p_landsat": [0.0], "p_modis": [0.0]},
            index=target.index,
        )

        features = sensor_series_features(
            self.frame, target, probabilities, feature_version=6
        )

        self.assertAlmostEqual(features.iloc[0]["era5_temp_c_interpolation"], 12.0)
        self.assertAlmostEqual(features.iloc[0]["era5_precip_mm_interpolation"], 2.0)

    def test_polygon_identity_has_stable_schema_and_unknown_fallback(self):
        model = SensorAwareLightGBMModel("sensor", {"feature_version": 7})
        model.polygon_ids = ["A", "B"]
        rows = pd.DataFrame({"anon_polygon_id": ["B", "NEW"]}, index=[5, 6])

        features = model._polygon_features(rows)

        self.assertEqual(list(features.columns), ["polygon_A", "polygon_B"])
        self.assertEqual(features.loc[5].tolist(), [0.0, 1.0])
        self.assertEqual(features.loc[6].tolist(), [0.0, 0.0])

    def test_season_summary_excludes_hidden_target(self):
        context = self.frame.copy()
        context["year"] = context["date"].dt.year
        target = context.iloc[[1]].copy()
        context.loc[target.index, "primary_ndvi"] = np.nan
        probabilities = pd.DataFrame(
            {"p_s2": [1.0], "p_landsat": [0.0], "p_modis": [0.0]},
            index=target.index,
        )

        features = sensor_series_features(
            context, target, probabilities, feature_version=8
        )

        self.assertAlmostEqual(
            features.iloc[0]["primary_season_mean"], (0.2 + 0.6 + 0.7) / 3
        )
        self.assertEqual(features.iloc[0]["primary_season_count"], 3.0)

    def test_season_identity_has_unknown_fallback(self):
        model = SensorAwareLightGBMModel("sensor", {"feature_version": 9})
        model.season_ids = ["A__2024", "A__2025"]
        rows = pd.DataFrame(
            {"anon_polygon_id": ["A", "B"], "year": [2025, 2025]}, index=[7, 8]
        )

        features = model._season_identity_features(rows)

        self.assertEqual(features.loc[7].tolist(), [0.0, 1.0])
        self.assertEqual(features.loc[8].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
