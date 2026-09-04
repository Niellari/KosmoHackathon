"""Проверки общего поведения обучаемых моделей."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.config import GapMaskingConfig, TrainingConfig
from src.models.boosting import HistoryRoutedLightGBMModel, TabularGapModel


class RecordingEstimator:
    def __init__(self):
        self.fitted_target: np.ndarray | None = None
        self.fitted_sample_weight: np.ndarray | None = None
        self.fitted_columns: list[str] | None = None
        self.prediction = np.array([0.1])

    def fit(self, matrix, target, sample_weight=None):
        self.fitted_columns = list(matrix.columns)
        self.fitted_target = target.to_numpy(float)
        if sample_weight is not None:
            self.fitted_sample_weight = np.asarray(sample_weight, dtype=float)
        return self

    def predict(self, matrix):
        return np.resize(self.prediction, len(matrix))


class TestGapModel(TabularGapModel):
    __test__ = False

    def _create_estimator(self):
        return RecordingEstimator()


class TestHistoryRoutedModel(HistoryRoutedLightGBMModel):
    __test__ = False

    def _create_lightgbm(self, params: dict):
        estimator = RecordingEstimator()
        estimator.prediction = np.array([params["prediction"]])
        return estimator


class ResidualTrainingTests(unittest.TestCase):
    def test_external_sample_weights_are_passed_to_estimator(self):
        train = pd.DataFrame(
            {"_sample_weight": [1.0, 0.25]}, index=[10, 11]
        )
        features = MagicMock()
        features.build_training_set.return_value = (
            pd.DataFrame({"linear": [0.4, 0.5]}, index=[10, 11]),
            pd.Series([0.45, 0.4], index=[10, 11]),
        )
        model = TestGapModel("test")

        model.fit(train, features)

        self.assertTrue(
            np.allclose(model.estimator.fitted_sample_weight, [1.0, 0.25])
        )

    def test_test_like_strategy_uses_prediction_feature_path(self):
        train = pd.DataFrame(
            {
                "anon_polygon_id": ["AOI-X"] * 5,
                "date": pd.date_range("2025-04-01", periods=5, freq="D"),
                "year": [2025] * 5,
                "doy": range(91, 96),
                "crop_type": ["зерновые"] * 5,
                "primary_ndvi": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        features = MagicMock()

        def prediction_features(context, targets):
            return (
                pd.DataFrame({"linear": [0.25] * len(targets)}, index=targets.index),
                pd.DataFrame(
                    {"prediction": [0.25] * len(targets)}, index=targets.index
                ),
            )

        features.build_prediction_set.side_effect = prediction_features
        model = TestGapModel(
            "test",
            training=TrainingConfig(
                gap_masking=GapMaskingConfig(
                    strategy="test_like_blocks",
                    target_fraction=0.2,
                    replicas=2,
                    block_length_weights={1: 1.0},
                    random_state=3,
                )
            ),
        )

        model.fit(train, features)

        features.build_training_set.assert_not_called()
        self.assertEqual(features.build_prediction_set.call_count, 2)
        self.assertEqual(len(model.estimator.fitted_target), 2)

    def test_model_learns_residual_and_adds_baseline_back(self):
        features = MagicMock()
        features.build_training_set.return_value = (
            pd.DataFrame({"linear": [0.4, 0.5]}, index=[10, 11]),
            pd.Series([0.45, 0.4], index=[10, 11]),
        )
        features.build_prediction_set.return_value = (
            pd.DataFrame({"linear": [0.6]}, index=[20]),
            pd.DataFrame({"prediction": [0.55]}, index=[20]),
        )
        model = TestGapModel(
            "test",
            training=TrainingConfig(
                target_mode="residual", residual_baseline="linear"
            ),
        )

        model.fit(pd.DataFrame(), features)
        result = model.predict(pd.DataFrame(), pd.DataFrame(index=[20]), features)

        self.assertTrue(np.allclose(model.estimator.fitted_target, [0.05, -0.1]))
        self.assertAlmostEqual(float(result.at[20, "prediction"]), 0.7)
        self.assertEqual(result.at[20, "target_mode"], "residual")

    def test_missing_residual_baseline_uses_heuristic_prediction(self):
        features = MagicMock()
        features.build_training_set.return_value = (
            pd.DataFrame({"linear": [0.4]}, index=[10]),
            pd.Series([0.45], index=[10]),
        )
        features.build_prediction_set.return_value = (
            pd.DataFrame({"linear": [np.nan]}, index=[20]),
            pd.DataFrame({"prediction": [0.33]}, index=[20]),
        )
        model = TestGapModel(
            "test",
            training=TrainingConfig(
                target_mode="residual", residual_baseline="linear"
            ),
        )

        model.fit(pd.DataFrame(), features)
        result = model.predict(pd.DataFrame(), pd.DataFrame(index=[20]), features)

        self.assertAlmostEqual(float(result.at[20, "prediction"]), 0.33)


class HistoryRoutedModelTests(unittest.TestCase):
    def test_two_experts_use_different_rows_and_columns(self):
        train = pd.DataFrame(index=[10, 11, 12])
        features = MagicMock()
        features.build_training_set.return_value = (
            pd.DataFrame(
                {
                    "linear": [0.4, 0.3, 0.5],
                    "historical": [0.45, np.nan, 0.55],
                    "historical_std": [0.02, np.nan, 0.03],
                    "n_reference_years_calc": [3.0, 0.0, 2.0],
                    "temperature": [15.0, 17.0, 19.0],
                },
                index=[10, 11, 12],
            ),
            pd.Series([0.46, 0.31, 0.56], index=[10, 11, 12]),
        )
        features.build_prediction_set.return_value = (
            pd.DataFrame(
                {
                    "linear": [0.5, 0.2],
                    "historical": [0.48, np.nan],
                    "historical_std": [0.02, np.nan],
                    "n_reference_years_calc": [4.0, 0.0],
                    "temperature": [18.0, 20.0],
                },
                index=[20, 21],
            ),
            pd.DataFrame({"prediction": [0.5, 0.2]}, index=[20, 21]),
        )
        model = TestHistoryRoutedModel(
            "routed",
            params={
                "history_rich_params": {"prediction": 0.8},
                "cold_start_params": {"prediction": 0.25},
            },
        )

        model.fit(train, features)
        result = model.predict(pd.DataFrame(), pd.DataFrame(), features)

        self.assertTrue(
            np.allclose(model.history_estimator.fitted_target, [0.46, 0.56])
        )
        self.assertTrue(
            np.allclose(
                model.cold_start_estimator.fitted_target,
                [0.46, 0.31, 0.56],
            )
        )
        self.assertIn("historical", model.history_estimator.fitted_columns)
        self.assertNotIn("historical", model.cold_start_estimator.fitted_columns)
        self.assertTrue(np.allclose(result["prediction"], [0.8, 0.25]))
        self.assertEqual(result["route"].tolist(), ["history_rich", "cold_start"])


if __name__ == "__main__":
    unittest.main()
