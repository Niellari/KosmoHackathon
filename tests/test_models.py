"""Проверки общего поведения обучаемых моделей."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.config import TrainingConfig
from src.models.boosting import TabularGapModel


class RecordingEstimator:
    def __init__(self):
        self.fitted_target: np.ndarray | None = None
        self.prediction = np.array([0.1])

    def fit(self, matrix, target):
        self.fitted_target = target.to_numpy(float)
        return self

    def predict(self, matrix):
        return np.resize(self.prediction, len(matrix))


class TestGapModel(TabularGapModel):
    __test__ = False

    def _create_estimator(self):
        return RecordingEstimator()


class ResidualTrainingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
