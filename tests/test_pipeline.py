"""Проверки ключевой бизнес-логики без обращения к внешним API."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.interpolation import GapInterpolator
from src.submission import validate_submission


def sample_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2024-04-03"]
    )
    frame = pd.DataFrame(
        {
            "anon_polygon_id": ["AOI-X"] * 5,
            "date": dates,
            "primary_ndvi": [0.2, np.nan, np.nan, 0.8, 0.45],
            "crop_type": ["зерновые"] * 5,
        }
    )
    frame["year"] = frame["date"].dt.year
    frame["doy"] = frame["date"].dt.dayofyear
    return frame


class DataTests(unittest.TestCase):
    def test_loader_restores_calendar_columns(self):
        frame = sample_frame().drop(columns=["year", "doy"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            loaded = load_dataset(path)
        self.assertEqual(int(loaded.loc[0, "year"]), 2024)
        self.assertEqual(int(loaded.loc[0, "doy"]), 94)


class InterpolationTests(unittest.TestCase):
    def test_consecutive_gaps_use_surrounding_observations(self):
        frame = sample_frame()
        targets = frame.iloc[1:3]
        result = GapInterpolator(frame).predict(
            targets, method="baseline", exclude_all_targets=True
        )
        self.assertTrue(np.allclose(result["prediction"], [0.5, 0.5]))

    def test_ensemble_returns_finite_values(self):
        frame = sample_frame()
        result = GapInterpolator(frame).predict(frame.iloc[1:3])
        self.assertTrue(np.isfinite(result["prediction"]).all())
        self.assertTrue(result["confidence"].between(0, 1).all())


class SubmissionTests(unittest.TestCase):
    def test_valid_submission(self):
        expected = pd.DataFrame(
            {"anon_polygon_id": ["AOI-X"], "date": ["2025-04-02"]}
        )
        submission = expected.copy()
        submission["primary_ndvi_true"] = [0.4]
        validate_submission(submission, expected)

    def test_nan_is_rejected(self):
        expected = pd.DataFrame(
            {"anon_polygon_id": ["AOI-X"], "date": ["2025-04-02"]}
        )
        submission = expected.copy()
        submission["primary_ndvi_true"] = [np.nan]
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_submission(submission, expected)


if __name__ == "__main__":
    unittest.main()
