"""Проверки ключевой бизнес-логики без обращения к внешним API."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import combine_context, load_dataset
from src.config import load_config, select_model
from src.interpolation import GapInterpolator, local_quadratic_value, pchip_value
from src.pipeline import AnalysisPipeline
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

    def test_external_rows_are_training_only(self):
        current = sample_frame()
        reference = sample_frame().copy()
        reference["anon_polygon_id"] = "AOI-REF"
        external = sample_frame().copy()
        external["anon_polygon_id"] = "EXT-1"
        external["_sample_weight"] = 0.25
        pipeline = AnalysisPipeline(
            current, reference=reference, training_extra=external
        )

        context = combine_context(pipeline.data, pipeline.reference)
        training = pipeline._training_source(pipeline.reference)

        self.assertNotIn("EXT-1", set(context["anon_polygon_id"]))
        self.assertIn("EXT-1", set(training["anon_polygon_id"]))
        weight = training.loc[
            training["anon_polygon_id"] == "EXT-1", "_sample_weight"
        ]
        self.assertTrue((weight == 0.25).all())

    def test_context_labels_can_be_used_for_transductive_training(self):
        current = sample_frame()
        current["anon_polygon_id"] = "AOI-NEW"
        reference = sample_frame()
        config = load_config("config.yaml")
        pipeline = AnalysisPipeline(current, reference=reference, config=config)
        context = combine_context(pipeline.data, pipeline.reference)

        training = pipeline._training_source(reference, context)

        self.assertIn("AOI-NEW", set(training["anon_polygon_id"]))
        self.assertEqual(
            training.loc[
                training["anon_polygon_id"] == "AOI-NEW", "primary_ndvi"
            ].notna().sum(),
            current["primary_ndvi"].notna().sum(),
        )


class InterpolationTests(unittest.TestCase):
    def test_pchip_preserves_monotonic_local_shape(self):
        points = [(-2, 0.2), (-1, 0.4), (1, 0.65), (2, 0.7)]

        prediction = pchip_value(points)

        self.assertGreaterEqual(prediction, 0.4)
        self.assertLessEqual(prediction, 0.65)

    def test_quadratic_recovers_local_peak(self):
        points = [(-2, 0.2), (-1, 0.5), (1, 0.5), (2, 0.2)]

        prediction = local_quadratic_value(points)

        self.assertAlmostEqual(prediction, 0.6)

    def test_interpolators_require_points_on_both_sides(self):
        points = [(-3, 0.2), (-1, 0.4)]

        self.assertTrue(np.isnan(pchip_value(points)))
        self.assertTrue(np.isnan(local_quadratic_value(points)))

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

    def test_pipeline_uses_model_selected_in_config(self):
        frame = sample_frame()
        config = select_model(load_config("config.yaml"), "baseline")
        pipeline = AnalysisPipeline(frame, config=config)

        result = pipeline.predict_targets(frame["primary_ndvi"].isna())

        self.assertEqual(set(result["model"]), {"baseline"})


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
