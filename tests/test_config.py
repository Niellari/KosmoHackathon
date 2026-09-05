"""Проверки YAML-конфигурации и выбора моделей."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.config import AppConfig, load_config, select_model
from src.features import FeatureBuilder
from src.external_validation import (
    external_experiment_config,
    with_external_source_weights,
    with_external_weight,
)
from src.models import create_model
from src.models.boosting import HistoryRoutedLightGBMModel


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = load_config("config.yaml")

        self.assertEqual(config.models.selected, "routed_lightgbm")
        self.assertEqual(config.training.target_mode, "direct")
        self.assertEqual(config.training.residual_baseline, "linear")
        self.assertEqual(config.training.gap_masking.strategy, "leave_one_out")
        self.assertEqual(config.training.gap_masking.replicas, 5)
        self.assertTrue(config.features.temporal_dynamics.enabled)
        self.assertIn("catboost", config.models.available)
        self.assertIn("routed_lightgbm", config.models.available)
        self.assertEqual(config.predict.prediction_column, "primary_ndvi_true")
        self.assertFalse(config.data.external.enabled)
        self.assertEqual(len(config.data.external.sources), 2)
        self.assertTrue(config.data.external.sources[0].path.is_absolute())
        self.assertEqual(config.data.external.sources[0].sample_weight, 0.5)

    def test_cli_model_override_does_not_mutate_original(self):
        config = load_config("config.yaml")
        changed = select_model(config, "baseline")

        self.assertEqual(config.models.selected, "routed_lightgbm")
        self.assertEqual(changed.models.selected, "baseline")

    def test_unknown_config_fields_are_rejected(self):
        config = load_config("config.yaml").model_dump()
        config["server"]["unknown_option"] = True

        with self.assertRaises(ValidationError):
            AppConfig.model_validate(config)

    def test_residual_requires_enabled_baseline_feature(self):
        config = load_config("config.yaml").model_dump()
        config["training"]["target_mode"] = "residual"
        config["features"]["interpolation"]["linear"] = False

        with self.assertRaisesRegex(ValidationError, "requires|требует"):
            AppConfig.model_validate(config)

    def test_registry_builds_selected_model(self):
        config = select_model(load_config("config.yaml"), "baseline")
        definition = config.models.available["baseline"]

        model = create_model("baseline", definition)

        self.assertEqual(model.name, "baseline")

    def test_registry_builds_history_routed_lightgbm(self):
        config = select_model(load_config("config.yaml"), "routed_lightgbm")
        definition = config.models.available["routed_lightgbm"]

        model = create_model("routed_lightgbm", definition, config.training)

        self.assertIsInstance(model, HistoryRoutedLightGBMModel)
        self.assertEqual(model.min_reference_years, 1)

    def test_default_feature_schema_is_stable(self):
        builder = FeatureBuilder(load_config("config.yaml").features)

        self.assertEqual(len(builder.feature_names), 36)
        self.assertIn("linear", builder.feature_names)
        self.assertIn("pchip_prediction", builder.feature_names)
        self.assertNotIn("local_quadratic_prediction", builder.feature_names)
        self.assertNotIn("interpolation_range", builder.feature_names)
        self.assertIn("local_acceleration", builder.feature_names)
        self.assertIn("neighbor_range", builder.feature_names)
        self.assertIn("crop_winter_wheat", builder.feature_names)

    def test_prediction_schema_stays_numeric_without_polygon_history(self):
        builder = FeatureBuilder(load_config("config.yaml").features)
        context = pd.DataFrame(
            {
                "anon_polygon_id": ["new", "new", "new"],
                "date": pd.to_datetime(
                    ["2025-04-01", "2025-04-02", "2025-04-03"]
                ),
                "year": [2025, 2025, 2025],
                "doy": [91, 92, 93],
                "crop_type": ["зерновые"] * 3,
                "primary_ndvi": [0.2, np.nan, 0.3],
            }
        )
        targets = context.loc[[1]].copy()

        matrix, _ = builder.build_prediction_set(context, targets)

        self.assertTrue(
            all(np.issubdtype(dtype, np.number) for dtype in matrix.dtypes)
        )
        self.assertTrue(np.isnan(matrix.at[1, "historical"]))

    def test_recomputed_history_is_identical_in_train_and_prediction(self):
        config = load_config("config.yaml")
        history = config.features.polygon_history.model_copy(
            update={
                "calculation": "leave_one_season_out",
                "expanded_statistics": True,
            }
        )
        feature_config = config.features.model_copy(
            update={"polygon_history": history}
        )
        builder = FeatureBuilder(feature_config)
        frame = pd.DataFrame(
            {
                "anon_polygon_id": ["field"] * 6,
                "date": pd.to_datetime(
                    [
                        "2022-04-10",
                        "2022-04-20",
                        "2023-04-10",
                        "2023-04-20",
                        "2024-04-10",
                        "2024-04-20",
                    ]
                ),
                "year": [2022, 2022, 2023, 2023, 2024, 2024],
                "doy": [100, 110, 100, 110, 101, 111],
                "crop_type": ["зерновые"] * 6,
                "primary_ndvi": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            },
            index=[10, 11, 20, 21, 30, 31],
        )
        train_matrix, _ = builder.build_training_set(frame)
        targets = frame.loc[[20]].copy()
        context = frame.copy()
        context.loc[20, "primary_ndvi"] = np.nan

        prediction_matrix, _ = builder.build_prediction_set(context, targets)

        history_columns = [
            column
            for column in builder.feature_names
            if HistoryRoutedLightGBMModel._is_history_feature(column)
        ]
        self.assertEqual(len(builder.feature_names), 59)
        self.assertTrue(
            np.allclose(
                train_matrix.loc[20, history_columns],
                prediction_matrix.loc[20, history_columns],
                equal_nan=True,
            )
        )
        self.assertEqual(train_matrix.at[20, "n_reference_years_calc"], 2.0)
        self.assertTrue(
            np.isfinite(train_matrix.at[20, "historical_year_trend"])
        )

    def test_external_experiment_is_isolated_from_production_config(self):
        config = load_config("config.yaml")

        experiment = external_experiment_config(config)

        self.assertFalse(config.data.external.enabled)
        self.assertTrue(experiment.data.external.enabled)
        self.assertFalse(experiment.features.crop_type.enabled)
        self.assertFalse(experiment.features.crop_curve.enabled)
        self.assertIsNone(
            experiment.models.available[experiment.models.selected].artifact_path
        )

    def test_external_weight_override_is_validated(self):
        config = load_config("config.yaml")

        changed = with_external_weight(config, 0.5)

        weights = {
            source.name: source.sample_weight
            for source in changed.data.external.sources
        }
        self.assertEqual(weights["zernograd_osm"], 0.5)
        self.assertEqual(weights["egorlykskaya_osm"], 0.5)
        with self.assertRaises(ValidationError):
            with_external_weight(config, 1.5)

    def test_named_external_weight_override_is_independent(self):
        config = load_config("config.yaml")

        changed = with_external_source_weights(
            config, {"egorlykskaya_osm": 0.1}
        )

        self.assertEqual(changed.data.external.sources[0].sample_weight, 0.5)
        self.assertEqual(changed.data.external.sources[1].sample_weight, 0.1)
        with self.assertRaisesRegex(ValueError, "Неизвестные"):
            with_external_source_weights(config, {"missing": 0.1})


if __name__ == "__main__":
    unittest.main()
