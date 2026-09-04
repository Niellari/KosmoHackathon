"""Проверки YAML-конфигурации и выбора моделей."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config import AppConfig, load_config, select_model
from src.features import FeatureBuilder
from src.external_validation import external_experiment_config
from src.models import create_model


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = load_config("config.yaml")

        self.assertEqual(config.models.selected, "lightgbm")
        self.assertEqual(config.training.target_mode, "direct")
        self.assertEqual(config.training.residual_baseline, "linear")
        self.assertEqual(config.training.gap_masking.strategy, "leave_one_out")
        self.assertEqual(config.training.gap_masking.replicas, 5)
        self.assertTrue(config.features.temporal_dynamics.enabled)
        self.assertIn("catboost", config.models.available)
        self.assertEqual(config.predict.prediction_column, "primary_ndvi_true")
        self.assertFalse(config.data.external.enabled)
        self.assertEqual(len(config.data.external.paths), 1)
        self.assertTrue(config.data.external.paths[0].is_absolute())
        self.assertEqual(config.data.external.sample_weight, 0.25)

    def test_cli_model_override_does_not_mutate_original(self):
        config = load_config("config.yaml")
        changed = select_model(config, "baseline")

        self.assertEqual(config.models.selected, "lightgbm")
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


if __name__ == "__main__":
    unittest.main()
