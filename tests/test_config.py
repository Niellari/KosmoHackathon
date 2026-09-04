"""Проверки YAML-конфигурации и выбора моделей."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config import AppConfig, load_config, select_model
from src.features import FeatureBuilder
from src.models import create_model


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = load_config("config.yaml")

        self.assertEqual(config.models.selected, "lightgbm")
        self.assertIn("catboost", config.models.available)
        self.assertEqual(config.predict.prediction_column, "primary_ndvi_true")

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

    def test_registry_builds_selected_model(self):
        config = select_model(load_config("config.yaml"), "baseline")
        definition = config.models.available["baseline"]

        model = create_model("baseline", definition)

        self.assertEqual(model.name, "baseline")

    def test_default_feature_schema_is_stable(self):
        builder = FeatureBuilder(load_config("config.yaml").features)

        self.assertEqual(len(builder.feature_names), 20)
        self.assertIn("linear", builder.feature_names)
        self.assertIn("crop_winter_wheat", builder.feature_names)


if __name__ == "__main__":
    unittest.main()
