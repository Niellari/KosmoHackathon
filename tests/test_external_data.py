"""Изоляция и веса именованных external-источников."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.config import ExternalDataConfig, ExternalSourceConfig
from src.data import load_external_training_data


class NamedExternalDataTests(unittest.TestCase):
    @staticmethod
    def _write_source(path: Path, polygon_id: str) -> None:
        pd.DataFrame(
            {
                "anon_polygon_id": [polygon_id],
                "date": ["2024-04-01"],
                "primary_ndvi": [0.42],
                "crop_type": [""],
            }
        ).to_csv(path, index=False)

    def test_sources_keep_independent_names_and_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zgd_path = root / "zgd.csv"
            egl_path = root / "egl.csv"
            self._write_source(zgd_path, "EXT-ZGD-0001")
            self._write_source(egl_path, "EXT-EGL-0001")
            config = ExternalDataConfig(
                enabled=True,
                sources=[
                    ExternalSourceConfig(
                        name="zernograd_osm",
                        path=zgd_path,
                        polygon_id_prefix="EXT-ZGD-",
                        sample_weight=0.5,
                    ),
                    ExternalSourceConfig(
                        name="egorlykskaya_osm",
                        path=egl_path,
                        polygon_id_prefix="EXT-EGL-",
                        sample_weight=0.1,
                    ),
                ],
            )

            result = load_external_training_data(config)

            self.assertEqual(
                set(result["_data_source"]),
                {"external:zernograd_osm", "external:egorlykskaya_osm"},
            )
            weights = result.groupby("_data_source")["_sample_weight"].first()
            self.assertEqual(weights["external:zernograd_osm"], 0.5)
            self.assertEqual(weights["external:egorlykskaya_osm"], 0.1)

    def test_zero_weight_excludes_source_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zgd_path = root / "zgd.csv"
            missing_path = root / "does-not-need-to-exist.csv"
            self._write_source(zgd_path, "EXT-ZGD-0001")
            config = ExternalDataConfig(
                enabled=True,
                sources=[
                    ExternalSourceConfig(
                        name="zernograd_osm",
                        path=zgd_path,
                        polygon_id_prefix="EXT-ZGD-",
                        sample_weight=0.5,
                    ),
                    ExternalSourceConfig(
                        name="disabled_by_weight",
                        path=missing_path,
                        sample_weight=0,
                    ),
                ],
            )

            result = load_external_training_data(config)

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["_data_source"], "external:zernograd_osm")


if __name__ == "__main__":
    unittest.main()
