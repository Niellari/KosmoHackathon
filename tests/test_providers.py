"""Тесты разбора контуров и склейки primary_ndvi. Сеть не требуется."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.providers import load_features, merge_primary
from src.providers.gee import OUTPUT_COLUMNS


SQUARE = {
    "type": "Polygon",
    "coordinates": [[[42.14, 47.14], [42.16, 47.14], [42.16, 47.16], [42.14, 47.16], [42.14, 47.14]]],
}


class MergePrimaryTest(unittest.TestCase):
    def test_priority_is_s2_then_landsat_then_modis(self):
        frame = pd.DataFrame(
            {
                "s2_ndvi": [0.5, np.nan, np.nan, np.nan],
                "landsat_ndvi": [0.6, 0.61, np.nan, np.nan],
                "modis_ndvi": [0.7, 0.71, 0.72, np.nan],
            }
        )
        result = merge_primary(frame)
        self.assertEqual(list(result[:3]), [0.5, 0.61, 0.72])
        self.assertTrue(np.isnan(result.iloc[3]))

    def test_missing_columns_do_not_break_merge(self):
        frame = pd.DataFrame({"modis_ndvi": [0.4, np.nan]})
        result = merge_primary(frame)
        self.assertEqual(result.iloc[0], 0.4)
        self.assertTrue(np.isnan(result.iloc[1]))

    def test_non_numeric_values_become_nan(self):
        frame = pd.DataFrame({"s2_ndvi": ["мусор", "0.3"]})
        result = merge_primary(frame)
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertAlmostEqual(result.iloc[1], 0.3)


class LoadFeaturesTest(unittest.TestCase):
    def test_feature_collection_keeps_identifier_and_crop(self):
        source = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": "AOI-EXT-1", "crop_type": "подсолнечник"},
                    "geometry": SQUARE,
                }
            ],
        }
        features = load_features(source)
        self.assertEqual(features[0]["anon_polygon_id"], "AOI-EXT-1")
        self.assertEqual(features[0]["crop_type"], "подсолнечник")

    def test_identifiers_are_generated_when_absent(self):
        source = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": SQUARE},
                {"type": "Feature", "properties": {}, "geometry": SQUARE},
            ],
        }
        features = load_features(source, default_crop="зерновые")
        self.assertEqual(
            [item["anon_polygon_id"] for item in features],
            ["AOI-EXT-0001", "AOI-EXT-0002"],
        )
        self.assertEqual(features[0]["crop_type"], "зерновые")

    def test_bare_geometry_is_accepted(self):
        features = load_features(SQUARE)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["geometry"], SQUARE)

    def test_path_to_geojson_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.geojson"
            path.write_text(json.dumps(SQUARE), encoding="utf-8")
            features = load_features(path)
        self.assertEqual(len(features), 1)

    def test_invalid_source_is_rejected(self):
        with self.assertRaises(ValueError):
            load_features(42)


class SchemaTest(unittest.TestCase):
    def test_output_schema_matches_competition_columns(self):
        competition = {
            "anon_polygon_id", "date", "s2_ndvi", "s2_evi", "s2_ndwi",
            "landsat_ndvi", "landsat_evi", "landsat_ndwi", "modis_ndvi",
            "modis_evi", "era5_temp_c", "era5_precip_mm", "year",
            "primary_ndvi", "doy", "crop_type",
        }
        self.assertEqual(set(OUTPUT_COLUMNS), competition)

    def test_example_regions_file_parses(self):
        features = load_features(Path("examples/regions.geojson"))
        self.assertEqual(len(features), 3)
        self.assertTrue(all(item["geometry"]["type"] == "Polygon" for item in features))


if __name__ == "__main__":
    unittest.main()
