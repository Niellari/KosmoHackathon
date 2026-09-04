from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.collection.base import RAW_COLUMNS
from src.collection.config import (
    BoundingBoxConfig,
    CollectionConfig,
    CollectionOutputConfig,
    PeriodConfig,
    PolygonsConfig,
    RegionConfig,
    load_collection_config,
)
from src.collection.osm_fields import build_field_collection
from src.collection.prepare import build_external_dataset
from src.collection.runner import collect_observations, load_polygon_features


def _feature(polygon_id: str, lon: float = 40.3) -> dict:
    return {
        "type": "Feature",
        "properties": {"polygon_id": polygon_id, "crop_type": "неизвестно"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon, 46.85],
                    [lon + 0.01, 46.85],
                    [lon + 0.01, 46.86],
                    [lon, 46.86],
                    [lon, 46.85],
                ]
            ],
        },
    }


class FakeProvider:
    def __init__(self):
        self.calls: list[list[str]] = []

    def collect(self, features, id_property, start_date, end_date):
        ids = [str(item["properties"][id_property]) for item in features]
        self.calls.append(ids)
        rows = []
        for polygon_id in ids:
            rows.append(
                {
                    "polygon_id": polygon_id,
                    "date": start_date.isoformat(),
                    "sensor": "sentinel2",
                    "scene_id": "SCENE-1",
                    "ndvi": 0.42,
                    "evi": 0.25,
                    "ndwi": -0.31,
                    "valid_fraction": 0.9,
                    "pixel_count": 100,
                    "scene_cloud_percent": 5.0,
                }
            )
        return pd.DataFrame(rows, columns=RAW_COLUMNS)


class CollectionConfigTests(unittest.TestCase):
    def test_pilot_config_resolves_paths_from_config_directory(self):
        config = load_collection_config("configs/collection-pilot.yaml")
        project_root = Path(__file__).resolve().parent.parent
        self.assertEqual(
            config.polygons.path, project_root / "data/external/polygons.geojson"
        )
        self.assertEqual(config.period.start, date(2024, 4, 1))
        self.assertEqual(config.polygons.id_prefix, "EXT-ZGD-")
        self.assertTrue(config.sensors.sentinel2.enabled)

    def test_invalid_period_is_rejected(self):
        with self.assertRaises(ValueError):
            PeriodConfig(start=date(2024, 10, 30), end=date(2024, 4, 1))


class CollectionRunnerTests(unittest.TestCase):
    def _write_geojson(self, directory: Path) -> Path:
        path = directory / "polygons.geojson"
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [_feature("EXT-1"), _feature("EXT-2", 40.5)],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def _config(self, directory: Path, polygon_path: Path) -> CollectionConfig:
        return CollectionConfig(
            region=RegionConfig(
                name="test",
                bbox=BoundingBoxConfig(
                    min_lon=39.5,
                    min_lat=46.5,
                    max_lon=42.5,
                    max_lat=47.8,
                ),
            ),
            polygons=PolygonsConfig(path=polygon_path),
            period=PeriodConfig(start=date(2024, 4, 1), end=date(2024, 10, 30)),
            output=CollectionOutputConfig(
                raw_directory=directory / "raw",
                manifest_directory=directory / "manifests",
                dataset_path=directory / "external.csv",
            ),
        )

    def test_collect_writes_csv_manifest_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            polygon_path = self._write_geojson(root)
            config = self._config(root, polygon_path)
            config = config.model_copy(
                update={
                    "execution": config.execution.model_copy(
                        update={"batch_size": 1}
                    )
                }
            )
            output = root / "raw" / "sentinel2.csv"
            provider = FakeProvider()

            collect_observations(
                config,
                provider,
                "sentinel2",
                config.period.start,
                config.period.end,
                output,
                limit=2,
            )
            self.assertEqual(provider.calls, [["EXT-1"], ["EXT-2"]])
            self.assertEqual(len(pd.read_csv(output)), 2)

            manifest_path = root / "manifests" / "sentinel2.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["completed_polygon_ids"], ["EXT-1", "EXT-2"])
            self.assertEqual(manifest["rows"], 2)

            resumed_provider = FakeProvider()
            collect_observations(
                config,
                resumed_provider,
                "sentinel2",
                config.period.start,
                config.period.end,
                output,
                limit=2,
            )
            self.assertEqual(resumed_provider.calls, [])
            self.assertEqual(len(pd.read_csv(output)), 2)

    def test_duplicate_polygon_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicates.geojson"
            path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [_feature("EXT-1"), _feature("EXT-1", 40.5)],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Повторяющийся polygon_id"):
                load_polygon_features(path, "polygon_id")


class FieldDiscoveryTests(unittest.TestCase):
    def test_build_field_collection_filters_area_and_assigns_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CollectionRunnerTests()._config(root, root / "fields.geojson")
            payload = {
                "elements": [
                    {
                        "id": 123,
                        "geometry": [
                            {"lon": 40.30, "lat": 46.84},
                            {"lon": 40.31, "lat": 46.84},
                            {"lon": 40.31, "lat": 46.85},
                            {"lon": 40.30, "lat": 46.85},
                            {"lon": 40.30, "lat": 46.84},
                        ],
                    },
                    {"id": 456, "geometry": [{"lon": 40.3, "lat": 46.8}]},
                ]
            }

            result = build_field_collection(payload, config=config, limit=10)

            self.assertEqual(len(result["features"]), 1)
            properties = result["features"][0]["properties"]
            self.assertEqual(properties["polygon_id"], "EXT-0001")
            self.assertEqual(properties["osm_way_id"], 123)
            self.assertEqual(properties["source_license"], "ODbL")


class ExternalDatasetTests(unittest.TestCase):
    def test_daily_grid_aggregates_tiles_and_marks_crop_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CollectionRunnerTests()._config(root, root / "fields.geojson")
            config = config.model_copy(
                update={
                    "period": PeriodConfig(
                        start=date(2024, 4, 1), end=date(2024, 4, 2)
                    )
                }
            )
            raw_path = root / "sentinel2.csv"
            pd.DataFrame(
                {
                    "polygon_id": ["EXT-1", "EXT-1"],
                    "date": ["2024-04-01", "2024-04-01"],
                    "ndvi": [0.2, 0.4],
                    "evi": [0.1, 0.3],
                    "ndwi": [-0.4, -0.2],
                }
            ).to_csv(raw_path, index=False)

            result = build_external_dataset(config, raw_path)

            self.assertEqual(len(result), 2)
            self.assertEqual(result["anon_polygon_id"].nunique(), 1)
            self.assertAlmostEqual(result.loc[0, "s2_ndvi"], 0.3)
            self.assertAlmostEqual(result.loc[0, "primary_ndvi"], 0.3)
            self.assertTrue(pd.isna(result.loc[1, "primary_ndvi"]))
            self.assertEqual(set(result["crop_type"]), {"неизвестно"})

    def test_multiyear_dataset_builds_cross_year_climatology(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CollectionRunnerTests()._config(root, root / "fields.geojson")
            paths = []
            for year, ndvi in [(2023, 0.4), (2024, 0.6)]:
                path = root / f"sentinel2-{year}.csv"
                pd.DataFrame(
                    {
                        "polygon_id": ["EXT-1"],
                        "date": [f"{year}-04-01"],
                        "ndvi": [ndvi],
                        "evi": [ndvi / 2],
                        "ndwi": [-ndvi],
                    }
                ).to_csv(path, index=False)
                paths.append(path)

            result = build_external_dataset(config, paths)
            observed = result[result["primary_ndvi"].notna()].sort_values("year")

            self.assertEqual(len(result), 2 * 213)
            self.assertEqual(observed["n_reference_years"].tolist(), [1, 1])
            self.assertAlmostEqual(
                float(observed.iloc[0]["ndvi_climatology_mean"]), 0.6
            )
            self.assertAlmostEqual(
                float(observed.iloc[1]["ndvi_climatology_mean"]), 0.4
            )

    def test_multiyear_climatology_can_assign_text_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CollectionRunnerTests()._config(root, root / "fields.geojson")
            paths = []
            for year, values in [(2023, [0.3, 0.5]), (2024, [0.6, 0.8])]:
                path = root / f"sentinel2-{year}.csv"
                pd.DataFrame(
                    {
                        "polygon_id": ["EXT-1", "EXT-1"],
                        "date": [f"{year}-04-01", f"{year}-04-02"],
                        "ndvi": values,
                        "evi": values,
                        "ndwi": [-value for value in values],
                    }
                ).to_csv(path, index=False)
                paths.append(path)

            result = build_external_dataset(config, paths)
            observed = result[result["primary_ndvi"].notna()]

            self.assertTrue(observed["status"].notna().all())


if __name__ == "__main__":
    unittest.main()
