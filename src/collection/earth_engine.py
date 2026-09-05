"""Получение Sentinel-2 статистик по полигонам через Google Earth Engine."""

from __future__ import annotations

from datetime import date, timedelta
import os

import numpy as np
import pandas as pd

from src.collection.base import RAW_COLUMNS
from src.collection.config import CollectionConfig


class EarthEngineUnavailableError(RuntimeError):
    """Earth Engine не установлен или не авторизован."""


class EarthEngineSentinel2Provider:
    def __init__(self, config: CollectionConfig):
        self.config = config
        try:
            import ee
        except ImportError as error:
            raise EarthEngineUnavailableError(
                "Не установлен earthengine-api. Выполните: "
                "pip install -r requirements-collect.txt"
            ) from error

        project_env = config.provider.project_id_env
        project_id = os.environ.get(project_env)
        if not project_id:
            raise EarthEngineUnavailableError(
                f"Не задана переменная окружения {project_env} с ID Google Cloud проекта"
            )
        try:
            ee.Initialize(project=project_id)
        except Exception as error:  # SDK использует разные классы ошибок по версиям.
            raise EarthEngineUnavailableError(
                "Не удалось инициализировать Earth Engine. Сначала выполните "
                "earthengine authenticate и проверьте Google Cloud project."
            ) from error
        self.ee = ee

    def _feature_collection(self, features: list[dict], id_property: str):
        ee_features = []
        for feature in features:
            polygon_id = str(feature["properties"][id_property])
            geometry = self.ee.Geometry(feature["geometry"])
            ee_features.append(
                self.ee.Feature(geometry, {"polygon_id": polygon_id})
            )
        return self.ee.FeatureCollection(ee_features)

    def _prepare_image(self, image):
        ee = self.ee
        sensor = self.config.sensors.sentinel2
        scl = image.select("SCL")
        valid = ee.Image.constant(1)
        for value in sensor.excluded_scl:
            valid = valid.And(scl.neq(value))

        blue = image.select("B2").multiply(0.0001)
        green = image.select("B3").multiply(0.0001)
        red = image.select("B4").multiply(0.0001)
        nir = image.select("B8").multiply(0.0001)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            .rename("evi")
        )
        ndwi = green.subtract(nir).divide(green.add(nir)).rename("ndwi")
        indices = ee.Image.cat([ndvi, evi, ndwi]).updateMask(valid)

        source_mask = image.select("B8").mask()
        valid_pixel = ee.Image.constant(1).rename("valid_pixel").updateMask(valid)
        total_pixel = (
            ee.Image.constant(1).rename("total_pixel").updateMask(source_mask)
        )
        return indices.addBands(valid_pixel).addBands(total_pixel).copyProperties(
            image, image.propertyNames()
        )

    def cropland_fractions(
        self, features: list[dict], id_property: str
    ) -> dict[str, float | None]:
        """Доля cropland WorldCover внутри каждого точного контура поля."""

        if not features:
            return {}
        validation = self.config.field_validation
        polygons = self._feature_collection(features, id_property)
        landcover = (
            self.ee.ImageCollection(validation.worldcover_collection)
            .first()
            .select(validation.worldcover_band)
        )
        cropland = landcover.eq(validation.cropland_class).rename(
            "cropland_fraction"
        )
        reduced = cropland.reduceRegions(
            collection=polygons,
            reducer=self.ee.Reducer.mean(),
            scale=validation.scale_m,
        ).getInfo()
        result: dict[str, float | None] = {}
        for feature in reduced.get("features", []):
            properties = feature.get("properties", {})
            polygon_id = str(properties.get("polygon_id"))
            result[polygon_id] = _number(
                properties.get("cropland_fraction", properties.get("mean"))
            )
        return result

    def collect(
        self,
        features: list[dict],
        id_property: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        if not features:
            return pd.DataFrame(columns=RAW_COLUMNS)

        ee = self.ee
        sensor = self.config.sensors.sentinel2
        polygons = self._feature_collection(features, id_property)
        end_exclusive = end_date + timedelta(days=1)
        images = (
            ee.ImageCollection(sensor.collection)
            .filterDate(start_date.isoformat(), end_exclusive.isoformat())
            .filterBounds(polygons.geometry())
            .filter(
                ee.Filter.lte(
                    "CLOUDY_PIXEL_PERCENTAGE", sensor.max_scene_cloud_percent
                )
            )
            .map(self._prepare_image)
            .sort("system:time_start")
        )
        scene_count = int(images.size().getInfo())
        if scene_count == 0:
            return pd.DataFrame(columns=RAW_COLUMNS)

        reducer = ee.Reducer.median().combine(
            reducer2=ee.Reducer.count(), sharedInputs=True
        )
        image_list = images.toList(scene_count)
        records: list[dict] = []

        # Для пилота один серверный reduceRegions выполняется на сцену. Такой режим
        # проще возобновлять; для большого сбора его можно заменить batch-export.
        for index in range(scene_count):
            image = ee.Image(image_list.get(index))
            reduced = image.reduceRegions(
                collection=polygons,
                reducer=reducer,
                scale=sensor.scale_m,
            )
            enriched = reduced.map(
                lambda feature: feature.set(
                    {
                        "date": image.date().format("YYYY-MM-dd"),
                        "scene_id": image.id(),
                        "scene_cloud_percent": image.get(
                            "CLOUDY_PIXEL_PERCENTAGE"
                        ),
                    }
                )
            )
            payload = enriched.getInfo()
            for feature in payload.get("features", []):
                props = feature.get("properties", {})
                valid_count = _number(props.get("valid_pixel_count"))
                total_count = _number(props.get("total_pixel_count"))
                valid_fraction = (
                    valid_count / total_count
                    if total_count is not None and total_count > 0 and valid_count is not None
                    else np.nan
                )
                if valid_count is None or valid_count < sensor.min_pixel_count:
                    continue
                if np.isfinite(valid_fraction) and valid_fraction < sensor.min_valid_fraction:
                    continue
                records.append(
                    {
                        "polygon_id": str(props.get("polygon_id")),
                        "date": props.get("date"),
                        "sensor": "sentinel2",
                        "scene_id": props.get("scene_id"),
                        "ndvi": _number(props.get("ndvi_median")),
                        "evi": _number(props.get("evi_median")),
                        "ndwi": _number(props.get("ndwi_median")),
                        "valid_fraction": valid_fraction,
                        "pixel_count": int(valid_count),
                        "scene_cloud_percent": _number(
                            props.get("scene_cloud_percent")
                        ),
                    }
                )

        result = pd.DataFrame(records, columns=RAW_COLUMNS)
        if result.empty:
            return result
        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
        return result.sort_values(["polygon_id", "date", "scene_id"]).reset_index(
            drop=True
        )


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
