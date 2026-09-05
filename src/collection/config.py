"""Строгая конфигурация отдельного pipeline сбора данных."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBoxConfig(StrictModel):
    min_lon: float = Field(ge=-180, le=180)
    min_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> "BoundingBoxConfig":
        if self.min_lon >= self.max_lon or self.min_lat >= self.max_lat:
            raise ValueError("Минимальные координаты bbox должны быть меньше максимальных")
        return self


class RegionConfig(StrictModel):
    name: str = Field(min_length=1)
    bbox: BoundingBoxConfig


class PolygonSelectionConfig(StrictModel):
    limit: int = Field(default=10, ge=1)
    min_area_ha: float = Field(default=20, gt=0)
    max_area_ha: float = Field(default=500, gt=0)
    min_cropland_fraction: float = Field(default=0.8, ge=0, le=1)
    strategy: Literal["nearest_center", "spatially_balanced"] = "nearest_center"
    min_centroid_distance_km: float = Field(default=0, ge=0)
    grid_cell_km: float = Field(default=10, gt=0)
    max_per_grid_cell: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def area_range_is_ordered(self) -> "PolygonSelectionConfig":
        if self.min_area_ha >= self.max_area_ha:
            raise ValueError("min_area_ha должен быть меньше max_area_ha")
        return self


class PolygonsConfig(StrictModel):
    path: Path
    id_property: str = Field(default="polygon_id", min_length=1)
    id_prefix: str = Field(default="EXT-", min_length=1)
    selection: PolygonSelectionConfig = PolygonSelectionConfig()


class FieldValidationConfig(StrictModel):
    enabled: bool = False
    candidate_pool_size: int = Field(default=30, ge=1, le=200)
    worldcover_collection: str = "ESA/WorldCover/v200"
    worldcover_band: str = "Map"
    cropland_class: int = 40
    scale_m: int = Field(default=10, ge=10, le=1000)
    min_valid_sentinel2_observations: int = Field(default=12, ge=1)


class WeatherZoneSearchConfig(StrictModel):
    dataset: str = "ECMWF/ERA5_LAND/DAILY_AGGR"
    temperature_band: str = "temperature_2m"
    precipitation_band: str = "total_precipitation_sum"
    grid_step_degrees: float = Field(default=0.2, gt=0, le=2)
    sample_dates: int = Field(default=24, ge=4, le=120)
    min_profile_dates: int = Field(default=4, ge=2, le=120)
    top_zones: int = Field(default=6, ge=1, le=30)
    min_zone_distance_km: float = Field(default=30, ge=0)
    min_year: int = Field(default=2019, ge=1950, le=2100)
    max_year: int = Field(default=2024, ge=1950, le=2100)

    @model_validator(mode="after")
    def years_are_ordered(self) -> "WeatherZoneSearchConfig":
        if self.min_year > self.max_year:
            raise ValueError("weather_zones.min_year должен быть не больше max_year")
        if self.min_profile_dates > self.sample_dates:
            raise ValueError("min_profile_dates должен быть не больше sample_dates")
        return self


class PeriodConfig(StrictModel):
    start: date
    end: date

    @model_validator(mode="after")
    def period_is_ordered(self) -> "PeriodConfig":
        if self.start > self.end:
            raise ValueError("Начало периода сбора должно быть не позже окончания")
        return self


class ProviderConfig(StrictModel):
    type: Literal["earth_engine"] = "earth_engine"
    project_id_env: str = Field(default="EE_PROJECT_ID", min_length=1)


class Sentinel2Config(StrictModel):
    enabled: bool = True
    collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    scale_m: int = Field(default=20, ge=10, le=1000)
    reducer: Literal["median"] = "median"
    max_scene_cloud_percent: float = Field(default=80, ge=0, le=100)
    min_valid_fraction: float = Field(default=0.6, ge=0, le=1)
    min_pixel_count: int = Field(default=50, ge=1)
    excluded_scl: list[int] = Field(
        default_factory=lambda: [0, 1, 3, 7, 8, 9, 10, 11]
    )


class ToggleSensorConfig(StrictModel):
    enabled: bool = False


class SensorsConfig(StrictModel):
    sentinel2: Sentinel2Config = Sentinel2Config()
    landsat: ToggleSensorConfig = ToggleSensorConfig()
    modis: ToggleSensorConfig = ToggleSensorConfig()
    era5: ToggleSensorConfig = ToggleSensorConfig()


class CollectionOutputConfig(StrictModel):
    raw_directory: Path = Path("artifacts/collection/raw")
    manifest_directory: Path = Path("artifacts/collection/manifests")
    dataset_path: Path = Path("data/external/processed/external_dataset.csv")
    format: Literal["csv"] = "csv"


class ExecutionConfig(StrictModel):
    batch_size: int = Field(default=5, ge=1, le=100)
    retries: int = Field(default=3, ge=0, le=10)
    resume: bool = True
    overwrite: bool = False


class ProcessingConfig(StrictModel):
    climatology_window_days: int = Field(default=21, ge=1, le=90)


class CollectionConfig(StrictModel):
    region: RegionConfig
    polygons: PolygonsConfig
    field_validation: FieldValidationConfig = FieldValidationConfig()
    weather_zones: WeatherZoneSearchConfig = WeatherZoneSearchConfig()
    period: PeriodConfig
    provider: ProviderConfig = ProviderConfig()
    sensors: SensorsConfig = SensorsConfig()
    output: CollectionOutputConfig = CollectionOutputConfig()
    execution: ExecutionConfig = ExecutionConfig()
    processing: ProcessingConfig = ProcessingConfig()


def load_collection_config(path: Path | str) -> CollectionConfig:
    """Загружает collection YAML и разрешает пути относительно файла."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурация сбора не найдена: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Корень {config_path} должен быть YAML-объектом")

    config = CollectionConfig.model_validate(raw)
    base = config_path.resolve().parent
    polygons = config.polygons.model_copy(
        update={"path": _resolve_path(base, config.polygons.path)}
    )
    output = config.output.model_copy(
        update={
            "raw_directory": _resolve_path(base, config.output.raw_directory),
            "manifest_directory": _resolve_path(
                base, config.output.manifest_directory
            ),
            "dataset_path": _resolve_path(base, config.output.dataset_path),
        }
    )
    return config.model_copy(update={"polygons": polygons, "output": output})


def _resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()
