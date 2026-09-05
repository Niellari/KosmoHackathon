"""Поиск вероятных погодных зон конкурсных AOI по ERA5-Land."""

from __future__ import annotations

from datetime import timedelta
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.collection.config import CollectionConfig, load_collection_config
from src.collection.earth_engine import EarthEngineUnavailableError, _number


def run_discover_weather_zones_command(args) -> Path:
    config = load_collection_config(args.config)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Файл уже существует: {output_path}. Используйте --force"
        )
    competition, sample_dates = load_weather_profiles(input_path, config)
    provider = EarthEngineWeatherGridProvider(config)
    candidates = provider.collect(sample_dates)
    result = match_weather_zones(competition, candidates, config, sample_dates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"ERA5-зоны сохранены: {output_path} | "
        f"зон: {len(result['features'])} | дат: {len(sample_dates)}"
    )
    return output_path


def load_weather_profiles(
    path: Path, config: CollectionConfig
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"anon_polygon_id", "date", "era5_temp_c", "era5_precip_mm"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"В {path} отсутствуют weather-колонки: {sorted(missing)}")
    weather = config.weather_zones
    usable = frame.loc[
        frame["date"].dt.year.between(weather.min_year, weather.max_year)
        & frame["date"].dt.month.between(
            config.period.start.month, config.period.end.month
        )
    ].copy()
    usable["era5_temp_c"] = pd.to_numeric(
        usable["era5_temp_c"], errors="coerce"
    )
    usable["era5_precip_mm"] = pd.to_numeric(
        usable["era5_precip_mm"], errors="coerce"
    )
    usable = usable.dropna(subset=["era5_temp_c", "era5_precip_mm"])
    counts = usable.groupby("date")["anon_polygon_id"].nunique().sort_index()
    available_dates = counts[counts >= 3].index
    if len(available_dates) < weather.sample_dates:
        raise ValueError(
            f"Недостаточно общих ERA5-дат: {len(available_dates)}, "
            f"требуется {weather.sample_dates}"
        )
    indices = np.linspace(
        0, len(available_dates) - 1, weather.sample_dates, dtype=int
    )
    sample_dates = [pd.Timestamp(available_dates[index]) for index in indices]
    return usable.loc[usable["date"].isin(sample_dates)].copy(), sample_dates


class EarthEngineWeatherGridProvider:
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
                f"Не задана переменная окружения {project_env}"
            )
        try:
            ee.Initialize(project=project_id)
        except Exception as error:
            raise EarthEngineUnavailableError(
                "Не удалось инициализировать Earth Engine"
            ) from error
        self.ee = ee

    def collect(self, sample_dates: list[pd.Timestamp]) -> pd.DataFrame:
        ee = self.ee
        search = self.config.weather_zones
        bbox = self.config.region.bbox
        points = []
        latitudes = np.arange(
            bbox.min_lat, bbox.max_lat + search.grid_step_degrees / 2,
            search.grid_step_degrees,
        )
        longitudes = np.arange(
            bbox.min_lon, bbox.max_lon + search.grid_step_degrees / 2,
            search.grid_step_degrees,
        )
        for lat_index, latitude in enumerate(latitudes):
            for lon_index, longitude in enumerate(longitudes):
                candidate_id = f"G{lat_index:03d}-{lon_index:03d}"
                points.append(
                    ee.Feature(
                        ee.Geometry.Point([float(longitude), float(latitude)]),
                        {
                            "candidate_id": candidate_id,
                            "longitude": float(longitude),
                            "latitude": float(latitude),
                        },
                    )
                )
        point_collection = ee.FeatureCollection(points)
        collection = ee.ImageCollection(search.dataset)
        records: list[dict] = []
        for timestamp in sample_dates:
            day = timestamp.date()
            image = ee.Image(
                collection.filterDate(
                    day.isoformat(), (day + timedelta(days=1)).isoformat()
                ).first()
            )
            weather_image = ee.Image.cat(
                [
                    image.select(search.temperature_band)
                    .subtract(273.15)
                    .rename("era5_temp_c"),
                    image.select(search.precipitation_band)
                    .multiply(1000)
                    .rename("era5_precip_mm"),
                ]
            )
            payload = weather_image.sampleRegions(
                collection=point_collection,
                properties=["candidate_id", "longitude", "latitude"],
                scale=11132,
                geometries=False,
            ).getInfo()
            for feature in payload.get("features", []):
                props = feature.get("properties", {})
                records.append(
                    {
                        "candidate_id": str(props.get("candidate_id")),
                        "longitude": _number(props.get("longitude")),
                        "latitude": _number(props.get("latitude")),
                        "date": pd.Timestamp(day),
                        "era5_temp_c": _number(props.get("era5_temp_c")),
                        "era5_precip_mm": _number(props.get("era5_precip_mm")),
                    }
                )
        result = pd.DataFrame(records)
        return result.dropna().reset_index(drop=True)


def match_weather_zones(
    competition: pd.DataFrame,
    candidates: pd.DataFrame,
    config: CollectionConfig,
    sample_dates: list[pd.Timestamp],
) -> dict:
    """Сопоставляет каждый AOI с ближайшей ERA5-погодной подписью."""

    required_candidate_rows = config.weather_zones.min_profile_dates
    candidate_groups = {
        str(candidate_id): group.set_index("date")
        for candidate_id, group in candidates.groupby("candidate_id")
        if group["date"].nunique() >= required_candidate_rows
    }
    matches: list[dict] = []
    for polygon_id, profile in competition.groupby("anon_polygon_id"):
        profile = profile.set_index("date")
        best: dict | None = None
        for candidate_id, candidate in candidate_groups.items():
            joined = profile[["era5_temp_c", "era5_precip_mm"]].join(
                candidate[
                    ["era5_temp_c", "era5_precip_mm", "longitude", "latitude"]
                ],
                how="inner",
                lsuffix="_target",
                rsuffix="_candidate",
            )
            if len(joined) < required_candidate_rows:
                continue
            temp_error = np.sqrt(
                np.mean(
                    np.square(
                        joined["era5_temp_c_target"]
                        - joined["era5_temp_c_candidate"]
                    )
                )
            )
            target_precip = np.log1p(joined["era5_precip_mm_target"].clip(lower=0))
            candidate_precip = np.log1p(
                joined["era5_precip_mm_candidate"].clip(lower=0)
            )
            precip_error = np.sqrt(np.mean(np.square(target_precip - candidate_precip)))
            distance = float(temp_error / 3.0 + precip_error)
            match = {
                "anon_polygon_id": str(polygon_id),
                "candidate_id": candidate_id,
                "longitude": float(joined["longitude"].iloc[0]),
                "latitude": float(joined["latitude"].iloc[0]),
                "weather_distance": distance,
                "temperature_rmse_c": float(temp_error),
                "log_precipitation_rmse": float(precip_error),
            }
            if best is None or (distance, candidate_id) < (
                best["weather_distance"], best["candidate_id"]
            ):
                best = match
        if best is not None:
            matches.append(best)
    if not matches:
        raise ValueError("Не удалось сопоставить ни один AOI с ERA5-сеткой")

    zones = _aggregate_zones(matches, config)
    features = []
    for index, zone in enumerate(zones, 1):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "zone_id": f"ERA5-ZONE-{index:02d}",
                    "matched_aoi_count": len(zone["matches"]),
                    "matched_aoi_ids": sorted(
                        item["anon_polygon_id"] for item in zone["matches"]
                    ),
                    "mean_weather_distance": float(
                        np.mean(
                            [item["weather_distance"] for item in zone["matches"]]
                        )
                    ),
                    "grid_step_degrees": config.weather_zones.grid_step_degrees,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [zone["longitude"], zone["latitude"]],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": f"{config.region.name}_era5_matches",
        "sample_dates": [timestamp.date().isoformat() for timestamp in sample_dates],
        "matched_aoi_count": len(matches),
        "features": features,
    }


def _aggregate_zones(matches: list[dict], config: CollectionConfig) -> list[dict]:
    radius = config.weather_zones.min_zone_distance_km
    centers = []
    for match in matches:
        members = [
            item
            for item in matches
            if _distance_km(
                match["longitude"],
                match["latitude"],
                item["longitude"],
                item["latitude"],
            )
            <= radius
        ]
        centers.append(
            {
                "longitude": match["longitude"],
                "latitude": match["latitude"],
                "matches": members,
            }
        )
    centers.sort(
        key=lambda zone: (
            -len(zone["matches"]),
            np.mean([item["weather_distance"] for item in zone["matches"]]),
            zone["latitude"],
            zone["longitude"],
        )
    )
    selected = []
    used_aoi: set[str] = set()
    for center in centers:
        members = [
            item
            for item in center["matches"]
            if item["anon_polygon_id"] not in used_aoi
        ]
        if not members:
            continue
        if any(
            _distance_km(
                center["longitude"],
                center["latitude"],
                item["longitude"],
                item["latitude"],
            )
            < radius
            for item in selected
        ):
            continue
        selected.append({**center, "matches": members})
        used_aoi.update(item["anon_polygon_id"] for item in members)
        if len(selected) >= config.weather_zones.top_zones:
            break
    return selected


def _distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))
