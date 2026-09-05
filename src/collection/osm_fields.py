"""Поиск открытых OSM-контуров сельхозземель для пилотного сбора."""

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.collection.config import load_collection_config
from src.collection.runner import _geometry_area_ha


OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def run_discover_fields_command(args) -> Path:
    config = load_collection_config(args.config)
    output = Path(args.output) if args.output else config.polygons.path
    if output.exists() and not args.force:
        raise FileExistsError(f"Файл уже существует: {output}. Используйте --force")

    limit = args.limit or config.polygons.selection.limit
    validation_enabled = (
        config.field_validation.enabled
        and not getattr(args, "skip_quality_validation", False)
    )
    candidates_path = getattr(args, "candidates", None)
    if candidates_path:
        result = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
        if result.get("type") != "FeatureCollection" or not isinstance(
            result.get("features"), list
        ):
            raise ValueError("--candidates должен быть GeoJSON FeatureCollection")
    else:
        bbox = config.region.bbox
        payload = _fetch_osm_farmland(bbox)
        pool_limit = (
            max(limit, config.field_validation.candidate_pool_size)
            if validation_enabled
            else limit
        )
        result = build_field_collection(payload, config=config, limit=pool_limit)
    if validation_enabled and result["features"]:
        from src.collection.earth_engine import EarthEngineSentinel2Provider

        provider = EarthEngineSentinel2Provider(config)
        result = validate_field_collection(
            result, config=config, provider=provider, limit=limit
        )
    if not result["features"]:
        if validation_enabled:
            quality = result.get("quality_validation", {})
            raise ValueError(
                "Ни один OSM-кандидат не прошёл WorldCover/Sentinel-2 "
                f"quality gate (кандидатов: {quality.get('candidates', 0)})"
            )
        raise ValueError("OpenStreetMap не вернул подходящих контуров")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Контуры сохранены: {output} | полигонов: {len(result['features'])} | "
        "источник: OpenStreetMap (ODbL)"
    )
    return output


def _fetch_osm_farmland(bbox, tile_degrees: float = 0.35) -> dict:
    """Запрашивает Overpass небольшими тайлами и дедуплирует ways."""

    lon_tiles = max(
        1, math.ceil((bbox.max_lon - bbox.min_lon - 1e-9) / tile_degrees)
    )
    lat_tiles = max(
        1, math.ceil((bbox.max_lat - bbox.min_lat - 1e-9) / tile_degrees)
    )
    elements: dict[int, dict] = {}
    for lat_index in range(lat_tiles):
        min_lat = bbox.min_lat + (bbox.max_lat - bbox.min_lat) * lat_index / lat_tiles
        max_lat = bbox.min_lat + (bbox.max_lat - bbox.min_lat) * (lat_index + 1) / lat_tiles
        for lon_index in range(lon_tiles):
            min_lon = bbox.min_lon + (bbox.max_lon - bbox.min_lon) * lon_index / lon_tiles
            max_lon = bbox.min_lon + (bbox.max_lon - bbox.min_lon) * (lon_index + 1) / lon_tiles
            query = (
                "[out:json][timeout:90];"
                f'way["landuse"="farmland"]'
                f"({min_lat},{min_lon},{max_lat},{max_lon});"
                "out tags geom;"
            )
            error: Exception | None = None
            for endpoint in OVERPASS_URLS:
                request = Request(
                    endpoint,
                    data=urlencode({"data": query}).encode("utf-8"),
                    headers={"User-Agent": "AgroPulse-KosmoHackathon/1.0"},
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=120) as response:
                        tile_payload = json.load(response)
                    for element in tile_payload.get("elements", []):
                        elements[int(element["id"])] = element
                    error = None
                    break
                except Exception as caught:
                    error = caught
            if error is not None:
                raise RuntimeError(
                    f"Не удалось получить OSM tile "
                    f"({min_lon:.3f},{min_lat:.3f})..({max_lon:.3f},{max_lat:.3f})"
                ) from error
    return {"elements": list(elements.values())}


def build_field_collection(payload: dict, config, limit: int) -> dict:
    """Фильтрует OSM ways и выбирает поля по заданной spatial-стратегии."""

    bbox = config.region.bbox
    selection = config.polygons.selection
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    lon_scale = math.cos(math.radians(center_lat))
    candidates: list[dict] = []

    for element in payload.get("elements", []):
        geometry_nodes = element.get("geometry") or []
        if len(geometry_nodes) < 4:
            continue
        coordinates = [
            [float(node["lon"]), float(node["lat"])] for node in geometry_nodes
        ]
        if coordinates[0] != coordinates[-1]:
            continue
        if not all(
            bbox.min_lon <= lon <= bbox.max_lon
            and bbox.min_lat <= lat <= bbox.max_lat
            for lon, lat in coordinates
        ):
            continue

        geometry = {"type": "Polygon", "coordinates": [coordinates]}
        area_ha = _geometry_area_ha(geometry)
        if not selection.min_area_ha <= area_ha <= selection.max_area_ha:
            continue
        centroid_lon = sum(point[0] for point in coordinates) / len(coordinates)
        centroid_lat = sum(point[1] for point in coordinates) / len(coordinates)
        distance = (
            ((centroid_lon - center_lon) * lon_scale) ** 2
            + (centroid_lat - center_lat) ** 2
        )
        candidates.append(
            {
                "center_distance": distance,
                "osm_id": int(element["id"]),
                "area_ha": area_ha,
                "geometry": geometry,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
            }
        )

    selected = _select_candidates(candidates, config=config, limit=limit)
    features = []
    for index, candidate in enumerate(selected, 1):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "polygon_id": f"{config.polygons.id_prefix}{index:04d}",
                    "source": "OpenStreetMap",
                    "source_license": "ODbL",
                    "osm_way_id": candidate["osm_id"],
                    "landuse": "farmland",
                    "area_ha": round(candidate["area_ha"], 2),
                    "centroid_lon": round(candidate["centroid_lon"], 7),
                    "centroid_lat": round(candidate["centroid_lat"], 7),
                },
                "geometry": candidate["geometry"],
            }
        )
    return {
        "type": "FeatureCollection",
        "name": config.region.name,
        "features": features,
    }


def _select_candidates(candidates: list[dict], config, limit: int) -> list[dict]:
    selection = config.polygons.selection
    ordered = sorted(
        candidates, key=lambda item: (item["center_distance"], item["osm_id"])
    )
    if selection.strategy == "nearest_center":
        return ordered[:limit]

    bbox = config.region.bbox
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    lon_km = 111.32 * math.cos(math.radians(center_lat))
    selected: list[dict] = []
    cell_counts: dict[tuple[int, int], int] = {}
    remaining = ordered.copy()
    while remaining and len(selected) < limit:
        eligible: list[tuple[float, float, int, dict, tuple[int, int]]] = []
        for candidate in remaining:
            cell = (
                int((candidate["centroid_lon"] - bbox.min_lon) * lon_km / selection.grid_cell_km),
                int((candidate["centroid_lat"] - bbox.min_lat) * 111.32 / selection.grid_cell_km),
            )
            if cell_counts.get(cell, 0) >= selection.max_per_grid_cell:
                continue
            min_distance = min(
                (
                    _distance_km(
                        candidate["centroid_lon"],
                        candidate["centroid_lat"],
                        item["centroid_lon"],
                        item["centroid_lat"],
                    )
                    for item in selected
                ),
                default=float("inf"),
            )
            if min_distance < selection.min_centroid_distance_km:
                continue
            spread_score = min_distance if selected else -candidate["center_distance"]
            eligible.append(
                (
                    spread_score,
                    -candidate["center_distance"],
                    -candidate["osm_id"],
                    candidate,
                    cell,
                )
            )
        if not eligible:
            break
        _, _, _, chosen, cell = max(eligible, key=lambda item: item[:3])
        selected.append(chosen)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        remaining.remove(chosen)
    return selected


def validate_field_collection(collection: dict, config, provider, limit: int) -> dict:
    """Добавляет WorldCover/Sentinel-2 quality и отбрасывает слабые поля."""

    features = collection.get("features", [])
    id_property = config.polygons.id_property
    fractions = provider.cropland_fractions(features, id_property)
    observations = provider.collect(
        features, id_property, config.period.start, config.period.end
    )
    valid_counts = (
        observations.groupby("polygon_id")["date"].nunique().to_dict()
        if not observations.empty
        else {}
    )
    accepted = []
    rejected = []
    for feature in features:
        properties = feature["properties"]
        polygon_id = str(properties[id_property])
        cropland_fraction = fractions.get(polygon_id)
        valid_observations = int(valid_counts.get(polygon_id, 0))
        properties["cropland_fraction"] = cropland_fraction
        properties["sentinel2_valid_observations"] = valid_observations
        reasons = []
        if cropland_fraction is None:
            reasons.append("missing_worldcover")
        elif cropland_fraction < config.polygons.selection.min_cropland_fraction:
            reasons.append("low_cropland_fraction")
        if valid_observations < config.field_validation.min_valid_sentinel2_observations:
            reasons.append("insufficient_sentinel2_observations")
        if reasons:
            rejected.append(
                {
                    "polygon_id": polygon_id,
                    "cropland_fraction": cropland_fraction,
                    "sentinel2_valid_observations": valid_observations,
                    "reasons": reasons,
                }
            )
            continue
        accepted.append(feature)
    return {
        **collection,
        "features": accepted[:limit],
        "quality_validation": {
            "candidates": len(features),
            "accepted": len(accepted),
            "rejected": rejected,
            "worldcover_collection": config.field_validation.worldcover_collection,
            "sentinel2_period": [
                config.period.start.isoformat(),
                config.period.end.isoformat(),
            ],
        },
    }


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
