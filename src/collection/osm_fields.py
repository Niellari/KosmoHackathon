"""Поиск открытых OSM-контуров сельхозземель для пилотного сбора."""

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.collection.config import load_collection_config
from src.collection.runner import _geometry_area_ha


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def run_discover_fields_command(args) -> Path:
    config = load_collection_config(args.config)
    output = Path(args.output) if args.output else config.polygons.path
    if output.exists() and not args.force:
        raise FileExistsError(f"Файл уже существует: {output}. Используйте --force")

    bbox = config.region.bbox
    query = (
        "[out:json][timeout:90];"
        f'way["landuse"="farmland"]'
        f"({bbox.min_lat},{bbox.min_lon},{bbox.max_lat},{bbox.max_lon});"
        "out tags geom;"
    )
    request = Request(
        OVERPASS_URL,
        data=urlencode({"data": query}).encode("utf-8"),
        headers={"User-Agent": "AgroPulse-KosmoHackathon/1.0"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)

    limit = args.limit or config.polygons.selection.limit
    result = build_field_collection(payload, config=config, limit=limit)
    if not result["features"]:
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


def build_field_collection(payload: dict, config, limit: int) -> dict:
    """Фильтрует замкнутые OSM ways и выбирает ближайшие к центру bbox."""

    bbox = config.region.bbox
    selection = config.polygons.selection
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    lon_scale = math.cos(math.radians(center_lat))
    candidates: list[tuple[float, int, float, dict]] = []

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
            (distance, int(element["id"]), area_ha, geometry)
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    features = []
    for index, (_, osm_id, area_ha, geometry) in enumerate(candidates[:limit], 1):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "polygon_id": f"{config.polygons.id_prefix}{index:04d}",
                    "source": "OpenStreetMap",
                    "source_license": "ODbL",
                    "osm_way_id": osm_id,
                    "landuse": "farmland",
                    "area_ha": round(area_ha, 2),
                },
                "geometry": geometry,
            }
        )
    return {
        "type": "FeatureCollection",
        "name": config.region.name,
        "features": features,
    }
