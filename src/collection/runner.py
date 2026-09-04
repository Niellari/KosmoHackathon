"""Оркестрация повторяемого и возобновляемого сбора наблюдений."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.collection.base import ObservationProvider, RAW_COLUMNS
from src.collection.config import CollectionConfig, load_collection_config


def run_collect_command(args) -> Path:
    config = load_collection_config(args.config)
    start_date = date.fromisoformat(args.start) if args.start else config.period.start
    end_date = date.fromisoformat(args.end) if args.end else config.period.end
    if start_date > end_date:
        raise ValueError("--start должен быть не позже --end")
    if args.sensor != "sentinel2":
        raise ValueError("На первом этапе реализован только sensor=sentinel2")
    if not config.sensors.sentinel2.enabled:
        raise ValueError("Sentinel-2 отключён в collection-конфигурации")

    from src.collection.earth_engine import EarthEngineSentinel2Provider

    provider = EarthEngineSentinel2Provider(config)
    output_path = (
        Path(args.output)
        if args.output
        else config.output.raw_directory / "sentinel2.csv"
    )
    limit = args.limit or config.polygons.selection.limit
    return collect_observations(
        config=config,
        provider=provider,
        sensor=args.sensor,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        limit=limit,
        force=args.force or config.execution.overwrite,
    )


def collect_observations(
    config: CollectionConfig,
    provider: ObservationProvider,
    sensor: str,
    start_date: date,
    end_date: date,
    output_path: Path,
    limit: int,
    force: bool = False,
) -> Path:
    features, _ = load_polygon_features(
        config.polygons.path, config.polygons.id_property
    )
    features = _select_features(config, features)[:limit]
    if not features:
        raise ValueError("В GeoJSON нет полигонов для сбора")

    manifest_path = config.output.manifest_directory / f"{output_path.stem}.json"
    fingerprint = _fingerprint(config, sensor, start_date, end_date, features)
    manifest = _load_manifest(manifest_path)

    if force:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        manifest = None
    if manifest and manifest.get("fingerprint") != fingerprint:
        raise ValueError(
            "Параметры сбора изменились. Используйте другой output или --force"
        )
    if output_path.exists() and not manifest:
        raise ValueError(
            f"Найден {output_path} без совместимого manifest. "
            "Используйте другой output или --force"
        )

    completed = set(manifest.get("completed_polygon_ids", [])) if manifest else set()
    if not config.execution.resume and completed:
        raise ValueError("Сбор уже начат, а execution.resume=false")

    state = manifest or {
        "fingerprint": fingerprint,
        "sensor": sensor,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "polygon_source": str(config.polygons.path),
        "completed_polygon_ids": [],
        "rows": 0,
    }
    pending = [
        feature
        for feature in features
        if str(feature["properties"][config.polygons.id_property]) not in completed
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = config.execution.batch_size
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        frame = _collect_with_retries(
            provider,
            batch,
            config.polygons.id_property,
            start_date,
            end_date,
            config.execution.retries,
        )
        frame = frame.reindex(columns=RAW_COLUMNS)
        if not frame.empty:
            written = _append_new_records(output_path, frame)
            state["rows"] = int(state.get("rows", 0)) + written
        ids = [str(f["properties"][config.polygons.id_property]) for f in batch]
        state["completed_polygon_ids"].extend(ids)
        _write_manifest(manifest_path, state)

    if not output_path.exists():
        pd.DataFrame(columns=RAW_COLUMNS).to_csv(output_path, index=False)
    print(
        f"Сбор завершён: {output_path} | полигонов: {len(features)} | "
        f"строк: {state['rows']}"
    )
    return output_path


def load_polygon_features(path: Path, id_property: str) -> tuple[list[dict], dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"GeoJSON с полигонами не найден: {path}. "
            "Создайте его по образцу data/external/polygons.example.geojson"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Ожидается GeoJSON FeatureCollection")

    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection не содержит список features")
    seen: set[str] = set()
    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Поддерживаются только Polygon и MultiPolygon")
        properties = feature.get("properties") or {}
        if id_property not in properties:
            raise ValueError(f"У каждого полигона должно быть свойство {id_property!r}")
        polygon_id = str(properties[id_property])
        if polygon_id in seen:
            raise ValueError(f"Повторяющийся polygon_id: {polygon_id}")
        seen.add(polygon_id)
    return features, payload


def _select_features(config: CollectionConfig, features: list[dict]) -> list[dict]:
    """Применяет локальные bbox/area-фильтры до обращения к API."""

    bbox = config.region.bbox
    selection = config.polygons.selection
    selected: list[dict] = []
    for feature in features:
        coordinates = list(_coordinate_pairs(feature["geometry"]))
        if not coordinates:
            continue
        if not all(
            bbox.min_lon <= lon <= bbox.max_lon
            and bbox.min_lat <= lat <= bbox.max_lat
            for lon, lat in coordinates
        ):
            continue
        area_ha = _geometry_area_ha(feature["geometry"])
        if not selection.min_area_ha <= area_ha <= selection.max_area_ha:
            continue
        cropland_fraction = feature.get("properties", {}).get("cropland_fraction")
        if cropland_fraction is not None:
            try:
                if float(cropland_fraction) < selection.min_cropland_fraction:
                    continue
            except (TypeError, ValueError):
                continue
        selected.append(feature)
    return selected


def _coordinate_pairs(geometry: dict):
    def visit(value):
        if (
            isinstance(value, list)
            and len(value) >= 2
            and all(isinstance(item, (int, float)) for item in value[:2])
        ):
            yield float(value[0]), float(value[1])
            return
        if isinstance(value, list):
            for child in value:
                yield from visit(child)

    yield from visit(geometry.get("coordinates", []))


def _geometry_area_ha(geometry: dict) -> float:
    polygons = (
        [geometry["coordinates"]]
        if geometry.get("type") == "Polygon"
        else geometry["coordinates"]
    )
    area_m2 = sum(_polygon_area_m2(polygon) for polygon in polygons)
    return area_m2 / 10_000


def _polygon_area_m2(rings: list) -> float:
    if not rings:
        return 0.0
    outer = _ring_area_m2(rings[0])
    holes = sum(_ring_area_m2(ring) for ring in rings[1:])
    return max(0.0, outer - holes)


def _ring_area_m2(ring: list) -> float:
    if len(ring) < 4:
        return 0.0
    radius = 6_371_008.8
    latitude = sum(float(point[1]) for point in ring) / len(ring)
    cos_latitude = math.cos(math.radians(latitude))
    projected = [
        (
            radius * math.radians(float(point[0])) * cos_latitude,
            radius * math.radians(float(point[1])),
        )
        for point in ring
    ]
    twice_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(projected, projected[1:] + projected[:1])
    )
    return abs(twice_area) / 2


def _collect_with_retries(
    provider: ObservationProvider,
    features: list[dict],
    id_property: str,
    start_date: date,
    end_date: date,
    retries: int,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return provider.collect(features, id_property, start_date, end_date)
        except Exception as error:
            last_error = error
            if attempt == retries:
                break
            print(f"Ошибка сбора, повтор {attempt + 1}/{retries}: {error}")
    assert last_error is not None
    raise last_error


def _append_new_records(path: Path, frame: pd.DataFrame) -> int:
    """Добавляет записи идемпотентно, в том числе после сбоя до manifest."""

    keys = ["polygon_id", "date", "sensor", "scene_id"]
    candidate = frame.drop_duplicates(keys).copy()
    if path.exists():
        existing = pd.read_csv(path, usecols=keys, dtype=str)
        existing_keys = set(map(tuple, existing[keys].fillna("").to_numpy()))
        candidate_keys = candidate[keys].fillna("").astype(str).apply(tuple, axis=1)
        candidate = candidate.loc[~candidate_keys.isin(existing_keys)]
    if candidate.empty:
        return 0
    exists = path.exists()
    candidate.to_csv(
        path,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
    )
    return len(candidate)


def _fingerprint(
    config: CollectionConfig,
    sensor: str,
    start_date: date,
    end_date: date,
    features: list[dict],
) -> str:
    selected_ids = [
        str(feature["properties"][config.polygons.id_property]) for feature in features
    ]
    payload: dict[str, Any] = {
        "provider": config.provider.model_dump(mode="json"),
        "sensor": sensor,
        "sensor_config": config.sensors.sentinel2.model_dump(mode="json"),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "selected_ids": selected_ids,
        "features": features,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Повреждён manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Некорректный manifest: {path}")
    return payload


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
