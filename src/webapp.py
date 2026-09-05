"""Лёгкий веб-адаптер над общим аналитическим pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import json
import os
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pathlib import Path
import socket
from threading import Lock, Timer
from uuid import uuid4
import webbrowser

import numpy as np
import pandas as pd
from bottle import Bottle, HTTPError, request, response, static_file

from src.config import AppConfig
from src.pipeline import AnalysisPipeline
from src.web_geometry import validate_geometry
from src.monitoring.config import load_monitoring_config, DEFAULT_CONFIG
from src.monitoring.store import JobStore, QueueFullError


def _json_value(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


@dataclass
class AppState:
    pipeline: AnalysisPipeline
    active_model: str
    custom_polygons: dict[str, dict] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)


def create_app(
    data_path: Path,
    train_path: Path | None = None,
    config: AppConfig | None = None,
    polygons_path: Path | None = None,
    monitoring_config=None,
) -> Bottle:
    pipeline = AnalysisPipeline.from_csv(data_path, train_path, config=config)
    active_model = pipeline.prepare_model()
    print(f"Модель веб-сервиса: {active_model}")
    state = AppState(pipeline=pipeline, active_model=active_model)
    app = Bottle()
    web_root = Path(__file__).resolve().parent.parent / "web"
    storage = polygons_path or web_root.parent / "artifacts" / "web-polygons.json"
    if storage.exists():
        state.custom_polygons = json.loads(storage.read_text(encoding="utf-8"))
    monitoring_config = monitoring_config or load_monitoring_config()
    if polygons_path is not None:
        monitoring_config = monitoring_config.model_copy(
            update={"database": storage.parent / "monitoring-test.sqlite3"}
        )
    jobs = JobStore(monitoring_config)

    @app.get("/api/monitoring")
    def monitoring_info():
        return {
            "project_id": monitoring_config.project_id,
            "history_years": monitoring_config.history_years,
            "max_area_ha": monitoring_config.max_area_ha,
            "max_period_days": monitoring_config.max_period_days,
            "sources": ["Sentinel-2", "ERA5-Land"],
        }

    @app.post("/api/polygons/<polygon_id>/analyses")
    def start_analysis(polygon_id):
        # Запрос из чужой веб-страницы не должен расходовать квоту владельца.
        origin = request.headers.get("Origin")
        if (
            origin
            and origin.rstrip("/")
            != f"{request.urlparts.scheme}://{request.urlparts.netloc}"
        ):
            raise HTTPError(403, "Запускайте анализ со страницы приложения")
        payload = request.json or {}
        if not isinstance(payload, dict):
            raise HTTPError(400, "Ожидается объект JSON")
        with state.lock:
            polygon = state.custom_polygons.get(polygon_id)
        if polygon is None:
            raise HTTPError(404, "Сначала сохраните поле в «Мои поля»")
        try:
            job, reused = jobs.submit(polygon, payload.get("start"), payload.get("end"))
        except QueueFullError as error:
            raise HTTPError(429, str(error)) from error
        except ValueError as error:
            raise HTTPError(400, str(error)) from error
        response.status = 200 if reused else 202
        return {"job": job, "reused": reused}

    @app.get("/api/polygons/<polygon_id>/analyses/latest")
    def latest_analysis(polygon_id):
        return {"job": jobs.latest(polygon_id)}

    @app.get("/api/analyses/<job_id>")
    def get_analysis(job_id):
        job = jobs.get(job_id, with_result=True)
        if job is None:
            raise HTTPError(404, "Задание не найдено")
        return job

    def persist(items):
        storage.parent.mkdir(parents=True, exist_ok=True)
        temporary = storage.with_suffix(".tmp")
        temporary.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, storage)
        state.custom_polygons = items

    def remote_json(url, data=None):
        try:
            req = Request(
                url, data=data, headers={"User-Agent": "AgroPulse-KosmoHackathon/1.0"}
            )
            with urlopen(req, timeout=30) as incoming:
                return json.load(incoming)
        except Exception as error:
            raise HTTPError(
                502, "Источник временно недоступен. Попробуйте ещё раз"
            ) from error

    @app.get("/api/regions")
    def regions():
        query = request.query.get("q", "").strip()
        if not 2 <= len(query) <= 150:
            raise HTTPError(400, "Введите название места от 2 символов")
        payload = remote_json(
            "https://nominatim.openstreetmap.org/search?"
            + urlencode(
                {"q": query, "format": "json", "limit": 5, "accept-language": "ru"}
            )
        )
        return {
            "items": [
                {
                    "name": p["display_name"],
                    "lat": float(p["lat"]),
                    "lon": float(p["lon"]),
                }
                for p in payload
            ]
        }

    @app.get("/api/fields")
    def fields():
        raw = request.query.get("bbox")
        if not raw:
            path = web_root.parent / "data" / "external" / "polygons.geojson"
            return (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {"type": "FeatureCollection", "features": []}
            )
        try:
            south, west, north, east = map(float, raw.split(","))
            if not (
                -85 <= south < north <= 85
                and -180 <= west < east <= 180
                and north - south <= 0.3
                and east - west <= 0.5
            ):
                raise ValueError()
        except ValueError as error:
            raise HTTPError(400, "Приблизьте карту для поиска полей") from error
        query = f'[out:json][timeout:20];way["landuse"="farmland"]({south},{west},{north},{east});out tags geom 150;'
        payload = remote_json(
            "https://overpass-api.de/api/interpreter",
            urlencode({"data": query}).encode(),
        )
        features = []
        for item in payload.get("elements", []):
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [[p["lon"], p["lat"]] for p in item.get("geometry", [])]
                ],
            }
            try:
                area = validate_geometry(geometry)
            except ValueError:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "polygon_id": f'OSM-{item["id"]}',
                        "name": item.get("tags", {}).get("name", "Поле OSM"),
                        "area_ha": area,
                        "source": "OpenStreetMap",
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    @app.hook("after_request")
    def enable_cors() -> None:
        response.headers["Cache-Control"] = "no-store"

    @app.get("/")
    def index():
        return static_file("index.html", root=str(web_root))

    @app.get("/static/<filepath:path>")
    def assets(filepath: str):
        return static_file(filepath, root=str(web_root))

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "rows": len(state.pipeline.data),
            "model": state.active_model,
        }

    @app.get("/api/meta")
    def meta():
        data = state.pipeline.data
        return {
            "rows": len(data),
            "polygons": int(data["anon_polygon_id"].nunique()),
            "date_min": data["date"].min().strftime("%Y-%m-%d"),
            "date_max": data["date"].max().strftime("%Y-%m-%d"),
        }

    @app.get("/api/polygons")
    def polygons():
        data = state.pipeline.data
        records = []
        for polygon_id, group in data.groupby("anon_polygon_id", sort=True):
            records.append(
                {
                    "id": polygon_id,
                    "crop_type": str(group["crop_type"].iloc[0]),
                    "years": sorted(int(year) for year in group["year"].unique()),
                    "source": "competition_dataset",
                }
            )
        with state.lock:
            records.extend(state.custom_polygons.values())
        return {"items": records}

    @app.post("/api/polygons")
    def add_polygon():
        payload = request.json or {}
        if not isinstance(payload, dict):
            raise HTTPError(400, "Ожидается объект JSON")
        geometry = payload.get("geometry")
        try:
            area = validate_geometry(geometry)
        except ValueError as error:
            raise HTTPError(400, str(error)) from error
        polygon_id = f"CUSTOM-{uuid4().hex[:6].upper()}"
        item = {
            "id": polygon_id,
            "name": str(payload.get("name") or polygon_id).strip()[:80] or polygon_id,
            "crop_type": str(payload.get("crop_type") or "не указан"),
            "years": [],
            "source": "user_geometry",
            "geometry": geometry,
            "area_ha": area,
        }
        with state.lock:
            persist({**state.custom_polygons, polygon_id: item})
        response.status = 201
        return item

    @app.patch("/api/polygons/<polygon_id>")
    def rename_polygon(polygon_id: str):
        payload = request.json or {}
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("name"), str)
            or not payload["name"].strip()
        ):
            raise HTTPError(400, "Введите название поля")
        with state.lock:
            if polygon_id not in state.custom_polygons:
                raise HTTPError(404, "Сохранённое поле не найдено")
            item = {
                **state.custom_polygons[polygon_id],
                "name": payload["name"].strip()[:80],
            }
            persist({**state.custom_polygons, polygon_id: item})
        return item

    @app.delete("/api/polygons/<polygon_id>")
    def delete_polygon(polygon_id: str):
        with state.lock:
            if polygon_id not in state.custom_polygons:
                raise HTTPError(404, "Можно удалять только пользовательские полигоны")
            persist(
                {
                    key: item
                    for key, item in state.custom_polygons.items()
                    if key != polygon_id
                }
            )
        return {"deleted": polygon_id}

    @app.get("/api/series/<polygon_id>")
    def series(polygon_id: str):
        raw_year = request.query.get("year")
        try:
            year = int(raw_year) if raw_year else None
        except ValueError as error:
            raise HTTPError(400, "Некорректный сезон") from error
        try:
            analyzed = state.pipeline.analyze_polygon(polygon_id, year=year)
        except KeyError as error:
            raise HTTPError(404, str(error)) from error

        points = []
        for _, row in analyzed.iterrows():
            points.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "observed": _json_value(row.get("primary_ndvi")),
                    "filled": _json_value(row.get("primary_ndvi_filled")),
                    "climatology": _json_value(row.get("climatology_mean_calc")),
                    "climatology_std": _json_value(row.get("climatology_std_calc")),
                    "zscore": _json_value(row.get("ndvi_zscore_calc")),
                    "status": _json_value(row.get("status_calc")),
                    "explanation": _json_value(row.get("anomaly_explanation")),
                    "temperature": _json_value(row.get("era5_temp_c")),
                    "precipitation": _json_value(row.get("era5_precip_mm")),
                }
            )

        anomalous = analyzed[
            analyzed["status_calc"].isin(["Угнетение биомассы", "Критическая аномалия"])
        ]
        return {
            "polygon_id": polygon_id,
            "year": int(analyzed["year"].iloc[0]),
            "crop_type": str(analyzed["crop_type"].iloc[0]),
            "summary": {
                "observations": int(analyzed["primary_ndvi"].notna().sum()),
                "restored": int(analyzed["primary_ndvi"].isna().sum()),
                "anomalies": int(len(anomalous)),
                "mean_ndvi": _json_value(analyzed["primary_ndvi_filled"].mean()),
            },
            "points": points,
        }

    return app


def find_available_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    """Возвращает предпочтительный или ближайший свободный TCP-порт."""

    for port in range(preferred_port, preferred_port + attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((host, port))
            return port
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                continue
            raise
    raise RuntimeError(
        f"Не найден свободный порт в диапазоне "
        f"{preferred_port}–{preferred_port + attempts - 1}"
    )


def open_in_browser(url: str) -> None:
    """Открывает интерфейс системным браузером, не прерывая сервер при ошибке."""

    try:
        opened = webbrowser.open(url, new=2)
        if not opened:
            print(f"Не удалось открыть браузер автоматически. Откройте вручную: {url}")
    except webbrowser.Error as error:
        print(f"Не удалось открыть браузер автоматически ({error}). Откройте: {url}")


def run_server(
    data_path: Path,
    train_path: Path | None,
    host: str,
    port: int,
    debug: bool = False,
    open_browser: bool = True,
    auto_select_port: bool = True,
    config: AppConfig | None = None,
) -> None:
    app = create_app(data_path=data_path, train_path=train_path, config=config)
    selected_port = find_available_port(host, port) if auto_select_port else port
    if selected_port != port:
        print(f"Порт {port} занят; выбран свободный порт {selected_port}.")
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{browser_host}:{selected_port}/"
    print(f"АгроПульс запущен: {url}")
    if open_browser:
        # Даём WSGI-серверу время начать принимать подключения.
        browser_timer = Timer(0.5, open_in_browser, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()
    # Отдельный процесс переживает обновление страницы и не блокирует HTTP.
    # SQLite и месячный кэш позволяют продолжить после перезапуска сервера.
    log_path = DEFAULT_CONFIG.parent.parent / "artifacts" / "monitoring" / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.monitoring.worker",
                "--config",
                str(DEFAULT_CONFIG),
            ],
            cwd=str(DEFAULT_CONFIG.parent.parent),
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            app.run(host=host, port=selected_port, debug=debug, reloader=False)
        finally:
            worker.terminate()
            worker.wait(timeout=10)
