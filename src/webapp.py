"""Лёгкий веб-адаптер над общим аналитическим pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
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
) -> Bottle:
    pipeline = AnalysisPipeline.from_csv(data_path, train_path, config=config)
    active_model = pipeline.prepare_model()
    print(f"Модель веб-сервиса: {active_model}")
    state = AppState(pipeline=pipeline, active_model=active_model)
    app = Bottle()
    web_root = Path(__file__).resolve().parent.parent / "web"

    @app.hook("after_request")
    def enable_cors() -> None:
        response.headers["Access-Control-Allow-Origin"] = "*"
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
        geometry = payload.get("geometry")
        if not geometry or geometry.get("type") != "Polygon":
            raise HTTPError(400, "Ожидается GeoJSON Polygon")
        polygon_id = f"CUSTOM-{uuid4().hex[:6].upper()}"
        item = {
            "id": polygon_id,
            "name": str(payload.get("name") or polygon_id),
            "crop_type": str(payload.get("crop_type") or "не указан"),
            "years": [],
            "source": "user_geometry",
            "geometry": geometry,
        }
        with state.lock:
            state.custom_polygons[polygon_id] = item
        response.status = 201
        return item

    @app.delete("/api/polygons/<polygon_id>")
    def delete_polygon(polygon_id: str):
        with state.lock:
            if polygon_id not in state.custom_polygons:
                raise HTTPError(404, "Можно удалять только пользовательские полигоны")
            del state.custom_polygons[polygon_id]
        return {"deleted": polygon_id}

    @app.get("/api/series/<polygon_id>")
    def series(polygon_id: str):
        raw_year = request.query.get("year")
        year = int(raw_year) if raw_year else None
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
    app.run(host=host, port=selected_port, debug=debug, reloader=False)
