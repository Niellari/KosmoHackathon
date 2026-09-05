"""Один отдельный процесс читает устойчивую очередь, сохраняя каждый месяц."""

import argparse
from contextlib import contextmanager
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import time

from src.monitoring.config import load_monitoring_config, DEFAULT_CONFIG
from src.monitoring.store import JobStore, PIPELINE_VERSION
from src.monitoring.earth_engine import (
    EarthEngineProvider,
    ProviderError,
    connect,
    classify_error,
)
from src.monitoring.analysis import analyze_observations


def months(start, end):
    while start <= end:
        following = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield start, min(end, following - timedelta(days=1))
        start = following


def shift_year(day, offset):
    try:
        return day.replace(year=day.year - offset)
    except ValueError:
        return day.replace(year=day.year - offset, day=28)


class MonitorRunner:
    def __init__(self, config, store, provider_factory=EarthEngineProvider):
        self.config, self.store, self.provider_factory = config, store, provider_factory

    def cached_month(self, provider, params, source, start, end):
        identity = {
            key: params[key]
            for key in (
                "geometry",
                "project_id",
                "scale_m",
                "min_valid_fraction",
                "min_pixel_count",
            )
        }
        identity.update(
            source=source, start=str(start), end=str(end), version=PIPELINE_VERSION
        )
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        path = self.config.cache_directory / (fingerprint + ".json")
        ttl = 86400 if end >= date.today() - timedelta(days=90) else 30 * 86400
        if path.exists() and time.time() - path.stat().st_mtime < ttl:
            return json.loads(path.read_text(encoding="utf-8"))["records"]
        records = provider.collect_month(source, params["geometry"], start, end)
        self.config.cache_directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"request": identity, "records": records},
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return records

    def run(self, job):
        job_id, params = job["id"], job["params"]
        try:
            # Задание фиксирует настройки: смена YAML не меняет уже созданный запрос.
            configured = self.config.model_copy(
                update={
                    key: params[key]
                    for key in (
                        "project_id",
                        "scale_m",
                        "min_valid_fraction",
                        "min_pixel_count",
                    )
                }
            )
            provider = self.provider_factory(configured)
            start, end = date.fromisoformat(params["start"]), date.fromisoformat(
                params["end"]
            )
            pieces = [("sentinel2", a, b) for a, b in months(start, end)]
            for offset in range(1, params["history_years"] + 1):
                first = max(
                    date(2017, 3, 28), shift_year(start, offset) - timedelta(days=21)
                )
                last = min(
                    start - timedelta(days=1),
                    shift_year(end, offset) + timedelta(days=21),
                )
                pieces.extend(("sentinel2", a, b) for a, b in months(first, last))
            pieces.extend(
                ("era5", a, b) for a, b in months(start - timedelta(days=13), end)
            )
            satellite, weather, warnings = [], [], []
            weather_failed = False
            for index, (source, a, b) in enumerate(pieces):
                label = "Sentinel-2" if source == "sentinel2" else "ERA5-Land"
                self.store.update(
                    job_id,
                    f"{label}: {a:%m.%Y} · {index+1}/{len(pieces)}",
                    int(index / len(pieces) * 90),
                )
                if source == "era5" and weather_failed:
                    continue
                try:
                    records = self.cached_month(provider, params, source, a, b)
                    (satellite if source == "sentinel2" else weather).extend(records)
                except ProviderError as error:
                    if source == "sentinel2":
                        raise
                    weather_failed = True
                    warnings.append(
                        f"Погодный источник недоступен: {error}. Анализ NDVI выполнен без полного погодного контекста"
                    )
            self.store.update(job_id, "Восстановление ряда и поиск отклонений", 95)
            result = analyze_observations(params, satellite, weather, warnings)
            result["collected_at"] = time.time()
            self.store.update(
                job_id, "Анализ завершён", 100, status="completed", result=result
            )
        except ProviderError as error:
            self.store.update(
                job_id,
                "Сбор остановлен",
                0,
                status="failed",
                error_code=error.code,
                message=str(error),
            )
        except ValueError as error:
            self.store.update(
                job_id,
                "Нет данных для анализа",
                0,
                status="failed",
                error_code="no_data",
                message=str(error),
            )
        except Exception as error:
            # В журнале нет сырых ответов провайдера, которые могут содержать реквизиты.
            print(f"Job {job_id}: {type(error).__name__}", flush=True)
            self.store.update(
                job_id,
                "Ошибка обработки",
                0,
                status="failed",
                error_code="processing_error",
                message="Не удалось обработать данные. Собранные месяцы сохранены; повторите анализ",
            )


@contextmanager
def worker_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Обработчик уже работает для этой базы") from None
        yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_monitoring_config(args.config)
    if args.check:
        try:
            ee = connect(config.project_id)
            ee.Number(1).getInfo()
            print("Earth Engine: доступ подтверждён; проект " + config.project_id)
        except Exception as error:
            safe = error if isinstance(error, ProviderError) else classify_error(error)
            print(f"Earth Engine: {safe.code}: {safe}")
            raise SystemExit(1)
        return
    store = JobStore(config)
    with worker_lock(config.database.with_suffix(".lock")):
        store.recover()
        runner = MonitorRunner(config, store)
        while True:
            job = store.claim()
            if job:
                runner.run(job)
            if args.once:
                break
            if not job:
                time.sleep(1)


if __name__ == "__main__":
    main()
