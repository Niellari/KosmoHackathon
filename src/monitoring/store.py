"""SQLite хранит задания, результаты и прогресс между перезапусками."""

from contextlib import closing
from datetime import date, timedelta
import hashlib
import json
import sqlite3
import time
from uuid import uuid4

from src.web_geometry import validate_geometry

PIPELINE_VERSION = "gee-monitor-v1"


class QueueFullError(ValueError):
    pass


class JobStore:
    def __init__(self, config):
        self.config = config
        config.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL,
                    polygon_id TEXT NOT NULL, params TEXT NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, progress INTEGER NOT NULL,
                    error_code TEXT, message TEXT, result TEXT,
                    created REAL NOT NULL, updated REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_polygon ON jobs(polygon_id, created);
            """
            )

    def connect(self):
        db = sqlite3.connect(self.config.database, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def submit(self, polygon, start, end):
        area = validate_geometry(polygon["geometry"])
        if area > self.config.max_area_ha:
            raise ValueError(f"Максимальная площадь: {self.config.max_area_ha:g} га")
        try:
            start, end = date.fromisoformat(start), date.fromisoformat(end)
        except (ValueError, TypeError):
            raise ValueError("Укажите даты в формате YYYY-MM-DD") from None
        if start < date(2018, 1, 1) or end >= date.today() or start > end:
            raise ValueError("Период должен быть между 01.01.2018 и вчерашним днём")
        if (end - start).days + 1 > self.config.max_period_days:
            raise ValueError(f"Выберите период до {self.config.max_period_days} дней")
        params = {
            "polygon_id": polygon["id"],
            "geometry": polygon["geometry"],
            "start": str(start),
            "end": str(end),
            "project_id": self.config.project_id,
            "history_years": self.config.history_years,
            "scale_m": self.config.scale_m,
            "min_valid_fraction": self.config.min_valid_fraction,
            "min_pixel_count": self.config.min_pixel_count,
            "max_interpolation_gap_days": self.config.max_interpolation_gap_days,
            "pipeline_version": PIPELINE_VERSION,
        }
        encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        now = time.time()
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT * FROM jobs WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
            ttl = 86400 if end >= date.today() - timedelta(days=90) else 30 * 86400
            if previous and (
                previous["status"] in ("queued", "running")
                or (
                    previous["status"] == "completed"
                    and now - previous["updated"] < ttl
                )
            ):
                return self.public(previous), True
            queued = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if queued >= self.config.max_queued_jobs:
                raise QueueFullError(
                    "Очередь заполнена. Дождитесь завершения текущих анализов"
                )
            job_id = previous["id"] if previous else uuid4().hex
            db.execute(
                """INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fingerprint) DO UPDATE SET status='queued',stage='Ожидает обработки',
                progress=0,error_code=NULL,message=NULL,result=NULL,updated=excluded.updated""",
                (
                    job_id,
                    fingerprint,
                    polygon["id"],
                    encoded,
                    "queued",
                    "Ожидает обработки",
                    0,
                    None,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self.public(row), False

    @staticmethod
    def public(row):
        if row is None:
            return None
        result = {
            key: row[key]
            for key in (
                "id",
                "polygon_id",
                "status",
                "stage",
                "progress",
                "error_code",
                "message",
                "created",
                "updated",
            )
        }
        params = json.loads(row["params"])
        result.update(start=params["start"], end=params["end"])
        return result

    def get(self, job_id, with_result=False):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            result = self.public(row)
            if result and with_result:
                result["result"] = json.loads(row["result"]) if row["result"] else None
            return result

    def latest(self, polygon_id):
        with closing(self.connect()) as db:
            return self.public(
                db.execute(
                    "SELECT * FROM jobs WHERE polygon_id=? ORDER BY updated DESC LIMIT 1",
                    (polygon_id,),
                ).fetchone()
            )

    def claim(self):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY updated LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='running',stage='Подключение к Earth Engine',updated=? WHERE id=?",
                (time.time(), row["id"]),
            )
            return {"id": row["id"], "params": json.loads(row["params"])}

    def update(
        self,
        job_id,
        stage,
        progress,
        status="running",
        result=None,
        error_code=None,
        message=None,
    ):
        with closing(self.connect()) as db, db:
            db.execute(
                "UPDATE jobs SET stage=?,progress=?,status=?,result=?,error_code=?,message=?,updated=? WHERE id=?",
                (
                    stage,
                    progress,
                    status,
                    (
                        json.dumps(result, ensure_ascii=False, allow_nan=False)
                        if result is not None
                        else None
                    ),
                    error_code,
                    message,
                    time.time(),
                    job_id,
                ),
            )

    def recover(self):
        # Вызывается только после получения эксклюзивной блокировки worker.
        with closing(self.connect()) as db, db:
            db.execute(
                "UPDATE jobs SET status='queued',stage='Возобновление после перезапуска',updated=? WHERE status='running'",
                (time.time(),),
            )
