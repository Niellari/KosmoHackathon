"""Очередь, восстановление после сбоя и честная обработка отсутствующих данных."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import tempfile
import unittest

from src.monitoring.config import load_monitoring_config
from src.monitoring.store import JobStore, QueueFullError
from src.monitoring.worker import MonitorRunner, months, shift_year
from src.monitoring.earth_engine import ProviderError
from src.monitoring.analysis import analyze_observations


POLYGON = {
    "id": "TEST",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[40, 47], [40.01, 47], [40.01, 47.01], [40, 47]]],
    },
}


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.config = load_monitoring_config().model_copy(
            update={
                "database": root / "jobs.sqlite3",
                "cache_directory": root / "cache",
                "history_years": 2,
            }
        )
        self.store = JobStore(self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def submit(self):
        return self.store.submit(POLYGON, "2024-04-01", "2024-04-10")[0]

    def test_concurrent_clicks_create_one_job(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = list(pool.map(lambda _: self.submit(), range(8)))
        self.assertEqual(len({job["id"] for job in jobs}), 1)
        claimed = self.store.claim()
        self.assertIsNone(self.store.claim())
        self.store.recover()
        self.assertEqual(self.store.claim()["id"], claimed["id"])

    def test_limits_and_retry(self):
        job = self.submit()
        for start, end in [
            ("2016-01-01", "2016-02-01"),
            ("2024-01-01", "2025-03-01"),
            ("2024-04-02", "2024-04-01"),
            ("not-date", "2024-04-01"),
            ("2099-01-01", "2099-02-01"),
        ]:
            with self.assertRaises(ValueError):
                self.store.submit(POLYGON, start, end)
        self.store.update(job["id"], "Ошибка", 0, status="failed", error_code="test")
        retried, reused = self.store.submit(POLYGON, "2024-04-01", "2024-04-10")
        self.assertFalse(reused)
        self.assertEqual(retried["id"], job["id"])
        limited = JobStore(self.config.model_copy(update={"max_queued_jobs": 1}))
        with self.assertRaises(QueueFullError):
            limited.submit(POLYGON, "2024-05-01", "2024-05-02")

    def test_worker_completion_weather_failure_and_cache(self):
        calls = []

        class FakeProvider:
            def __init__(self, config):
                pass

            def collect_month(self, source, geometry, start, end):
                calls.append((source, start, end))
                if source == "era5":
                    raise ProviderError("provider_unavailable", "Погода недоступна")
                return [
                    {"date": str(start), "ndvi": 0.4, "evi": 0.2, "ndwi": 0.1},
                    {"date": str(end), "ndvi": 0.5, "evi": 0.3, "ndwi": 0.1},
                ]

        job = self.submit()
        runner = MonitorRunner(self.config, self.store, FakeProvider)
        runner.run(self.store.claim())
        result = self.store.get(job["id"], with_result=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["summary"]["observations"], 2)
        self.assertEqual(result["result"]["summary"]["restored"], 8)
        self.assertTrue(
            any("Погодный источник" in value for value in result["result"]["warnings"])
        )
        before = sum(source == "sentinel2" for source, _, _ in calls)
        self.store.update(job["id"], "retry", 0, status="failed")
        self.submit()
        runner.run(self.store.claim())
        self.assertEqual(sum(source == "sentinel2" for source, _, _ in calls), before)
        self.assertEqual(self.store.get(job["id"])["status"], "completed")

    def test_missing_auth_is_reported_without_fake_results(self):
        def unavailable(config):
            raise ProviderError("authentication_required", "Войдите в Google")

        job = self.submit()
        MonitorRunner(self.config, self.store, unavailable).run(self.store.claim())
        result = self.store.get(job["id"], with_result=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "authentication_required")
        self.assertIsNone(result["result"])

    def test_long_gaps_edges_and_missing_weather_are_not_fabricated(self):
        params = {
            "polygon_id": "X",
            "start": "2024-04-01",
            "end": "2024-06-30",
            "max_interpolation_gap_days": 30,
        }
        satellite = [
            {"date": "2024-04-10", "ndvi": 0.2},
            {"date": "2024-06-10", "ndvi": 0.2},
        ]
        for year in (2022, 2023):
            satellite.extend(
                {"date": f"{year}-04-{day:02d}", "ndvi": 0.8} for day in (1, 10, 20)
            )
        result = analyze_observations(params, satellite, [])
        self.assertEqual(result["summary"]["restored"], 0)
        self.assertEqual(result["summary"]["observations"], 2)
        self.assertIsNone(result["points"][0]["filled"])
        self.assertFalse(
            any("дефицит влаги" in point["explanation"] for point in result["points"])
        )
        self.assertTrue(
            all(point["precipitation"] is None for point in result["points"])
        )

    def test_month_boundaries_and_leap_year(self):
        self.assertEqual(
            list(months(date(2024, 2, 28), date(2024, 3, 2))),
            [
                (date(2024, 2, 28), date(2024, 2, 29)),
                (date(2024, 3, 1), date(2024, 3, 2)),
            ],
        )
        self.assertEqual(shift_year(date(2024, 2, 29), 1), date(2023, 2, 28))


if __name__ == "__main__":
    unittest.main()
