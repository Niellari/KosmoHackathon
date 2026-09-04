"""Проверки изолированной части отправщика без запуска браузера."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from api.config import ValidationConfig
from api.credentials import CredentialsError, load_credentials
from api.result import SubmissionReceipt, append_history, was_submitted
from api.session import CookieStore
from api.validation import SubmissionValidationError, validate_submission_file


class SubmitterValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = ValidationConfig(
            expected_columns=("anon_polygon_id", "date", "prediction"),
            identity_columns=("anon_polygon_id", "date"),
            target_column="prediction",
            test_data_path=None,
        )

    def tearDown(self):
        self.directory.cleanup()

    def write_csv(self, rows):
        path = self.root / "submission.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.config.expected_columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_file_returns_hash_and_row_count(self):
        path = self.write_csv(
            [
                {
                    "anon_polygon_id": "AOI-1",
                    "date": "2026-01-02",
                    "prediction": "0.42",
                }
            ]
        )
        result = validate_submission_file(path, self.config)
        self.assertEqual(result.rows, 1)
        self.assertEqual(len(result.sha256), 64)

    def test_duplicate_key_is_rejected(self):
        row = {
            "anon_polygon_id": "AOI-1",
            "date": "2026-01-02",
            "prediction": "0.42",
        }
        path = self.write_csv([row, row])
        with self.assertRaisesRegex(SubmissionValidationError, "Повторяющийся"):
            validate_submission_file(path, self.config)

    def test_non_finite_prediction_is_rejected(self):
        path = self.write_csv(
            [
                {
                    "anon_polygon_id": "AOI-1",
                    "date": "2026-01-02",
                    "prediction": "nan",
                }
            ]
        )
        with self.assertRaisesRegex(SubmissionValidationError, "конечным"):
            validate_submission_file(path, self.config)


class SubmissionHistoryTests(unittest.TestCase):
    def test_successful_hash_is_found_in_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            receipt = SubmissionReceipt.create(
                status="submitted",
                file="submission.csv",
                sha256="abc",
                rows=1,
                url="https://example.test/result",
            )
            append_history(path, receipt)
            self.assertTrue(was_submitted(path, "abc"))
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "submitted")


class FakeDriver:
    def __init__(self, cookies=None):
        self.cookies = list(cookies or [])

    def get_cookies(self):
        return self.cookies

    def add_cookie(self, cookie):
        self.cookies.append(cookie)


class CookieStoreTests(unittest.TestCase):
    def test_cookies_survive_between_driver_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session" / "cookies.json"
            store = CookieStore(path)
            source = FakeDriver(
                [{"name": "session", "value": "secret", "domain": "example.test"}]
            )

            self.assertEqual(store.save(source), 1)
            destination = FakeDriver()
            self.assertEqual(store.restore(destination), 1)
            self.assertEqual(destination.cookies[0]["value"], "secret")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_broken_cookie_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cookies.json"
            path.write_text("not-json", encoding="utf-8")
            self.assertEqual(CookieStore(path).restore(FakeDriver()), 0)


class CredentialsTests(unittest.TestCase):
    def test_credentials_are_loaded_without_logging_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.env"
            path.write_text(
                "COSMO_EMAIL=user@example.test\nCOSMO_PASSWORD=secret=value\n",
                encoding="utf-8",
            )
            credentials = load_credentials(path)
            self.assertEqual(credentials.email, "user@example.test")
            self.assertEqual(credentials.password, "secret=value")

    def test_placeholder_credentials_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.env"
            path.write_text(
                "COSMO_EMAIL=your-email@example.com\nCOSMO_PASSWORD=your-password\n",
                encoding="utf-8",
            )
            with self.assertRaises(CredentialsError):
                load_credentials(path)


if __name__ == "__main__":
    unittest.main()
