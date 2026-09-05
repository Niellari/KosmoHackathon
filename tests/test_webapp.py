"""Проверки запуска локального веб-сервиса."""

from __future__ import annotations

import errno
import io
import json
import tempfile
from pathlib import Path
from wsgiref.util import setup_testing_defaults
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch

from src.webapp import create_app, find_available_port, open_in_browser


class PolygonApiTests(unittest.TestCase):
    def request(self, app, method, path, body=None):
        env = {}
        setup_testing_defaults(env)
        data = json.dumps(body).encode() if body is not None else b""
        env.update(
            REQUEST_METHOD=method,
            PATH_INFO=path,
            CONTENT_TYPE="application/json",
            CONTENT_LENGTH=str(len(data)),
        )
        env["wsgi.input"] = io.BytesIO(data)
        status = []
        output = b"".join(
            app(env, lambda value, headers, exc_info=None: status.append(value))
        )
        return int(status[0].split()[0]), output

    @patch("src.webapp.AnalysisPipeline.from_csv")
    def test_save_reload_delete_and_reject_bad_geometry(self, factory):
        pipeline = MagicMock()
        pipeline.data = pd.DataFrame(columns=["anon_polygon_id", "crop_type", "year"])
        factory.return_value = pipeline
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polygons.json"
            app = create_app(Path("unused.csv"), polygons_path=path)
            geometry = {
                "type": "Polygon",
                "coordinates": [[[40, 47], [40.01, 47], [40.01, 47.01], [40, 47]]],
            }
            status, body = self.request(
                app,
                "POST",
                "/api/polygons",
                {"geometry": geometry, "name": "Тестовое поле"},
            )
            self.assertEqual(status, 201)
            item = json.loads(body)
            self.assertGreater(item["area_ha"], 0)
            restarted = create_app(Path("unused.csv"), polygons_path=path)
            status, body = self.request(restarted, "GET", "/api/polygons")
            self.assertEqual(json.loads(body)["items"][0]["name"], "Тестовое поле")
            status, body = self.request(
                restarted,
                "PATCH",
                "/api/polygons/" + item["id"],
                {"name": "Новое название"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["name"], "Новое название")
            status, body = self.request(
                restarted,
                "POST",
                "/api/polygons/" + item["id"] + "/analyses",
                {"start": "2024-06-01", "end": "2024-06-14"},
            )
            self.assertEqual(status, 202)
            job = json.loads(body)["job"]
            status, body = self.request(restarted, "GET", "/api/analyses/" + job["id"])
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["status"], "queued")
            status, body = self.request(
                restarted, "GET", "/api/polygons/" + item["id"] + "/analyses/latest"
            )
            self.assertEqual(json.loads(body)["job"]["id"], job["id"])
            bad = {
                "type": "Polygon",
                "coordinates": [[[40, 47], [40.01, 47], [40.01, 47.01]]],
            }
            status, _ = self.request(
                restarted, "POST", "/api/polygons", {"geometry": bad}
            )
            self.assertEqual(status, 400)
            status, _ = self.request(restarted, "DELETE", "/api/polygons/" + item["id"])
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {})


class ServerPortTests(unittest.TestCase):
    def test_busy_preferred_port_falls_back_to_next_port(self):
        probe = MagicMock()
        probe.__enter__.return_value.bind.side_effect = [
            OSError(errno.EADDRINUSE, "Address already in use"),
            None,
        ]

        with patch("src.webapp.socket.socket", return_value=probe):
            selected = find_available_port("127.0.0.1", 8000)

        self.assertEqual(selected, 8001)
        self.assertEqual(probe.__enter__.return_value.bind.call_count, 2)

    @patch("src.webapp.webbrowser.open", return_value=True)
    def test_browser_is_opened_in_new_tab(self, browser_open):
        open_in_browser("http://127.0.0.1:8001/")

        browser_open.assert_called_once_with("http://127.0.0.1:8001/", new=2)


if __name__ == "__main__":
    unittest.main()
