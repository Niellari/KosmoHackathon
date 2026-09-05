import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wsgiref.util import setup_testing_defaults
import pandas as pd
from src.webapp import create_app


class SearchTests(unittest.TestCase):
    @patch("src.webapp.AnalysisPipeline.from_csv")
    def test_saved_names_are_searched_from_storage(self, factory):
        factory.return_value.data = pd.DataFrame(
            columns=["anon_polygon_id", "crop_type", "year"]
        )
        with tempfile.TemporaryDirectory() as folder:
            storage = Path(folder) / "plots.json"
            storage.write_text(
                json.dumps(
                    {
                        "a": {
                            "id": "a",
                            "name": "Берёзовый участок",
                            "source": "user_geometry",
                        }
                    }
                ),
                encoding="utf-8",
            )
            app = create_app(Path("unused.csv"), polygons_path=storage)
            from urllib.parse import urlencode

            for query, count in [("БЕРЕЗ", 1), ("нет такого", 0)]:
                env = {}
                setup_testing_defaults(env)
                env.update(
                    PATH_INFO="/api/search/plots", QUERY_STRING=urlencode({"q": query})
                )
                env["wsgi.input"] = io.BytesIO()
                status = []
                body = b"".join(app(env, lambda s, h, exc_info=None: status.append(s)))
                self.assertTrue(status[0].startswith("200"))
                self.assertEqual(len(json.loads(body)["items"]), count)
