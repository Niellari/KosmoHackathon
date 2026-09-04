"""Общие типы провайдеров внешних наблюдений."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


RAW_COLUMNS = [
    "polygon_id",
    "date",
    "sensor",
    "scene_id",
    "ndvi",
    "evi",
    "ndwi",
    "valid_fraction",
    "pixel_count",
    "scene_cloud_percent",
]


class ObservationProvider(Protocol):
    def collect(
        self,
        features: list[dict],
        id_property: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Возвращает сырые наблюдения для переданных GeoJSON Feature."""

