"""Сбор спутниковых и метеорядов по произвольному контуру через Earth Engine.

Задача провайдера — воспроизвести конвейер организаторов, а не построить свой.
Это принципиально: модель `lightgbm_sensor` опирается на систематические
смещения между сенсорами, и если собрать данные другими формулами, смещения
окажутся другими, а признаки источника начнут врать.

Что удалось зафиксировать по следам в конкурсных файлах:

* `modis_ndvi` не кратен шагу 1e-4, хотя продукт MOD13Q1 хранит целые значения
  с этой шкалой. Значит берётся среднее по площади контура, а не пиксель;
* `s2_ndwi` имеет медиану около -0.37 и почти целиком отрицателен — это
  формула Макфитерса `(Green - NIR) / (Green + NIR)`, а не индекс Гао по SWIR;
* `landsat_ndvi` выходит за `[-1, 1]` (до 1.84), а `landsat_ndwi` доходит до
  -18.5. Так бывает только если к Collection 2 Level-2 применены штатные
  коэффициенты `0.0000275` и сдвиг `-0.2`: над тёмными целями отражение
  становится отрицательным, знаменатель проходит через ноль и индекс
  разлетается. Клиппинга организаторы не делают, поэтому не делаем и мы;
* `s2_evi` содержит редкие выбросы порядка 1e11-1e12 при медиане 0.157. Это
  тот же механизм: знаменатель `NIR + 6*RED - 7.5*BLUE + 1` изредка проходит
  через ноль. Сама медиана согласуется с отражением в долях единицы.

Порядок сборки `primary_ndvi` — приоритет s2 → landsat → modis — взят из
`data/column_description.md` и подтверждён на конкурсных файлах.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Диапазон дня года, которым ограничены конкурсные файлы: 1 апреля — 30 октября.
SEASON_START_DOY = 91
SEASON_END_DOY = 304

# Классы маски SCL Sentinel-2, которые исключаются как непригодные:
# нет данных, дефектный пиксель, тень облака, облако средней и высокой
# вероятности, перистое облако, снег.
S2_REJECTED_SCL = (0, 1, 3, 8, 9, 10, 11)

OUTPUT_COLUMNS = (
    "anon_polygon_id",
    "date",
    "s2_ndvi",
    "s2_evi",
    "s2_ndwi",
    "landsat_ndvi",
    "landsat_evi",
    "landsat_ndwi",
    "modis_ndvi",
    "modis_evi",
    "era5_temp_c",
    "era5_precip_mm",
    "year",
    "primary_ndvi",
    "doy",
    "crop_type",
)


@dataclass(frozen=True)
class ProviderSettings:
    """Настройки сбора. Значения по умолчанию повторяют конкурсный конвейер."""

    key_path: Path | None = None
    project: str | None = None
    # Масштаб редукции: 10 м для Sentinel-2, 30 м для Landsat, 250 м для MODIS,
    # 11132 м — шаг сетки ERA5-Land.
    s2_scale: int = 10
    landsat_scale: int = 30
    modis_scale: int = 250
    # Сетка ERA5-Land крупнее любого поля, поэтому редуцировать надо на мелком
    # масштабе: при 11 км контур меньше пикселя не попадает ни в один центр и
    # редукция возвращает пусто. Значение ячейки от пересэмплирования не меняется.
    era5_scale: int = 1000
    season_only: bool = True


class GeeProvider:
    """Возвращает ряд по контуру в схеме конкурсного датасета."""

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self._ee = None

    # ------------------------------------------------------------------ auth

    def initialise(self):
        """Публичная точка входа: нужна модулям, работающим с той же сессией."""

        return self._initialise()

    def _initialise(self):
        if self._ee is not None:
            return self._ee
        import json

        import ee

        key_path = self.settings.key_path
        if key_path is not None:
            key_path = Path(key_path)
            if not key_path.exists():
                raise FileNotFoundError(
                    f"Ключ сервис-аккаунта не найден: {key_path}"
                )
            key = json.loads(key_path.read_text(encoding="utf-8"))
            project = self.settings.project or key.get("project_id")
            credentials = ee.ServiceAccountCredentials(
                key["client_email"], str(key_path)
            )
            ee.Initialize(credentials, project=project)
        else:
            project = self.settings.project or os.environ.get("EE_PROJECT_ID")
            if not project:
                raise ValueError(
                    "Укажите collect.project в config.yaml или EE_PROJECT_ID"
                )
            # Использует Application Default Credentials / earthengine authenticate.
            ee.Initialize(project=project)
        self._ee = ee
        return ee

    # -------------------------------------------------------------- geometry

    def geometry(self, source):
        """Принимает GeoJSON, путь к нему, bbox из четырёх чисел или точку."""

        ee = self._initialise()
        if isinstance(source, (str, Path)) and Path(str(source)).exists():
            import json

            source = json.loads(Path(str(source)).read_text(encoding="utf-8"))
        if isinstance(source, dict):
            if source.get("type") == "FeatureCollection":
                source = source["features"][0]["geometry"]
            elif source.get("type") == "Feature":
                source = source["geometry"]
            return ee.Geometry(source)
        if isinstance(source, (list, tuple)) and len(source) == 4:
            return ee.Geometry.Rectangle([float(v) for v in source])
        if isinstance(source, (list, tuple)) and len(source) == 2:
            # Точка разворачивается в квадрат ~1 км: усреднять по точке нечего.
            lon, lat = float(source[0]), float(source[1])
            half = 0.005
            return ee.Geometry.Rectangle(
                [lon - half, lat - half, lon + half, lat + half]
            )
        raise ValueError("Не удалось разобрать геометрию контура")

    # ------------------------------------------------------------ reductions

    def _reduce(self, collection, geom, scale: int, bands: list[str]) -> pd.DataFrame:
        """Среднее по контуру для каждого снимка коллекции."""

        ee = self._initialise()

        def measure(image):
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=scale,
                maxPixels=int(1e9),
                bestEffort=True,
            )
            return ee.Feature(None, stats).set(
                "date", image.date().format("YYYY-MM-dd")
            )

        features = ee.FeatureCollection(collection.map(measure)).getInfo()["features"]
        records = []
        for item in features:
            properties = item["properties"]
            row = {"date": properties.get("date")}
            for band in bands:
                row[band] = properties.get(band)
            records.append(row)

        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=["date", *bands])
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.dropna(subset=bands, how="all")
        # За один день может быть несколько сцен: усредняем их.
        return frame.groupby("date", as_index=False)[bands].mean()

    # -------------------------------------------------------------- sentinel

    def _sentinel2(self, geom, start: str, end: str) -> pd.DataFrame:
        ee = self._initialise()

        def prepare(image):
            scl = image.select("SCL")
            valid = ee.Image.constant(1)
            for code in S2_REJECTED_SCL:
                valid = valid.And(scl.neq(code))
            optical = image.select(["B2", "B3", "B4", "B8"]).divide(10000)
            blue = optical.select("B2")
            green = optical.select("B3")
            red = optical.select("B4")
            nir = optical.select("B8")
            ndvi = nir.subtract(red).divide(nir.add(red))
            evi = (
                nir.subtract(red)
                .multiply(2.5)
                .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            )
            ndwi = green.subtract(nir).divide(green.add(nir))
            stack = ndvi.rename("s2_ndvi").addBands(
                [evi.rename("s2_evi"), ndwi.rename("s2_ndwi")]
            )
            return stack.updateMask(valid).copyProperties(image, ["system:time_start"])

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filterBounds(geom)
            .map(prepare)
        )
        return self._reduce(
            collection, geom, self.settings.s2_scale, ["s2_ndvi", "s2_evi", "s2_ndwi"]
        )

    # --------------------------------------------------------------- landsat

    def _landsat(self, geom, start: str, end: str) -> pd.DataFrame:
        ee = self._initialise()

        def prepare(image):
            qa = image.select("QA_PIXEL")
            # Биты QA_PIXEL: 1 — расширенное облако, 3 — облако, 4 — тень.
            clear = (
                qa.bitwiseAnd(1 << 1)
                .eq(0)
                .And(qa.bitwiseAnd(1 << 3).eq(0))
                .And(qa.bitwiseAnd(1 << 4).eq(0))
            )
            optical = (
                image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5"])
                .multiply(0.0000275)
                .add(-0.2)
            )
            blue = optical.select("SR_B2")
            green = optical.select("SR_B3")
            red = optical.select("SR_B4")
            nir = optical.select("SR_B5")
            ndvi = nir.subtract(red).divide(nir.add(red))
            evi = (
                nir.subtract(red)
                .multiply(2.5)
                .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            )
            ndwi = green.subtract(nir).divide(green.add(nir))
            stack = ndvi.rename("landsat_ndvi").addBands(
                [evi.rename("landsat_evi"), ndwi.rename("landsat_ndwi")]
            )
            return stack.updateMask(clear).copyProperties(image, ["system:time_start"])

        eight = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        nine = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        collection = (
            eight.merge(nine).filterDate(start, end).filterBounds(geom).map(prepare)
        )
        return self._reduce(
            collection,
            geom,
            self.settings.landsat_scale,
            ["landsat_ndvi", "landsat_evi", "landsat_ndwi"],
        )

    # ----------------------------------------------------------------- modis

    def _modis(self, geom, start: str, end: str) -> pd.DataFrame:
        ee = self._initialise()

        def prepare(image):
            scaled = image.select(["NDVI", "EVI"]).multiply(0.0001)
            return scaled.rename(["modis_ndvi", "modis_evi"]).copyProperties(
                image, ["system:time_start"]
            )

        collection = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(start, end)
            .filterBounds(geom)
            .map(prepare)
        )
        return self._reduce(
            collection, geom, self.settings.modis_scale, ["modis_ndvi", "modis_evi"]
        )

    # ------------------------------------------------------------------ era5

    def _era5(self, geom, start: str, end: str) -> pd.DataFrame:
        ee = self._initialise()

        def prepare(image):
            temp = image.select("temperature_2m").subtract(273.15).rename("era5_temp_c")
            precip = (
                image.select("total_precipitation_sum")
                .multiply(1000)
                .rename("era5_precip_mm")
            )
            return temp.addBands(precip).copyProperties(image, ["system:time_start"])

        collection = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start, end)
            .map(prepare)
        )
        return self._reduce(
            collection,
            geom,
            self.settings.era5_scale,
            ["era5_temp_c", "era5_precip_mm"],
        )

    # ---------------------------------------------------------------- сборка

    def fetch(
        self,
        source,
        start: str,
        end: str,
        polygon_id: str = "AOI-EXT-0001",
        crop_type: str = "неизвестно",
    ) -> pd.DataFrame:
        """Собирает полный ряд по контуру в схеме конкурсного датасета."""

        geom = self.geometry(source)
        grid = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
        for part in (
            self._sentinel2(geom, start, end),
            self._landsat(geom, start, end),
            self._modis(geom, start, end),
            self._era5(geom, start, end),
        ):
            if not part.empty:
                grid = grid.merge(part, on="date", how="left")

        for column in OUTPUT_COLUMNS:
            if column not in grid.columns:
                grid[column] = np.nan

        grid["anon_polygon_id"] = polygon_id
        grid["crop_type"] = crop_type
        grid["year"] = grid["date"].dt.year
        grid["doy"] = grid["date"].dt.dayofyear
        if self.settings.season_only:
            grid = grid[grid["doy"].between(SEASON_START_DOY, SEASON_END_DOY)].copy()

        grid["primary_ndvi"] = merge_primary(grid)
        return grid[list(OUTPUT_COLUMNS)].sort_values("date").reset_index(drop=True)

    def fetch_many(self, source, start: str, end: str) -> pd.DataFrame:
        """Собирает конкурсную схему для всех объектов GeoJSON."""

        features = load_features(source)
        frames = [
            self.fetch(
                item["geometry"],
                start,
                end,
                polygon_id=item["anon_polygon_id"],
                crop_type=item["crop_type"],
            )
            for item in features
        ]
        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        return pd.concat(frames, ignore_index=True).sort_values(
            ["anon_polygon_id", "date"]
        ).reset_index(drop=True)


def load_features(source, default_crop: str = "неизвестно") -> list[dict]:
    """Разбирает GeoJSON в список контуров с идентификатором и культурой.

    Принимает FeatureCollection, Feature или голую геометрию. Идентификатор и
    культура берутся из свойств объекта, если они там есть, — это позволяет
    описать сразу несколько регионов одним файлом.
    """

    import json

    if isinstance(source, (str, Path)):
        source = json.loads(Path(str(source)).read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("Ожидался GeoJSON или путь к нему")

    if source.get("type") == "FeatureCollection":
        features = source.get("features", [])
    elif source.get("type") == "Feature":
        features = [source]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": source}]

    result = []
    for position, feature in enumerate(features, start=1):
        properties = feature.get("properties") or {}
        identifier = (
            properties.get("anon_polygon_id")
            or properties.get("id")
            or properties.get("name")
            or f"AOI-EXT-{position:04d}"
        )
        result.append(
            {
                "anon_polygon_id": str(identifier),
                "crop_type": str(properties.get("crop_type", default_crop)),
                "geometry": feature["geometry"],
            }
        )
    return result


def merge_primary(frame: pd.DataFrame) -> pd.Series:
    """Приоритетная склейка s2 → landsat → modis, как у организаторов."""

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in ("modis_ndvi", "landsat_ndvi", "s2_ndvi"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            result = values.where(values.notna(), result)
    return result
