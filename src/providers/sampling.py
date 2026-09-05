"""Отбор контуров пашни для расширения обучающей выборки.

Точки берутся только там, где ESA WorldCover относит пиксель к классу 40
(cropland). Каждая точка разворачивается в квадрат размером с типичное поле, и
квадрат оставляется только если он почти целиком лежит на пашне: иначе в
выборку попадают лесополосы, дороги и края угодий, у которых сезонная кривая
устроена иначе, чем у конкурсных полигонов.

Размер контура выбран не произвольно. `modis_ndvi` в конкурсных файлах не
кратен шагу 1e-4 продукта MOD13Q1, то есть является средним по нескольким
пикселям 250 м. Квадрат со стороной около 700 м покрывает примерно восемь
таких пикселей и воспроизводит это свойство.
"""

from __future__ import annotations

WORLDCOVER = "ESA/WorldCover/v200/2021"
CROPLAND_CLASS = 40

# Радиус буфера в метрах: сторона итогового квадрата примерно вдвое больше.
DEFAULT_RADIUS_M = 350
# Доля пашни внутри контура, ниже которой контур отбрасывается.
DEFAULT_MIN_PURITY = 0.9


def cropland_fields(
    provider,
    bbox: tuple[float, float, float, float],
    count: int,
    seed: int = 42,
    radius_m: int = DEFAULT_RADIUS_M,
    min_purity: float = DEFAULT_MIN_PURITY,
    scale: int = 100,
    oversample: int = 10,
    crop_type: str = "неизвестно",
    prefix: str = "AOI-EXT",
) -> list[dict]:
    """Возвращает контуры пашни в формате, который принимает `fetch_many`."""

    ee = provider.initialise()
    region = ee.Geometry.Rectangle([float(v) for v in bbox])
    cropland = ee.Image(WORLDCOVER).select("Map").eq(CROPLAND_CLASS)

    points = cropland.selfMask().sample(
        region=region,
        scale=scale,
        numPixels=count * oversample,
        seed=seed,
        geometries=True,
        dropNulls=True,
    )
    squares = points.map(lambda item: ee.Feature(item.geometry().buffer(radius_m).bounds()))
    with_purity = cropland.reduceRegions(
        collection=squares, reducer=ee.Reducer.mean(), scale=10
    )
    selected = with_purity.filter(ee.Filter.gte("mean", min_purity)).limit(count)

    features = selected.getInfo()["features"]
    result = []
    for position, feature in enumerate(features, start=1):
        result.append(
            {
                "anon_polygon_id": f"{prefix}-{position:04d}",
                "crop_type": crop_type,
                "geometry": feature["geometry"],
                "purity": feature["properties"].get("mean"),
            }
        )
    return result
