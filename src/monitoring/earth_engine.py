"""Ограниченные по месяцу запросы к GEE; секреты остаются вне сайта."""

from datetime import date, timedelta
import os


class ProviderError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def classify_error(error):
    text = str(error).lower()
    if any(
        word in text
        for word in (
            "authenticate",
            "credential",
            "invalid_grant",
            "refresh",
            "unauthorized",
        )
    ):
        return ProviderError(
            "authentication_required",
            "Нужен вход в Google на сервере: earthengine authenticate --auth_mode=localhost",
        )
    if any(
        word in text
        for word in (
            "permission",
            "403",
            "not registered",
            "not enabled",
            "access denied",
            "not found",
        )
    ):
        return ProviderError(
            "project_access",
            "Проверьте регистрацию проекта в Earth Engine, включение API и права аккаунта",
        )
    if any(word in text for word in ("quota", "429", "too many")):
        return ProviderError(
            "quota_exceeded",
            "Лимит Earth Engine исчерпан. Повторите позже; собранные месяцы сохранены",
        )
    return ProviderError(
        "provider_unavailable",
        "Earth Engine временно недоступен или запрос слишком тяжёлый. Повторите анализ позже",
    )


def connect(project_id):
    try:
        import ee
    except ImportError:
        raise ProviderError(
            "dependency_missing",
            "Установите зависимости: pip install -r requirements-collect.txt",
        ) from None
    try:
        # ADC позволяет использовать сервисный аккаунт в Docker/облаке.
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            import google.auth

            credentials, _ = google.auth.default(
                scopes=[
                    "https://www.googleapis.com/auth/earthengine",
                    "https://www.googleapis.com/auth/cloud-platform",
                ]
            )
            ee.Initialize(credentials=credentials, project=project_id)
        else:
            ee.Initialize(project=project_id)
        ee.data.setDeadline(90000)
        return ee
    except Exception as error:
        raise classify_error(error) from None


class EarthEngineProvider:
    satellite_collection = "COPERNICUS/S2_SR_HARMONIZED"
    weather_collection = "ECMWF/ERA5_LAND/DAILY_AGGR"

    def __init__(self, config):
        self.config = config
        self.ee = connect(config.project_id)

    def collect_month(self, source, geometry, start, end):
        ee = self.ee
        region = ee.Geometry(geometry)
        images = (
            ee.ImageCollection(
                self.satellite_collection
                if source == "sentinel2"
                else self.weather_collection
            )
            .filterDate(str(start), str(end + timedelta(days=1)))
            .filterBounds(region)
            .sort("system:time_start")
        )
        if source == "sentinel2":
            images = images.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))

        def reduce_image(raw):
            image = ee.Image(raw)
            if source == "sentinel2":
                scl = image.select("SCL")
                valid = scl.neq(0)
                for value in (1, 3, 7, 8, 9, 10, 11):
                    valid = valid.And(scl.neq(value))
                red = image.select("B4").multiply(0.0001)
                nir = image.select("B8").multiply(0.0001)
                blue = image.select("B2").multiply(0.0001)
                green = image.select("B3").multiply(0.0001)
                indices = ee.Image.cat(
                    [
                        nir.subtract(red).divide(nir.add(red)).rename("ndvi"),
                        nir.subtract(red)
                        .multiply(2.5)
                        .divide(
                            nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
                        )
                        .rename("evi"),
                        green.subtract(nir).divide(green.add(nir)).rename("ndwi"),
                    ]
                ).updateMask(valid)
                # Счётчики наследуют проекцию и маску B8, а не проекцию constant(1).
                total = image.select("B8").multiply(0).add(1).rename("total_pixel")
                valid_pixels = total.updateMask(valid).rename("valid_pixel")
                stats = (
                    indices.addBands(total)
                    .addBands(valid_pixels)
                    .reduceRegion(
                        reducer=ee.Reducer.median().combine(
                            ee.Reducer.count(), sharedInputs=True
                        ),
                        geometry=region,
                        scale=self.config.scale_m,
                        maxPixels=20_000_000,
                        tileScale=4,
                    )
                )
            else:
                # ERA5 имеет сетку около 11 км. Берём ячейку центра поля;
                # значение является региональным погодным контекстом.
                stats = image.select(
                    ["temperature_2m", "total_precipitation_sum"]
                ).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=region.centroid(10),
                    scale=11132,
                    maxPixels=10000,
                )
            return ee.Feature(None, stats).set(
                {"date": image.date().format("YYYY-MM-dd"), "scene_id": image.id()}
            )

        try:
            # Для поля до 5000 га месячная коллекция ограничена; усечения данных нет.
            count = int(images.size().getInfo())
            if count > 300:
                raise ProviderError(
                    "request_too_large",
                    "Слишком много сцен за месяц. Выберите меньшую территорию",
                )
            if not count:
                return []
            payload = ee.FeatureCollection(
                images.toList(count).map(reduce_image)
            ).getInfo()
        except ProviderError:
            raise
        except Exception as error:
            raise classify_error(error) from None
        result = []
        for feature in payload.get("features", []):
            props = feature.get("properties", {})
            if source == "sentinel2":
                count = props.get("valid_pixel_count", 0) or 0
                total = props.get("total_pixel_count", 0) or 0
                fraction = count / total if total else 0
                if (
                    count < self.config.min_pixel_count
                    or fraction < self.config.min_valid_fraction
                    or props.get("ndvi_median") is None
                ):
                    continue
                result.append(
                    {
                        "date": props["date"],
                        "scene_id": props["scene_id"],
                        "ndvi": props["ndvi_median"],
                        "evi": props.get("evi_median"),
                        "ndwi": props.get("ndwi_median"),
                        "valid_fraction": fraction,
                        "pixel_count": count,
                    }
                )
            else:
                temp = props.get("temperature_2m")
                precip = props.get("total_precipitation_sum")
                result.append(
                    {
                        "date": props["date"],
                        "temperature": temp - 273.15 if temp is not None else None,
                        "precipitation": (
                            precip * 1000
                            if precip is not None and precip >= 0
                            else None
                        ),
                    }
                )
        return result
