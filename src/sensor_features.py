"""Признаки источника primary_ndvi и отдельных сенсорных рядов."""

from __future__ import annotations

import numpy as np
import pandas as pd


EPOCH = pd.Timestamp("2010-01-01")
SENSORS = (
    ("s2", "s2_ndvi", "p_s2"),
    ("landsat", "landsat_ndvi", "p_landsat"),
    ("modis", "modis_ndvi", "p_modis"),
)
MODULI = (5, 8, 16)
PROBABILITY_COLUMNS = tuple(sensor[2] for sensor in SENSORS)
AUXILIARY_SERIES = (
    ("s2_evi", "s2_evi"),
    ("s2_ndwi", "s2_ndwi"),
    ("landsat_evi", "landsat_evi"),
    ("landsat_ndwi", "landsat_ndwi"),
    ("modis_evi", "modis_evi"),
)
WEATHER_COLUMNS = ("era5_temp_c", "era5_precip_mm")


def source_labels(frame: pd.DataFrame) -> np.ndarray:
    """Источник primary_ndvi с приоритетом Sentinel-2 -> Landsat -> MODIS."""

    return np.where(
        frame["s2_ndvi"].notna(),
        0,
        np.where(
            frame["landsat_ndvi"].notna(),
            1,
            np.where(frame["modis_ndvi"].notna(), 2, -1),
        ),
    )


def _epoch_days(frame: pd.DataFrame) -> np.ndarray:
    return (frame["date"] - EPOCH).dt.days.to_numpy()


def calendar_features(
    context: pd.DataFrame, rows: pd.DataFrame, feature_version: int = 2
) -> pd.DataFrame:
    """Оценивает орбитальный календарь без признаков скрытых строк."""

    context_days = _epoch_days(context)
    row_days = _epoch_days(rows)
    context_polygons = context["anon_polygon_id"].to_numpy()
    row_polygons = rows["anon_polygon_id"].to_numpy()
    context_doy = context["date"].dt.dayofyear.to_numpy()
    row_doy = rows["date"].dt.dayofyear.to_numpy()
    result: dict[str, np.ndarray] = {}

    for short_name, column, _ in SENSORS:
        present = context[column].notna().to_numpy()
        for modulus in MODULI:
            rates = pd.Series(present).groupby(
                [pd.Series(context_polygons), pd.Series(context_days % modulus)]
            ).mean()
            index = pd.MultiIndex.from_arrays(
                [row_polygons, row_days % modulus]
            )
            result[f"{short_name}_rate_m{modulus}"] = rates.reindex(index).to_numpy()

        observed = pd.DataFrame(
            {"polygon": context_polygons[present], "day": context_days[present]}
        )
        by_polygon = {
            polygon: np.sort(group["day"].to_numpy())
            for polygon, group in observed.groupby("polygon", sort=False)
        }
        previous = np.full(len(rows), np.nan)
        following = np.full(len(rows), np.nan)
        for position, (polygon, day) in enumerate(zip(row_polygons, row_days)):
            days = by_polygon.get(polygon)
            if days is None or not len(days):
                continue
            insertion = int(np.searchsorted(days, day))
            if insertion:
                previous[position] = day - days[insertion - 1]
            if insertion < len(days):
                following[position] = days[insertion] - day
        result[f"{short_name}_calendar_prev_days"] = previous
        result[f"{short_name}_calendar_next_days"] = following

        # Погода и покрытие сцены коррелируют между полями одного региона.
        # Доля доступных наблюдений на точную дату переносит этот сигнал, но
        # целевая строка уже скрыта и потому не создаёт утечки.
        if feature_version >= 2:
            date_rates = pd.Series(present).groupby(
                context["date"].reset_index(drop=True)
            ).mean()
            result[f"{short_name}_date_availability"] = date_rates.reindex(
                rows["date"]
            ).to_numpy()

    # Источник primary_ndvi часто повторяет орбитный шаблон поля между годами.
    # Дополняем остатки календаря причинными частотами из доступного контекста.
    if feature_version >= 2:
        labels = source_labels(context)
        observed_source = context["primary_ndvi"].notna().to_numpy() & (labels >= 0)
        polygon_doy_index = pd.MultiIndex.from_arrays([row_polygons, row_doy])
        for label, name in enumerate(("s2", "landsat", "modis")):
            indicator = (labels[observed_source] == label).astype(float)
            polygon_doy_rates = pd.Series(indicator).groupby(
                [
                    pd.Series(context_polygons[observed_source]),
                    pd.Series(context_doy[observed_source]),
                ]
            ).mean()
            result[f"p_{name}_polygon_doy"] = polygon_doy_rates.reindex(
                polygon_doy_index
            ).to_numpy()

            date_rates = pd.Series(indicator).groupby(
                context.loc[observed_source, "date"].reset_index(drop=True)
            ).mean()
            result[f"p_{name}_date"] = date_rates.reindex(rows["date"]).to_numpy()

            doy_rates = pd.Series(indicator).groupby(
                pd.Series(context_doy[observed_source])
            ).mean()
            result[f"p_{name}_doy"] = doy_rates.reindex(row_doy).to_numpy()

    result["is_modis_doy"] = (row_doy % 16 == 1).astype(float)
    for modulus in MODULI:
        result[f"orbit_m{modulus}"] = (row_days % modulus).astype(float)
    return pd.DataFrame(result, index=rows.index)


def fit_source_classifier(
    batches: list[tuple[pd.DataFrame, pd.DataFrame, np.ndarray]],
    params: dict | None = None,
    feature_version: int = 2,
):
    """Обучает классификатор источника только на уже скрытых примерах."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        from src.models.base import ModelUnavailableError

        raise ModelUnavailableError("Для sensor-модели установите lightgbm") from error

    matrices: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    for context, targets, target_labels in batches:
        known = target_labels >= 0
        matrices.append(
            calendar_features(context, targets, feature_version).loc[known]
        )
        labels.append(target_labels[known])
    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "random_state": 42,
    }
    classifier = LGBMClassifier(**{**defaults, **(params or {})})
    classifier.fit(pd.concat(matrices, ignore_index=True), np.concatenate(labels))
    return classifier


def source_probabilities(
    classifier,
    context: pd.DataFrame,
    rows: pd.DataFrame,
    feature_version: int = 2,
) -> pd.DataFrame:
    probabilities = np.asarray(
        classifier.predict_proba(calendar_features(context, rows, feature_version)),
        dtype=float,
    )
    result = pd.DataFrame(0.0, index=rows.index, columns=PROBABILITY_COLUMNS)
    for position, label in enumerate(classifier.classes_):
        result.iloc[:, int(label)] = probabilities[:, position]
    return result


def _sensor_neighbors(
    context: pd.DataFrame, rows: pd.DataFrame, column: str
) -> tuple[np.ndarray, ...]:
    row_days = _epoch_days(rows)
    row_polygons = rows["anon_polygon_id"].to_numpy()
    observed = context[context[column].notna()]
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for polygon, group in observed.groupby("anon_polygon_id", sort=False):
        days = _epoch_days(group)
        order = np.argsort(days)
        series[str(polygon)] = (days[order], group[column].to_numpy(float)[order])

    arrays = [np.full(len(rows), np.nan) for _ in range(4)]
    previous_value, next_value, previous_days, next_days = arrays
    for position, (polygon, day) in enumerate(zip(row_polygons, row_days)):
        entry = series.get(str(polygon))
        if entry is None:
            continue
        days, values = entry
        insertion = int(np.searchsorted(days, day))
        if insertion:
            previous_value[position] = values[insertion - 1]
            previous_days[position] = day - days[insertion - 1]
        if insertion < len(days):
            next_value[position] = values[insertion]
            next_days[position] = days[insertion] - day
    return tuple(arrays)


def _pairwise_sensor_offsets(context: pd.DataFrame) -> np.ndarray:
    """Медианная поправка при переводе значения одного сенсора в другой."""

    columns = [column for _, column, _ in SENSORS]
    offsets = np.zeros((len(columns), len(columns)), dtype=float)
    for target, target_column in enumerate(columns):
        for source, source_column in enumerate(columns):
            if target == source:
                continue
            difference = (
                pd.to_numeric(context[target_column], errors="coerce")
                - pd.to_numeric(context[source_column], errors="coerce")
            ).dropna()
            offsets[target, source] = float(difference.median()) if len(difference) else 0.0
    return offsets


def _harmonized_primary_features(
    context: pd.DataFrame,
    rows: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Интерполирует плотный primary-ряд в шкале вероятного сенсора."""

    observed = context[context["primary_ndvi"].notna()].copy()
    observed["_source"] = source_labels(observed)
    observed = observed[observed["_source"] >= 0]
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for polygon, group in observed.groupby("anon_polygon_id", sort=False):
        days = _epoch_days(group)
        order = np.argsort(days)
        series[str(polygon)] = (
            days[order],
            group["primary_ndvi"].to_numpy(float)[order],
            group["_source"].to_numpy(int)[order],
        )

    row_days = _epoch_days(rows)
    previous = np.full(len(rows), np.nan)
    following = np.full(len(rows), np.nan)
    previous_days = np.full(len(rows), np.nan)
    next_days = np.full(len(rows), np.nan)
    previous_source = np.full(len(rows), -1, dtype=int)
    next_source = np.full(len(rows), -1, dtype=int)
    for position, (polygon, day) in enumerate(
        zip(rows["anon_polygon_id"].to_numpy(), row_days)
    ):
        entry = series.get(str(polygon))
        if entry is None:
            continue
        days, values, sources = entry
        insertion = int(np.searchsorted(days, day))
        if insertion:
            previous[position] = values[insertion - 1]
            previous_days[position] = day - days[insertion - 1]
            previous_source[position] = sources[insertion - 1]
        if insertion < len(days):
            following[position] = values[insertion]
            next_days[position] = days[insertion] - day
            next_source[position] = sources[insertion]

    offsets = _pairwise_sensor_offsets(context)
    interpolations = []
    result: dict[str, np.ndarray] = {}
    for target, (name, _, _) in enumerate(SENSORS):
        adjusted_previous = previous.copy()
        adjusted_following = following.copy()
        has_previous = previous_source >= 0
        has_next = next_source >= 0
        adjusted_previous[has_previous] += offsets[target, previous_source[has_previous]]
        adjusted_following[has_next] += offsets[target, next_source[has_next]]
        total = previous_days + next_days
        usable = np.isfinite(total) & (total > 0)
        fraction = np.divide(
            previous_days, total, out=np.zeros_like(total), where=usable
        )
        linear = adjusted_previous + (
            adjusted_following - adjusted_previous
        ) * fraction
        fallback = np.where(
            np.isfinite(adjusted_previous), adjusted_previous, adjusted_following
        )
        interpolation = np.where(usable & np.isfinite(linear), linear, fallback)
        result[f"harmonized_{name}_interpolation"] = interpolation
        interpolations.append(interpolation)

    for source, (name, _, _) in enumerate(SENSORS):
        result[f"primary_prev_is_{name}"] = (previous_source == source).astype(float)
        result[f"primary_next_is_{name}"] = (next_source == source).astype(float)

    values = np.stack(interpolations, axis=1)
    weights = probabilities.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    available = np.isfinite(values)
    effective_weights = np.where(available, weights, 0.0)
    total_weight = effective_weights.sum(axis=1)
    result["harmonized_source_interpolation"] = np.where(
        total_weight > 0,
        (np.where(available, values, 0.0) * effective_weights).sum(axis=1)
        / np.where(total_weight > 0, total_weight, 1.0),
        np.nan,
    )
    return pd.DataFrame(result, index=rows.index)


def _auxiliary_series_features(
    context: pd.DataFrame,
    rows: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Соседние EVI/NDWI дополняют NDVI сигналом формы и влажности."""

    result: dict[str, np.ndarray] = {}
    interpolations: dict[str, np.ndarray] = {}
    for name, column in AUXILIARY_SERIES:
        previous, following, previous_days, next_days = _sensor_neighbors(
            context, rows, column
        )
        total = previous_days + next_days
        usable = np.isfinite(total) & (total > 0)
        fraction = np.divide(
            previous_days, total, out=np.zeros_like(total), where=usable
        )
        linear = previous + (following - previous) * fraction
        fallback = np.where(np.isfinite(previous), previous, following)
        interpolation = np.where(usable & np.isfinite(linear), linear, fallback)
        result[f"{name}_value_prev"] = previous
        result[f"{name}_value_next"] = following
        result[f"{name}_interpolation"] = interpolation
        interpolations[name] = interpolation

    weights = probabilities.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    for index_name, names in (
        ("evi", ("s2_evi", "landsat_evi", "modis_evi")),
        ("ndwi", ("s2_ndwi", "landsat_ndwi")),
    ):
        values = np.column_stack([interpolations[name] for name in names])
        source_weights = weights[:, : len(names)]
        available = np.isfinite(values)
        effective = np.where(available, source_weights, 0.0)
        total_weight = effective.sum(axis=1)
        result[f"source_{index_name}_interpolation"] = np.where(
            total_weight > 0,
            (np.where(available, values, 0.0) * effective).sum(axis=1)
            / np.where(total_weight > 0, total_weight, 1.0),
            np.nan,
        )
    return pd.DataFrame(result, index=rows.index)


def _weather_features(context: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Восстанавливает скрытую погоду по времени поля и срезу региона."""

    result: dict[str, np.ndarray] = {}
    for column in WEATHER_COLUMNS:
        previous, following, previous_days, next_days = _sensor_neighbors(
            context, rows, column
        )
        total = previous_days + next_days
        usable = np.isfinite(total) & (total > 0)
        fraction = np.divide(
            previous_days, total, out=np.zeros_like(total), where=usable
        )
        linear = previous + (following - previous) * fraction
        fallback = np.where(np.isfinite(previous), previous, following)
        result[f"{column}_value_prev"] = previous
        result[f"{column}_value_next"] = following
        result[f"{column}_interpolation"] = np.where(
            usable & np.isfinite(linear), linear, fallback
        )

        observed = context[context[column].notna()]
        by_date = observed.groupby("date")[column].agg(["median", "std"])
        result[f"{column}_date_median"] = by_date["median"].reindex(
            rows["date"]
        ).to_numpy()
        result[f"{column}_date_std"] = by_date["std"].reindex(
            rows["date"]
        ).to_numpy()
    return pd.DataFrame(result, index=rows.index)


def _season_summary_features(context: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Уровень и амплитуда доступной части сезона конкретного поля."""

    target_index = pd.MultiIndex.from_arrays(
        [rows["anon_polygon_id"].astype(str), rows["year"]]
    )
    result: dict[str, np.ndarray] = {}
    primary = context[context["primary_ndvi"].notna()]
    grouped = primary.groupby(["anon_polygon_id", "year"])["primary_ndvi"]
    summary = grouped.agg(["mean", "std", "min", "max", "count"])
    summary["q25"] = grouped.quantile(0.25)
    summary["q75"] = grouped.quantile(0.75)
    for statistic in ("mean", "std", "min", "max", "q25", "q75", "count"):
        result[f"primary_season_{statistic}"] = summary[statistic].reindex(
            target_index
        ).to_numpy()

    for name, column, _ in SENSORS:
        observed = context[context[column].notna()]
        sensor_summary = observed.groupby(["anon_polygon_id", "year"])[column].agg(
            ["mean", "std", "count"]
        )
        for statistic in ("mean", "std", "count"):
            result[f"{name}_season_{statistic}"] = sensor_summary[
                statistic
            ].reindex(target_index).to_numpy()
    return pd.DataFrame(result, index=rows.index)


def sensor_series_features(
    context: pd.DataFrame,
    rows: pd.DataFrame,
    probabilities: pd.DataFrame,
    feature_version: int = 2,
) -> pd.DataFrame:
    """Строит соседей и интерполяции независимо для каждого сенсора."""

    result: dict[str, np.ndarray] = {}
    interpolations: list[np.ndarray] = []
    for short_name, column, _ in SENSORS:
        previous, following, previous_days, next_days = _sensor_neighbors(
            context, rows, column
        )
        total = previous_days + next_days
        usable = np.isfinite(total) & (total > 0)
        weight = np.divide(
            previous_days, total, out=np.zeros_like(total), where=usable
        )
        linear = previous + (following - previous) * weight
        fallback = np.where(np.isfinite(previous), previous, following)
        interpolation = np.where(usable & np.isfinite(linear), linear, fallback)
        result[f"{short_name}_value_prev"] = previous
        result[f"{short_name}_value_next"] = following
        result[f"{short_name}_prev_days"] = previous_days
        result[f"{short_name}_next_days"] = next_days
        result[f"{short_name}_interpolation"] = interpolation
        interpolations.append(interpolation)

        if feature_version >= 3:
            observed = context[context[column].notna()]
            by_date = observed.groupby("date")[column].agg(["mean", "median", "std"])
            for statistic in ("mean", "median", "std"):
                result[f"{short_name}_date_{statistic}"] = by_date[
                    statistic
                ].reindex(rows["date"]).to_numpy()

            crop_date = observed.groupby(["crop_type", "date"])[column].median()
            target_crop_date = pd.MultiIndex.from_arrays(
                [rows["crop_type"].to_numpy(), rows["date"].to_numpy()]
            )
            result[f"{short_name}_crop_date_median"] = crop_date.reindex(
                target_crop_date
            ).to_numpy()

            history = {
                str(polygon): (
                    group["date"].dt.year.to_numpy(),
                    group["date"].dt.dayofyear.to_numpy(),
                    group[column].to_numpy(float),
                )
                for polygon, group in observed.groupby("anon_polygon_id", sort=False)
            }
            historical = np.full(len(rows), np.nan)
            historical_std = np.full(len(rows), np.nan)
            historical_count = np.zeros(len(rows), dtype=float)
            for position, row in enumerate(rows.itertuples(index=False)):
                values = history.get(str(row.anon_polygon_id))
                if values is None:
                    continue
                years, days, ndvi = values
                selected = (years != row.date.year) & (
                    np.abs(days - row.date.dayofyear) <= 10
                )
                if selected.any():
                    historical[position] = float(np.median(ndvi[selected]))
                    historical_std[position] = float(np.std(ndvi[selected]))
                    historical_count[position] = float(selected.sum())
            result[f"{short_name}_historical"] = historical
            result[f"{short_name}_historical_std"] = historical_std
            result[f"{short_name}_historical_count"] = historical_count

    frame = pd.DataFrame(result, index=rows.index)
    values = np.stack(interpolations, axis=1)
    weights = probabilities.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    available = np.isfinite(values)
    effective_weights = np.where(available, weights, 0.0)
    total_weight = effective_weights.sum(axis=1)
    frame["source_interpolation"] = np.where(
        total_weight > 0,
        (np.where(available, values, 0.0) * effective_weights).sum(axis=1)
        / np.where(total_weight > 0, total_weight, 1.0),
        np.nan,
    )
    best = weights.argmax(axis=1)
    frame["best_source_interpolation"] = values[np.arange(len(values)), best]
    if feature_version >= 4:
        frame = pd.concat(
            [frame, _harmonized_primary_features(context, rows, probabilities)],
            axis=1,
        )
    if feature_version >= 5:
        frame = pd.concat(
            [frame, _auxiliary_series_features(context, rows, probabilities)],
            axis=1,
        )
    if feature_version >= 6:
        frame = pd.concat([frame, _weather_features(context, rows)], axis=1)
    if feature_version >= 8:
        frame = pd.concat([frame, _season_summary_features(context, rows)], axis=1)
    return frame
