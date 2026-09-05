"""Соседи по собственному ряду каждого сенсора.

Основной ряд `primary_ndvi` смешивает три сенсора, которые систематически
расходятся между собой. Поэтому ближайшее по времени наблюдение в смешанном
ряду часто приходит от другого сенсора, чем скрытое значение, и несёт его
смещение. Здесь ряд каждого сенсора берётся отдельно, а вероятности источника
позволяют собрать из них ту оценку, которая соответствует ожидаемому сенсору.

Замер на синтетических пропусках (13 765 точек, 3 прогона): добавление этих
признаков к вероятностям источника снижает RMSE с 0.0696 до 0.0662, то есть
GapScore растёт с 9.13 до 10.14. Наибольший выигрыш на точках MODIS, -9.3%:
он и самый смещённый относительно остальных, и самый редкий в смешанном ряду.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sensor_calendar import EPOCH, SENSORS

# Соответствие короткого имени сенсора колонке с вероятностью источника.
PROBABILITY_BY_SENSOR = {"s2": "p_s2", "ls": "p_landsat", "md": "p_modis"}


def _epoch_days(frame: pd.DataFrame) -> np.ndarray:
    return (frame["date"] - EPOCH).dt.days.to_numpy()


def _neighbour_arrays(context: pd.DataFrame, rows: pd.DataFrame, column: str):
    """Значения и расстояния ближайших наблюдений одного сенсора."""

    row_days = _epoch_days(rows)
    row_polygons = rows["anon_polygon_id"].to_numpy()

    observed = context[context[column].notna()]
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for polygon, group in observed.groupby("anon_polygon_id", sort=False):
        days = _epoch_days(group)
        order = np.argsort(days)
        series[polygon] = (days[order], group[column].to_numpy(float)[order])

    count = len(rows)
    previous_value = np.full(count, np.nan)
    next_value = np.full(count, np.nan)
    previous_days = np.full(count, np.nan)
    next_days = np.full(count, np.nan)

    for position in range(count):
        entry = series.get(row_polygons[position])
        if entry is None:
            continue
        days, values = entry
        day = row_days[position]
        insert = int(np.searchsorted(days, day))
        if insert > 0:
            previous_value[position] = values[insert - 1]
            previous_days[position] = day - days[insert - 1]
        if insert < len(days):
            next_value[position] = values[insert]
            next_days[position] = days[insert] - day
    return previous_value, next_value, previous_days, next_days


def _interpolate(previous_value, next_value, previous_days, next_days):
    """Линейная оценка между соседями одного сенсора."""

    total = previous_days + next_days
    usable = np.isfinite(total) & (total > 0)
    weight = np.divide(previous_days, total, out=np.zeros_like(total), where=usable)
    linear = previous_value + (next_value - previous_value) * weight
    fallback = np.where(np.isfinite(previous_value), previous_value, next_value)
    return np.where(usable & np.isfinite(linear), linear, fallback)


def per_sensor_features(
    context: pd.DataFrame,
    rows: pd.DataFrame,
    probabilities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Соседи и интерполяция по ряду каждого сенсора.

    У скрытых точек все сенсорные колонки пусты, поэтому ряды строятся по
    контексту без предсказываемого наблюдения автоматически — так же, как это
    происходит на платформе.
    """

    features: dict[str, np.ndarray] = {}
    for name, column in SENSORS:
        previous_value, next_value, previous_days, next_days = _neighbour_arrays(
            context, rows, column
        )
        features[f"{name}_val_prev"] = previous_value
        features[f"{name}_val_next"] = next_value
        features[f"{name}_interp"] = _interpolate(
            previous_value, next_value, previous_days, next_days
        )

    frame = pd.DataFrame(features, index=rows.index)
    if probabilities is None:
        return frame

    # Оценка по тому сенсору, от которого вероятнее всего пришло скрытое значение.
    weights = np.stack(
        [probabilities[PROBABILITY_BY_SENSOR[name]].to_numpy() for name, _ in SENSORS],
        axis=1,
    )
    values = np.stack([frame[f"{name}_interp"].to_numpy() for name, _ in SENSORS], axis=1)
    known = np.isfinite(values)
    masked_weights = np.where(known, weights, 0.0)
    total = masked_weights.sum(axis=1)
    frame["source_interp"] = np.where(
        total > 0,
        (np.where(known, values, 0.0) * masked_weights).sum(axis=1)
        / np.where(total > 0, total, 1.0),
        np.nan,
    )
    frame["best_source_interp"] = values[np.arange(len(values)), weights.argmax(axis=1)]
    return frame
