"""Определение сенсора-источника `primary_ndvi` по орбитальному календарю.

`primary_ndvi` собран из трёх сенсоров по приоритету `s2 -> landsat -> modis`,
и сенсоры систематически расходятся: на совпадающих датах медианная разница
s2-landsat равна -0.036, s2-modis -0.075, landsat-modis -0.047. Поэтому
интерполяция смешанного ряда переоценивает точки Sentinel-2 и недооценивает
точки MODIS. Замер на синтетических пропусках: смещение предсказания по
источникам расходится на 0.041 NDVI, и знание источника снимает это смещение.

Источник восстановим из одной только даты, потому что съёмка идёт по орбитам:

* MODIS в данных встречается строго при `doy % 16 == 1` (даты 97, 113, ..., 273);
* даты Landsat у каждого полигона сидят на двух остатках по модулю 8
  (циклы Landsat 8 и 9), доля top-2 остатков равна 0.96;
* даты Sentinel-2 — на двух остатках по модулю 5, доля top-2 равна 1.00.

Классификатор по этим признакам определяет источник скрытой точки с точностью
0.959 на полигонах, не участвовавших в обучении.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2010-01-01")
SENSORS = (("s2", "s2_ndvi"), ("ls", "landsat_ndvi"), ("md", "modis_ndvi"))
MODULI = (5, 8, 16)
SOURCE_NAMES = ("s2", "landsat", "modis")
PROBABILITY_COLUMNS = ("p_s2", "p_landsat", "p_modis")

DEFAULT_CLASSIFIER_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbosity": -1,
    "random_state": 42,
}


def _epoch_days(frame: pd.DataFrame) -> np.ndarray:
    return (frame["date"] - EPOCH).dt.days.to_numpy()


def source_labels(frame: pd.DataFrame) -> np.ndarray:
    """Источник `primary_ndvi`: 0 — Sentinel-2, 1 — Landsat, 2 — MODIS, -1 — нет."""

    return np.where(
        frame["s2_ndvi"].notna(),
        0,
        np.where(frame["landsat_ndvi"].notna(), 1, np.where(frame["modis_ndvi"].notna(), 2, -1)),
    )


def calendar_features(context: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Признаки орбитального календаря, считаемые только по контексту.

    В контексте у скрытых точек сенсорные колонки пусты, поэтому ни доли, ни
    расстояния не используют предсказываемое наблюдение.
    """

    context_days = _epoch_days(context)
    row_days = _epoch_days(rows)
    context_polygons = context["anon_polygon_id"].to_numpy()
    row_polygons = rows["anon_polygon_id"].to_numpy()
    features: dict[str, np.ndarray] = {}

    for name, column in SENSORS:
        present = context[column].notna().to_numpy()

        # Доля дат с наблюдением этого сенсора для каждого остатка орбиты.
        for modulus in MODULI:
            rates = pd.Series(present).groupby(
                [pd.Series(context_polygons), pd.Series(context_days % modulus)]
            ).mean()
            index = pd.MultiIndex.from_arrays([row_polygons, row_days % modulus])
            features[f"{name}_rate_m{modulus}"] = rates.reindex(index).to_numpy()

        # Расстояние до ближайшего наблюдения этого сенсора слева и справа.
        previous = np.full(len(rows), np.nan)
        following = np.full(len(rows), np.nan)
        observed = pd.DataFrame(
            {"polygon": context_polygons[present], "day": context_days[present]}
        )
        by_polygon = {
            polygon: np.sort(group["day"].to_numpy())
            for polygon, group in observed.groupby("polygon")
        }
        for position, (polygon, day) in enumerate(zip(row_polygons, row_days)):
            days = by_polygon.get(polygon)
            if days is None or not len(days):
                continue
            insert = int(np.searchsorted(days, day))
            if insert > 0:
                previous[position] = day - days[insert - 1]
            if insert < len(days):
                following[position] = days[insert] - day
        features[f"{name}_prev_days"] = previous
        features[f"{name}_next_days"] = following

    doy = rows["date"].dt.dayofyear.to_numpy()
    features["is_modis_doy"] = (doy % 16 == 1).astype(float)
    for modulus in MODULI:
        features[f"orbit_m{modulus}"] = (row_days % modulus).astype(float)
    return pd.DataFrame(features, index=rows.index)


def build_source_training(
    pool: pd.DataFrame, repeats: int, rate: float, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Обучающие пары для классификатора источника.

    Точки прячутся теми же масками, что и на инференсе. Это принципиально:
    если учиться на неспрятанных наблюдениях, расстояние до собственного
    сенсора равно нулю, классификатор выучивает эту утечку и на реальных
    пропусках падает с 0.959 до 0.778.
    """

    from src.synthetic import MaskSpec, apply_mask, generate_mask

    features: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    for step in range(repeats):
        mask = generate_mask(pool, MaskSpec(rate=rate, seed=seed + step))
        flags = mask.to_numpy()
        context = apply_mask(pool, mask)
        truth = source_labels(pool.loc[flags])
        known = truth >= 0
        features.append(calendar_features(context, context.loc[flags]).loc[known])
        labels.append(truth[known])
    return pd.concat(features, ignore_index=True), np.concatenate(labels)


def fit_source_classifier(
    pool: pd.DataFrame,
    repeats: int = 6,
    rate: float = 0.15,
    seed: int = 5000,
    params: dict | None = None,
):
    """Обучает классификатор источника на замаскированных точках."""

    from lightgbm import LGBMClassifier

    features, labels = build_source_training(pool, repeats, rate, seed)
    model = LGBMClassifier(**{**DEFAULT_CLASSIFIER_PARAMS, **(params or {})})
    model.fit(features, labels)
    return model


def source_probabilities(model, context: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Вероятности источника для строк со скрытыми сенсорными колонками."""

    proba = np.asarray(model.predict_proba(calendar_features(context, rows)), dtype=float)
    frame = pd.DataFrame(0.0, index=rows.index, columns=list(PROBABILITY_COLUMNS))
    for position, label in enumerate(model.classes_):
        frame.iloc[:, int(label)] = proba[:, position]
    return frame
