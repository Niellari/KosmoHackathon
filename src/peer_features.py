"""Признаки по остальным полигонам на ту же дату.

Все конкурсные поля лежат в одном регионе и делят погоду и фазу сезона.
Платформа скрывает у контрольной точки её собственные спутниковые и погодные
колонки, но не трогает соседние полигоны, поэтому в ту же дату остаётся
доступным срез по остальным полям. Из него восстанавливается то, что скрыто:

* `peer_temp`, `peer_precip` — погода дня. Собственные `era5_temp_c` и
  `era5_precip_mm` у контрольной точки замаскированы, а у соседей нет;
* `peer_ndvi_mean` — региональный уровень вегетации в этот день;
* `peer_ndvi_dev` — отклонение этого уровня от сезонной нормы по всем полям.
  Отрицательное значение означает, что день плохой для всего региона, то есть
  засуха или заморозок, а не особенность конкретного поля;
* `peer_crop_mean` — то же по полям с той же культурой, у них совпадает
  календарь сева и уборки.

Всё считается только по контексту, где контрольные точки уже скрыты, поэтому
значение самой предсказываемой точки в агрегаты не попадает.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PEER_COLUMNS = (
    "peer_ndvi_mean",
    "peer_ndvi_std",
    "peer_ndvi_count",
    "peer_ndvi_dev",
    "peer_crop_mean",
    "peer_temp",
    "peer_precip",
)


def _date_aggregate(context: pd.DataFrame, column: str) -> pd.Series:
    """Среднее по всем полигонам для каждой даты."""

    values = pd.to_numeric(context.get(column), errors="coerce")
    if values is None:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({"date": context["date"], "value": values}).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby("date")["value"].mean()


def peer_features(context: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Строит срез по соседним полигонам для каждой целевой строки."""

    ndvi = pd.to_numeric(context["primary_ndvi"], errors="coerce")
    observed = pd.DataFrame(
        {
            "date": context["date"],
            "doy": context["doy"],
            "crop_type": context["crop_type"].astype(str),
            "value": ndvi,
        }
    ).dropna(subset=["value"])

    by_date = observed.groupby("date")["value"].agg(["mean", "std", "count"])
    # Сезонная норма региона: среднее по всем полям и годам для дня года.
    by_doy = observed.groupby("doy")["value"].mean()
    by_crop = observed.groupby(["date", "crop_type"])["value"].mean()

    dates = rows["date"].to_numpy()
    doys = pd.to_numeric(rows["doy"], errors="coerce").to_numpy()
    crops = rows["crop_type"].astype(str).to_numpy()

    result = pd.DataFrame(index=rows.index)
    result["peer_ndvi_mean"] = by_date["mean"].reindex(dates).to_numpy()
    result["peer_ndvi_std"] = by_date["std"].reindex(dates).to_numpy()
    result["peer_ndvi_count"] = (
        by_date["count"].reindex(dates).fillna(0).to_numpy(dtype=float)
    )
    seasonal = by_doy.reindex(doys).to_numpy()
    result["peer_ndvi_dev"] = result["peer_ndvi_mean"].to_numpy() - seasonal
    result["peer_crop_mean"] = (
        by_crop.reindex(pd.MultiIndex.from_arrays([dates, crops])).to_numpy()
    )

    # Погода дня по соседям: у самой точки она замаскирована платформой.
    for name, column in (("peer_temp", "era5_temp_c"), ("peer_precip", "era5_precip_mm")):
        aggregate = _date_aggregate(context, column)
        result[name] = (
            aggregate.reindex(dates).to_numpy()
            if len(aggregate)
            else np.full(len(rows), np.nan)
        )

    return result[list(PEER_COLUMNS)]
