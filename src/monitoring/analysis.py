"""Консервативный анализ реальных наблюдений, независимый от конкурсного ML."""

import numpy as np
import pandas as pd


def finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def analyze_observations(params, satellite, weather, warnings=None):
    grid = pd.date_range(params["start"], params["end"], freq="D")
    raw = pd.DataFrame(satellite)
    if raw.empty:
        raise ValueError(
            "Нет пригодных Sentinel-2 наблюдений. Выберите другой период или поле"
        )
    raw["date"] = pd.to_datetime(raw["date"])
    if "scene_id" in raw:
        raw = raw.drop_duplicates(["date", "scene_id"])
    daily = raw.groupby("date")["ndvi"].median().sort_index()
    daily = daily.where(daily.between(-1, 1)).dropna()
    frame = pd.DataFrame(index=grid)
    frame["observed"] = daily.reindex(grid)
    known = frame["observed"].dropna()
    if known.empty:
        raise ValueError(
            "В выбранном периоде нет пригодных наблюдений. Попробуйте другой период"
        )
    frame["filled"] = frame["observed"].interpolate(method="time", limit_area="inside")
    # Не восстанавливаем края и длинные облачные интервалы с ложной уверенностью.
    before = pd.Series(pd.NaT, index=grid, dtype="datetime64[ns]")
    before.loc[known.index] = known.index
    span = (before.bfill() - before.ffill()).dt.days
    frame.loc[
        frame["observed"].isna()
        & (span.isna() | (span > params["max_interpolation_gap_days"])),
        "filled",
    ] = np.nan
    history = daily[daily.index < grid[0]]
    frame["climatology"] = np.nan
    frame["climatology_std"] = np.nan
    frame["reference_years"] = 0
    for current in grid:
        distance = abs(history.index.dayofyear - current.dayofyear)
        candidates = history[np.minimum(distance, 366 - distance) <= 21]
        count = candidates.index.year.nunique()
        if count >= 2 and len(candidates) >= 3:
            frame.at[current, "climatology"] = candidates.median()
            frame.at[current, "climatology_std"] = max(candidates.std(ddof=0), 0.03)
            frame.at[current, "reference_years"] = count
    frame["zscore"] = (frame["filled"] - frame["climatology"]) / frame[
        "climatology_std"
    ]
    weather_frame = pd.DataFrame(weather)
    weather_index = pd.date_range(grid[0] - pd.Timedelta(days=13), grid[-1])
    if weather_frame.empty:
        weather_frame = pd.DataFrame(
            index=weather_index, columns=["temperature", "precipitation"], dtype=float
        )
    else:
        weather_frame["date"] = pd.to_datetime(weather_frame["date"])
        weather_frame = (
            weather_frame.groupby("date")[["temperature", "precipitation"]]
            .mean()
            .reindex(weather_index)
        )
    rain = weather_frame["precipitation"].rolling(14, min_periods=14).sum()
    heat = weather_frame["temperature"].rolling(7, min_periods=7).mean()
    points = []
    for current, row in frame.iterrows():
        z = finite(row["zscore"])
        status = (
            "Недостаточно данных"
            if z is None
            else (
                "Критическая аномалия"
                if z < -2
                else "Угнетение биомассы" if z < -1 else "Штатное развитие"
            )
        )
        explanation = (
            "Недостаточно наблюдений или исторической нормы"
            if z is None
            else "NDVI находится в пределах сезонной нормы"
        )
        if z is not None and z < -1:
            reasons = [f"NDVI ниже исторической нормы: Z={z:.2f}"]
            if pd.notna(rain.loc[current]) and rain.loc[current] < 5:
                reasons.append("возможен дефицит влаги: менее 5 мм осадков за 14 дней")
            if pd.notna(heat.loc[current]) and heat.loc[current] > 28:
                reasons.append(
                    "возможен тепловой стресс: средняя температура за 7 дней выше 28 °C"
                )
            if pd.isna(row["observed"]):
                reasons.append(
                    "NDVI восстановлен интерполяцией; требуется подтверждение наблюдением"
                )
            explanation = "; ".join(reasons)
        points.append(
            {
                "date": str(current.date()),
                **{
                    key: finite(row[key])
                    for key in (
                        "observed",
                        "filled",
                        "climatology",
                        "climatology_std",
                        "zscore",
                    )
                },
                "status": status,
                "explanation": explanation,
                "temperature": finite(weather_frame.at[current, "temperature"]),
                "precipitation": finite(weather_frame.at[current, "precipitation"]),
                "reference_years": int(row["reference_years"]),
            }
        )
    notes = list(warnings or [])
    if frame["climatology"].isna().any():
        notes.append(
            "Для части дат недостаточно истории минимум за два года; аномалии там не оцениваются"
        )
    if frame["filled"].isna().any():
        notes.append(
            f'Края ряда и интервалы между наблюдениями более {params["max_interpolation_gap_days"]} дней оставлены без восстановления'
        )
    if weather_frame.loc[grid].isna().any().any():
        notes.append(
            "Погода доступна не на все даты; отсутствие осадков в данных не считается засухой"
        )
    return {
        "polygon_id": params["polygon_id"],
        "year": grid[0].year,
        "start": params["start"],
        "end": params["end"],
        "crop_type": "неизвестна",
        "summary": {
            "observations": int(frame["observed"].notna().sum()),
            "restored": int((frame["observed"].isna() & frame["filled"].notna()).sum()),
            "missing": int(frame["filled"].isna().sum()),
            "anomalies": int((frame["zscore"] < -1).sum()),
            "mean_ndvi": finite(frame["filled"].mean()),
        },
        "points": points,
        "warnings": list(dict.fromkeys(notes)),
        "method": "Линейная интерполяция по времени, без экстраполяции. Конкурсная ML-модель не применяется.",
        "sources": ["COPERNICUS/S2_SR_HARMONIZED", "ECMWF/ERA5_LAND/DAILY_AGGR"],
        "weather_note": "ERA5-Land: ячейка центра поля, сетка около 11 км; региональный контекст, не измерение на поле.",
        "attribution": "Contains modified Copernicus Sentinel data; ERA5-Land: Copernicus Climate Change Service (C3S), ECMWF.",
    }
