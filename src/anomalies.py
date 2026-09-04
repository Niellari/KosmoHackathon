"""Детекция и объяснение негативных аномалий вегетации."""

from __future__ import annotations

import numpy as np
import pandas as pd


STATUS_NORMAL = "Штатное развитие"
STATUS_STRESS = "Угнетение биомассы"
STATUS_CRITICAL = "Критическая аномалия"


def add_climatology(frame: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает устойчивую сезонную норму по другим годам того же поля."""

    result = frame.copy()
    result["climatology_mean_calc"] = np.nan
    result["climatology_std_calc"] = np.nan

    observed = result[result["primary_ndvi_filled"].notna()]
    for polygon_id, group in result.groupby("anon_polygon_id", sort=False):
        history = observed[observed["anon_polygon_id"] == polygon_id]
        for index, row in group.iterrows():
            candidates = history[
                (history["year"] != row["year"])
                & ((history["doy"] - row["doy"]).abs() <= 7)
            ]["primary_ndvi_filled"]
            if len(candidates) < 3:
                continue
            result.at[index, "climatology_mean_calc"] = float(candidates.median())
            std = float(candidates.std(ddof=0))
            result.at[index, "climatology_std_calc"] = max(std, 0.03)

    # Для короткой истории используем предоставленную организаторами норму.
    if "ndvi_climatology_mean" in result:
        result["climatology_mean_calc"] = result["climatology_mean_calc"].fillna(
            result["ndvi_climatology_mean"]
        )
    if "ndvi_climatology_std" in result:
        result["climatology_std_calc"] = result["climatology_std_calc"].fillna(
            result["ndvi_climatology_std"]
        )
    return result


def detect_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавляет Z-score, статус и детерминированное объяснение."""

    result = add_climatology(frame)
    std = result["climatology_std_calc"].where(
        result["climatology_std_calc"] > 0.01
    )
    result["ndvi_zscore_calc"] = (
        result["primary_ndvi_filled"] - result["climatology_mean_calc"]
    ) / std

    result["status_calc"] = STATUS_NORMAL
    result.loc[result["ndvi_zscore_calc"] < -1, "status_calc"] = STATUS_STRESS
    result.loc[result["ndvi_zscore_calc"] < -2, "status_calc"] = STATUS_CRITICAL
    result.loc[result["ndvi_zscore_calc"].isna(), "status_calc"] = "Недостаточно данных"

    precipitation = pd.to_numeric(result.get("era5_precip_mm"), errors="coerce")
    temperature = pd.to_numeric(result.get("era5_temp_c"), errors="coerce")
    result["precip_14d"] = precipitation.fillna(0).rolling(14, min_periods=3).sum()
    result["temp_7d"] = temperature.rolling(7, min_periods=3).mean()

    explanations = []
    for _, row in result.iterrows():
        status = row["status_calc"]
        if status == "Недостаточно данных":
            explanations.append("Недостаточно исторических наблюдений для оценки нормы")
            continue
        if status == STATUS_NORMAL:
            explanations.append("NDVI находится в пределах сезонной нормы")
            continue

        reasons = [f"NDVI ниже нормы: Z={row['ndvi_zscore_calc']:.2f}"]
        if pd.notna(row.get("precip_14d")) and row["precip_14d"] < 5:
            reasons.append("мало осадков за последние 14 дней")
        if pd.notna(row.get("temp_7d")) and row["temp_7d"] > 28:
            reasons.append("высокая средняя температура за 7 дней")
        explanations.append("; ".join(reasons))
    result["anomaly_explanation"] = explanations
    return result
