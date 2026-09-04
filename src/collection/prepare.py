"""Нормализация внешних наблюдений в схему конкурсного train."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.collection.config import CollectionConfig, load_collection_config


EXTERNAL_COLUMNS = [
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
    "ndvi_climatology_mean",
    "ndvi_climatology_std",
    "ndvi_zscore",
    "n_reference_years",
    "status",
    "crop_type",
]


def run_prepare_external_command(args) -> Path:
    config = load_collection_config(args.config)
    sentinel_path = (
        Path(args.sentinel2)
        if args.sentinel2
        else config.output.raw_directory / "sentinel2.csv"
    )
    output_path = Path(args.output) if args.output else config.output.dataset_path
    frame = build_external_dataset(config, sentinel_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    observed = int(frame["primary_ndvi"].notna().sum())
    print(
        f"Внешний датасет сохранён: {output_path} | строк: {len(frame)} | "
        f"полигонов: {frame['anon_polygon_id'].nunique()} | "
        f"наблюдений primary_ndvi: {observed}"
    )
    return output_path


def build_external_dataset(
    config: CollectionConfig, sentinel_path: Path | str
) -> pd.DataFrame:
    sentinel_path = Path(sentinel_path)
    if not sentinel_path.exists():
        raise FileNotFoundError(f"Raw Sentinel-2 не найден: {sentinel_path}")
    raw = pd.read_csv(sentinel_path, parse_dates=["date"])
    required = {"polygon_id", "date", "ndvi", "evi", "ndwi"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"В raw Sentinel-2 отсутствуют колонки: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Raw Sentinel-2 не содержит наблюдений")

    daily = (
        raw.groupby(["polygon_id", "date"], as_index=False)
        .agg(
            s2_ndvi=("ndvi", "median"),
            s2_evi=("evi", "median"),
            s2_ndwi=("ndwi", "median"),
        )
        .rename(columns={"polygon_id": "anon_polygon_id"})
    )
    polygon_ids = sorted(daily["anon_polygon_id"].astype(str).unique())
    dates = pd.date_range(config.period.start, config.period.end, freq="D")
    grid = pd.MultiIndex.from_product(
        [polygon_ids, dates], names=["anon_polygon_id", "date"]
    ).to_frame(index=False)
    frame = grid.merge(daily, on=["anon_polygon_id", "date"], how="left")

    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["doy"] = frame["date"].dt.dayofyear.astype("int16")
    frame["primary_ndvi"] = frame["s2_ndvi"]
    frame["n_reference_years"] = 0
    frame["crop_type"] = "неизвестно"
    for column in EXTERNAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[EXTERNAL_COLUMNS].sort_values(
        ["anon_polygon_id", "date"]
    ).reset_index(drop=True)
