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
    sentinel_paths = (
        [Path(path) for path in args.sentinel2]
        if args.sentinel2
        else [config.output.raw_directory / "sentinel2.csv"]
    )
    output_path = Path(args.output) if args.output else config.output.dataset_path
    frame = build_external_dataset(config, sentinel_paths)
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
    config: CollectionConfig, sentinel_paths: Path | str | list[Path | str]
) -> pd.DataFrame:
    if isinstance(sentinel_paths, (str, Path)):
        paths = [Path(sentinel_paths)]
    else:
        paths = [Path(path) for path in sentinel_paths]
    if not paths:
        raise ValueError("Не задан ни один raw Sentinel-2 CSV")
    raw_frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Raw Sentinel-2 не найден: {path}")
        raw_frames.append(pd.read_csv(path, parse_dates=["date"]))
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
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
    years = sorted(int(year) for year in daily["date"].dt.year.unique())
    grids = []
    for year in years:
        dates = pd.date_range(
            config.period.start.replace(year=year),
            config.period.end.replace(year=year),
            freq="D",
        )
        grids.append(
            pd.MultiIndex.from_product(
                [polygon_ids, dates], names=["anon_polygon_id", "date"]
            ).to_frame(index=False)
        )
    grid = pd.concat(grids, ignore_index=True)
    frame = grid.merge(daily, on=["anon_polygon_id", "date"], how="left")

    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["doy"] = frame["date"].dt.dayofyear.astype("int16")
    frame["primary_ndvi"] = frame["s2_ndvi"]
    frame["n_reference_years"] = 0
    frame["crop_type"] = "неизвестно"
    for column in EXTERNAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame["status"] = frame["status"].astype("object")
    frame = _add_climatology(
        frame, window_days=config.processing.climatology_window_days
    )
    return frame[EXTERNAL_COLUMNS].sort_values(
        ["anon_polygon_id", "date"]
    ).reset_index(drop=True)


def _add_climatology(frame: pd.DataFrame, window_days: int) -> pd.DataFrame:
    result = frame.copy()
    known = result[result["primary_ndvi"].notna()]
    for index, row in known.iterrows():
        candidates = known[
            (known["anon_polygon_id"] == row["anon_polygon_id"])
            & (known["year"] != row["year"])
            & ((known["doy"] - row["doy"]).abs() <= window_days)
        ]
        if candidates.empty:
            continue
        values = candidates["primary_ndvi"].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        result.at[index, "ndvi_climatology_mean"] = mean
        result.at[index, "ndvi_climatology_std"] = std
        result.at[index, "n_reference_years"] = int(candidates["year"].nunique())
        if std > 0:
            zscore = (float(row["primary_ndvi"]) - mean) / std
            result.at[index, "ndvi_zscore"] = zscore
            result.at[index, "status"] = (
                "Критическая аномалия"
                if zscore < -2
                else "Угнетение биомассы"
                if zscore < -1
                else "Штатное развитие"
            )
    return result
