"""Генерация синтетических пропусков, воспроизводящих контрольные точки платформы.

Процесс маскирования организаторов восстановлен по `data/test_dataset.csv`:
каждое наблюдение `primary_ndvi` скрывается независимо с вероятностью 0.15.
Проверки, подтверждающие это (см. `benchmark --profile`):

* доля скрытых наблюдений — 3112 / 20753 = 0.1500;
* длины серий подряд идущих пропусков совпадают с геометрическим
  распределением `p^(k-1)(1-p)`: 2252/340/41/8 против ожидаемых
  2249/337/51/8, chi2 = 1.86 при ~3 степенях свободы;
* доля масок стабильна по годам, по фазе сезона и по полигонам
  (z-оценки долей: mean 0.08, std 1.07, |z|>2 у 5 полигонов из 78);
* доля пропусков на MODIS-датах (`doy % 16 == 1`) равна 0.234 против
  базовой доли таких наблюдений 0.224, то есть стратификации по сенсорам нет.

Поэтому генератор намеренно прост: независимый Бернулли по наблюдаемым точкам.
Дополнительная стратификация по сенсорному календарю или по группам полигонов
воспроизводила бы не тот процесс, который используется на платформе.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Колонки, которые организаторы скрывают в контрольных строках. Проверено:
# в gap-строках test заполнены только anon_polygon_id, date, is_synthetic_gap
# и crop_type. `year` и `doy` перечислены для полноты, но load_dataset всегда
# пересчитывает их из `date`, поэтому на пайплайн их маскирование не влияет.
MASKED_COLUMNS = (
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
    "primary_ndvi",
    "ndvi_climatology_mean",
    "ndvi_climatology_std",
    "ndvi_zscore",
    "n_reference_years",
    "status",
    "year",
    "doy",
)

# Доля скрытых наблюдений на платформе.
PLATFORM_MASK_RATE = 0.15


@dataclass(frozen=True)
class MaskSpec:
    """Параметры генерации одного набора синтетических пропусков."""

    rate: float = PLATFORM_MASK_RATE
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.rate < 1.0:
            raise ValueError(f"rate должен быть в (0, 1), получено {self.rate}")


def generate_mask(frame: pd.DataFrame, spec: MaskSpec) -> pd.Series:
    """Независимо скрывает долю `spec.rate` наблюдаемых точек.

    Маскируются только строки с известным `primary_ndvi`: контрольная точка по
    построению является настоящим спутниковым наблюдением, поэтому она сохраняет
    нерегулярное положение в ряду и остаток по орбитальному циклу сенсора.
    """

    observed = frame["primary_ndvi"].notna().to_numpy()
    rng = np.random.default_rng(spec.seed)
    draw = rng.random(len(frame)) < spec.rate
    return pd.Series(observed & draw, index=frame.index, name="is_synthetic_gap")


def apply_mask(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Возвращает копию с маскированием ровно тех же колонок, что и платформа.

    `year` и `doy` скрываются вместе с остальными, но тут же восстанавливаются
    из `date` — ровно так же поступает `load_dataset` с реальным test-файлом,
    поэтому оставлять их пустыми означало бы моделировать более тяжёлые условия,
    чем на платформе.
    """

    if len(mask) != len(frame):
        raise ValueError("Размер маски не совпадает с датасетом")
    masked = frame.copy()
    flags = mask.to_numpy()
    for column in MASKED_COLUMNS:
        if column in masked.columns:
            masked.loc[flags, column] = np.nan
    masked["year"] = masked["date"].dt.year.astype("int16")
    masked["doy"] = masked["date"].dt.dayofyear.astype("int16")
    masked["is_synthetic_gap"] = flags
    return masked


@dataclass(frozen=True)
class MaskedSplit:
    """Один воспроизводимый прогон: контекст, цели и истинные значения.

    * `context` — данные в том виде, в каком их видит модель на инференсе:
      контрольные точки скрыты вместе со всеми вычисляемыми признаками.
    * `train_source` — подвыборка контекста для обучения. Полигоны из holdout
      исключены целиком, поэтому модель не видит их целевых значений.
    * `targets` — строки, которые нужно предсказать (`is_synthetic_gap = True`).
    * `truth` — истинные `primary_ndvi` для `targets`, выровненные по индексу.
    * `holdout_polygons` — полигоны, не участвовавшие в обучении.
    """

    context: pd.DataFrame
    train_source: pd.DataFrame
    targets: pd.DataFrame
    truth: pd.Series
    holdout_polygons: frozenset[str]

    @property
    def unseen_targets(self) -> pd.Series:
        """Булева маска целей, лежащих в holdout-полигонах."""

        return self.targets["anon_polygon_id"].isin(self.holdout_polygons)


def holdout_polygons(
    frame: pd.DataFrame, fraction: float, seed: int
) -> frozenset[str]:
    """Отбирает полигоны, полностью исключаемые из обучения.

    В test 39 из 78 полигонов отсутствуют в train, поэтому проверка на невидимых
    полигонах — не дополнительная строгость, а воспроизведение половины
    реального тестового набора.
    """

    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction должен быть в [0, 1), получено {fraction}")
    polygons = np.sort(frame["anon_polygon_id"].unique())
    count = int(round(len(polygons) * fraction))
    if count == 0:
        return frozenset()
    rng = np.random.default_rng(seed)
    return frozenset(rng.choice(polygons, size=count, replace=False).tolist())


def make_split(
    frame: pd.DataFrame,
    spec: MaskSpec,
    holdout_fraction: float = 0.5,
) -> MaskedSplit:
    """Собирает один прогон синтетической валидации.

    Маскирование выполняется до вычисления любых признаков, поэтому
    климатология, соседи и агрегаты по полигону строятся уже без скрытых точек.
    """

    mask = generate_mask(frame, spec)
    if not mask.any():
        raise ValueError("Маска пуста: в датасете нет наблюдаемых primary_ndvi")

    truth = frame.loc[mask.to_numpy(), "primary_ndvi"].astype(float)
    context = apply_mask(frame, mask)
    unseen = holdout_polygons(frame, holdout_fraction, spec.seed)
    train_source = context[~context["anon_polygon_id"].isin(unseen)].copy()
    targets = context.loc[mask.to_numpy()].copy()
    return MaskedSplit(
        context=context,
        train_source=train_source,
        targets=targets,
        truth=truth,
        holdout_polygons=unseen,
    )


def iter_splits(
    frame: pd.DataFrame,
    repeats: int,
    spec: MaskSpec,
    holdout_fraction: float = 0.5,
):
    """Несколько прогонов с разными масками — динамическое маскирование.

    Один фиксированный набор масок скрывает лишь ~15% точек, поэтому оценка на
    нём шумная, а модель, обученная на нём, подстраивается под конкретный набор.
    Каждый повтор сдвигает seed и перегенерирует и маску, и holdout.
    """

    for repeat in range(repeats):
        yield make_split(
            frame,
            MaskSpec(rate=spec.rate, seed=spec.seed + repeat),
            holdout_fraction,
        )


def gap_profile(frame: pd.DataFrame, gap_column: str = "is_synthetic_gap") -> dict:
    """Считает характеристики набора пропусков для сравнения с реальными.

    Все величины считаются в календаре наблюдений полигона: подряд идущими
    считаются пропуски, между которыми нет уцелевших наблюдений.
    """

    if gap_column not in frame.columns:
        raise ValueError(f"В датасете нет колонки {gap_column}")

    gap = frame[gap_column].fillna(False).astype(bool).to_numpy()
    observed = frame["primary_ndvi"].notna().to_numpy()
    total = int(gap.sum() + observed.sum())
    if total == 0:
        raise ValueError("В датасете нет ни наблюдений, ни пропусков")

    runs: list[int] = []
    distances: list[tuple[float, float]] = []
    working = frame.assign(_gap=gap, _obs=observed)
    for _, group in working.groupby("anon_polygon_id", sort=False):
        group = group[group["_gap"] | group["_obs"]].sort_values("date")
        flags = group["_gap"].to_numpy()
        position = 0
        while position < len(flags):
            if flags[position]:
                end = position
                while end < len(flags) and flags[end]:
                    end += 1
                runs.append(end - position)
                position = end
            else:
                position += 1

        days = group["date"].to_numpy("datetime64[D]").astype(int)
        observed_days = days[~flags]
        if not len(observed_days):
            distances += [(np.nan, np.nan)] * int(flags.sum())
            continue
        insert = np.searchsorted(observed_days, days[flags])
        for day, index in zip(days[flags], insert):
            previous = day - observed_days[index - 1] if index > 0 else np.nan
            following = (
                observed_days[index] - day if index < len(observed_days) else np.nan
            )
            distances.append((previous, following))

    run_series = pd.Series(runs, dtype=float)
    distance_frame = pd.DataFrame(distances, columns=["previous", "next"])
    doy = frame.loc[gap, "date"].dt.dayofyear

    return {
        "n_gaps": int(gap.sum()),
        "mask_rate": round(float(gap.sum() / total), 4),
        "run_share_1": round(float((run_series == 1).mean()), 3),
        "run_share_2": round(float((run_series == 2).mean()), 3),
        "run_share_3plus": round(float((run_series >= 3).mean()), 3),
        "prev_days_median": float(distance_frame["previous"].median()),
        "next_days_median": float(distance_frame["next"].median()),
        "prev_days_p90": float(distance_frame["previous"].quantile(0.9)),
        "next_days_p90": float(distance_frame["next"].quantile(0.9)),
        "edge_no_prev": int(distance_frame["previous"].isna().sum()),
        "edge_no_next": int(distance_frame["next"].isna().sum()),
        "modis_doy_share": round(float((doy % 16 == 1).mean()), 3),
    }


def compare_profiles(reference: dict, generated: dict) -> pd.DataFrame:
    """Сводит два профиля в таблицу для проверки реалистичности генератора."""

    keys = [key for key in reference if key in generated]
    return pd.DataFrame(
        {
            "реальные": [reference[key] for key in keys],
            "синтетика": [generated[key] for key in keys],
        },
        index=keys,
    )
