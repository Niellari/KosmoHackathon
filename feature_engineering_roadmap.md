# План улучшения признаков и модели

Документ фиксирует идеи для развития модели восстановления `primary_ndvi`.
Статус каждого направления отмечается отдельно.

## Исходные наблюдения

- В test находится 3 112 контрольных строк с `is_synthetic_gap = True`.
- 95,9% контрольных точек имеют известное значение `primary_ndvi` и слева, и справа.
- Медианное расстояние до ближайшего наблюдения составляет 3 дня.
- Медианный интервал между левым и правым наблюдением составляет 7 дней.
- 84,7% пропусков одиночные; максимальная последовательность содержит 5 точек.
- 85,1% контрольных точек относятся к полигонам, которых нет в train.
- В контрольной строке скрываются `primary_ndvi`, спутниковые, погодные и вычисляемые признаки. Доступны `anon_polygon_id`, `date`, `crop_type` и `is_synthetic_gap`.

Из этого следует, что основной приоритет — локальная форма временного ряда и переносимость на новые полигоны. Любой новый признак должен быть доступен при инференсе и рассчитываться без использования скрытого значения текущей строки.

## 1. Маскирование, соответствующее test

Текущий leave-one-out подход в основном скрывает отдельные известные точки. Обучение и валидацию необходимо приблизить к структуре test.

План:

- генерировать одиночные пропуски и блоки длиной 2–5 точек;
- воспроизводить наблюдаемое в test распределение длин блоков;
- одновременно исключать все точки одного блока из контекста;
- маскировать те же динамические и вычисляемые признаки, что и организаторы;
- `year` и `doy` восстанавливать только из `date`;
- отдельно считать качество на одиночных и последовательных пропусках.

Это изменение относится к подготовке обучающих примеров, а не только к feature engineering, но имеет наивысший приоритет.

## 2. Предсказание поправки к интерполяции

**Статус: реализовано, но не включено по умолчанию.** Режим переключается через
`training.target_mode`. На контрольной маске из 1000 точек direct показал RMSE
0,072809, residual-linear — 0,073607, residual-neighbor-mean — 0,073970. Поэтому
до дополнительной настройки residual текущим default остаётся direct.

Вместо прямого предсказания `primary_ndvi` проверить обучение на остатке относительно простого baseline.

```text
residual = primary_ndvi_true - baseline_prediction
primary_ndvi_pred = baseline_prediction + predicted_residual
```

Первый кандидат для baseline — линейная интерполяция. Дополнительно следует сравнить среднее соседей и PCHIP.

Планируемый конфиг:

```yaml
training:
  target_mode: residual       # direct | residual
  residual_baseline: linear   # neighbor_mean | linear | pchip | ensemble
```

При `target_mode: direct` модель продолжает предсказывать сам `primary_ndvi`. Для `baseline` и `ensemble` этот блок не применяется.

## 3. Локальная динамика NDVI

Текущая модель получает соседние значения, но не все характеристики формы кривой.

Планируемые признаки:

```text
gap_span_days
gap_position
neighbor_asymmetry
slope_before
slope_after
slope_between
slope_change
local_acceleration
neighbor_mean
neighbor_std
neighbor_min
neighbor_max
neighbor_range
```

Примеры формул:

```text
gap_span_days = previous_days + next_days

gap_position = previous_days / gap_span_days

slope_before =
    (previous_1 - previous_2) /
    days_between_previous_points

slope_after =
    (next_2 - next_1) /
    days_between_next_points

slope_change = slope_after - slope_before
```

Эти признаки должны помочь различать рост, плато, сезонный максимум и снижение NDVI.

## 4. Альтернативные методы интерполяции

Модель может использовать результаты нескольких интерполяторов как независимые признаки:

```text
neighbor_mean_prediction
linear_prediction
pchip_prediction
local_quadratic_prediction
crop_curve_prediction
historical_prediction
```

Приоритет — PCHIP, поскольку он сохраняет монотонность и обычно создаёт меньше выбросов, чем обычный кубический spline.

Каждый метод должен корректно работать при отсутствии одного из соседей и при блоках из нескольких пропущенных точек.

## 5. Состояние текущего сезона относительно нормы

Одной исторической средней недостаточно. Нужно описать, насколько весь текущий сезон отклоняется от типичной динамики поля.

Планируемые признаки:

```text
previous_1_climatology_residual
previous_2_climatology_residual
next_1_climatology_residual
next_2_climatology_residual
season_residual_mean
season_residual_median
season_residual_trend
season_observations_before
```

Пример:

```text
previous_1_climatology_residual =
    previous_1 - historical_value_on_previous_date
```

Если весь текущий сезон развивается ниже нормы, прогноз скрытой точки также должен быть скорректирован вниз.

## 6. Расширенные исторические признаки

Историю полигона следует описывать не только средним и стандартным отклонением.

Планируемые признаки:

```text
historical_mean
historical_median
historical_std
historical_q25
historical_q75
historical_iqr
historical_min
historical_max
historical_years_count
historical_recent_weighted
historical_year_trend
```

Необходимо сравнить исторические окна `doy ± 7`, `± 14` и `± 21` день. Для последних лет можно использовать больший вес.

## 7. Статистика других полей на ту же дату

Для каждой даты и культуры можно построить агрегаты по другим полигонам:

```text
same_date_crop_mean
same_date_crop_median
same_date_crop_std
same_date_crop_count
same_date_global_median
same_date_global_count
```

При обучении значение текущего полигона должно исключаться из агрегата. Предпочтительная схема — leave-one-polygon-out.

Такие признаки могут отражать общую сезонную или погодную динамику, но их необходимо отдельно проверить на переносимость между регионами.

## 8. Источник соседнего primary_ndvi

В train `primary_ndvi` формируется из спутниковых источников по приоритету:

```text
Sentinel-2 -> Landsat -> MODIS
```

Планируемые категориальные признаки:

```text
previous_1_source
previous_2_source
next_1_source
next_2_source
same_source_neighbors
source_switch
```

Возможные значения источника:

```text
sentinel2
landsat
modis
unknown
```

Также стоит оценить систематические различия сенсоров:

```text
landsat_minus_sentinel_median
modis_minus_sentinel_median
```

Поправки нужно рассчитывать по близким датам и без использования целевой строки.

## 9. Контекст EVI и NDWI

EVI и NDWI самой контрольной строки скрыты. Разрешено использовать только ближайшие доступные наблюдения других дат.

Планируемые признаки:

```text
previous_s2_evi
next_s2_evi
previous_s2_ndwi
next_s2_ndwi
previous_landsat_evi
next_landsat_evi
previous_landsat_ndwi
next_landsat_ndwi
previous_modis_evi
next_modis_evi
distance_to_previous_sensor_observation
distance_to_next_sensor_observation
```

Перед использованием нужна очистка данных:

- физически невозможные значения превращать в `NaN`;
- не подавать экстремальные EVI/NDWI в модель без проверки;
- добавить индикаторы отсутствующих наблюдений;
- проверить устойчивые границы отдельно для каждого сенсора и индекса.

## 10. Погодный контекст

Погода в synthetic gap скрыта, поэтому признаки должны строиться из доступной истории, восстановленного ряда ERA5 или заново полученных данных.

Планируемые признаки:

```text
temp_mean_7d
temp_mean_14d
temp_mean_30d
temp_min_14d
temp_max_14d
growing_degree_days_30d
precip_sum_7d
precip_sum_14d
precip_sum_30d
dry_days_14d
dry_days_30d
days_since_rain
```

Погодные окна необходимо считать по календарным дням, а не по количеству строк. Эти признаки особенно важны для интерпретации аномалий и длинных пропусков.

## Ограничения против утечки таргета

Нельзя использовать как признаки:

- `s2_ndvi`, `landsat_ndvi` или `modis_ndvi` из самой целевой строки;
- `ndvi_zscore` целевой строки;
- `status` целевой строки;
- незамаскированный `primary_ndvi` любой точки обучающего блока;
- агрегат других полей, если в нём осталось значение текущего полигона;
- прямое категориальное кодирование `anon_polygon_id`, поскольку половина полигонов test отсутствует в train.

## Планируемое расширение YAML

```yaml
features:
  temporal_dynamics:
    enabled: true
    slopes: true
    acceleration: true
    gap_geometry: true
    local_statistics: true

  interpolation:
    baseline: true
    linear: true
    pchip: true
    local_quadratic: false

  season_state:
    enabled: true
    climatology_residuals: true
    residual_trend: true

  polygon_history:
    enabled: true
    doy_windows: [7, 14, 21]
    statistics: [mean, median, std, q25, q75]
    recent_year_weighting: true

  peer_context:
    enabled: true
    group_by_crop: true
    leave_one_polygon_out: true

  sensor_context:
    enabled: true
    include_source: true
    include_evi: true
    include_ndwi: true
    max_distance_days: 30
    invalid_values_to_nan: true

  weather_context:
    enabled: false
    windows_days: [7, 14, 30]
    temperature: true
    precipitation: true

training:
  target_mode: residual
  residual_baseline: linear
  gap_masking:
    strategy: test_like_blocks
    max_block_length: 5
```

## Последовательность экспериментов

| Версия | Изменение |
|---|---|
| V0 | Текущий LightGBM |
| V1 | Маскирование блоками по структуре test |
| V2 | Предсказание residual относительно linear |
| V3 | Наклоны, ускорение и геометрия пропуска |
| V4 | PCHIP и дополнительные интерполяции |
| V5 | Отклонение текущего сезона от нормы |
| V6 | Статистика других полей на ту же дату |
| V7 | Источник соседних NDVI |
| V8 | Очищенные EVI/NDWI соседних дат |
| V9 | Погодные календарные окна |

Каждое изменение нужно добавлять отдельно и сравнивать с предыдущей версией на одинаковых масках.

## Метрики и срезы валидации

Кроме общего RMSE необходимо сохранять качество:

- по длине блока пропусков;
- по расстоянию до соседних наблюдений;
- по известным и новым полигонам;
- по типам культур;
- по годам и последнему сезону;
- по фазам сезона;
- по источнику истинного `primary_ndvi`;
- отдельно для одиночных и краевых пропусков.

Приоритет ближайшей реализации: V1–V4. Эти изменения лучше всего соответствуют структуре test и не требуют внешних источников данных.
