# Журнал экспериментов

Все платформенные значения записываются только из ответа submitter. Локальные
оценки помечаются отдельно и не подменяют конкурсную метрику.

| Дата | Модель | Изменение | Локальная оценка | RMSE платформы | Статус |
|---|---|---|---:|---:|---|
| 2026-09-05 | `routed_lightgbm` | baseline ветки `main` | — | 0.0763 | отправлено |
| 2026-09-05 | `sensor_lightgbm` | общие признаки + орбитальный классификатор + per-sensor интерполяции | — | 0.0672 | отправлено |
| 2026-09-05 | `sensor_lightgbm` v2 | + polygon/DOY и cross-field date-признаки источника | 0.07707 (strict holdout, v1 0.07781) | — | кандидат |
| 2026-09-05 | `sensor_lightgbm` v3 | + cross-field NDVI и per-sensor history | 0.07605 (strict holdout, v2 0.07707) | — | лучший локальный кандидат |
| 2026-09-05 | routed blend v1/v3 | вес v3 0.85 для известных и 0.78 для новых полигонов | 0.07596 (strict holdout) | — | готов к отправке |
| 2026-09-05 | `sensor_lightgbm` v4 | v3 + три source-эксперта, смесь 0.49 | 0.07535 (strict holdout) | — | лучший локальный кандидат |
| 2026-09-05 | field-offset ablation | поправка полигона к cross-field медиане | 0.07573 (strict holdout) | — | отклонено |
| 2026-09-05 | multi-seed ablation | усреднение seed 42/137/911 | 0.07539 (strict holdout) | — | отклонено |
| 2026-09-05 | recent/trend ablation | recent-weighted и межгодовой тренд сенсора | 0.07539 (strict holdout) | — | отклонено |
| 2026-09-05 | best routed blend | вес v4 0.95 для известных и 0.87 для новых полигонов | 0.07531 (strict holdout) | 0.0651 | отправлено |
| 2026-09-05 | harmonized sensor interpolation | перевод соседних primary в шкалу вероятного сенсора | 0.06949 против 0.06939 (50% polygon holdout, seed 42) | — | отдельно отклонено |
| 2026-09-05 | blend base/harmonized | вес harmonized 0.40 | 0.06930 против 0.06939 (50% polygon holdout, seed 42) | — | слабое улучшение |
| 2026-09-05 | transductive sensor v3 | обучение также на известных точках контекста | 0.06619 против 0.06939; unseen 0.07491 против 0.07778 | — | готов к отправке |
| 2026-09-05 | training-source A/B | train-only / test-only / combined | 0.06950 / 0.06876 / 0.06606 (test holdout, seed 73) | — | combined выбран |
| 2026-09-05 | routed transductive blend | вес transductive 0.95 seen, 0.85 unseen | 0.06597 против 0.06606 (proxy blend, seed 73) | — | готов к отправке |
| 2026-09-05 | transductive sensor v5 | + per-sensor EVI/NDWI соседи и интерполяции | 0.06574 против 0.06619; seen 0.05533, unseen 0.07458 | — | лучший локальный кандидат |
| 2026-09-05 | routed transductive v5 blend | v5 с весом 0.95 seen, 0.85 unseen; остаток — platform-best 0.0651 | proxy: v5 лучше v3 на 0.00046 | — | готов к отправке |
| 2026-09-05 | transductive sensor v6 | + temporal/date-level ERA5 temperature and precipitation | 0.06547 против 0.06574; seen 0.05500, unseen 0.07434 | — | лучший локальный кандидат |
| 2026-09-05 | routed transductive v6 blend | v6 с весом 0.95 seen, 0.85 unseen; остаток — platform-best 0.0651 | proxy: v6 лучше v5 на 0.00027 | 0.0597 | отправлено |
| 2026-09-05 | transductive sensor v7 | + one-hot идентификатор полигона | 0.06549; seen 0.05536, unseen 0.07411 | — | только для unseen routing |
| 2026-09-05 | routed v6/v7 blend | v6 для seen, v7 для unseen; затем веса 0.95/0.85 к platform-best | proxy: unseen v7 лучше v6 на 0.00023 | — | готов к отправке |
| 2026-09-05 | transductive sensor v8 | + агрегаты доступной части polygon-year сезона | 0.06529; seen 0.05482, unseen 0.07417 | — | v8 seen / v7 unseen |
| 2026-09-05 | routed v8/v7 blend | v8 для seen, v7 для unseen; затем веса 0.95/0.85 к прежнему 0.0651 | proxy: лучшие group-wise RMSE 0.05482/0.07411 | — | готов к отправке |
