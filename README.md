# АгроПульс

Веб-сервис и воспроизводимый batch-pipeline для восстановления пропусков `primary_ndvi`, визуализации сезонной динамики сельскохозяйственных полей и объяснения негативных аномалий.

## Возможности текущего MVP

- единая точка входа `main.py` с режимами `predict`, `serve` и `validate`;
- формирование конкурсного `submission.csv` только для `is_synthetic_gap = True`;
- строгая проверка схемы, количества строк, ключей, `NaN` и бесконечных значений;
- baseline из двух ближайших наблюдений;
- LightGBM-метамодель над соседними наблюдениями, историей полигона и сезонной кривой культуры;
- восстановление календарных признаков из `date`;
- детекция аномалий по отклонению от сезонной нормы;
- объяснения с использованием Z-score, температуры и осадков;
- веб-интерфейс с выбором полигона и сезона, картой, графиком и списком аномалий;
- добавление пользовательского GeoJSON-полигона через карту;
- Docker Compose и автоматические тесты.

## Структура

```text
main.py                 единая точка входа
config.yaml             данные, признаки, модели и режимы запуска
src/config.py           строгая схема и загрузка YAML
src/data.py             загрузка и нормализация CSV
src/interpolation.py    восстановление primary_ndvi
src/features.py         единое построение признаков
src/training.py         генератор test-like обучающих пропусков
src/models/             реестр и реализации моделей
src/predictor.py        обучение, кэширование и общий инференс
src/anomalies.py        климатическая норма и аномалии
src/pipeline.py         общий pipeline для CLI и веб-сервиса
src/submission.py       генерация и проверка submission.csv
src/webapp.py           HTTP API и раздача интерфейса
web/                    HTML, CSS и JavaScript
tests/                  тесты бизнес-логики
data/                   конкурсные датасеты
artifacts/              создаваемые результаты
```

## Требования

- Python 3.11 или новее;
- около 500 МБ свободной оперативной памяти;
- интернет в браузере для загрузки OpenStreetMap и Leaflet. Аналитический pipeline и график NDVI работают локально.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Задача 1: создание submission.csv

Основной алгоритм:

```bash
python main.py predict \
  --config config.yaml \
  --input data/test_dataset.csv \
  --train data/train_dataset.csv \
  --output artifacts/submission.csv \
  --model lightgbm
```

Воспроизведение baseline:

```bash
python main.py predict \
  --config config.yaml \
  --model baseline \
  --output artifacts/submission_baseline.csv
```

Формат результата:

```csv
anon_polygon_id,date,primary_ndvi_true
AOI-0001,2010-08-29,0.190719
```

Фактический валидатор платформы ожидает имя целевой колонки
`primary_ndvi_true`, несмотря на указанное в PDF имя `primary_ndvi_pred`.
Поэтому генератор по умолчанию создаёт именно этот заголовок.

Если организаторы снова изменят шаблон, имя можно передать явно:

```bash
python main.py predict --target-column primary_ndvi_pred
```

## Локальная валидация

Команда временно скрывает известные значения train и сравнивает baseline с метамоделью:

```bash
python main.py validate --train data/train_dataset.csv --sample-size 3000 --seed 42
```

Этот режим предназначен для быстрой проверки pipeline. Для итогового исследовательского отчёта необходимо дополнительно проводить разбиение по полигонам и целым сезонам.

## Отправка submission через браузер

Selenium-отправщик изолирован в каталоге `api/` и не входит в основной
Docker-образ. Установите дополнительные зависимости:

```bash
pip install -r requirements-submit.txt
```

Адрес страницы отправки
`https://рнд.космохакатон.рф/personal/solution` уже задан в `api/config.yaml`.
Перед реальной отправкой необходимо проверить CSS-селекторы формы загрузки по
фактической разметке платформы.

Скопируйте шаблон реквизитов и заполните его:

```bash
cp api/credentials.env.example api/credentials.env
```

```dotenv
COSMO_EMAIL=ваш-логин
COSMO_PASSWORD=ваш-пароль
```

`api/credentials.env` исключён из Git и не должен добавляться в репозиторий.
После загрузки страницы отправщик ждёт одну секунду. Если присутствуют поля
`#email` и `#password`, он заполняет их, нажимает `button.btnLoginCSS` и сохраняет
полученную сессию. Между вводом логина, вводом пароля и нажатием кнопки
выдерживаются паузы по 0,25 секунды; значение задаётся параметром
`authentication.input_delay`.

Проверка без нажатия финальной кнопки:

```bash
./submit.sh artifacts/submission.csv --dry-run
```

Если платформа не принимает автоматический вход, выполните авторизацию вручную
в отдельном окне Selenium:

```bash
./submit.sh artifacts/submission.csv --dry-run --manual-login
```

Скрипт ждёт ручной вход до пяти минут, сохраняет cookies и продолжает работу.
При следующих запусках сохранённая сессия используется автоматически.

Для реальной отправки:

```bash
./submit.sh artifacts/submission.csv
```

На странице решения отправщик выбирает файл через `#solution-file`, затем
нажимает кнопку `#save-solution-btn-metrika` и ожидает ответ после отправки
HTML-формы. Через три секунды он читает верхнюю строку
`tbody tr.team-row:first-child`: первую ячейку как дату, вторую как статус и
третью как значение метрики. Новая строка сравнивается с верхней строкой до
отправки, поэтому старый результат не будет принят за новый.

Selenium использует отдельный постоянный профиль
`artifacts/submissions/session/chrome-profile`, не связанный с обычным Chrome.
Cookies дополнительно сохраняются в
самом профиле Chrome и автоматически доступны при следующем запуске. Отдельный
`cookies.json` не используется, чтобы не создавать конфликтующие копии
сессионных и CSRF-cookies. Если окно предыдущего запуска Selenium ещё открыто,
его необходимо закрыть: два процесса не могут одновременно изменять один
профиль.

Чтобы сбросить только сессию платформы в Selenium-профиле и заново войти
вручную:

```bash
./submit.sh artifacts/submission.csv --dry-run --reset-session --manual-login
```

Команда очищает cookies, `localStorage` и `sessionStorage` текущего домена.
Профиль обычного Chrome не затрагивается.

В режиме `--dry-run` файл выбирается в форме, но финальная кнопка не нажимается,
а окно браузера остаётся открытым. Закройте его вручную перед следующим запуском,
иначе постоянный профиль будет занят. Скриншоты автоматически не создаются.

В `artifacts/submissions/` ведётся журнал успешных отправок. Повторная отправка
того же файла блокируется; её можно разрешить параметром `--allow-duplicate`.
Для машинно-читаемого ответа используйте `--json`.

Пошаговая диагностика записывается без реквизитов и cookies в файлы:

- `artifacts/submissions/logs/submitter.log` — этапы сценария и ошибки;
- `artifacts/submissions/logs/chromedriver.log` — сообщения ChromeDriver.

## Задача 2: веб-сервис

```bash
python main.py serve \
  --config config.yaml \
  --data data/test_dataset.csv \
  --train data/train_dataset.csv \
  --host 127.0.0.1 \
  --port 8000
```

Открыть: <http://127.0.0.1:8000>

Если порт 8000 уже занят, приложение автоматически проверит следующие порты
и напечатает фактически выбранный адрес, например `http://127.0.0.1:8001`.
Страница откроется в системном браузере автоматически. Для запуска без открытия
браузера используйте `python main.py serve --no-browser`.

## Конфигурация и выбор модели

Общие признаки и модель задаются в `config.yaml` и одинаково используются в
`predict`, `validate` и `serve`. Активная модель выбирается в YAML:

```yaml
models:
  selected: routed_lightgbm
```

Доступны `baseline`, `ensemble`, `lightgbm`, `routed_lightgbm`,
`random_forest` и `catboost`.
Выбор можно временно переопределить без редактирования файла:

```bash
python main.py validate --config config.yaml --model random_forest
python main.py serve --config config.yaml --model baseline
```

Параметры каждой модели находятся в `models.available`. LightGBM и Random
Forest устанавливаются из `requirements.txt`. CatBoost является опциональным:

```bash
pip install catboost
```

`routed_lightgbm` содержит два LightGBM-эксперта в одном артефакте. Для строки
с историей полигона используется history-rich эксперт со всеми признаками. Если
исторических сезонов нет, автоматически выбирается cold-start эксперт, который
не видит `historical`, `historical_std` и `n_reference_years_calc`. Оба эксперта
обучаются и применяются одним обычным запуском `predict`, `validate` или `serve`.
Порог доступной истории и параметры экспертов задаются независимо:

```yaml
models:
  available:
    routed_lightgbm:
      type: history_routed_lightgbm
      params:
        min_reference_years: 1
        common_params: {n_estimators: 350, learning_rate: 0.035}
        history_rich_params: {}
        cold_start_params: {}
```

Обученная модель сохраняется в `artifacts/models/` и повторно используется,
только если конфигурация признаков, параметры и сигнатура train не изменились.

Обучаемые модели поддерживают прямой прогноз и прогноз поправки к интерполяции:

```yaml
training:
  target_mode: direct       # direct | residual
  residual_baseline: linear
  gap_masking:
    strategy: leave_one_out  # leave_one_out | test_like_blocks
    target_fraction: 0.15
    replicas: 5
    block_length_weights: {1: 0.922, 2: 0.070, 3: 0.007, 4: 0.0005, 5: 0.0005}
    random_state: 42
```

В режиме `residual` таргетом служит `primary_ndvi - linear_prediction`, а при
инференсе предсказанная поправка прибавляется обратно к `linear_prediction`.
Режим реализован, но текущий default оставлен `direct`: в исходном сравнении до
внедрения test-like обучения на контрольной маске из 1000 точек direct показал
RMSE 0.072809, residual-linear — 0.073607, а residual-neighbor-mean — 0.073970.
Изменение блока `training` инвалидирует кэш модели и вызывает переобучение.

Опциональный режим `test_like_blocks` создаёт несколько обучающих реплик и одновременно скрывает
целые блоки наблюдений. Длины блоков повторяют распределение test. В скрытых
строках очищаются `primary_ndvi`, спутниковые, погодные и вычисляемые поля;
календарные `year` и `doy` безопасно восстанавливаются из `date`.

Локальная динамика настраивается отдельно:

```yaml
features:
  interpolation:
    baseline: true
    linear: true
    pchip: true
    local_quadratic: false
    differences: false
    agreement: false
  temporal_dynamics:
    enabled: true
    gap_geometry: true
    slopes: true
    acceleration: true
    local_statistics: true
```

В этот блок входят геометрия пропуска, наклоны до/после него, изменение наклона,
локальное ускорение и статистики четырёх ближайших известных значений. Один и
тот же расчёт используется при обучении, batch-предсказании и в веб-сервисе.

PCHIP используется как дополнительный признак LightGBM, а не как безусловная
замена итогового прогноза. На фиксированной маске из 1000 точек он улучшил RMSE
с 0,072058607 до 0,072020307. Реализованные quadratic, разности и статистики
согласия оставлены выключенными, поскольку их абляции ухудшили локальный RMSE.

Основной сценарий:

1. Выбрать `anon_polygon_id`.
2. Выбрать сезон.
3. Нажать «Проанализировать».
4. Сравнить исходный, восстановленный NDVI и сезонную норму.
5. Просмотреть критические отклонения и их объяснения.
6. При необходимости нарисовать и сохранить пользовательский контур.

Конкурсные полигоны анонимизированы и не содержат координат. Поэтому они отображаются в селекторе, а карта предназначена для пользовательских контуров. Автоматический сбор спутниковых данных для нового GeoJSON-контура должен подключаться через отдельный provider и требует учётных данных соответствующего внешнего API.

## Пилотный сбор Sentinel-2

Offline-сборщик расширения train изолирован от инференса и использует отдельный
конфиг `configs/collection-pilot.yaml`. Основные команды продолжают работать без
Earth Engine и его зависимостей.

Установить опциональный клиент:

```bash
pip install -r requirements-collect.txt
```

Зарегистрировать Google Cloud project для Earth Engine, авторизовать локальный
клиент и передать ID проекта:

```bash
earthengine authenticate
export EE_PROJECT_ID="ваш-google-cloud-project"
```

Получить 10 открытых OSM-контуров сельхозземель, проходящих ограничения площади
и bbox из collection-конфига:

```bash
python main.py discover-fields --config configs/collection-pilot.yaml
```

Команда использует объекты OpenStreetMap `landuse=farmland` (лицензия ODbL),
выбирает ближайшие к центру заданного региона и сохраняет источник, OSM way ID и
расчётную площадь. Существующий файл не перезаписывается без `--force`.

Либо скопировать образец GeoJSON и заменить демонстрационную геометрию вручную.
Координаты должны быть в WGS84, а каждый объект должен иметь уникальное свойство
`polygon_id`:

```bash
cp data/external/polygons.example.geojson data/external/polygons.geojson
```

Демонстрационный прямоугольник в образце показывает только формат и не является
проверенным сельскохозяйственным полем.

Запустить пилотный сбор:

```bash
python main.py collect --config configs/collection-pilot.yaml
```

Собрать те же поля за несколько сезонов без захвата зимних дат:

```bash
python main.py collect-history \
  --config configs/collection-pilot.yaml \
  --years 2019 2020 2021 2022 2023 \
  --limit 5
```

Каждый сезон сохраняется отдельно как `sentinel2-YYYY.csv` и имеет собственный
manifest для безопасного возобновления.

По умолчанию обрабатываются до 10 полигонов за сезон 2024 года. Результат
появляется в `artifacts/collection/raw/sentinel2.csv`, а состояние выполнения —
в `artifacts/collection/manifests/sentinel2.json`. Завершённые полигоны при
повторном запуске не запрашиваются. Изменившийся период или набор полигонов
требует нового output либо явного `--force`.

Параметры одного эксперимента можно переопределить из CLI:

```bash
python main.py collect \
  --config configs/collection-pilot.yaml \
  --start 2024-05-01 \
  --end 2024-09-30 \
  --limit 5 \
  --output artifacts/collection/raw/sentinel2-may-september.csv
```

Сборщик проверяет GeoJSON, уникальность идентификаторов, попадание геометрии в
настроенный bbox и площадь полей. Если у Feature присутствует свойство
`cropland_fraction`, применяется также минимальный порог из YAML. На первом
этапе реализован только Sentinel-2; Landsat, MODIS и ERA5 добавляются отдельными
детерминированными этапами после сравнения пилотных наблюдений с конкурсным train.

Преобразовать scene-level наблюдения в ежедневную таблицу схемы train:

```bash
python main.py prepare-external \
  --config configs/collection-pilot.yaml \
  --sentinel2 \
    artifacts/collection/raw/sentinel2-2019.csv \
    artifacts/collection/raw/sentinel2-2020.csv \
    artifacts/collection/raw/sentinel2-2021.csv \
    artifacts/collection/raw/sentinel2-2022.csv \
    artifacts/collection/raw/sentinel2-2023.csv \
    artifacts/collection/raw/sentinel2.csv
```

Сцены и тайлы одного поля за одну дату объединяются медианой. Для каждого поля
создаётся полная ежедневная сетка периода, `primary_ndvi` заполняется из
`s2_ndvi`, а `crop_type` получает честное значение `неизвестно`. Для нескольких
лет климатология каждой известной точки считается только по другим годам того
же поля в окне `doy ± 21`. Результат
сохраняется в `data/external/processed/external_dataset.csv` и остаётся
изолированным от конкурсного train до A/B-валидации.

Подключение к обучению задаётся независимо в основном конфиге:

```yaml
data:
  external:
    enabled: false
    paths: [data/external/processed/external_dataset.csv]
    usage: training_only
    crop_type_fallback: неизвестно
    polygon_id_prefix: EXT-
    sample_weight: 0.25
```

При `enabled: true` external-строки добавляются только в обучающую выборку и не
попадают в исторический контекст тестовых полигонов. До подтверждения пользы в
production-конфиге флаг остаётся выключенным. A/B-проверка использует файлы из
`paths` независимо от production-флага, отключает признаки культуры и оценивает
обе модели на одинаковых известных точках исходного train:

```bash
python main.py validate-external --config config.yaml
```

Вес external можно переопределить для абляции без редактирования YAML:

```bash
python main.py validate-external --config config.yaml --external-weight 0.10
```

Сформировать отдельный submission экспериментальной external-моделью, не меняя
production-флаг и не перезаписывая кэш основной модели:

```bash
python main.py predict \
  --config config.yaml \
  --with-external \
  --output artifacts/submission_external.csv
```

## Docker

```bash
docker compose up --build
```

После запуска интерфейс доступен по адресу <http://127.0.0.1:8000>.

Batch-инференс внутри контейнера:

```bash
docker compose run --rm agropulse \
  python main.py predict \
  --input data/test_dataset.csv \
  --train data/train_dataset.csv \
  --output artifacts/submission.csv
```

## Тесты

```bash
python -m unittest discover -s tests -v
```

## Алгоритм восстановления

Для каждой скрытой точки pipeline использует только доступный контекст:

1. Находит ближайшие известные `primary_ndvi` слева и справа в том же сезоне.
2. Строит baseline как среднее двух соседних наблюдений.
3. Рассчитывает историческую оценку по другим годам того же полигона в окне `doy ± 21`.
4. Рассчитывает сглаженную сезонную кривую для `crop_type`.
5. Рассчитывает геометрию пропуска, локальные наклоны, ускорение и статистики соседей.
6. Передаёт все оценки и признаки в LightGBM-метамодель.

Все synthetic gaps одной последовательности исключаются из контекста одновременно. Это не позволяет соседним контрольным точкам использовать скрытые значения друг друга.

## Детекция аномалий

```text
Z = (primary_ndvi_filled - climatology_mean) / climatology_std
```

| Значение | Статус |
|---|---|
| `Z >= -1` | Штатное развитие |
| `-2 <= Z < -1` | Угнетение биомассы |
| `Z < -2` | Критическая аномалия |

Интерпретатор дополняет статус погодным контекстом: суммой осадков за 14 дней и средней температурой за 7 дней.

## Известные ограничения MVP

- Offline-сбор Sentinel-2 реализован, но пока не связан с пользовательскими контурами веб-интерфейса.
- Конкурсные данные не содержат реальной геометрии полигонов.
- Для дальнейшего улучшения RMSE нужны абляции и расширенная валидация по целым сезонам и полигонам.
- Пользовательские полигоны хранятся в памяти процесса и исчезают после перезапуска.
- В исходных колонках встречаются экстремальные выбросы `s2_evi`, `landsat_ndvi`, `landsat_evi` и `landsat_ndwi`; текущий baseline использует согласованный `primary_ndvi` без глобального клиппинга.

## Воспроизводимость

- Один и тот же `AnalysisPipeline` используется CLI и веб-сервисом.
- `year` и `doy` всегда пересчитываются из `date`.
- Все контрольные строки валидируются перед записью submission.
- Алгоритм детерминирован; параметр `seed` нужен только для выбора точек локальной валидации.
- Комментарии к бизнес-логике написаны на русском языке.
