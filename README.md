# АгроПульс

Веб-сервис и воспроизводимый batch-pipeline для восстановления пропусков `primary_ndvi`, визуализации сезонной динамики сельскохозяйственных полей и объяснения негативных аномалий.

## Возможности текущего MVP

- единая точка входа `main.py` с режимами `predict`, `serve` и `validate`;
- формирование конкурсного `submission.csv` только для `is_synthetic_gap = True`;
- строгая проверка схемы, количества строк, ключей, `NaN` и бесконечных значений;
- baseline из двух ближайших наблюдений;
- LightGBM-метамодель над соседними наблюдениями, историей полигона и сезонной кривой культуры;
- sensor-aware LightGBM с орбитальным классификатором и отдельными рядами Sentinel-2, Landsat и MODIS;
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

Для строгого сравнения используется benchmark, который одновременно скрывает
наблюдения, исключает часть полигонов из обучения и отдельно показывает RMSE
на видимых и новых полях:

```bash
python main.py benchmark --model sensor_lightgbm
python main.py benchmark --model lightgbm_sensor
python main.py benchmark --profile
```

`lightgbm_sensor` — отдельный экспериментальный вариант из ветки `combined`:
он добавляет агрегаты остальных полигонов на ту же дату (`peer_date`) и может
использовать LightGBM, ExtraTrees либо CatBoost. Основной
`sensor_lightgbm` при этом сохранён без замены.

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
  selected: sensor_lightgbm
```

Доступны `baseline`, `ensemble`, `lightgbm`, `routed_lightgbm`, `sensor_lightgbm`,
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
  use_context_labels: true  # учить маски также на известных строках test
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

`use_context_labels` включает трансдуктивное обучение: известные `primary_ndvi`
из входного ряда используются как дополнительные разрешённые обучающие точки.
Строки `is_synthetic_gap=true` остаются пустыми и в обучение не попадают.

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

1. Найти регион или населённый пункт в строке поиска. Используются Nominatim и OpenStreetMap; поиск запускается только по отправке формы.
2. Нажать на готовое поле на карте. Кнопка «Найти поля на карте» запрашивает контуры OSM через Overpass для видимой области; для больших областей нужно приблизить карту. При старте показаны реальные пилотные контуры из `data/external/polygons.geojson`.
3. Либо нажать «Нарисовать поле», поставить минимум три вершины и замкнуть границу нажатием на первую вершину или кнопкой «Замкнуть». Вершины перетаскиваются; доступны отмена точки и отмена рисования (также Escape). Пересечения блокируют сохранение. Площадь приблизительная.
4. Сохранить поле в «Мои поля». Его можно выбрать повторно, переименовать или удалить. Контуры сохраняются на сервере в `artifacts/web-polygons.json` и переживают перезапуск. Для Docker этот файл должен находиться в сохраняемом томе.
5. Для анализа доступного конкурсного ряда раскрыть «Конкурсные временные ряды», выбрать полигон и сезон, затем нажать «Проанализировать». Конкурсные идентификаторы не привязываются к реальной геометрии.
6. Нижняя панель показывает наблюдения точками, восстановленный NDVI линией, историческую норму с полосой ±1 стандартное отклонение и аномальные интервалы. Наведение или стрелки клавиатуры на графике раскрывают значения. Карточки отклонений выделяют соответствующий период. Период объединяет последовательные отрицательные отклонения без пропущенных календарных дней; это представление существующего детектора, а не отдельная новая модель.

Конкурсные полигоны анонимизированы и не содержат координат. Они доступны в отдельном селекторе. Для полей с реальной геометрией кнопка анализа запускает фоновый сбор Sentinel-2 и ERA5-Land через Google Earth Engine.

## Реальные поля: Google Earth Engine

Проект задаётся в `configs/monitoring.yaml`: `sunlit-arcade-460421-j9`. Это публичный идентификатор, не API-токен. Переменная `EE_PROJECT_ID` переопределяет его. На проекте должны быть включены Earth Engine API, регистрация Earth Engine и права выбранного аккаунта.

Установить клиент и один раз войти в Google (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-collect.txt
.\.venv\Scripts\earthengine.exe authenticate --auth_mode=localhost
.\.venv\Scripts\python.exe -m src.monitoring.worker --check
.\.venv\Scripts\python.exe main.py serve --no-browser
```

Не помещайте OAuth-токены и ключи сервисных аккаунтов в репозиторий или браузер. Для сервера поддерживаются Application Default Credentials: задайте `GOOGLE_APPLICATION_CREDENTIALS` как путь к защищённому файлу сервисного аккаунта или используйте предоставляемую средой авторизацию. При запуске в Docker локальная авторизация автоматически не переносится: передайте файл как read-only secret/mount и укажите путь внутри контейнера. Docker-образ уже включает клиент Earth Engine; каталог `artifacts` сохраняется томом в `docker-compose.yml`.

На карте выберите или нарисуйте поле, укажите начало и конец периода и нажмите «Проанализировать». Готовый OSM-контур при первом анализе сохраняется автоматически. Обработка продолжается при обновлении страницы: повторный выбор сохранённого поля показывает последнее задание или результат.

Архитектура:

- `src/monitoring/store.py`: SQLite-очередь, дедупликация одинаковых запросов, сохранение прогресса и результата;
- `src/monitoring/worker.py`: отдельный процесс, автоматически запускаемый командой `serve`; одна активная обработка, эксклюзивная блокировка worker и возобновление незавершённых заданий после перезапуска;
- `src/monitoring/earth_engine.py`: помесячное получение Sentinel-2 и погоды, серверная авторизация;
- `src/monitoring/analysis.py`: независимый анализ реального ряда;
- `artifacts/monitoring/jobs.sqlite3`: задания и результаты; `cache/`: наблюдения и параметры каждого успешно собранного месяца; `worker.log`: технические типы ошибок без секретов.

HTTP API: `POST /api/polygons/{id}/analyses` принимает `start` и `end`, возвращает задание (202 для нового, 200 для повторного); `GET /api/analyses/{id}` возвращает прогресс и готовый результат; `GET /api/polygons/{id}/analyses/latest` восстанавливает состояние интерфейса. Ошибку можно повторить той же кнопкой, сохраняя месячный кэш. Параллельные одинаковые запросы не создают несколько заданий.

Для самостоятельного запуска обработчика: `python -m src.monitoring.worker`; два обработчика одной базы одновременно не допускаются. `--once` обрабатывает одно задание. Текущая архитектура рассчитана на один сервер приложения и одну SQLite-базу.

Источники и расчёты:

- Sentinel-2 SR Harmonized (`COPERNICUS/S2_SR_HARMONIZED`), пространственное разрешение расчёта 20 м, медиана индексов по полю; SCL исключает классы 0, 1, 3, 7, 8, 9, 10, 11; порог облачности сцены 80%, пригодных пикселей не менее 60% и 10 штук. NDVI=(B8−B4)/(B8+B4), EVI=2.5×(B8−B4)/(B8+6×B4−7.5×B2+1), NDWI=(B3−B8)/(B3+B8); отражательная способность масштабируется на 0.0001.
- ERA5-Land Daily Aggregated (`ECMWF/ERA5_LAND/DAILY_AGGR`): `temperature_2m` переводится из K в °C, `total_precipitation_sum` из м в мм; отрицательные осадки считаются отсутствующими. Берётся ячейка центра поля на сетке около 11 км. Это региональный реанализ с задержкой обновления, не измерение на поле.
- Для нормы собираются аналогичные периоды за три предыдущих года с запасом ±21 день. Медиана и стандартное отклонение вычисляются по историческим наблюдениям, не по предсказаниям; необходимы минимум два разных года и три наблюдения, минимальный std=0.03.
- Реальный NDVI восстанавливается линейно по времени только между известными наблюдениями с интервалом до 30 дней. Края и длинные пропуски остаются пустыми. Конкурсная ML-модель не применяется без отдельной проверки переноса на реальные данные.
- Z-score и погодные объяснения показывают возможные причины, не доказанный диагноз. Дефицит осадков оценивается только при наличии всех 14 суточных значений, жара — всех 7 температур. При недоступной погоде результат NDVI сохраняется с предупреждением.

Ограничения MVP: до 5000 га на поле, до 366 дней на запрос, даты с 2018 года до вчерашнего дня, максимум 10 заданий в очереди. Параметры меняются в `configs/monitoring.yaml`. Свежие результаты/месяцы (окончание в последние 90 дней) кэшируются на сутки, исторические — на 30 дней. Повторное открытие результата не расходует квоту. После истечения срока кнопка «Проверить обновления» ставит повторное задание. Автоматического периодического мониторинга без действия пользователя нет.

Источники: [Earth Engine authentication](https://developers.google.com/earth-engine/guides/auth), [Sentinel-2 SR](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED), [ERA5-Land](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR), [квоты](https://developers.google.com/earth-engine/guides/usage). Атрибуция: Contains modified Copernicus Sentinel data; ERA5-Land: Copernicus Climate Change Service (C3S), ECMWF. Исходные наблюдения и результаты не являются данными конкурсного train и не подмешиваются в обучение автоматически.

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

Сначала найти вероятные погодные зоны test-AOI по ERA5-Land. Поиск использует
только входные погодные признаки, проходит по сетке Ростовской области и
сохраняет разнесённые центры зон с количеством сопоставленных AOI:

```bash
EE_PROJECT_ID=your-project-id python main.py discover-weather-zones \
  --config configs/weather-zone-search.yaml \
  --input data/test_dataset.csv \
  --output artifacts/collection/weather-zones.geojson
```

Для выбранного центра используется отдельный collection-конфиг. Получить
пространственно разнесённые OSM-контуры, проверить WorldCover и число пригодных
Sentinel-2 дат:

```bash
EE_PROJECT_ID=your-project-id python main.py discover-fields \
  --config configs/collection-era5-zone02.yaml
```

Команда использует объекты OpenStreetMap `landuse=farmland` (лицензия ODbL),
разносит поля по grid-cell с минимальной дистанцией, сохраняет OSM way ID и
расчётную площадь. При `field_validation.enabled=true` в GeoJSON также
записываются `cropland_fraction` из ESA WorldCover и число чистых Sentinel-2
наблюдений. Существующий файл не перезаписывается без `--force`.

Если Overpass уже отдал кандидатов, их можно проверить повторно без сетевого
OSM-запроса:

```bash
EE_PROJECT_ID=your-project-id python main.py discover-fields \
  --config configs/collection-era5-zone02.yaml \
  --candidates artifacts/collection/era5-zone02-candidates.geojson \
  --force
```

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
настроенный bbox, площадь полей, минимальную долю cropland и число пригодных
Sentinel-2 наблюдений. Основной возобновляемый pipeline сохраняет сырые сцены
Sentinel-2, а отдельный multi-sensor режим сразу формирует ежедневную таблицу с
Sentinel-2, Landsat 8/9, MODIS и ERA5-Land:

```bash
EE_PROJECT_ID=your-project-id python main.py collect-multisensor \
  --regions data/external/polygons.geojson \
  --start 2019-04-01 \
  --end 2024-10-30 \
  --output artifacts/collection/multisensor.csv
```

По умолчанию команда использует авторизацию `earthengine authenticate`. Для
сервера можно задать `collect.key_path` и `collect.project` в `config.yaml`.
Каждый объект GeoJSON должен иметь `id`, `name` или `anon_polygon_id`; значение
`crop_type` переносится из properties, а при его отсутствии записывается
`неизвестно`.

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
же поля в окне `doy ± 21`. Результат зерноградского сбора сохраняется в
`data/external/processed/zernograd_2019_2024.csv` и остаётся
изолированным от конкурсного train до A/B-валидации.

Каждый географический кластер задаётся отдельным именованным источником со
своим префиксом полигонов и весом:

```yaml
data:
  external:
    enabled: false
    usage: training_only
    sources:
      - name: zernograd_osm
        enabled: true
        path: data/external/processed/zernograd_2019_2024.csv
        crop_type_fallback: неизвестно
        polygon_id_prefix: EXT-ZGD-
        sample_weight: 0.50
      - name: egorlykskaya_osm
        enabled: true
        path: data/external/processed/egorlykskaya_2019_2024.csv
        crop_type_fallback: неизвестно
        polygon_id_prefix: EXT-EGL-
        sample_weight: 0.50
```

При `enabled: true` external-строки добавляются только в обучающую выборку и не
попадают в исторический контекст тестовых полигонов. До подтверждения пользы в
production-конфиге флаг остаётся выключенным. A/B-проверка использует активные
`sources` независимо от production-флага, отключает признаки культуры и оценивает
обе модели на одинаковых известных точках исходного train:

```bash
python main.py validate-external --config config.yaml
```

Вес external можно переопределить для абляции без редактирования YAML:

```bash
python main.py validate-external --config config.yaml --external-weight 0.10
```

Для независимой проверки источников параметр можно повторять. Вес `0` полностью
исключает соответствующий источник из обучения:

```bash
python main.py predict \
  --config config.yaml \
  --with-external \
  --external-source-weight zernograd_osm=0.50 \
  --external-source-weight egorlykskaya_osm=0.10 \
  --output artifacts/submission_external_egl_w010.csv
```

Второй кластер собирается теми же командами с отдельным конфигом:

```bash
python main.py discover-fields \
  --config configs/collection-egorlykskaya.yaml --limit 10
EE_PROJECT_ID=your-project-id python main.py collect-history \
  --config configs/collection-egorlykskaya.yaml \
  --years 2019 2020 2021 2022 2023 2024 --limit 10
python main.py prepare-external \
  --config configs/collection-egorlykskaya.yaml \
  --sentinel2 artifacts/collection/egorlykskaya/raw/sentinel2-{2019,2020,2021,2022,2023,2024}.csv
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

- Реальный веб-сбор требует авторизации Earth Engine и доступной квоты проекта; отсутствие пригодных сцен обрабатывается явно.
- Конкурсные данные не содержат реальной геометрии полигонов.
- Для дальнейшего улучшения RMSE нужны абляции и расширенная валидация по целым сезонам и полигонам.
- Хранилище пользовательских полигонов рассчитано на один процесс веб-сервера; для нескольких процессов потребуется общая база данных.
- В исходных колонках встречаются экстремальные выбросы `s2_evi`, `landsat_ndvi`, `landsat_evi` и `landsat_ndwi`; текущий baseline использует согласованный `primary_ndvi` без глобального клиппинга.

## Воспроизводимость

- Один и тот же `AnalysisPipeline` используется CLI и веб-сервисом.
- `year` и `doy` всегда пересчитываются из `date`.
- Все контрольные строки валидируются перед записью submission.
- Алгоритм детерминирован; параметр `seed` нужен только для выбора точек локальной валидации.
- Комментарии к бизнес-логике написаны на русском языке.
