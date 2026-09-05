"""Единая точка входа для batch-инференса и веб-сервиса."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, select_model
from src.pipeline import AnalysisPipeline
from src.submission import create_submission


def _parse_external_source_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        name, separator, raw_weight = value.partition("=")
        if not separator or not name or not raw_weight:
            raise ValueError(
                "--external-source-weight ожидает значение NAME=WEIGHT"
            )
        if name in weights:
            raise ValueError(f"Вес source {name!r} задан несколько раз")
        try:
            weights[name] = float(raw_weight)
        except ValueError as error:
            raise ValueError(
                f"Некорректный вес external source {name!r}: {raw_weight!r}"
            ) from error
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Анализ временных рядов NDVI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Сформировать submission.csv")
    predict.add_argument("--config", default="config.yaml")
    predict.add_argument("--input")
    predict.add_argument("--train")
    predict.add_argument("--output")
    predict.add_argument("--model", help="Имя модели из models.available")
    predict.add_argument(
        "--with-external",
        action="store_true",
        help="Обучить экспериментальную модель с data.external",
    )
    predict.add_argument(
        "--external-weight", type=float, help="Вес external-строк для эксперимента"
    )
    predict.add_argument(
        "--external-source-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Вес отдельного external-источника; параметр можно повторять",
    )
    predict.add_argument(
        "--target-column",
        default=None,
        help="Имя целевой колонки, ожидаемое платформой",
    )
    predict.add_argument(
        "--method",
        choices=("baseline", "ensemble", "ml"),
        default=None,
        help="Устаревший псевдоним выбора модели; используйте --model",
    )

    serve = subparsers.add_parser("serve", help="Запустить веб-интерфейс")
    serve.add_argument("--config", default="config.yaml")
    serve.add_argument("--data")
    serve.add_argument("--train")
    serve.add_argument("--model", help="Имя модели из models.available")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--debug", action="store_true", default=None)
    serve.add_argument(
        "--no-browser",
        action="store_true",
        default=None,
        help="Не открывать веб-интерфейс в браузере автоматически",
    )

    validate = subparsers.add_parser("validate", help="Проверить алгоритм на synthetic gaps")
    validate.add_argument("--config", default="config.yaml")
    validate.add_argument("--train")
    validate.add_argument("--model", help="Имя модели из models.available")
    validate.add_argument("--sample-size", type=int)
    validate.add_argument("--seed", type=int)

    validate_external = subparsers.add_parser(
        "validate-external",
        help="Сравнить обучение с external data и без них",
    )
    validate_external.add_argument("--config", default="config.yaml")
    validate_external.add_argument("--train")
    validate_external.add_argument("--model", help="Имя модели из models.available")
    validate_external.add_argument("--sample-size", type=int)
    validate_external.add_argument("--seeds", type=int, nargs="+")
    validate_external.add_argument("--external-weight", type=float)
    validate_external.add_argument(
        "--external-source-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
    )

    collect = subparsers.add_parser(
        "collect", help="Собрать внешние спутниковые наблюдения"
    )
    collect.add_argument("--config", default="configs/collection-pilot.yaml")
    collect.add_argument(
        "--sensor", choices=("sentinel2",), default="sentinel2"
    )
    collect.add_argument("--start", help="Начало периода YYYY-MM-DD")
    collect.add_argument("--end", help="Конец периода YYYY-MM-DD")
    collect.add_argument("--limit", type=int, help="Ограничение числа полигонов")
    collect.add_argument("--output", help="Путь к raw CSV")
    collect.add_argument(
        "--force", action="store_true", help="Перезаписать существующий сбор"
    )

    history = subparsers.add_parser(
        "collect-history", help="Собрать одинаковые сезонные окна за несколько лет"
    )
    history.add_argument("--config", default="configs/collection-pilot.yaml")
    history.add_argument("--years", type=int, nargs="+", required=True)
    history.add_argument("--limit", type=int, help="Ограничение числа полигонов")
    history.add_argument(
        "--force", action="store_true", help="Перезаписать существующие сборы"
    )

    discover = subparsers.add_parser(
        "discover-fields", help="Найти открытые контуры сельхозполей"
    )
    discover.add_argument("--config", default="configs/collection-pilot.yaml")
    discover.add_argument("--limit", type=int, help="Количество полигонов")
    discover.add_argument("--output", help="Путь к GeoJSON")
    discover.add_argument(
        "--candidates",
        help="Готовый GeoJSON кандидатов; пропускает повторный запрос Overpass",
    )
    discover.add_argument(
        "--force", action="store_true", help="Перезаписать существующий GeoJSON"
    )
    discover.add_argument(
        "--skip-quality-validation",
        action="store_true",
        help="Не выполнять WorldCover/Sentinel-2 проверку кандидатов",
    )

    weather_zones = subparsers.add_parser(
        "discover-weather-zones",
        help="Найти вероятные зоны конкурсных AOI по ERA5-Land",
    )
    weather_zones.add_argument(
        "--config", default="configs/weather-zone-search.yaml"
    )
    weather_zones.add_argument("--input", default="data/test_dataset.csv")
    weather_zones.add_argument(
        "--output", default="artifacts/collection/weather-zones.geojson"
    )
    weather_zones.add_argument("--force", action="store_true")

    prepare_external = subparsers.add_parser(
        "prepare-external", help="Собрать нормализованный внешний датасет"
    )
    prepare_external.add_argument("--config", default="configs/collection-pilot.yaml")
    prepare_external.add_argument(
        "--sentinel2", nargs="+", help="Пути к raw Sentinel-2 CSV"
    )
    prepare_external.add_argument("--output", help="Путь к итоговому CSV")

    collect_multisensor = subparsers.add_parser(
        "collect-multisensor",
        help="Собрать Sentinel-2, Landsat, MODIS и ERA5 в конкурсной схеме",
    )
    collect_multisensor.add_argument("--config", default="config.yaml")
    collect_multisensor.add_argument(
        "--regions", default="examples/regions.geojson", help="GeoJSON контуров"
    )
    collect_multisensor.add_argument("--start", required=True)
    collect_multisensor.add_argument("--end", required=True)
    collect_multisensor.add_argument(
        "--output", default="artifacts/collection/multisensor.csv"
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Честная оценка на синтетических пропусках по спецификации платформы",
    )
    benchmark.add_argument("--config", default="config.yaml")
    benchmark.add_argument("--train")
    benchmark.add_argument("--model", help="Имя модели из models.available")
    benchmark.add_argument("--repeats", type=int)
    benchmark.add_argument("--mask-rate", type=float)
    benchmark.add_argument("--holdout-fraction", type=float)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument(
        "--profile",
        action="store_true",
        help="Сравнить профиль синтетики с реальными пропусками теста",
    )
    benchmark.add_argument(
        "--test",
        help="Тестовый датасет для эталонного профиля (по умолчанию data.test_path)",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-external":
        from src.collection.prepare import run_prepare_external_command

        run_prepare_external_command(args)
        return
    if args.command == "discover-fields":
        from src.collection.osm_fields import run_discover_fields_command

        run_discover_fields_command(args)
        return
    if args.command == "discover-weather-zones":
        from src.collection.weather_zones import run_discover_weather_zones_command

        run_discover_weather_zones_command(args)
        return
    if args.command == "collect":
        from src.collection.runner import run_collect_command

        run_collect_command(args)
        return
    if args.command == "collect-history":
        from src.collection.runner import run_collect_history_command

        run_collect_history_command(args)
        return

    config = load_config(args.config)

    requested_model = getattr(args, "model", None)
    legacy_method = getattr(args, "method", None)
    if requested_model and legacy_method:
        raise ValueError("Используйте только один из параметров: --model или --method")
    if legacy_method:
        requested_model = {
            "baseline": "baseline",
            "ensemble": "ensemble",
            "ml": None,
        }[legacy_method]
    config = select_model(config, requested_model)
    external_weight = getattr(args, "external_weight", None)
    if external_weight is not None:
        from src.external_validation import with_external_weight

        config = with_external_weight(config, external_weight)
    source_weight_values = getattr(args, "external_source_weight", [])
    if source_weight_values:
        from src.external_validation import with_external_source_weights

        config = with_external_source_weights(
            config, _parse_external_source_weights(source_weight_values)
        )

    if args.command == "collect-multisensor":
        from src.providers import GeeProvider, ProviderSettings

        provider = GeeProvider(
            ProviderSettings(
                key_path=config.collect.key_path,
                project=config.collect.project,
            )
        )
        frame = provider.fetch_many(args.regions, args.start, args.end)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(
            f"Multi-sensor dataset создан: {output_path} | "
            f"полигонов: {frame['anon_polygon_id'].nunique()} | строк: {len(frame)}"
        )
        return

    if args.command == "predict":
        if (
            args.external_weight is not None or args.external_source_weight
        ) and not args.with_external:
            raise ValueError(
                "Настройка external-весов требует --with-external"
            )
        if args.with_external:
            from src.external_validation import external_experiment_config

            config = external_experiment_config(config)
        input_path = Path(args.input) if args.input else config.data.test_path
        train_path = Path(args.train) if args.train else config.data.train_path
        output_path = Path(args.output) if args.output else config.predict.output_path
        target_column = args.target_column or config.predict.prediction_column
        result = create_submission(
            input_path=input_path,
            train_path=train_path,
            output_path=output_path,
            method=config.models.selected,
            target_column=target_column,
            config=config,
        )
        print(
            f"Submission создан: {result.path} | "
            f"модель: {result.model_name} | "
            f"строк: {result.rows} | диапазон: "
            f"{result.min_prediction:.6f}..{result.max_prediction:.6f}"
        )
        return

    if args.command == "validate":
        train_path = Path(args.train) if args.train else config.data.train_path
        sample_size = (
            args.sample_size
            if args.sample_size is not None
            else config.validation.sample_size
        )
        seed = args.seed if args.seed is not None else config.validation.seed
        pipeline = AnalysisPipeline.from_csv(train_path, config=config)
        metrics = pipeline.validate(
            sample_size=sample_size,
            seed=seed,
            model_name=config.models.selected,
        )
        print(f"Baseline RMSE: {metrics['baseline_rmse']:.6f}")
        print(f"{metrics['model']} RMSE: {metrics['model_rmse']:.6f}")
        return

    if args.command == "validate-external":
        from src.external_validation import validate_external_ab

        train_path = Path(args.train) if args.train else config.data.train_path
        sample_size = args.sample_size or config.validation.sample_size
        seeds = args.seeds or config.validation.external_ab.seeds
        result = validate_external_ab(
            train_path=train_path,
            config=config,
            model_name=config.models.selected,
            sample_size=sample_size,
            seeds=seeds,
        )
        for run in result["runs"]:
            print(
                f"seed={run['seed']}: без external={run['base_rmse']:.6f}, "
                f"с external={run['external_rmse']:.6f}, "
                f"изменение={run['change_percent']:+.3f}%"
            )
        print(
            f"Средний RMSE: без external={result['base_rmse_mean']:.6f}, "
            f"с external={result['external_rmse_mean']:.6f}, "
            f"изменение={result['change_percent']:+.3f}%"
        )
        return

    if args.command == "benchmark":
        from src.benchmark import run_benchmark
        from src.data import load_dataset
        from src.synthetic import compare_profiles, gap_profile

        train_path = Path(args.train) if args.train else config.data.train_path
        frame = load_dataset(train_path)

        if args.profile:
            test_path = Path(args.test) if args.test else config.data.test_path
            reference = gap_profile(load_dataset(test_path))
            spec_rate = args.mask_rate or config.benchmark.mask_rate
            from src.synthetic import MaskSpec, apply_mask, generate_mask

            mask = generate_mask(frame, MaskSpec(rate=spec_rate, seed=config.benchmark.seed))
            generated = gap_profile(
                apply_mask(frame, mask).assign(is_synthetic_gap=mask.to_numpy())
            )
            print(compare_profiles(reference, generated).to_string())
            return

        result = run_benchmark(
            frame,
            config,
            model_name=config.models.selected,
            repeats=args.repeats or config.benchmark.repeats,
            mask_rate=args.mask_rate or config.benchmark.mask_rate,
            seed=args.seed or config.benchmark.seed,
            holdout_fraction=(
                config.benchmark.holdout_fraction
                if args.holdout_fraction is None
                else args.holdout_fraction
            ),
        )
        print(result.render())
        return

    if args.command == "serve":
        from src.webapp import run_server

        data_path = Path(args.data) if args.data else config.data.test_path
        train_path = Path(args.train) if args.train else config.data.train_path
        run_server(
            data_path=data_path,
            train_path=train_path,
            host=args.host or config.server.host,
            port=args.port or config.server.port,
            debug=config.server.debug if args.debug is None else args.debug,
            open_browser=(
                config.server.open_browser if args.no_browser is None else False
            ),
            auto_select_port=config.server.auto_select_port,
            config=config,
        )


if __name__ == "__main__":
    main()
