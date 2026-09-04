"""Единая точка входа для batch-инференса и веб-сервиса."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, select_model
from src.pipeline import AnalysisPipeline
from src.submission import create_submission


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

    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    if args.command == "predict":
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
