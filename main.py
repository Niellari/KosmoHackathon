"""Единая точка входа для batch-инференса и веб-сервиса."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import AnalysisPipeline
from src.submission import create_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Анализ временных рядов NDVI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Сформировать submission.csv")
    predict.add_argument("--input", default="data/test_dataset.csv")
    predict.add_argument("--train", default="data/train_dataset.csv")
    predict.add_argument("--output", default="artifacts/submission.csv")
    predict.add_argument(
        "--target-column",
        default="primary_ndvi_true",
        help="Имя целевой колонки, ожидаемое платформой",
    )
    predict.add_argument(
        "--method",
        choices=("baseline", "ensemble", "ml"),
        default="ml",
        help="Алгоритм восстановления пропусков",
    )

    serve = subparsers.add_parser("serve", help="Запустить веб-интерфейс")
    serve.add_argument("--data", default="data/test_dataset.csv")
    serve.add_argument("--train", default="data/train_dataset.csv")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--debug", action="store_true")
    serve.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать веб-интерфейс в браузере автоматически",
    )

    validate = subparsers.add_parser("validate", help="Проверить алгоритм на synthetic gaps")
    validate.add_argument("--train", default="data/train_dataset.csv")
    validate.add_argument("--sample-size", type=int, default=3000)
    validate.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "predict":
        result = create_submission(
            input_path=Path(args.input),
            train_path=Path(args.train),
            output_path=Path(args.output),
            method=args.method,
            target_column=args.target_column,
        )
        print(
            f"Submission создан: {result.path} | "
            f"строк: {result.rows} | диапазон: "
            f"{result.min_prediction:.6f}..{result.max_prediction:.6f}"
        )
        return

    if args.command == "validate":
        pipeline = AnalysisPipeline.from_csv(Path(args.train))
        metrics = pipeline.validate(sample_size=args.sample_size, seed=args.seed)
        print(f"Baseline RMSE: {metrics['baseline_rmse']:.6f}")
        print(f"ML RMSE: {metrics['ml_rmse']:.6f}")
        return

    if args.command == "serve":
        from src.webapp import run_server

        run_server(
            data_path=Path(args.data),
            train_path=Path(args.train),
            host=args.host,
            port=args.port,
            debug=args.debug,
            open_browser=not args.no_browser,
        )


if __name__ == "__main__":
    main()
