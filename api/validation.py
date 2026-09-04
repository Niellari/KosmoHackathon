"""Проверка submission до открытия браузера."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import math
from pathlib import Path

from api.config import ValidationConfig


class SubmissionValidationError(ValueError):
    """Файл нельзя безопасно отправить на платформу."""


@dataclass(frozen=True)
class ValidatedSubmission:
    path: Path
    sha256: str
    rows: int


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise SubmissionValidationError("CSV не содержит заголовок")
            return list(reader.fieldnames), list(reader)
    except UnicodeDecodeError as error:
        raise SubmissionValidationError("CSV должен быть в кодировке UTF-8") from error


def _expected_keys(path: Path, identity_columns: tuple[str, ...]) -> set[tuple[str, ...]]:
    headers, rows = _read_csv(path)
    required = {*identity_columns, "is_synthetic_gap"}
    missing = required.difference(headers)
    if missing:
        raise SubmissionValidationError(
            f"В тестовом датасете отсутствуют колонки: {sorted(missing)}"
        )
    truthy = {"1", "true", "yes", "y", "да"}
    return {
        tuple(row[column].strip() for column in identity_columns)
        for row in rows
        if row["is_synthetic_gap"].strip().lower() in truthy
    }


def validate_submission_file(
    path: Path, config: ValidationConfig
) -> ValidatedSubmission:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SubmissionValidationError(f"Submission-файл не найден: {path}")
    if path.suffix.lower() != ".csv":
        raise SubmissionValidationError("Submission должен иметь расширение .csv")

    headers, rows = _read_csv(path)
    if headers != list(config.expected_columns):
        raise SubmissionValidationError(
            f"Неверные колонки: {headers}; ожидаются {list(config.expected_columns)}"
        )
    if not rows:
        raise SubmissionValidationError("Submission не содержит строк данных")

    seen: set[tuple[str, ...]] = set()
    for number, row in enumerate(rows, start=2):
        key = tuple(row[column].strip() for column in config.identity_columns)
        if not all(key):
            raise SubmissionValidationError(f"Пустой ключ в строке {number}")
        if key in seen:
            raise SubmissionValidationError(f"Повторяющийся ключ в строке {number}: {key}")
        seen.add(key)

        if "date" in config.identity_columns:
            try:
                date.fromisoformat(row["date"].strip())
            except ValueError as error:
                raise SubmissionValidationError(
                    f"Неверная дата в строке {number}: {row['date']!r}"
                ) from error

        try:
            prediction = float(row[config.target_column])
        except ValueError as error:
            raise SubmissionValidationError(
                f"Предсказание в строке {number} не является числом"
            ) from error
        if not math.isfinite(prediction):
            raise SubmissionValidationError(
                f"Предсказание в строке {number} не является конечным числом"
            )

    if config.test_data_path is not None:
        if not config.test_data_path.is_file():
            raise SubmissionValidationError(
                f"Тестовый датасет не найден: {config.test_data_path}"
            )
        expected = _expected_keys(config.test_data_path, config.identity_columns)
        if seen != expected:
            missing = len(expected - seen)
            extra = len(seen - expected)
            raise SubmissionValidationError(
                "Ключи submission не совпадают с synthetic gaps: "
                f"пропущено {missing}, лишних {extra}"
            )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ValidatedSubmission(path=path, sha256=digest, rows=len(rows))

