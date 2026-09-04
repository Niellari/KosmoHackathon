"""Загрузка конфигурации браузерного отправщика."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class SubmitterConfigError(ValueError):
    """Конфигурация отправщика отсутствует или некорректна."""


@dataclass(frozen=True)
class BrowserConfig:
    headless: bool
    profile_dir: Path
    profile_name: str
    cookies_path: Path | None
    settle_delay: float
    page_timeout: int
    login_timeout: int
    cooldown_timeout: int
    cooldown_poll_interval: float
    result_timeout: int
    result_settle_delay: float
    driver_log_path: Path


@dataclass(frozen=True)
class SelectorConfig:
    file_input: str
    submit_button: str
    success_marker: str | None
    failure_marker: str | None
    submission_id: str | None
    score: str | None
    result_row: str
    cooldown_alert: str | None = None
    cooldown_timer: str | None = None


@dataclass(frozen=True)
class AuthenticationConfig:
    credentials_path: Path
    email_selector: str
    password_selector: str
    submit_button_selector: str
    error_selector: str | None
    input_delay: float
    manual_login_timeout: int


@dataclass(frozen=True)
class ValidationConfig:
    expected_columns: tuple[str, ...]
    identity_columns: tuple[str, ...]
    target_column: str
    test_data_path: Path | None


@dataclass(frozen=True)
class SubmitterConfig:
    submission_url: str
    browser: BrowserConfig
    authentication: AuthenticationConfig
    selectors: SelectorConfig
    validation: ValidationConfig
    history_path: Path
    log_path: Path


def _required(mapping: dict, key: str, section: str) -> object:
    value = mapping.get(key)
    if value in (None, ""):
        raise SubmitterConfigError(f"В секции {section!r} не задано поле {key!r}")
    return value


def _optional_selector(mapping: dict, key: str) -> str | None:
    value = mapping.get(key)
    return str(value) if value else None


def _project_path(project_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def load_submitter_config(path: Path) -> SubmitterConfig:
    path = path.resolve()
    if not path.exists():
        raise SubmitterConfigError(f"Конфигурация не найдена: {path}")

    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise SubmitterConfigError("Корень конфигурации должен быть YAML-объектом")

    project_root = path.parent.parent
    browser_raw = raw.get("browser") or {}
    selectors_raw = raw.get("selectors") or {}
    authentication_raw = raw.get("authentication") or {}
    validation_raw = raw.get("validation") or {}
    output_raw = raw.get("output") or {}
    if not all(
        isinstance(value, dict)
        for value in (
            browser_raw,
            selectors_raw,
            authentication_raw,
            validation_raw,
            output_raw,
        )
    ):
        raise SubmitterConfigError("Секции конфигурации должны быть YAML-объектами")

    submission_url = str(_required(raw, "submission_url", "root"))
    if submission_url.startswith("CHANGE_ME"):
        raise SubmitterConfigError(
            "Укажите реальный submission_url в api/config.yaml"
        )

    expected_columns = tuple(validation_raw.get("expected_columns") or ())
    identity_columns = tuple(validation_raw.get("identity_columns") or ())
    target_column = str(_required(validation_raw, "target_column", "validation"))
    if not expected_columns:
        raise SubmitterConfigError("validation.expected_columns не может быть пустым")
    if not identity_columns:
        raise SubmitterConfigError("validation.identity_columns не может быть пустым")
    unknown_identity = set(identity_columns).difference(expected_columns)
    if unknown_identity:
        raise SubmitterConfigError(
            "validation.identity_columns содержит неизвестные колонки: "
            f"{sorted(unknown_identity)}"
        )
    if target_column not in expected_columns:
        raise SubmitterConfigError(
            "validation.target_column должен входить в expected_columns"
        )

    test_path_value = validation_raw.get("test_data_path")
    return SubmitterConfig(
        submission_url=submission_url,
        browser=BrowserConfig(
            headless=bool(browser_raw.get("headless", False)),
            profile_dir=_project_path(
                project_root,
                browser_raw.get(
                    "profile_dir", "artifacts/submissions/session/chrome-profile"
                ),
            ),
            profile_name=str(browser_raw.get("profile_name", "Default")),
            cookies_path=(
                _project_path(project_root, browser_raw["cookies_path"])
                if browser_raw.get("cookies_path")
                else None
            ),
            settle_delay=float(browser_raw.get("settle_delay", 1.0)),
            page_timeout=int(browser_raw.get("page_timeout", 30)),
            login_timeout=int(browser_raw.get("login_timeout", 180)),
            cooldown_timeout=int(browser_raw.get("cooldown_timeout", 90)),
            cooldown_poll_interval=float(
                browser_raw.get("cooldown_poll_interval", 1.0)
            ),
            result_timeout=int(browser_raw.get("result_timeout", 180)),
            result_settle_delay=float(browser_raw.get("result_settle_delay", 4.0)),
            driver_log_path=_project_path(
                project_root,
                browser_raw.get(
                    "driver_log_path",
                    "artifacts/submissions/logs/chromedriver.log",
                ),
            ),
        ),
        authentication=AuthenticationConfig(
            credentials_path=_project_path(
                project_root,
                authentication_raw.get("credentials_path", "api/credentials.env"),
            ),
            email_selector=str(
                _required(authentication_raw, "email_selector", "authentication")
            ),
            password_selector=str(
                _required(authentication_raw, "password_selector", "authentication")
            ),
            submit_button_selector=str(
                _required(
                    authentication_raw,
                    "submit_button_selector",
                    "authentication",
                )
            ),
            error_selector=_optional_selector(authentication_raw, "error_selector"),
            input_delay=float(authentication_raw.get("input_delay", 0.25)),
            manual_login_timeout=int(
                authentication_raw.get("manual_login_timeout", 300)
            ),
        ),
        selectors=SelectorConfig(
            file_input=str(_required(selectors_raw, "file_input", "selectors")),
            submit_button=str(
                _required(selectors_raw, "submit_button", "selectors")
            ),
            success_marker=_optional_selector(selectors_raw, "success_marker"),
            failure_marker=_optional_selector(selectors_raw, "failure_marker"),
            submission_id=_optional_selector(selectors_raw, "submission_id"),
            score=_optional_selector(selectors_raw, "score"),
            result_row=str(
                _required(selectors_raw, "result_row", "selectors")
            ),
            cooldown_alert=_optional_selector(
                selectors_raw, "cooldown_alert"
            ),
            cooldown_timer=_optional_selector(
                selectors_raw, "cooldown_timer"
            ),
        ),
        validation=ValidationConfig(
            expected_columns=expected_columns,
            identity_columns=identity_columns,
            target_column=target_column,
            test_data_path=(
                _project_path(project_root, test_path_value)
                if test_path_value
                else None
            ),
        ),
        history_path=_project_path(
            project_root,
            output_raw.get("history_path", "artifacts/submissions/history.jsonl"),
        ),
        log_path=_project_path(
            project_root,
            output_raw.get("log_path", "artifacts/submissions/logs/submitter.log"),
        ),
    )
