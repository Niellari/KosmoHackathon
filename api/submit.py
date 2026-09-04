"""CLI для проверки и отправки submission.csv через Selenium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from api.config import SubmitterConfigError, load_submitter_config
from api.credentials import CredentialsError, load_credentials
from api.logging_setup import configure_file_logging, get_logger
from api.result import SubmissionReceipt, append_history, was_submitted
from api.selenium_client import SeleniumUnavailableError, create_driver
from api.session import CookieStore
from api.submission_page import PlatformSubmissionError, SubmissionPage
from api.validation import SubmissionValidationError, validate_submission_file


class DuplicateSubmissionError(RuntimeError):
    """Этот файл уже был успешно отправлен."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Отправить конкурсный submission через Selenium"
    )
    parser.add_argument("file", type=Path, help="Путь к submission.csv")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Конфигурация платформы",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Загрузить файл в форму, но не нажимать кнопку отправки",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--manual-login",
        action="store_true",
        help="Выполнить вход вручную и сохранить полученную сессию",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Очистить cookies и web storage платформы перед входом",
    )
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="Вывести машинно-читаемый результат"
    )
    return parser


def run(args: argparse.Namespace) -> SubmissionReceipt:
    config = load_submitter_config(args.config)
    logger = configure_file_logging(config.log_path)
    logger.info(
        "Запуск submitter: file=%s dry_run=%s manual_login=%s reset_session=%s",
        args.file,
        args.dry_run,
        args.manual_login,
        args.reset_session,
    )
    validated = validate_submission_file(args.file, config.validation)
    logger.info(
        "Submission проверен: rows=%s sha256=%s",
        validated.rows,
        validated.sha256,
    )
    if (
        not args.dry_run
        and not args.allow_duplicate
        and was_submitted(config.history_path, validated.sha256)
    ):
        raise DuplicateSubmissionError(
            "Этот файл уже был отправлен. Для повторной отправки используйте "
            "--allow-duplicate"
        )

    driver = create_driver(
        config.browser,
        force_headless=args.headless,
        detach=args.dry_run,
    )
    page = SubmissionPage(driver, config.selectors)
    cookie_store = (
        CookieStore(config.browser.cookies_path)
        if config.browser.cookies_path is not None
        else None
    )
    try:
        page.open(
            config.submission_url,
            timeout=config.browser.page_timeout,
            settle_delay=config.browser.settle_delay,
        )
        if args.reset_session:
            logger.info("Очистка cookies и web storage текущего домена")
            driver.delete_all_cookies()
            try:
                driver.execute_script("window.localStorage.clear();")
                driver.execute_script("window.sessionStorage.clear();")
            except Exception:
                logger.warning("Не удалось полностью очистить web storage")
            page.open(
                config.submission_url,
                timeout=config.browser.page_timeout,
                settle_delay=config.browser.settle_delay,
            )
            logger.info("Чистая сессия платформы создана")
        restored_cookies = cookie_store.restore(driver) if cookie_store else 0
        if restored_cookies:
            driver.refresh()
            page.open(
                config.submission_url,
                timeout=config.browser.page_timeout,
                settle_delay=config.browser.settle_delay,
            )
        authenticated = False
        if page.requires_authentication(config.authentication):
            if args.manual_login:
                if args.headless or config.browser.headless:
                    raise PlatformSubmissionError(
                        "Ручная авторизация недоступна в headless-режиме"
                    )
                print(
                    "Выполните вход вручную в открытом окне браузера...",
                    file=sys.stderr,
                )
                page.wait_for_manual_authentication(
                    config.authentication,
                    timeout=config.authentication.manual_login_timeout,
                )
                authenticated = True
            else:
                credentials = load_credentials(
                    config.authentication.credentials_path
                )
                authenticated = page.authenticate_if_needed(
                    config.authentication,
                    credentials,
                    timeout=config.browser.page_timeout,
                )
        if authenticated and cookie_store:
            cookie_store.save(driver)
        page.upload(validated.path, timeout=config.browser.login_timeout)
        if args.dry_run:
            receipt = SubmissionReceipt.create(
                status="dry-run",
                file=str(validated.path),
                sha256=validated.sha256,
                rows=validated.rows,
                url=driver.current_url,
                message=(
                    "Файл выбран в форме; финальная кнопка не нажата; "
                    "браузер оставлен открытым"
                ),
            )
        else:
            previous_url, previous_result, submitted_button = page.submit(
                timeout=config.browser.page_timeout
            )
            platform_result = page.wait_for_result(
                timeout=config.browser.result_timeout,
                previous_url=previous_url,
                previous_result=previous_result,
                submitted_button=submitted_button,
                settle_delay=config.browser.result_settle_delay,
            )
            receipt = SubmissionReceipt.create(
                status="submitted",
                file=str(validated.path),
                sha256=validated.sha256,
                rows=validated.rows,
                url=driver.current_url,
                platform_date=platform_result.date,
                platform_status=platform_result.status,
                score=platform_result.metric,
            )
        append_history(config.history_path, receipt)
        logger.info("Сценарий завершён: status=%s url=%s", receipt.status, receipt.url)
        return receipt
    except Exception:
        logger.exception("Сценарий завершился с ошибкой")
        raise
    finally:
        try:
            if cookie_store:
                cookie_store.save(driver)
        except Exception:
            # Ошибка сохранения cookies не должна скрывать результат отправки.
            pass
        if args.dry_run:
            logger.info("Dry-run завершён; браузер оставлен открытым")
        else:
            driver.quit()


def main() -> None:
    args = build_parser().parse_args()
    try:
        receipt = run(args)
    except (
        SubmitterConfigError,
        CredentialsError,
        SubmissionValidationError,
        DuplicateSubmissionError,
        SeleniumUnavailableError,
        PlatformSubmissionError,
    ) as error:
        get_logger().error("Ошибка запуска submitter: %s", error)
        if args.json:
            print(
                json.dumps(
                    {"status": "error", "error": str(error)}, ensure_ascii=False
                )
            )
        else:
            print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if args.json:
        print(json.dumps(receipt.as_dict(), ensure_ascii=False))
    else:
        print(f"Статус: {receipt.status}")
        print(f"Файл: {receipt.file}")
        print(f"Строк: {receipt.rows}")
        if receipt.submission_id:
            print(f"ID отправки: {receipt.submission_id}")
        if receipt.platform_date:
            print(f"Дата на платформе: {receipt.platform_date}")
        if receipt.platform_status:
            print(f"Статус платформы: {receipt.platform_status}")
        if receipt.score:
            print(f"Метрика: {receipt.score}")


if __name__ == "__main__":
    main()
