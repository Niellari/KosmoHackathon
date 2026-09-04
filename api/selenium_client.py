"""Создание Selenium WebDriver с настроенным постоянным профилем Chrome."""

from __future__ import annotations

from api.config import BrowserConfig
from api.logging_setup import get_logger


class SeleniumUnavailableError(RuntimeError):
    """Selenium или совместимый браузер недоступен."""


def create_driver(
    config: BrowserConfig,
    force_headless: bool = False,
    detach: bool = False,
):
    try:
        from selenium import webdriver
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.chrome.service import Service
    except ImportError as error:
        raise SeleniumUnavailableError(
            "Selenium не установлен. Выполните: "
            "python -m pip install -r requirements-submit.txt"
        ) from error

    profile_lock = config.profile_dir / "SingletonLock"
    if profile_lock.exists() or profile_lock.is_symlink():
        raise SeleniumUnavailableError(
            "Профиль Selenium сейчас используется другим процессом Chrome. "
            "Закройте окно предыдущего запуска и повторите submit.sh"
        )
    config.profile_dir.mkdir(parents=True, exist_ok=True)
    config.driver_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger()
    logger.info(
        "Запуск Chrome: profile_dir=%s profile_name=%s headless=%s",
        config.profile_dir,
        config.profile_name,
        config.headless or force_headless,
    )
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", detach)
    options.add_argument(f"--user-data-dir={config.profile_dir}")
    options.add_argument(f"--profile-directory={config.profile_name}")
    options.add_argument("--no-first-run")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1440,1000")
    if config.headless or force_headless:
        options.add_argument("--headless=new")

    try:
        service = Service(log_output=str(config.driver_log_path))
        driver = webdriver.Chrome(options=options, service=service)
        driver.set_page_load_timeout(config.page_timeout)
        logger.info("ChromeDriver подключён, session_id=%s", driver.session_id)
        return driver
    except WebDriverException as error:
        logger.exception("ChromeDriver не смог создать браузерную сессию")
        raise SeleniumUnavailableError(
            "Не удалось запустить Chrome/Chromium. Проверьте установку браузера "
            "и закройте другой процесс с тем же профилем."
        ) from error
