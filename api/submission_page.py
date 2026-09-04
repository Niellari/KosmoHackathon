"""Page Object страницы загрузки решения."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from api.config import AuthenticationConfig, SelectorConfig
from api.credentials import Credentials
from api.logging_setup import get_logger


class PlatformSubmissionError(RuntimeError):
    """Платформа отклонила файл или не показала ожидаемый результат."""


@dataclass(frozen=True)
class PlatformResult:
    date: str
    status: str
    metric: str


class SubmissionPage:
    def __init__(self, driver, selectors: SelectorConfig):
        self.driver = driver
        self.selectors = selectors

    def open(self, url: str, timeout: int, settle_delay: float) -> None:
        from selenium.webdriver.support.ui import WebDriverWait

        logger = get_logger()
        logger.info("Переход на страницу: %s", url)
        self.driver.get(url)
        WebDriverWait(self.driver, timeout).until(
            lambda driver: driver.execute_script("return document.readyState")
            == "complete"
        )
        time.sleep(settle_delay)
        logger.info(
            "Страница загружена: url=%s title=%r",
            self.driver.current_url,
            self.driver.title,
        )

    def authenticate_if_needed(
        self,
        config: AuthenticationConfig,
        credentials: Credentials | None,
        timeout: int,
    ) -> bool:
        """Авторизуется, только если на странице присутствует форма входа."""

        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait

        email_fields = self.driver.find_elements(By.CSS_SELECTOR, config.email_selector)
        password_fields = self.driver.find_elements(
            By.CSS_SELECTOR, config.password_selector
        )
        if not email_fields and not password_fields:
            get_logger().info("Форма входа отсутствует: используется текущая сессия")
            return False
        if not email_fields or not password_fields:
            raise PlatformSubmissionError("Форма входа отображается не полностью")
        if credentials is None:
            raise PlatformSubmissionError("Для авторизации не переданы реквизиты")

        email_field = email_fields[0]
        password_field = password_fields[0]
        email_field.clear()
        email_field.send_keys(credentials.email)
        time.sleep(config.input_delay)
        password_field.clear()
        password_field.send_keys(credentials.password)
        time.sleep(config.input_delay)

        buttons = self.driver.find_elements(
            By.CSS_SELECTOR, config.submit_button_selector
        )
        if not buttons:
            raise PlatformSubmissionError("Кнопка входа не найдена")
        buttons[0].click()
        get_logger().info("Форма входа отправлена; ожидается переход")

        # Обычная HTML-форма делает полноценный POST-переход. В короткий момент
        # между старым и новым document Chrome отвечает `target frame detached`.
        # Это нормальная часть навигации, а не ошибка авторизации.
        def login_finished(driver) -> bool:
            try:
                if driver.execute_script("return document.readyState") != "complete":
                    return False
                return not driver.find_elements(
                    By.CSS_SELECTOR, config.email_selector
                )
            except WebDriverException:
                return False

        try:
            WebDriverWait(self.driver, timeout, poll_frequency=0.25).until(login_finished)
        except TimeoutException as error:
            platform_message = self._authentication_error(config)
            suffix = f": {platform_message}" if platform_message else ""
            raise PlatformSubmissionError(
                "После отправки формы платформа снова показала страницу входа"
                f"{suffix}. Проверьте логин и пароль"
            ) from error
        return True

    def _authentication_error(self, config: AuthenticationConfig) -> str | None:
        if not config.error_selector:
            return None
        try:
            from selenium.webdriver.common.by import By

            for element in self.driver.find_elements(
                By.CSS_SELECTOR, config.error_selector
            ):
                message = element.text.strip()
                if message:
                    return message
        except Exception:
            return None
        return None

    def requires_authentication(self, config: AuthenticationConfig) -> bool:
        from selenium.webdriver.common.by import By

        return bool(
            self.driver.find_elements(By.CSS_SELECTOR, config.email_selector)
            or self.driver.find_elements(By.CSS_SELECTOR, config.password_selector)
        )

    def wait_for_manual_authentication(
        self, config: AuthenticationConfig, timeout: int
    ) -> None:
        """Ждёт, пока пользователь самостоятельно завершит вход в браузере."""

        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait

        def login_finished(driver) -> bool:
            try:
                if driver.execute_script("return document.readyState") != "complete":
                    return False
                return not driver.find_elements(
                    By.CSS_SELECTOR, config.email_selector
                )
            except WebDriverException:
                return False

        try:
            WebDriverWait(self.driver, timeout, poll_frequency=0.25).until(login_finished)
        except TimeoutException as error:
            raise PlatformSubmissionError(
                f"Ручная авторизация не завершена за {timeout} секунд"
            ) from error
        get_logger().info("Ручная авторизация завершена")

    def upload(self, file_path: Path, timeout: int) -> None:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as conditions
        from selenium.webdriver.support.ui import WebDriverWait

        try:
            element = WebDriverWait(self.driver, timeout).until(
                conditions.presence_of_element_located(
                    (By.CSS_SELECTOR, self.selectors.file_input)
                )
            )
        except TimeoutException as error:
            raise PlatformSubmissionError(
                "Поле загрузки не найдено. Войдите на платформу в открытом "
                "браузере или обновите selectors.file_input"
            ) from error
        element.send_keys(str(file_path))
        get_logger().info("Файл выбран в форме: %s", file_path)

    def submit(self, timeout: int):
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as conditions
        from selenium.webdriver.support.ui import WebDriverWait

        try:
            button = WebDriverWait(self.driver, timeout).until(
                conditions.element_to_be_clickable(
                    (By.CSS_SELECTOR, self.selectors.submit_button)
                )
            )
        except TimeoutException as error:
            raise PlatformSubmissionError(
                "Кнопка отправки не найдена или недоступна. Обновите "
                "selectors.submit_button"
            ) from error
        previous_url = self.driver.current_url
        previous_result = self._read_latest_result()
        button.click()
        get_logger().info("Нажата кнопка отправки submission")
        return previous_url, previous_result, button

    def wait_for_result(
        self,
        timeout: int,
        previous_url: str,
        previous_result: PlatformResult | None,
        submitted_button,
        settle_delay: float,
    ) -> PlatformResult:
        from selenium.common.exceptions import (
            StaleElementReferenceException,
            TimeoutException,
            WebDriverException,
        )
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait

        def result_is_ready(driver):
            try:
                failure = self._failure_message()
                if failure:
                    raise PlatformSubmissionError(failure)
                if self.selectors.success_marker and driver.find_elements(
                    By.CSS_SELECTOR, self.selectors.success_marker
                ):
                    return True
                if driver.current_url != previous_url:
                    return True
                # При POST на тот же URL адрес не меняется, но исходный DOM и
                # кнопка формы становятся устаревшими.
                submitted_button.is_enabled()
                return False
            except StaleElementReferenceException:
                return True
            except WebDriverException:
                # Во время полноценной навигации документ кратковременно
                # отсоединяется; продолжаем ждать новый DOM.
                return False

        try:
            WebDriverWait(self.driver, timeout).until(result_is_ready)
        except TimeoutException as error:
            raise PlatformSubmissionError(
                "После нажатия кнопки платформа не начала переход и не показала "
                "результат за отведённое время"
            ) from error

        WebDriverWait(self.driver, timeout).until(
            lambda driver: driver.execute_script("return document.readyState")
            == "complete"
        )
        time.sleep(settle_delay)
        failure = self._failure_message()
        if failure:
            raise PlatformSubmissionError(failure)
        get_logger().info(
            "Платформа ответила на отправку: url=%s title=%r",
            self.driver.current_url,
            self.driver.title,
        )

        deadline = time.monotonic() + timeout
        while True:
            latest = self._read_latest_result()
            if latest is not None and latest != previous_result:
                if latest.status.lower() == "completed" and latest.metric:
                    get_logger().info(
                        "Получена метрика: date=%s status=%s metric=%s",
                        latest.date,
                        latest.status,
                        latest.metric,
                    )
                    return latest
                if latest.status.lower() in {"failed", "error", "rejected"}:
                    raise PlatformSubmissionError(
                        "Платформа завершила обработку со статусом "
                        f"{latest.status!r}"
                    )

            if time.monotonic() + settle_delay >= deadline:
                raise PlatformSubmissionError(
                    "Новое решение не появилось в верхней строке таблицы "
                    f"за {timeout} секунд"
                )

            # Загружаем исходную GET-страницу, а не обновляем ответ POST — это
            # исключает повторную отправку формы браузером.
            self.driver.get(previous_url)
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script("return document.readyState")
                == "complete"
            )
            time.sleep(settle_delay)

    def _read_latest_result(self) -> PlatformResult | None:
        from selenium.webdriver.common.by import By

        rows = self.driver.find_elements(By.CSS_SELECTOR, self.selectors.result_row)
        if not rows:
            return None
        cells = rows[0].find_elements(By.CSS_SELECTOR, "td")
        if len(cells) < 3:
            raise PlatformSubmissionError(
                "В верхней строке таблицы результата меньше трёх ячеек"
            )
        return PlatformResult(
            date=cells[0].text.strip(),
            status=cells[1].text.strip(),
            metric=cells[2].text.strip(),
        )

    def _failure_message(self) -> str | None:
        if not self.selectors.failure_marker:
            return None
        from selenium.webdriver.common.by import By

        for element in self.driver.find_elements(
            By.CSS_SELECTOR, self.selectors.failure_marker
        ):
            message = element.text.strip()
            if message:
                return message
        return None

    def _optional_text(self, selector: str | None) -> str | None:
        if not selector:
            return None
        from selenium.webdriver.common.by import By

        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
        if not elements:
            return None
        return elements[0].text.strip() or None
