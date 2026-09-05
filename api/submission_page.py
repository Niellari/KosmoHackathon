"""Page Object страницы загрузки решения."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
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

    def wait_until_submission_allowed(
        self, timeout: int, poll_interval: float
    ) -> None:
        """Wait until the upload form appears after a platform cooldown."""

        from selenium.common.exceptions import (
            StaleElementReferenceException,
            TimeoutException,
            WebDriverException,
        )
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait

        forms = self.driver.find_elements(
            By.CSS_SELECTOR, self.selectors.file_input
        )
        if forms:
            get_logger().info("Форма отправки доступна; кулдаун отсутствует")
            return

        alerts = (
            self.driver.find_elements(
                By.CSS_SELECTOR, self.selectors.cooldown_alert
            )
            if self.selectors.cooldown_alert
            else []
        )
        remaining_text = None
        if alerts and self.selectors.cooldown_timer:
            timers = self.driver.find_elements(
                By.CSS_SELECTOR, self.selectors.cooldown_timer
            )
            if timers:
                remaining_text = timers[0].text.strip() or None
        until = alerts[0].get_attribute("data-until") if alerts else None
        remaining_seconds = None
        try:
            remaining_seconds = max(0, int(float(until) - time.time() + 0.999))
        except (TypeError, ValueError):
            pass

        shown_remaining = (
            f"{remaining_seconds} с"
            if remaining_seconds is not None
            else remaining_text or "неизвестно"
        )
        if alerts:
            get_logger().info(
                "Активен кулдаун отправки: осталось %s; ждём появления формы",
                shown_remaining,
            )
            print(
                f"Активен кулдаун платформы, осталось {shown_remaining}. "
                "Ожидаю форму отправки...",
                file=sys.stderr,
                flush=True,
            )
        else:
            get_logger().warning(
                "Форма отправки отсутствует без предупреждения о кулдауне"
            )
            print(
                "Форма отправки пока отсутствует. Ожидаю...",
                file=sys.stderr,
                flush=True,
            )

        def submission_form_appeared(driver) -> bool:
            try:
                return bool(
                    driver.find_elements(
                        By.CSS_SELECTOR, self.selectors.file_input
                    )
                )
            except (StaleElementReferenceException, WebDriverException):
                # Во время location.reload старый document отсоединяется.
                return False

        try:
            WebDriverWait(
                self.driver, timeout, poll_frequency=poll_interval
            ).until(
                submission_form_appeared
            )
        except TimeoutException as error:
            raise PlatformSubmissionError(
                f"Форма отправки не появилась за {timeout} секунд. "
                "Возможно, страница не перезагрузилась или платформа "
                "продлила блокировку"
            ) from error
        get_logger().info("Форма появилась; отправка доступна")

    def submit(self, timeout: int, post_click_delay: float):
        from selenium.common.exceptions import (
            ElementClickInterceptedException,
            TimeoutException,
            WebDriverException,
        )
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
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
                button,
            )
            button.click()
        except ElementClickInterceptedException:
            # После восстановления постоянной Chrome-сессии браузер иногда
            # сохраняет старую позицию прокрутки, и таблица перекрывает кнопку.
            # Повторно находим элемент и выполняем DOM-click после центрирования.
            get_logger().warning(
                "Обычный клик по кнопке перехвачен; повторяем через DOM-click"
            )
            button = WebDriverWait(self.driver, timeout).until(
                conditions.element_to_be_clickable(
                    (By.CSS_SELECTOR, self.selectors.submit_button)
                )
            )
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'center'});"
                "arguments[0].click();",
                button,
            )
        except WebDriverException as error:
            # Chrome иногда сообщает об отсоединённом frame уже после того, как
            # POST был принят. В этом случае продолжаем и проверяем таблицу.
            message = str(error).lower()
            transient_navigation_error = (
                "target frame detached" in message
                or "unable to receive message from renderer" in message
            )
            if not transient_navigation_error:
                raise
            get_logger().warning(
                "Chrome отсоединил старый document после клика; "
                "продолжаем ожидать результат"
            )
        finally:
            # Пауза гарантированно начинается непосредственно после попытки
            # клика, даже если Chrome вернул ошибку во время POST-навигации.
            get_logger().info(
                "После отправки ждём %.1f секунды перед проверкой результата",
                post_click_delay,
            )
            time.sleep(post_click_delay)
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
                latest = self._read_latest_result()
                if latest is not None and latest != previous_result:
                    return True
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

        def document_is_ready(driver) -> bool:
            try:
                return (
                    driver.execute_script("return document.readyState") == "complete"
                )
            except WebDriverException:
                return False

        WebDriverWait(self.driver, timeout, poll_frequency=0.25).until(
            document_is_ready
        )
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
