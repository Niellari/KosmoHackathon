"""Изолированное сохранение и восстановление cookies Selenium."""

from __future__ import annotations

import json
import os
from pathlib import Path


class CookieStore:
    def __init__(self, path: Path):
        self.path = path

    def restore(self, driver) -> int:
        """Добавляет сохранённые cookies после открытия домена платформы."""

        if not self.path.exists():
            return 0
        try:
            cookies = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(cookies, list):
            return 0

        restored = 0
        for cookie in cookies:
            if not isinstance(cookie, dict) or not cookie.get("name"):
                continue
            normalized = dict(cookie)
            # Chrome иногда возвращает поля, которые add_cookie не принимает.
            normalized.pop("sameParty", None)
            normalized.pop("sourcePort", None)
            normalized.pop("sourceScheme", None)
            try:
                driver.add_cookie(normalized)
                restored += 1
            except Exception:
                # Истёкшая cookie или cookie другого поддомена не должна
                # препятствовать восстановлению остальных данных сессии.
                continue
        return restored

    def save(self, driver) -> int:
        cookies = driver.get_cookies()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass

        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(cookies, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        temporary_path.replace(self.path)
        return len(cookies)

