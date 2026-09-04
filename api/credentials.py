"""Чтение реквизитов входа из лок игнорируемого env-файла."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class CredentialsError(ValueError):
    """Реквизиты входа отсутствуют или не заполнены."""


@dataclass(frozen=True)
class Credentials:
    email: str
    password: str


def load_credentials(path: Path) -> Credentials:
    if not path.is_file():
        raise CredentialsError(
            f"Файл реквизитов не найден: {path}. "
            "Скопируйте api/credentials.env.example в api/credentials.env"
        )

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    email = values.get("COSMO_EMAIL", "")
    password = values.get("COSMO_PASSWORD", "")
    placeholders = {"", "your-email@example.com", "your-password"}
    if email in placeholders or password in placeholders:
        raise CredentialsError(
            f"Заполните COSMO_EMAIL и COSMO_PASSWORD в {path}"
        )
    return Credentials(email=email, password=password)
