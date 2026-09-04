"""Модель результата и журнал отправок."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class SubmissionReceipt:
    status: str
    file: str
    sha256: str
    rows: int
    submitted_at: str
    url: str
    submission_id: str | None = None
    platform_date: str | None = None
    platform_status: str | None = None
    score: str | None = None
    message: str | None = None

    @classmethod
    def create(cls, **kwargs) -> "SubmissionReceipt":
        return cls(
            submitted_at=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def was_submitted(history_path: Path, digest: str) -> bool:
    if not history_path.exists():
        return False
    with history_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("sha256") == digest and item.get("status") == "submitted":
                return True
    return False


def append_history(history_path: Path, receipt: SubmissionReceipt) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt.as_dict(), ensure_ascii=False) + "\n")
