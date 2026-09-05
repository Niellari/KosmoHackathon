#!/usr/bin/env python3
"""Gracefully close Chrome that owns this repository's submitter profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE = (
    REPOSITORY_ROOT / "artifacts/submissions/session/chrome-profile"
).resolve()
CHROME_EXECUTABLE_MARKERS = ("chrome", "chromium")


def _find_windows_profile_chrome_processes(profile_dir: Path) -> list[int]:
    """Return matching Chrome PIDs using Windows process command lines."""

    command = (
        "Get-CimInstance Win32_Process "
        "| Where-Object { $_.Name -match '^(chrome|chromium)\\.exe$' } "
        "| Select-Object ProcessId, CommandLine "
        "| ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "неизвестная ошибка PowerShell"
        raise RuntimeError(f"Не удалось проверить процессы Chrome: {message}")
    if not result.stdout.strip():
        return []

    payload = json.loads(result.stdout)
    processes = payload if isinstance(payload, list) else [payload]
    expected = str(profile_dir.resolve()).casefold()
    pattern = re.compile(r'--user-data-dir=(?:"([^"]+)"|(\S+))', re.IGNORECASE)
    current_pid = os.getpid()
    found: list[int] = []
    for process in processes:
        pid = int(process.get("ProcessId") or 0)
        command_line = process.get("CommandLine") or ""
        match = pattern.search(command_line)
        if pid == current_pid or not match:
            continue
        configured_profile = match.group(1) or match.group(2)
        if str(Path(configured_profile).resolve()).casefold() == expected:
            found.append(pid)
    return sorted(found)


def _read_cmdline(path: Path) -> tuple[str, ...]:
    try:
        content = path.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ()
    return tuple(
        item.decode("utf-8", errors="replace")
        for item in content.split(b"\0")
        if item
    )


def find_profile_chrome_processes(
    profile_dir: Path, proc_root: Path = Path("/proc")
) -> list[int]:
    """Return Chrome PIDs whose exact user-data-dir is the submitter profile."""

    if os.name == "nt" and proc_root == Path("/proc"):
        return _find_windows_profile_chrome_processes(profile_dir)

    expected_argument = f"--user-data-dir={profile_dir.resolve()}"
    current_pid = os.getpid()
    found: list[int] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == current_pid:
            continue
        arguments = _read_cmdline(process_dir / "cmdline")
        if not arguments or expected_argument not in arguments:
            continue
        executable = Path(arguments[0]).name.lower()
        if not any(marker in executable for marker in CHROME_EXECUTABLE_MARKERS):
            continue
        found.append(pid)
    return sorted(found)


def _remove_stale_lock(profile_dir: Path) -> bool:
    lock = profile_dir / "SingletonLock"
    if not (lock.exists() or lock.is_symlink()):
        return False
    lock.unlink()
    return True


def close_profile_chrome(profile_dir: Path, timeout: float) -> list[int]:
    pids = find_profile_chrome_processes(profile_dir)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            raise RuntimeError(
                f"Нет разрешения завершить Chrome-процесс PID {pid}"
            ) from error

    deadline = time.monotonic() + timeout
    remaining = find_profile_chrome_processes(profile_dir)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = find_profile_chrome_processes(profile_dir)
    if remaining:
        raise RuntimeError(
            "Chrome сабмиттера не завершился после SIGTERM; оставшиеся PID: "
            + ", ".join(map(str, remaining))
        )

    # Если Chrome завершился аварийно, он мог оставить symlink блокировки.
    # Удаляем только известный lock-файл и только когда владельца профиля уже нет.
    _remove_stale_lock(profile_dir)
    return pids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Закрыть Chrome с профилем Selenium-сабмиттера"
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Только показать найденные PID, ничего не закрывать",
    )
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("--timeout не может быть отрицательным")

    profile_dir = DEFAULT_PROFILE
    pids = find_profile_chrome_processes(profile_dir)
    if args.check:
        print(
            "Chrome-профиль сабмиттера свободен"
            if not pids
            else "Chrome-профиль сабмиттера занят PID: "
            + ", ".join(map(str, pids))
        )
        return 0

    try:
        closed = close_profile_chrome(profile_dir, timeout=args.timeout)
    except RuntimeError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    if closed:
        print("Предыдущая Chrome-сессия сабмиттера закрыта")
    else:
        print("Активной Chrome-сессии сабмиттера нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
