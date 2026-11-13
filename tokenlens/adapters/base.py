from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..models import AgentSourceInfo, UsageEvent


def local_tz():
    return datetime.now().astimezone().tzinfo


def parse_iso_dt(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=local_tz())
        return dt.astimezone(local_tz())
    except Exception:
        return None


def parse_epoch_dt(value: int | float | str | None) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        num = float(value)
        if num <= 0:
            return None
        if num > 1e11:
            num /= 1000.0
        return datetime.fromtimestamp(num, tz=local_tz())
    except Exception:
        return None


def open_sqlite_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=3.0)


def get_app_data_dirs() -> list[Path]:
    dirs: list[Path] = []
    home = Path.home()
    if os.name == "nt":
        for env_key in ("APPDATA", "LOCALAPPDATA"):
            val = os.environ.get(env_key)
            if val:
                dirs.append(Path(val))
    elif os.name == "posix":
        dirs.append(home / "Library" / "Application Support")
        dirs.append(home / ".config")
        dirs.append(home / ".local" / "share")
    return dirs


def find_existing_paths(candidates: Iterable[Path]) -> list[Path]:
    return [p for p in candidates if p.exists()]


class BaseAdapter(ABC):
    agent_id: str = ""
    agent_name: str = ""
    color: str = "#6366f1"
    description: str = ""

    @abstractmethod
    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        """Scan local disk for token usage events and return source metadata."""
