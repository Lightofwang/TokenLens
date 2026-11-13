from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class UsageEvent:
    """Normalized single-turn or single-request token usage record."""

    agent_id: str
    agent_name: str
    session_id: str
    timestamp: datetime
    model: str = "unknown"
    uncached_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    reasoning: int = 0
    total_tokens: int = 0
    title: str = ""
    project_path: str = ""
    is_estimated: bool = False
    source_file: str = ""
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens <= 0:
            self.total_tokens = (
                self.uncached_input
                + self.cache_read
                + self.cache_write
                + self.output
            )


@dataclass(slots=True)
class AgentSourceInfo:
    """Discovery and health metadata for an AI agent adapter."""

    agent_id: str
    name: str
    color: str
    description: str
    status: str  # "exact" | "estimated" | "encrypted" | "unavailable" | "not_installed"
    status_label: str
    paths_checked: list[str] = field(default_factory=list)
    paths_found: list[str] = field(default_factory=list)
    events_count: int = 0
    sessions_count: int = 0
    total_tokens: int = 0
    notes: str = ""
