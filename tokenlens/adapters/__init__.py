from __future__ import annotations

from .antigravity import AntigravityAdapter
from .base import BaseAdapter
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .opencode import OpenCodeAdapter
from .qoder import QoderAdapter
from .trae import TraeAdapter
from .vscode import VSCodeAdapter
from .workbuddy import WorkBuddyAdapter

ALL_ADAPTERS: list[BaseAdapter] = [
    AntigravityAdapter(),
    CodexAdapter(),
    ClaudeCodeAdapter(),
    OpenCodeAdapter(),
    QoderAdapter(),
    VSCodeAdapter(),
    TraeAdapter(),
    WorkBuddyAdapter(),
]
