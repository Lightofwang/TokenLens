from __future__ import annotations

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, get_app_data_dirs


class WorkBuddyAdapter(BaseAdapter):
    agent_id = "workbuddy"
    agent_name = "WorkBuddy"
    color = "#ec4899"
    description = "WorkBuddy / CodeBuddy 客户端探测器"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        app_dirs = get_app_data_dirs()
        paths_checked: list[str] = []
        paths_found: list[str] = []

        for base in app_dirs:
            for name in ("WorkBuddy", "CodeBuddy", "CodeBuddyExtension"):
                p = base / name
                paths_checked.append(str(p))
                if p.exists():
                    paths_found.append(str(p))

        if not paths_found:
            return [], AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地目录",
                paths_checked=paths_checked[:4],
            )

        return [], AgentSourceInfo(
            agent_id=self.agent_id,
            name=self.agent_name,
            color=self.color,
            description=self.description,
            status="unavailable",
            status_label="本地未落盘 Token 数据",
            paths_checked=paths_checked[:6],
            paths_found=paths_found,
            events_count=0,
            sessions_count=0,
            total_tokens=0,
            notes="检测到客户端本地目录，但该客户端仅存储 UI 状态与运行日志，未在本地落盘接口返回的 Token 统计。",
        )
