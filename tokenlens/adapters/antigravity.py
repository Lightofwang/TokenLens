from __future__ import annotations

import json
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, parse_iso_dt


class AntigravityAdapter(BaseAdapter):
    agent_id = "antigravity"
    agent_name = "Antigravity"
    color = "#f97316"
    description = "Google Antigravity IDE / Agent transcripts"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        brain_dir = Path.home() / ".gemini" / "antigravity" / "brain"
        events: list[UsageEvent] = []
        paths_checked = [str(brain_dir)]
        paths_found: list[str] = []

        if not brain_dir.exists():
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地目录",
                paths_checked=paths_checked,
            )

        paths_found.append(str(brain_dir))
        sessions_seen: set[str] = set()

        for fp in brain_dir.rglob("transcript.jsonl"):
            # brain/<session_id>/.system_generated/logs/transcript.jsonl
            rel_parts = fp.relative_to(brain_dir).parts
            session_id = rel_parts[0] if rel_parts else fp.parent.name
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    first_user_title = ""
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if not first_user_title and obj.get("type") == "USER_INPUT":
                            content = obj.get("content", "")
                            if isinstance(content, str) and content.strip():
                                first_user_title = content.strip().replace("\n", " ")[:120]
                        if obj.get("type") != "PLANNER_RESPONSE":
                            continue
                        inp = int(obj.get("input_tokens") or 0)
                        out = int(obj.get("output_tokens") or 0)
                        cr = int(obj.get("cache_read_tokens") or 0)
                        if inp == 0 and out == 0 and cr == 0:
                            continue
                        dt = parse_iso_dt(obj.get("created_at"))
                        if not dt:
                            continue
                        model = obj.get("model") or "antigravity-agent"
                        sessions_seen.add(session_id)
                        events.append(
                            UsageEvent(
                                agent_id=self.agent_id,
                                agent_name=self.agent_name,
                                session_id=session_id,
                                timestamp=dt,
                                model=model,
                                uncached_input=inp,
                                cache_read=cr,
                                output=out,
                                title=first_user_title or f"Session {session_id[:8]}",
                                source_file=str(fp),
                            )
                        )
            except Exception:
                continue

        status = "exact" if events else "unavailable"
        label = "精确记录 (JSONL)" if events else "目录存在但暂无记录"
        return events, AgentSourceInfo(
            agent_id=self.agent_id,
            name=self.agent_name,
            color=self.color,
            description=self.description,
            status=status,
            status_label=label,
            paths_checked=paths_checked,
            paths_found=paths_found,
            events_count=len(events),
            sessions_count=len(sessions_seen),
            total_tokens=sum(e.total_tokens for e in events),
            notes="读取 ~/.gemini/antigravity/brain/**/transcript.jsonl 中的 PLANNER_RESPONSE Token 指标。",
        )
