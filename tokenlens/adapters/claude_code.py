from __future__ import annotations

import json
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, parse_iso_dt


class ClaudeCodeAdapter(BaseAdapter):
    agent_id = "claude_code"
    agent_name = "Claude Code"
    color = "#38bdf8"
    description = "Anthropic Claude Code CLI (~/.claude)"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        projects_dir = Path.home() / ".claude" / "projects"
        events: list[UsageEvent] = []
        paths_checked = [str(projects_dir)]
        paths_found: list[str] = []

        if not projects_dir.exists():
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地目录",
                paths_checked=paths_checked,
            )

        paths_found.append(str(projects_dir))
        sessions_seen: set[str] = set()

        for fp in projects_dir.rglob("*.jsonl"):
            session_id = fp.stem
            project_name = fp.parent.name
            first_title = ""
            seen_req_ids: set[str] = set()

            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        msg = obj.get("message")
                        if not first_title and obj.get("type") == "user" and isinstance(msg, dict):
                            c = msg.get("content")
                            if isinstance(c, str) and c.strip():
                                first_title = c.strip().replace("\n", " ")[:120]
                            elif isinstance(c, list) and c and isinstance(c[0], dict):
                                t = c[0].get("text", "")
                                if t:
                                    first_title = str(t).strip().replace("\n", " ")[:120]

                        if not isinstance(msg, dict) or "usage" not in msg:
                            continue
                        usage = msg.get("usage")
                        if not isinstance(usage, dict):
                            continue

                        # Deduplicate streamed chunks sharing the same message id / requestId
                        req_id = obj.get("requestId") or msg.get("id")
                        if req_id:
                            if req_id in seen_req_ids:
                                continue
                            seen_req_ids.add(str(req_id))

                        inp = int(usage.get("input_tokens") or 0)
                        out = int(usage.get("output_tokens") or 0)
                        cr = int(usage.get("cache_read_input_tokens") or 0)
                        cw = int(usage.get("cache_creation_input_tokens") or 0)
                        if inp + out + cr + cw <= 0:
                            continue

                        dt = parse_iso_dt(obj.get("timestamp"))
                        if not dt:
                            continue

                        model = str(msg.get("model") or "claude-code")
                        cwd = str(obj.get("cwd") or project_name)
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
                                cache_write=cw,
                                output=out,
                                title=first_title or f"Claude {session_id[:8]}",
                                project_path=cwd,
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
            notes="读取 ~/.claude/projects/**/*.jsonl 中的 message.usage 明细。",
        )
