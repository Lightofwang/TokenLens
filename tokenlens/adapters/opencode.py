from __future__ import annotations

import json
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, open_sqlite_ro, parse_epoch_dt


class OpenCodeAdapter(BaseAdapter):
    agent_id = "opencode"
    agent_name = "OpenCode"
    color = "#eab308"
    description = "OpenCode CLI / Desktop (opencode.db)"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        db_path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
        events: list[UsageEvent] = []
        paths_checked = [str(db_path)]
        paths_found: list[str] = []

        if not db_path.exists():
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地数据库",
                paths_checked=paths_checked,
            )

        paths_found.append(str(db_path))
        sessions_seen: set[str] = set()

        try:
            with open_sqlite_ro(db_path) as conn:
                session_map: dict[str, tuple[str, str]] = {}
                try:
                    for sid, title, directory in conn.execute(
                        "SELECT id, title, directory FROM session"
                    ).fetchall():
                        session_map[str(sid)] = (str(title or ""), str(directory or ""))
                except Exception:
                    pass

                rows = conn.execute(
                    "SELECT session_id, data, time_created FROM message"
                ).fetchall()
                for sid, data_str, tc in rows:
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue
                    t = obj.get("tokens")
                    if not isinstance(t, dict):
                        continue
                    inp = int(t.get("input") or 0)
                    out = int(t.get("output") or 0)
                    reasoning = int(t.get("reasoning") or 0)
                    cache = t.get("cache") or {}
                    cr = int(cache.get("read") or 0)
                    cw = int(cache.get("write") or 0)
                    total = int(t.get("total") or 0)
                    if inp + out + reasoning + cr + cw <= 0 and total <= 0:
                        continue
                    dt = parse_epoch_dt(tc)
                    if not dt:
                        continue
                    s_id = str(sid or "")
                    s_title, s_dir = session_map.get(s_id, ("", ""))
                    model = str(obj.get("modelID") or obj.get("model") or "opencode")
                    sessions_seen.add(s_id)
                    events.append(
                        UsageEvent(
                            agent_id=self.agent_id,
                            agent_name=self.agent_name,
                            session_id=s_id,
                            timestamp=dt,
                            model=model,
                            uncached_input=inp,
                            cache_read=cr,
                            cache_write=cw,
                            output=out + reasoning,
                            reasoning=reasoning,
                            total_tokens=total if total > 0 else 0,
                            title=s_title or f"OpenCode {s_id[:8]}",
                            project_path=s_dir,
                            source_file=str(db_path),
                        )
                    )
        except Exception:
            pass

        status = "exact" if events else "unavailable"
        label = "精确记录 (SQLite)" if events else "数据库存在但暂无记录"
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
            notes="读取 ~/.local/share/opencode/opencode.db 中的 message.tokens 字段。",
        )
