from __future__ import annotations

import json
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, get_app_data_dirs, open_sqlite_ro, parse_epoch_dt


class QoderAdapter(BaseAdapter):
    agent_id = "qoder"
    agent_name = "Qoder"
    color = "#06b6d4"
    description = "Qoder / QoderCN / QoderWork CN"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        app_dirs = get_app_data_dirs()
        events: list[UsageEvent] = []
        paths_checked: list[str] = []
        paths_found: list[str] = []
        sessions_seen: set[str] = set()

        # 1. Qoder & QoderCN desktop IDE
        for base in app_dirs:
            for app_name in ("Qoder", "QoderCN"):
                ldb = base / app_name / "SharedClientCache" / "cache" / "db" / "local.db"
                paths_checked.append(str(ldb))
                if not ldb.exists():
                    continue
                paths_found.append(str(ldb))
                try:
                    with open_sqlite_ro(ldb) as conn:
                        session_titles: dict[str, tuple[str, str]] = {}
                        try:
                            for sid, stitle, p_uri in conn.execute(
                                "SELECT session_id, session_title, project_uri FROM chat_session"
                            ).fetchall():
                                session_titles[str(sid)] = (
                                    str(stitle or "").strip()[:120],
                                    str(p_uri or ""),
                                )
                        except Exception:
                            pass

                        rows = conn.execute(
                            "SELECT session_id, token_info, model_info, gmt_create "
                            "FROM chat_message WHERE token_info IS NOT NULL AND length(token_info) > 2"
                        ).fetchall()
                        for sid, ti_str, mi_str, gmt in rows:
                            try:
                                ti = json.loads(ti_str)
                            except Exception:
                                continue
                            pt = int(ti.get("prompt_tokens") or 0)
                            ct = int(ti.get("completion_tokens") or 0)
                            cached = int(ti.get("cached_tokens") or 0)
                            if pt + ct <= 0:
                                continue
                            dt = parse_epoch_dt(gmt)
                            if not dt:
                                continue
                            model = "qoder"
                            if mi_str:
                                try:
                                    mi = json.loads(mi_str)
                                    model = str(mi.get("model_name") or mi.get("model_key") or "qoder")
                                except Exception:
                                    pass
                            uncached = max(0, pt - cached)
                            s_id = f"{app_name}:{sid}"
                            s_title, p_uri = session_titles.get(str(sid), ("", ""))
                            sessions_seen.add(s_id)
                            events.append(
                                UsageEvent(
                                    agent_id=self.agent_id,
                                    agent_name=self.agent_name,
                                    session_id=s_id,
                                    timestamp=dt,
                                    model=model,
                                    uncached_input=uncached,
                                    cache_read=cached,
                                    output=ct,
                                    title=s_title or f"{app_name} {str(sid)[:8]}",
                                    project_path=p_uri,
                                    source_file=str(ldb),
                                )
                            )
                except Exception:
                    continue

            # 2. QoderWork CN
            qw_db = base / "QoderWork CN" / "data" / "agents.db"
            paths_checked.append(str(qw_db))
            if qw_db.exists():
                paths_found.append(str(qw_db))
                try:
                    with open_sqlite_ro(qw_db) as conn:
                        chat_titles: dict[str, str] = {}
                        try:
                            for cid, cname in conn.execute("SELECT id, name FROM chats").fetchall():
                                chat_titles[str(cid)] = str(cname or "").strip()[:120]
                        except Exception:
                            pass

                        rows = conn.execute(
                            "SELECT chat_id, metadata, created_at FROM messages "
                            "WHERE role='assistant' AND metadata IS NOT NULL AND length(metadata) > 2"
                        ).fetchall()
                        for cid, meta_str, ca in rows:
                            try:
                                m = json.loads(meta_str)
                            except Exception:
                                continue
                            inp = int(m.get("inputTokens") or 0)
                            out = int(m.get("outputTokens") or 0)
                            is_est = False
                            if inp == 0 and m.get("contextUsageRatio"):
                                inp = int(round(float(m["contextUsageRatio"]) * 180000))
                                is_est = True
                            if inp + out <= 0:
                                continue
                            dt = parse_epoch_dt(ca)
                            if not dt:
                                continue
                            s_id = f"QoderWork:{cid}"
                            sessions_seen.add(s_id)
                            events.append(
                                UsageEvent(
                                    agent_id=self.agent_id,
                                    agent_name=self.agent_name,
                                    session_id=s_id,
                                    timestamp=dt,
                                    model="qoderwork",
                                    uncached_input=inp,
                                    output=out,
                                    title=chat_titles.get(str(cid), f"QoderWork {str(cid)[:8]}"),
                                    is_estimated=is_est,
                                    duration_ms=int(m.get("durationMs") or 0),
                                    source_file=str(qw_db),
                                )
                            )
                except Exception:
                    continue

        if not paths_found:
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地数据库",
                paths_checked=paths_checked[:4],
            )

        status = "exact" if events else "unavailable"
        label = "精确记录 (SQLite)" if events else "数据库存在但暂无记录"
        return events, AgentSourceInfo(
            agent_id=self.agent_id,
            name=self.agent_name,
            color=self.color,
            description=self.description,
            status=status,
            status_label=label,
            paths_checked=paths_checked[:6],
            paths_found=paths_found,
            events_count=len(events),
            sessions_count=len(sessions_seen),
            total_tokens=sum(e.total_tokens for e in events),
            notes="合并统计 Qoder / QoderCN (local.db chat_message.token_info) 与 QoderWork CN (agents.db)。",
        )
