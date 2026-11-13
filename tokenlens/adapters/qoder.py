from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, get_app_data_dirs, open_sqlite_ro, parse_epoch_dt, parse_iso_dt

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def _estimate_content_tokens(content: object) -> int:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("thinking"):
                    parts.append(str(item["thinking"]))
                elif item.get("input"):
                    parts.append(json.dumps(item["input"], ensure_ascii=False))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        return 0
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return max(1, int(round(cjk * 1.5 + other * 0.28)))


class QoderAdapter(BaseAdapter):
    agent_id = "qoder"
    agent_name = "Qoder"
    color = "#06b6d4"
    description = "Qoder / QoderCN (新版 Agent SDK + 旧版 local.db) / QoderWork CN"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        home = Path.home()
        app_dirs = get_app_data_dirs()
        events: list[UsageEvent] = []
        paths_checked: list[str] = []
        paths_found: list[str] = []
        sessions_seen: set[str] = set()

        # 1. New Qoder CN (Agent SDK ~/.qoder-cn + com.qodercn.app.stable/main.sqlite)
        qcn_projects = home / ".qoder-cn" / "projects"
        paths_checked.append(str(qcn_projects))
        if qcn_projects.exists():
            paths_found.append(str(qcn_projects))
            stable_titles: dict[str, tuple[str, str]] = {}
            for base in app_dirs:
                mdb = base / "com.qodercn.app.stable" / "main.sqlite"
                if mdb.exists():
                    paths_found.append(str(mdb))
                    try:
                        with open_sqlite_ro(mdb) as conn:
                            for sid, title, cwd in conn.execute(
                                "SELECT session_id, title, cwd FROM chat_sessions"
                            ).fetchall():
                                stable_titles[str(sid)] = (
                                    str(title or "").strip()[:120],
                                    str(cwd or ""),
                                )
                    except Exception:
                        pass

            for fp in qcn_projects.rglob("*.jsonl"):
                session_id = fp.stem
                if fp.parent.name == "subagents" and len(fp.parents) >= 2:
                    session_id = fp.parents[1].name
                s_title, s_cwd = stable_titles.get(session_id, ("", ""))
                first_user_title = s_title
                req_records: dict[str, tuple[dict, dict, str]] = {}

                try:
                    with fp.open("r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if not s_cwd and obj.get("cwd"):
                                s_cwd = str(obj["cwd"])
                            msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                            if not first_user_title and obj.get("type") == "user" and isinstance(msg, dict):
                                c = msg.get("content")
                                if isinstance(c, str) and c.strip():
                                    first_user_title = c.strip().replace("\n", " ")[:120]
                                elif isinstance(c, list) and c and isinstance(c[0], dict) and c[0].get("text"):
                                    first_user_title = str(c[0]["text"]).strip().replace("\n", " ")[:120]

                            u = msg.get("usage") or obj.get("usage")
                            if not isinstance(u, dict):
                                continue
                            rid = str(u.get("request_id") or obj.get("requestId") or obj.get("uuid") or "")
                            if not rid:
                                continue
                            req_records[rid] = (obj, msg, str(fp))
                except Exception:
                    continue

                for rid, (obj, msg, src_fp) in req_records.items():
                    u = msg.get("usage") or obj.get("usage") or {}
                    dt = parse_iso_dt(obj.get("timestamp"))
                    if not dt:
                        continue
                    inp = int(u.get("input_tokens") or 0)
                    cr = int(u.get("cache_read_input_tokens") or 0)
                    cw = int(u.get("cache_creation_input_tokens") or 0)
                    out = int(u.get("output_tokens") or 0)
                    credits = float(u.get("credits") or 0.0)
                    orig_credits = float(u.get("original_credits") or 0.0)
                    ratio = float(u.get("context_usage_ratio") or 0.0)
                    is_est = False

                    if inp + cr + cw + out == 0:
                        if ratio > 0:
                            inp = int(round(ratio * 200_000))
                            is_est = True
                        if out == 0:
                            out = _estimate_content_tokens(msg.get("content"))
                            if out > 0:
                                is_est = True

                    if inp + cr + cw + out <= 0 and credits <= 0:
                        continue

                    model = str(msg.get("model") or obj.get("model") or "qoder-cn")
                    full_sid = f"QoderCN:{session_id}"
                    sessions_seen.add(full_sid)
                    events.append(
                        UsageEvent(
                            agent_id=self.agent_id,
                            agent_name=self.agent_name,
                            session_id=full_sid,
                            timestamp=dt,
                            model=model,
                            uncached_input=inp,
                            cache_read=cr,
                            cache_write=cw,
                            output=out,
                            title=first_user_title or f"QoderCN {session_id[:8]}",
                            project_path=s_cwd,
                            is_estimated=is_est,
                            source_file=src_fp,
                            credits=credits,
                            original_credits=orig_credits,
                        )
                    )

        # 2. Legacy Qoder & QoderCN desktop IDE (SharedClientCache/cache/db/local.db)
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

            # 3. QoderWork CN
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
                status_label="未检测到本地数据",
                paths_checked=paths_checked[:4],
            )

        status = "exact" if events else "unavailable"
        label = "精确积分 + Token/比例还原" if events else "目录存在但暂无记录"
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
            notes=(
                "支持新版 Qoder CN (~/.qoder-cn 精确记录 credits 积分，官方模型未下发 input_tokens 时"
                "按 context_usage_ratio×200k 还原上下文 Token) 及旧版 local.db / QoderWork CN。"
            ),
        )
