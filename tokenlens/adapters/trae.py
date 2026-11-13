from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, get_app_data_dirs, open_sqlite_ro, parse_epoch_dt

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    cjk_count = len(_CJK_RE.findall(text))
    other_count = max(0, len(text) - cjk_count)
    return max(1, int(round(cjk_count * 1.5 + other_count * 0.28)))


class TraeAdapter(BaseAdapter):
    agent_id = "trae"
    agent_name = "Trae"
    color = "#a855f7"
    description = "Trae / Trae CN (加密数据库检测 + 明文历史估算)"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        app_dirs = get_app_data_dirs()
        events: list[UsageEvent] = []
        paths_checked: list[str] = []
        paths_found: list[str] = []
        sessions_seen: set[str] = set()
        encrypted_dbs: list[str] = []

        for base in app_dirs:
            for app_name in ("Trae", "Trae CN"):
                enc_db = base / app_name / "ModularData" / "ai-agent" / "database.db"
                paths_checked.append(str(enc_db))
                if enc_db.exists():
                    encrypted_dbs.append(str(enc_db))
                    paths_found.append(str(enc_db))

                ws_dir = base / app_name / "User" / "workspaceStorage"
                if not ws_dir.exists():
                    continue
                for vscdb in ws_dir.glob("*/state.vscdb"):
                    try:
                        with open_sqlite_ro(vscdb) as conn:
                            rows = conn.execute(
                                "SELECT key, value FROM ItemTable "
                                "WHERE key LIKE 'memento/icube-ai-ng-chat-storage-%' "
                                "OR key LIKE 'memento/icube-ai-agent-storage-%'"
                            ).fetchall()
                            for _, val in rows:
                                if isinstance(val, bytes):
                                    val = val.decode("utf-8", errors="ignore")
                                try:
                                    data = json.loads(val)
                                except Exception:
                                    continue
                                sess_list = data.get("list") if isinstance(data, dict) else None
                                if not isinstance(sess_list, list):
                                    continue
                                for sess in sess_list:
                                    if not isinstance(sess, dict):
                                        continue
                                    sid = str(sess.get("sessionId") or "")
                                    msgs = sess.get("messages") or []
                                    if not sid or not isinstance(msgs, list) or not msgs:
                                        continue
                                    s_title = str(sess.get("title") or "").strip()[:120]
                                    rolling_ctx = 0
                                    for m in msgs:
                                        if not isinstance(m, dict):
                                            continue
                                        role = str(m.get("role") or "")
                                        content = str(m.get("content") or "")
                                        if not s_title and role == "user" and content.strip():
                                            s_title = content.strip().replace("\n", " ")[:120]
                                        tok = estimate_tokens_from_text(content)
                                        if role == "user":
                                            rolling_ctx += tok + 800
                                            continue
                                        if role != "assistant" or tok <= 0:
                                            continue
                                        dt = parse_epoch_dt(m.get("timestamp"))
                                        if not dt:
                                            continue
                                        model = str(m.get("modelName") or m.get("displayModelName") or "trae-agent")
                                        full_sid = f"{app_name}:{sid}"
                                        sessions_seen.add(full_sid)
                                        # Estimate prompt context + completion
                                        prompt_est = max(rolling_ctx, 1200)
                                        rolling_ctx += tok
                                        events.append(
                                            UsageEvent(
                                                agent_id=self.agent_id,
                                                agent_name=self.agent_name,
                                                session_id=full_sid,
                                                timestamp=dt,
                                                model=model,
                                                uncached_input=prompt_est,
                                                output=tok,
                                                title=s_title or f"{app_name} {sid[:8]}",
                                                is_estimated=True,
                                                source_file=str(vscdb),
                                            )
                                        )
                    except Exception:
                        continue

        if not paths_found and not events:
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地目录",
                paths_checked=paths_checked,
            )

        if events:
            status = "estimated"
            label = "文本估算 (原生库已加密)"
        elif encrypted_dbs:
            status = "encrypted"
            label = "数据库 SQLCipher 加密"
        else:
            status = "unavailable"
            label = "暂无可解析数据"

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
            notes=(
                "Trae 核心库 ModularData/ai-agent/database.db 使用 SQLCipher 加密；"
                "当前从明文 state.vscdb 会话存储中按文本长度与累计上下文进行 Token 估算（标为估算值）。"
            ),
        )
