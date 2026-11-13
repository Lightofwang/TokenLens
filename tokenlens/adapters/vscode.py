from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, get_app_data_dirs, local_tz, parse_epoch_dt


class VSCodeAdapter(BaseAdapter):
    agent_id = "vscode"
    agent_name = "VS Code"
    color = "#6366f1"
    description = "VS Code GitHub Copilot Chat (chatSessions)"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        app_dirs = get_app_data_dirs()
        events: list[UsageEvent] = []
        paths_checked: list[str] = []
        paths_found: list[str] = []
        sessions_seen: set[str] = set()

        ws_roots: list[Path] = []
        for base in app_dirs:
            ws = base / "Code" / "User" / "workspaceStorage"
            paths_checked.append(str(ws))
            if ws.exists():
                paths_found.append(str(ws))
                ws_roots.append(ws)

        if not ws_roots:
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到 VS Code 数据目录",
                paths_checked=paths_checked,
            )

        for ws in ws_roots:
            for fp in ws.glob("*/chatSessions/*"):
                if not fp.is_file():
                    continue
                session_id = fp.stem
                if fp.suffix == ".json":
                    try:
                        obj = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        continue
                    cdate = parse_epoch_dt(obj.get("creationDate"))
                    s_title = str(obj.get("customTitle") or "").strip()[:120]
                    for idx, req in enumerate(obj.get("requests") or []):
                        if not isinstance(req, dict):
                            continue
                        res = req.get("result") if isinstance(req.get("result"), dict) else {}
                        meta = res.get("metadata") if isinstance(res.get("metadata"), dict) else {}
                        pt = int(req.get("promptTokens") or meta.get("promptTokens") or 0)
                        ct = int(req.get("completionTokens") or req.get("outputTokens") or meta.get("outputTokens") or 0)
                        if pt + ct <= 0:
                            continue
                        dt = parse_epoch_dt(req.get("timestamp")) or cdate
                        if not dt:
                            continue
                        model = str(req.get("modelId") or meta.get("modelId") or "copilot")
                        msg_txt = ""
                        if isinstance(req.get("message"), dict):
                            msg_txt = str(req["message"].get("text") or "").strip().replace("\n", " ")[:120]
                        sessions_seen.add(session_id)
                        events.append(
                            UsageEvent(
                                agent_id=self.agent_id,
                                agent_name=self.agent_name,
                                session_id=session_id,
                                timestamp=dt,
                                model=model,
                                uncached_input=pt,
                                output=ct,
                                title=s_title or msg_txt or f"VSCode {session_id[:8]}",
                                source_file=str(fp),
                            )
                        )
                elif fp.suffix == ".jsonl":
                    cdate = None
                    default_model = "copilot"
                    s_title = ""
                    req_dates: dict[int, datetime] = {}
                    req_pt: dict[int, int] = defaultdict(int)
                    req_ct: dict[int, int] = defaultdict(int)
                    req_model: dict[int, str] = {}
                    req_title: dict[int, str] = {}

                    try:
                        with fp.open("r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                obj = json.loads(line)
                                k = obj.get("k") or []
                                v = obj.get("v")
                                if obj.get("kind") == 0 and isinstance(v, dict):
                                    cdate = parse_epoch_dt(v.get("creationDate"))
                                    s_title = str(v.get("customTitle") or "").strip()[:120]
                                    sel_m = (v.get("inputState") or {}).get("selectedModel") or {}
                                    if isinstance(sel_m, dict):
                                        m_id = sel_m.get("identifier") or (sel_m.get("metadata") or {}).get("id")
                                        if m_id:
                                            default_model = str(m_id)
                                elif len(k) >= 3 and k[0] == "requests" and isinstance(k[1], int):
                                    idx, field = k[1], k[2]
                                    if field == "timestamp":
                                        dt = parse_epoch_dt(v)
                                        if dt:
                                            req_dates[idx] = dt
                                    elif field == "promptTokens":
                                        req_pt[idx] = max(req_pt[idx], int(v or 0))
                                    elif field in ("completionTokens", "outputTokens"):
                                        req_ct[idx] = max(req_ct[idx], int(v or 0))
                                    elif field == "modelId" and v:
                                        req_model[idx] = str(v)
                                    elif field == "message" and isinstance(v, dict):
                                        t = str(v.get("text") or "").strip().replace("\n", " ")[:120]
                                        if t:
                                            req_title[idx] = t
                                    elif field == "result" and isinstance(v, dict):
                                        m = v.get("metadata") or {}
                                        if isinstance(m, dict):
                                            req_pt[idx] = max(req_pt[idx], int(m.get("promptTokens") or 0))
                                            req_ct[idx] = max(req_ct[idx], int(m.get("outputTokens") or 0))
                    except Exception:
                        continue

                    fallback_dt = cdate or datetime.fromtimestamp(fp.stat().st_mtime, tz=local_tz())
                    for idx in set(req_pt.keys()) | set(req_ct.keys()):
                        pt = req_pt[idx]
                        ct = req_ct[idx]
                        if pt + ct <= 0:
                            continue
                        dt = req_dates.get(idx) or fallback_dt
                        sessions_seen.add(session_id)
                        events.append(
                            UsageEvent(
                                agent_id=self.agent_id,
                                agent_name=self.agent_name,
                                session_id=session_id,
                                timestamp=dt,
                                model=req_model.get(idx, default_model),
                                uncached_input=pt,
                                output=ct,
                                title=s_title or req_title.get(idx) or f"VSCode {session_id[:8]}",
                                source_file=str(fp),
                            )
                        )

        status = "exact" if events else "unavailable"
        label = "精确记录 (ChatSessions)" if events else "未检测到含 Token 的会话"
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
            notes="读取 VS Code workspaceStorage/*/chatSessions 中的 Copilot 对话 Token 记录。",
        )
