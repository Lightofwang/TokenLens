from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..models import AgentSourceInfo, UsageEvent
from .base import (
    BaseAdapter,
    close_sqlite_snapshot,
    get_app_data_dirs,
    local_tz,
    open_sqlite_snapshot,
    parse_epoch_dt,
    parse_iso_dt,
)


def _decode_file_uri(uri_or_path: str | None) -> str:
    raw = str(uri_or_path or "").strip()
    if not raw:
        return ""
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        path = unquote(parsed.path or "")
        # Windows file:///c:/foo -> C:\foo or /c:/foo -> c:/foo
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path
    return raw


def _parse_workspace_yaml(yaml_path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not yaml_path.exists():
        return info
    try:
        for line in yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    except Exception:
        pass
    return info


def _normalize_model_name(raw_model: str | None, fallback: str = "copilot") -> str:
    m = str(raw_model or "").strip()
    if not m:
        return fallback
    if m.startswith("customendpoint/"):
        m = m[len("customendpoint/"):]
    return m


def _extract_request_event(
    agent_id: str,
    agent_name: str,
    session_id: str,
    req: dict[str, Any],
    fallback_dt: datetime | None,
    fallback_model: str,
    session_title: str,
    source_file: Path,
) -> UsageEvent | None:
    res = req.get("result") if isinstance(req.get("result"), dict) else {}
    meta = res.get("metadata") if isinstance(res.get("metadata"), dict) else {}

    pt = int(req.get("promptTokens") or meta.get("promptTokens") or meta.get("inputTokens") or 0)
    ct = int(
        req.get("completionTokens")
        or req.get("outputTokens")
        or meta.get("outputTokens")
        or meta.get("completionTokens")
        or 0
    )
    cr = int(req.get("cacheReadTokens") or meta.get("cacheReadTokens") or meta.get("cachedInputTokens") or 0)
    cw = int(req.get("cacheWriteTokens") or meta.get("cacheWriteTokens") or 0)
    if pt + ct + cr + cw <= 0:
        return None

    dt = parse_epoch_dt(req.get("timestamp")) or fallback_dt
    if not dt:
        return None

    model = _normalize_model_name(req.get("modelId") or meta.get("modelId"), fallback=fallback_model)
    msg_txt = ""
    if isinstance(req.get("message"), dict):
        msg_txt = str(req["message"].get("text") or "").strip().replace("\n", " ")[:120]

    credits = float(req.get("copilotCredits") or meta.get("copilotCredits") or 0.0)
    elapsed_ms = int(req.get("elapsedMs") or (res.get("timings") or {}).get("totalElapsed") or 0)

    return UsageEvent(
        agent_id=agent_id,
        agent_name=agent_name,
        session_id=session_id,
        timestamp=dt,
        model=model,
        uncached_input=pt,
        cache_read=cr,
        cache_write=cw,
        output=ct,
        title=session_title or msg_txt or f"VSCode {session_id[:8]}",
        source_file=str(source_file),
        duration_ms=elapsed_ms,
        credits=credits,
        original_credits=credits,
    )


class VSCodeAdapter(BaseAdapter):
    agent_id = "vscode"
    agent_name = "VS Code"
    color = "#6366f1"
    description = "VS Code Copilot Chat / Agent Host / 自定义 BYOK 模型"

    def __init__(
        self,
        app_dirs: list[Path] | None = None,
        copilot_dir: Path | None = None,
    ) -> None:
        self._app_dirs = app_dirs
        self._copilot_dir = copilot_dir

    def _scan_copilot_session_state(
        self,
        copilot_ss_dir: Path,
        events: list[UsageEvent],
        sessions_seen: set[str],
        seen_api_calls: set[tuple[str, str]],
        seen_signatures: set[tuple[str, str, int, int]],
    ) -> None:
        if not copilot_ss_dir.exists():
            return
        for sess_dir in copilot_ss_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            ev_file = sess_dir / "events.jsonl"
            if not ev_file.is_file():
                continue

            ws_meta = _parse_workspace_yaml(sess_dir / "workspace.yaml")
            sdk_sid = ws_meta.get("id") or sess_dir.name
            project_path = ws_meta.get("cwd") or ""
            s_title = (ws_meta.get("name") or "").strip()[:120]
            current_model = "copilot"
            fallback_dt = parse_iso_dt(ws_meta.get("updated_at")) or parse_iso_dt(ws_meta.get("created_at"))
            first_user_msg = ""

            try:
                lines = ev_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                etype = obj.get("type")
                data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
                ev_dt = parse_iso_dt(obj.get("timestamp"))

                if etype == "session.start":
                    sdk_sid = str(data.get("sessionId") or sdk_sid)
                    ctx = data.get("context") if isinstance(data.get("context"), dict) else {}
                    if not project_path and ctx.get("cwd"):
                        project_path = str(ctx.get("cwd") or "")
                    if data.get("selectedModel"):
                        current_model = _normalize_model_name(str(data["selectedModel"]), current_model)
                    if not fallback_dt:
                        fallback_dt = parse_iso_dt(data.get("startTime")) or ev_dt
                elif etype == "session.model_change":
                    if data.get("newModel"):
                        current_model = _normalize_model_name(str(data["newModel"]), current_model)
                elif etype == "user.message":
                    if not first_user_msg and isinstance(data.get("content"), str):
                        first_user_msg = data["content"].strip().replace("\n", " ")[:120]
                elif etype == "session.usage_record":
                    usage = data.get("usage") if isinstance(data.get("usage"), dict) else data
                    in_tok = int(usage.get("inputTokens") or 0)
                    out_tok = int(usage.get("outputTokens") or 0)
                    cr_tok = int(usage.get("cacheReadTokens") or 0)
                    cw_tok = int(usage.get("cacheWriteTokens") or 0)
                    rs_tok = int(usage.get("reasoningTokens") or 0)
                    if in_tok + out_tok + cr_tok + cw_tok <= 0:
                        continue
                    dt = ev_dt or fallback_dt or datetime.fromtimestamp(ev_file.stat().st_mtime, tz=local_tz())
                    model = _normalize_model_name(
                        usage.get("model") or usage.get("modelDisplayName"),
                        fallback=current_model,
                    )
                    sessions_seen.add(sdk_sid)
                    event_id = str(obj.get("id") or "")
                    if event_id:
                        seen_api_calls.add((sdk_sid, event_id))
                    seen_signatures.add((sdk_sid, model, in_tok, out_tok))
                    events.append(
                        UsageEvent(
                            agent_id=self.agent_id,
                            agent_name=self.agent_name,
                            session_id=sdk_sid,
                            timestamp=dt,
                            model=model,
                            uncached_input=in_tok,
                            cache_read=cr_tok,
                            cache_write=cw_tok,
                            output=out_tok,
                            reasoning=rs_tok,
                            title=s_title or first_user_msg or f"VSCode {sdk_sid[:8]}",
                            project_path=project_path,
                            source_file=str(ev_file),
                            duration_ms=int(usage.get("duration") or 0),
                        )
                    )

    def _scan_agent_session_data(
        self,
        code_dir: Path,
        events: list[UsageEvent],
        sessions_seen: set[str],
        seen_api_calls: set[tuple[str, str]],
        seen_signatures: set[tuple[str, str, int, int]],
    ) -> list[str]:
        found_paths: list[str] = []
        session_root = code_dir / "agentSessionData"
        if not session_root.exists():
            return found_paths
        found_paths.append(str(session_root))

        # Map session URI tail id -> modified_time/start_time from agent-host.db
        host_times: dict[str, tuple[datetime | None, datetime | None]] = {}
        host_db = code_dir / "User" / "globalStorage" / "agent-host.db"
        if host_db.is_file():
            found_paths.append(str(host_db))
            conn = open_sqlite_snapshot(host_db)
            if conn is not None:
                try:
                    for uri, start_ms, mod_ms in conn.execute(
                        "SELECT session_uri, start_time, modified_time FROM sessions"
                    ).fetchall():
                        sid = str(uri).rsplit("/", 1)[-1].lstrip(":")
                        host_times[sid] = (parse_epoch_dt(mod_ms), parse_epoch_dt(start_ms))
                except Exception:
                    pass
                finally:
                    close_sqlite_snapshot(conn)

        for sdb in session_root.glob("*/session.db"):
            if not sdb.is_file():
                continue
            sid = sdb.parent.name
            conn = open_sqlite_snapshot(sdb)
            if conn is None:
                continue
            try:
                tables = {
                    r[0]
                    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                if "turn_usage" not in tables:
                    continue
                usage_rows = conn.execute("SELECT turn_id, usage FROM turn_usage").fetchall()
                if not usage_rows:
                    continue

                meta_map: dict[str, str] = {}
                if "session_metadata" in tables:
                    for k, v in conn.execute("SELECT key, value FROM session_metadata").fetchall():
                        meta_map[str(k)] = str(v)

                s_title = str(meta_map.get("customTitle") or "").strip()[:120]
                project_path = _decode_file_uri(meta_map.get("copilot.workingDirectory"))
                default_model = "copilot"
                if meta_map.get("copilot.model"):
                    try:
                        m_obj = json.loads(meta_map["copilot.model"])
                        if isinstance(m_obj, dict) and m_obj.get("id"):
                            default_model = _normalize_model_name(m_obj["id"], default_model)
                    except Exception:
                        pass

                sdk_sid = sid
                if meta_map.get("defaultChatProviderData"):
                    try:
                        p_obj = json.loads(meta_map["defaultChatProviderData"])
                        if isinstance(p_obj, dict) and p_obj.get("sdkSessionId"):
                            sdk_sid = str(p_obj["sdkSessionId"])
                    except Exception:
                        pass

                mod_dt, start_dt = host_times.get(sid, (None, None))
                fallback_dt = mod_dt or start_dt or datetime.fromtimestamp(sdb.stat().st_mtime, tz=local_tz())

                for _, raw_usage in usage_rows:
                    try:
                        u = json.loads(raw_usage) if isinstance(raw_usage, str) else {}
                    except Exception:
                        continue
                    if not isinstance(u, dict):
                        continue
                    in_tok = int(u.get("inputTokens") or 0)
                    out_tok = int(u.get("outputTokens") or 0)
                    cr_tok = int(u.get("cacheReadTokens") or 0)
                    cw_tok = int(u.get("cacheWriteTokens") or 0)
                    if in_tok + out_tok + cr_tok + cw_tok <= 0:
                        continue

                    meta = u.get("_meta") if isinstance(u.get("_meta"), dict) else {}
                    mcall = meta.get("vscode.modelCall") if isinstance(meta.get("vscode.modelCall"), dict) else {}
                    call_sdk_sid = str(mcall.get("sdkSessionId") or sdk_sid)
                    event_id = str(mcall.get("eventId") or "")
                    model = _normalize_model_name(u.get("model") or mcall.get("model"), fallback=default_model)

                    if event_id and (call_sdk_sid, event_id) in seen_api_calls:
                        continue
                    if (call_sdk_sid, model, in_tok, out_tok) in seen_signatures:
                        continue

                    seen_signatures.add((call_sdk_sid, model, in_tok, out_tok))
                    sessions_seen.add(call_sdk_sid)
                    events.append(
                        UsageEvent(
                            agent_id=self.agent_id,
                            agent_name=self.agent_name,
                            session_id=call_sdk_sid,
                            timestamp=fallback_dt,
                            model=model,
                            uncached_input=in_tok,
                            cache_read=cr_tok,
                            cache_write=cw_tok,
                            output=out_tok,
                            title=s_title or f"VSCode {sid[:8]}",
                            project_path=project_path,
                            source_file=str(sdb),
                            duration_ms=int(mcall.get("durationMs") or 0),
                        )
                    )
            except Exception:
                continue
            finally:
                close_sqlite_snapshot(conn)

        return found_paths

    def _scan_chat_sessions(
        self,
        ws_root: Path,
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> None:
        for fp in ws_root.glob("*/chatSessions/*"):
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
                for req in obj.get("requests") or []:
                    if not isinstance(req, dict):
                        continue
                    ev = _extract_request_event(
                        self.agent_id,
                        self.agent_name,
                        session_id,
                        req,
                        cdate,
                        "copilot",
                        s_title,
                        fp,
                    )
                    if ev is not None:
                        sessions_seen.add(session_id)
                        events.append(ev)

            elif fp.suffix == ".jsonl":
                cdate = None
                default_model = "copilot"
                s_title = ""
                requests_list: list[dict[str, Any]] = []

                try:
                    with fp.open("r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            kind = obj.get("kind")
                            k = obj.get("k") or []
                            v = obj.get("v")

                            if kind == 0 and not k and isinstance(v, dict):
                                cdate = parse_epoch_dt(v.get("creationDate"))
                                if v.get("customTitle"):
                                    s_title = str(v["customTitle"]).strip()[:120]
                                sel_m = (v.get("inputState") or {}).get("selectedModel") or {}
                                if isinstance(sel_m, dict):
                                    m_id = sel_m.get("identifier") or (sel_m.get("metadata") or {}).get("id")
                                    if m_id:
                                        default_model = _normalize_model_name(str(m_id), default_model)
                                init_reqs = v.get("requests")
                                if isinstance(init_reqs, list):
                                    requests_list = [dict(r) for r in init_reqs if isinstance(r, dict)]
                            elif k == ["customTitle"] and isinstance(v, str):
                                s_title = v.strip()[:120]
                            elif k == ["inputState", "selectedModel"] and isinstance(v, dict):
                                m_id = v.get("identifier") or (v.get("metadata") or {}).get("id")
                                if m_id:
                                    default_model = _normalize_model_name(str(m_id), default_model)
                            elif k == ["requests"] and isinstance(v, list):
                                items = [dict(r) for r in v if isinstance(r, dict)]
                                if kind == 1:
                                    requests_list = items
                                else:
                                    splice_idx = obj.get("i")
                                    if isinstance(splice_idx, int) and 0 <= splice_idx <= len(requests_list):
                                        requests_list = requests_list[:splice_idx] + items
                                    else:
                                        requests_list.extend(items)
                            elif len(k) >= 3 and k[0] == "requests" and isinstance(k[1], int):
                                idx, field = k[1], k[2]
                                while len(requests_list) <= idx:
                                    requests_list.append({})
                                if len(k) == 3:
                                    requests_list[idx][field] = v
                                elif field == "result" and len(k) >= 4 and isinstance(k[3], str):
                                    res = requests_list[idx].setdefault("result", {})
                                    if isinstance(res, dict):
                                        res[k[3]] = v
                except Exception:
                    continue

                fallback_dt = cdate or datetime.fromtimestamp(fp.stat().st_mtime, tz=local_tz())
                for req in requests_list:
                    ev = _extract_request_event(
                        self.agent_id,
                        self.agent_name,
                        session_id,
                        req,
                        fallback_dt,
                        default_model,
                        s_title,
                        fp,
                    )
                    if ev is not None:
                        sessions_seen.add(session_id)
                        events.append(ev)

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        app_dirs = self._app_dirs if self._app_dirs is not None else get_app_data_dirs()
        copilot_ss_dir = (
            self._copilot_dir
            if self._copilot_dir is not None
            else (Path.home() / ".copilot" / "session-state")
        )

        events: list[UsageEvent] = []
        paths_checked: list[str] = [str(copilot_ss_dir)]
        paths_found: list[str] = []
        sessions_seen: set[str] = set()
        seen_api_calls: set[tuple[str, str]] = set()
        seen_signatures: set[tuple[str, str, int, int]] = set()

        if copilot_ss_dir.exists():
            paths_found.append(str(copilot_ss_dir))
            self._scan_copilot_session_state(
                copilot_ss_dir, events, sessions_seen, seen_api_calls, seen_signatures
            )

        code_dirs_found = False
        for base in app_dirs:
            for edition in ("Code", "Code - Insiders"):
                code_dir = base / edition
                ws = code_dir / "User" / "workspaceStorage"
                paths_checked.append(str(ws))
                if code_dir.exists():
                    code_dirs_found = True
                if ws.exists():
                    paths_found.append(str(ws))
                    self._scan_chat_sessions(ws, events, sessions_seen)
                if code_dir.exists():
                    found_agent_paths = self._scan_agent_session_data(
                        code_dir, events, sessions_seen, seen_api_calls, seen_signatures
                    )
                    paths_found.extend(found_agent_paths)

        if not paths_found and not code_dirs_found:
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到 VS Code 数据目录",
                paths_checked=paths_checked,
            )

        status = "exact" if events else "unavailable"
        label = "精确记录 (ChatSessions + AgentHost)" if events else "未检测到含 Token 的会话"
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
                "读取 ~/.copilot/session-state、VS Code agentSessionData/*/session.db "
                "与 workspaceStorage/*/chatSessions，支持 Copilot 与自定义 BYOK 模型。"
            ),
        )
