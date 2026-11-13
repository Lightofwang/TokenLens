from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, get_app_data_dirs, open_sqlite_ro, parse_epoch_dt, parse_iso_dt

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))")
_TRACE_ID_RE = re.compile(r'trace_id="([^"]+)"')
_SESSION_EQ_RE = re.compile(r"session_id=([A-Za-z0-9_-]{8,})")
_TITLE_LINE_RE = re.compile(r'session_id:\s*"([^"]+)",\s*title:\s*"([^"]*)"')
_CUSTOM_MODEL_CFG_RE = re.compile(r'config_name:\s*"([^"]+)"')
_CUSTOM_MODEL_NAME_RE = re.compile(r'model_name:\s*"([^"]+)"')
_TOKEN_EVENT_RE = re.compile(r"TokenUsageEvent\s*\{([^}]+)\}")


def _extract_some_int(block: str, field: str) -> int:
    m = re.search(rf"\b{field}:\s*(?:Some\()?(\d+)\)?", block)
    return int(m.group(1)) if m else 0


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
    description = "Trae / Trae CN / MarsCode (精确日志解析 + 明文历史估算)"

    def __init__(
        self,
        app_dirs: list[Path] | None = None,
        marscode_dir: Path | None = None,
    ) -> None:
        self._app_dirs = app_dirs
        self._marscode_dir = marscode_dir

    def _scan_ai_agent_logs(
        self,
        log_root: Path,
        app_label: str,
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> int:
        if not log_root.exists():
            return 0
        added = 0
        for log_file in log_root.rglob("ai-agent*stdout.log"):
            if not log_file.is_file():
                continue
            try:
                lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            trace_models: dict[str, str] = {}
            trace_sessions: dict[str, str] = {}
            session_titles: dict[str, str] = {}
            last_model = "trae-agent"
            last_sid = log_file.stem

            for ln in lines:
                ts_m = _LOG_TS_RE.match(ln)
                trace_m = _TRACE_ID_RE.search(ln)
                trace_id = trace_m.group(1) if trace_m else ""

                sid_m = _SESSION_EQ_RE.search(ln)
                if sid_m:
                    last_sid = sid_m.group(1)
                    if trace_id:
                        trace_sessions[trace_id] = last_sid

                title_m = _TITLE_LINE_RE.search(ln)
                if title_m:
                    t_sid, t_val = title_m.group(1), title_m.group(2).strip()[:120]
                    if t_val:
                        session_titles[t_sid] = t_val

                if "model_info: CustomModel" in ln:
                    cfg_m = _CUSTOM_MODEL_CFG_RE.search(ln)
                    mod_m = _CUSTOM_MODEL_NAME_RE.search(ln)
                    cfg_val = cfg_m.group(1) if cfg_m else ""
                    mod_val = mod_m.group(1) if mod_m else ""
                    is_utility = cfg_val in ("title_generation", "custom_model_placeholder") or mod_val == "no_thinking_model"
                    picked = ""
                    if cfg_val and cfg_val not in ("title_generation", "custom_model_placeholder"):
                        picked = cfg_val
                    elif mod_val and mod_val != "no_thinking_model":
                        picked = mod_val
                    if picked:
                        if picked.startswith("customendpoint/"):
                            picked = picked[len("customendpoint/"):]
                        if not is_utility:
                            last_model = picked
                        if trace_id:
                            trace_models[trace_id] = picked
                    elif is_utility and trace_id:
                        trace_models[trace_id] = "trae-utility"
                    continue

                if "TokenUsageEvent" not in ln or not ts_m:
                    continue
                te_m = _TOKEN_EVENT_RE.search(ln)
                if not te_m:
                    continue
                dt = parse_iso_dt(ts_m.group(1))
                if not dt:
                    continue

                block = te_m.group(1)
                prompt_tok = _extract_some_int(block, "prompt_tokens")
                comp_tok = _extract_some_int(block, "completion_tokens")
                cache_read = _extract_some_int(block, "cache_read_input_tokens")
                cache_write = _extract_some_int(block, "cache_creation_input_tokens")
                reasoning = _extract_some_int(block, "reasoning_tokens")
                # Prefer cached vs uncached breakdown if prompt_tokens includes cache reads
                uncached_in = max(0, prompt_tok - cache_read) if prompt_tok >= cache_read else prompt_tok
                if uncached_in + cache_read + cache_write + comp_tok <= 0:
                    continue

                sid = trace_sessions.get(trace_id, last_sid)
                full_sid = f"{app_label}:{sid}"
                if "generate_session_title" in ln or "generate_commit_message" in ln:
                    task_model = trace_models.get(trace_id) or (
                        "trae-utility" if last_model == "trae-agent" else last_model
                    )
                else:
                    task_model = trace_models.get(trace_id) or last_model

                sessions_seen.add(full_sid)
                events.append(
                    UsageEvent(
                        agent_id=self.agent_id,
                        agent_name=self.agent_name,
                        session_id=full_sid,
                        timestamp=dt,
                        model=task_model,
                        uncached_input=uncached_in,
                        cache_read=cache_read,
                        cache_write=cache_write,
                        output=comp_tok,
                        reasoning=reasoning,
                        title=session_titles.get(sid) or f"{app_label} {sid[:10]}",
                        is_estimated=False,
                        source_file=str(log_file),
                    )
                )
                added += 1
        return added

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        app_dirs = self._app_dirs if self._app_dirs is not None else get_app_data_dirs()
        marscode_dir = (
            self._marscode_dir
            if self._marscode_dir is not None
            else (Path.home() / ".marscode")
        )

        events: list[UsageEvent] = []
        paths_checked: list[str] = []
        paths_found: list[str] = []
        sessions_seen: set[str] = set()
        encrypted_dbs: list[str] = []
        exact_log_events = 0
        estimated_db_events = 0

        for base in app_dirs:
            for app_name in ("Trae", "Trae CN"):
                enc_db = base / app_name / "ModularData" / "ai-agent" / "database.db"
                paths_checked.append(str(enc_db))
                if enc_db.exists():
                    encrypted_dbs.append(str(enc_db))
                    paths_found.append(str(enc_db))

                log_dir = base / app_name / "logs"
                paths_checked.append(str(log_dir))
                if log_dir.exists():
                    paths_found.append(str(log_dir))
                    exact_log_events += self._scan_ai_agent_logs(log_dir, app_name, events, sessions_seen)

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
                                        estimated_db_events += 1
                    except Exception:
                        continue

        # Also check VS Code MarsCode / Trae extension storage (~/.marscode)
        mc_db = marscode_dir / "ai-chat" / "AppData" / "vscode" / "ai-agent" / "database.db"
        paths_checked.append(str(mc_db))
        if mc_db.exists():
            encrypted_dbs.append(str(mc_db))
            paths_found.append(str(mc_db))
        mc_logs = marscode_dir / "logs"
        if mc_logs.exists():
            paths_found.append(str(mc_logs))
            exact_log_events += self._scan_ai_agent_logs(mc_logs, "MarsCode", events, sessions_seen)

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

        if exact_log_events and estimated_db_events:
            status = "estimated"
            label = "日志精确 + 文本估算 (库已加密)"
        elif exact_log_events:
            status = "exact"
            label = "日志精确记录 (部分库已加密)"
        elif estimated_db_events:
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
                "Trae / Trae CN 核心对话库 ModularData/ai-agent/database.db 使用 SQLCipher 加密；"
                "已解析 logs 中的精确 TokenUsageEvent 与明文 state.vscdb 估算记录（新版云端代理及自定义模型"
                "聊天正文仅存加密库中，客户端日志未完整输出响应 Token）。"
            ),
        )
