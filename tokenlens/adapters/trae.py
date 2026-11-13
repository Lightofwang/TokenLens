from __future__ import annotations

import json
import re
import struct
import zlib
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..models import AgentSourceInfo, UsageEvent
from .base import (
    BaseAdapter,
    estimate_tokens_from_text,
    get_app_data_dirs,
    local_tz,
    open_sqlite_ro,
    parse_epoch_dt,
    parse_iso_dt,
)

_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))")
_TRACE_ID_RE = re.compile(r'trace_id="([^"]+)"')
_SESSION_EQ_RE = re.compile(r"session_id=([A-Za-z0-9_-]{8,})")
_TITLE_LINE_RE = re.compile(r'session_id:\s*"([^"]+)",\s*title:\s*"([^"]*)"')
_CUSTOM_MODEL_CFG_RE = re.compile(r'config_name:\s*"([^"]+)"')
_CUSTOM_MODEL_NAME_RE = re.compile(r'model_name:\s*"([^"]+)"')
_TOKEN_EVENT_RE = re.compile(r"TokenUsageEvent\s*\{([^}]+)\}")
_RENDERER_TITLE_RE = re.compile(r'"chat_session_id"\s*:\s*"([^"]+)"[^}]*?"title"\s*:\s*"((?:[^"\\]|\\.)+)"')
_FETCH_BODY_LEN_RE = re.compile(r"\bbody_len=(\d+)")
_FETCH_URL_RE = re.compile(r"\burl=([^,\s]+)")
_SSE_MSG_COUNT_RE = re.compile(r"\bafter\s+(\d+)\s+messages\b")


def _extract_some_int(block: str, field: str) -> int:
    m = re.search(rf"\b{field}:\s*(?:Some\()?(\d+)\)?", block)
    return int(m.group(1)) if m else 0


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _unwrap_some(raw: str) -> str:
    s = raw.strip()
    if s.startswith('Some("') and s.endswith('")'):
        return s[6:-2]
    if s.startswith("Some(") and s.endswith(")"):
        return s[5:-1]
    return s


def _normalize_model_name(name: str) -> str:
    if name.startswith("customendpoint/"):
        return name[len("customendpoint/"):]
    return name


def _decode_cbor(buf: bytes, pos: int = 0) -> tuple[Any, int]:
    ib = buf[pos]
    pos += 1
    major, ai = ib >> 5, ib & 0x1F
    if ai < 24:
        val: int | None = ai
    elif ai == 24:
        val = buf[pos]
        pos += 1
    elif ai == 25:
        val = struct.unpack_from(">H", buf, pos)[0]
        pos += 2
    elif ai == 26:
        val = struct.unpack_from(">I", buf, pos)[0]
        pos += 4
    elif ai == 27:
        val = struct.unpack_from(">Q", buf, pos)[0]
        pos += 8
    elif ai == 31:
        val = None
    else:
        raise ValueError(f"Unsupported CBOR additional info: {ai}")

    if major == 0:
        return val, pos
    if major == 1:
        assert val is not None
        return -1 - val, pos
    if major == 2:
        if val is None:
            chunks: list[bytes] = []
            while buf[pos] != 0xFF:
                chunk, pos = _decode_cbor(buf, pos)
                if isinstance(chunk, bytes):
                    chunks.append(chunk)
            return b"".join(chunks), pos + 1
        end = pos + val
        if end > len(buf):
            raise EOFError("Truncated CBOR bytes")
        return buf[pos:end], end
    if major == 3:
        if val is None:
            parts: list[str] = []
            while buf[pos] != 0xFF:
                part, pos = _decode_cbor(buf, pos)
                parts.append(_to_text(part))
            return "".join(parts), pos + 1
        end = pos + val
        if end > len(buf):
            raise EOFError("Truncated CBOR string")
        return buf[pos:end].decode("utf-8", errors="replace"), end
    if major == 4:
        items: list[Any] = []
        if val is None:
            while buf[pos] != 0xFF:
                item, pos = _decode_cbor(buf, pos)
                items.append(item)
            return items, pos + 1
        for _ in range(val):
            item, pos = _decode_cbor(buf, pos)
            items.append(item)
        return items, pos
    if major == 5:
        mapping: dict[Any, Any] = {}
        if val is None:
            while buf[pos] != 0xFF:
                k, pos = _decode_cbor(buf, pos)
                v, pos = _decode_cbor(buf, pos)
                mapping[k] = v
            return mapping, pos + 1
        for _ in range(val):
            k, pos = _decode_cbor(buf, pos)
            v, pos = _decode_cbor(buf, pos)
            mapping[k] = v
        return mapping, pos
    if major == 6:
        item, pos = _decode_cbor(buf, pos)
        return item, pos
    if major == 7:
        if ai == 20:
            return False, pos
        if ai == 21:
            return True, pos
        if ai == 22 or ai == 23:
            return None, pos
        return ai, pos
    raise ValueError(f"Unsupported CBOR major type: {major}")


def _decompress_alaudalog(path: Path) -> bytes:
    """Decompress Trae 3.4+ `.alaudalog` raw-DEFLATE container frames."""
    try:
        raw = path.read_bytes()
    except OSError:
        return b""
    out_chunks: list[bytes] = []
    pos = 0
    while True:
        idx = raw.find(b"AALG", pos)
        if idx == -1 or idx + 20 > len(raw):
            break
        ver_len = struct.unpack_from("<I", raw, idx + 16)[0]
        if ver_len > 256:
            pos = idx + 4
            continue
        start = idx + 20 + ver_len
        if start >= len(raw):
            break
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        try:
            out_chunks.append(d.decompress(raw[start:]))
            out_chunks.append(d.flush())
        except zlib.error:
            pass
        if d.eof and d.unused_data:
            pos = len(raw) - len(d.unused_data)
        else:
            break
    return b"".join(out_chunks)


def _iter_cbor_maps(buf: bytes) -> Iterator[dict[str, Any]]:
    pos = 0
    n = len(buf)
    while pos < n:
        if (buf[pos] >> 5) == 5:
            try:
                obj, npos = _decode_cbor(buf, pos)
                if isinstance(obj, dict) and ("fields" in obj or "name" in obj):
                    yield obj
                pos = npos
                continue
            except Exception:
                pass
        pos += 1


def _parse_cbor_ts(ts_list: Any) -> datetime | None:
    if not isinstance(ts_list, list) or len(ts_list) < 6:
        return None
    try:
        y, doy, h, m, s, ns = (int(ts_list[i]) for i in range(6))
        dt_utc = datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=doy - 1, hours=h, minutes=m, seconds=s, microseconds=ns // 1000
        )
        return dt_utc.astimezone(local_tz())
    except Exception:
        return None


def _collect_renderer_titles(session_log_root: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    if not session_log_root.exists():
        return titles
    for rlog in session_log_root.glob("window*/renderer.log"):
        if not rlog.is_file():
            continue
        try:
            text = rlog.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _RENDERER_TITLE_RE.finditer(text):
            sid = m.group(1).strip()
            raw_title = m.group(2)
            try:
                decoded = json.loads(f'"{raw_title}"')
            except Exception:
                decoded = raw_title
            cleaned = str(decoded).strip().replace("\n", " ")[:120]
            if sid and cleaned:
                titles[sid] = cleaned
    return titles


class TraeAdapter(BaseAdapter):
    agent_id = "trae"
    agent_name = "Trae"
    color = "#a855f7"
    description = "Trae / Trae CN / MarsCode (精确日志解析 + 明文/隧道估算)"

    def __init__(
        self,
        app_dirs: list[Path] | None = None,
        marscode_dir: Path | None = None,
    ) -> None:
        self._app_dirs = app_dirs
        self._marscode_dir = marscode_dir

    def _scan_stdout_log(
        self,
        log_file: Path,
        app_label: str,
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> int:
        try:
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return 0

        added = 0
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
                    picked = _normalize_model_name(picked)
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

    def _scan_alaudalog(
        self,
        log_file: Path,
        app_label: str,
        renderer_titles: dict[str, str],
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> int:
        payload = _decompress_alaudalog(log_file)
        if not payload:
            return 0

        trace_models: dict[str, str] = {}
        trace_sessions: dict[str, str] = {}
        span_parents: dict[int, int] = {}
        span_req_ids: dict[int, str] = {}
        resp_logid_ctx: dict[str, tuple[str, str]] = {}
        custom_calls: dict[int, dict[str, Any]] = {}
        cloud_tasks: dict[str, list[dict[str, Any]]] = {}

        for rec in _iter_cbor_maps(payload):
            span_name = _to_text(rec.get("name"))
            if span_name:
                sid = rec.get("sid")
                psid = rec.get("psid")
                if isinstance(sid, int) and isinstance(psid, int):
                    span_parents[sid] = psid
                tags = rec.get("tags")
                if span_name == "custom_model_request" and isinstance(sid, int) and isinstance(tags, list):
                    for kv in tags:
                        if isinstance(kv, list) and len(kv) == 2 and _to_text(kv[0]) == "request_id":
                            span_req_ids[sid] = _unwrap_some(_to_text(kv[1]))
                continue

            fields = rec.get("fields")
            if not isinstance(fields, dict):
                continue

            msg = _to_text(fields.get("message"))
            trace_id = _to_text(fields.get("trace_id"))
            span_id = rec.get("span")
            dt = _parse_cbor_ts(rec.get("time"))

            sess_id = _to_text(fields.get("session_id")) or _to_text(fields.get("chat_session_id"))
            if sess_id and trace_id and not sess_id.startswith("__"):
                trace_sessions[trace_id] = sess_id
                for t in cloud_tasks.get(trace_id, ()):
                    if not t["session_id"]:
                        t["session_id"] = sess_id

            for mkey in ("custom_model_model_name", "custom_model_config_name", "model_name", "config_name"):
                if mkey in fields:
                    mval = _unwrap_some(_to_text(fields[mkey]))
                    if mval and mval != "None":
                        if mkey == "config_name" and "//" not in mval:
                            prov = _unwrap_some(_to_text(fields.get("provider")))
                            if prov and prov != "None":
                                mval = f"{prov}//{mval}"
                        mval = _normalize_model_name(mval)
                        if trace_id:
                            prev_m = trace_models.get(trace_id, "")
                            if not prev_m or prev_m.lower() == "auto" or "//" in mval:
                                trace_models[trace_id] = mval
                                for t in cloud_tasks.get(trace_id, ()):
                                    if not t["model"] or t["model"].lower() == "auto" or "//" in mval:
                                        t["model"] = mval
                        break

            url_field = _to_text(fields.get("url"))
            if "cloud stream response headers" in msg and ("create_agent_task" in url_field or "create_task" in url_field):
                resp_logid = _to_text(fields.get("resp_logid"))
                if resp_logid:
                    resp_logid_ctx[resp_logid] = (
                        trace_sessions.get(trace_id, ""),
                        trace_models.get(trace_id, ""),
                    )
                    if trace_id and trace_id in cloud_tasks:
                        cloud_tasks[trace_id][-1]["resp_logid"] = resp_logid
                continue

            if "calling Fetch" in msg and "method=POST" in msg:
                um = _FETCH_URL_RE.search(msg)
                bm = _FETCH_BODY_LEN_RE.search(msg)
                fetch_url = um.group(1) if um else ""
                # 1) CustomModelProxy local/custom endpoint call (preferred when present)
                if isinstance(span_id, int):
                    cur: int | None = span_id
                    cm_span: int | None = None
                    while cur is not None:
                        if cur in span_req_ids:
                            cm_span = cur
                            break
                        cur = span_parents.get(cur)
                    if cm_span is not None and bm and dt is not None:
                        custom_calls[cm_span] = {
                            "dt": dt,
                            "request_id": span_req_ids[cm_span],
                            "body_len": int(bm.group(1)),
                            "url": fetch_url,
                            "sse_messages": 0,
                            "duration_ms": 0,
                        }
                        continue
                # 2) Cloud create_agent_task call (captures Trae built-in cloud models too)
                if trace_id and ("create_agent_task" in fetch_url or "/create_task" in fetch_url) and bm and dt is not None:
                    cloud_tasks.setdefault(trace_id, []).append(
                        {
                            "dt": dt,
                            "end_dt": None,
                            "body_len": int(bm.group(1)),
                            "session_id": trace_sessions.get(trace_id, ""),
                            "model": trace_models.get(trace_id, ""),
                            "resp_logid": "",
                            "server_replies": 0,
                            "completed": False,
                        }
                    )
                continue

            if trace_id and trace_id in cloud_tasks:
                cur_task = cloud_tasks[trace_id][-1]
                if "recording history entry" in msg:
                    if _to_text(fields.get("history_source")) == "Server" and _to_text(fields.get("content_source")) != "UserInput":
                        cur_task["server_replies"] += 1
                elif "resolving explicit Cloud task completion as terminal" in msg or "chat: runner completed" in msg:
                    status_val = _to_text(fields.get("status"))
                    if dt is not None:
                        cur_task["end_dt"] = dt
                    if not status_val or "Err" not in status_val:
                        cur_task["completed"] = True

            if isinstance(span_id, int) and span_id in custom_calls:
                if "SSE stream ended normally after" in msg:
                    mm = _SSE_MSG_COUNT_RE.search(msg)
                    if mm:
                        custom_calls[span_id]["sse_messages"] = int(mm.group(1))
                elif "[CustomModelProxy] sse.end dispatched" in msg:
                    if fields.get("duration_ms") is not None:
                        try:
                            custom_calls[span_id]["duration_ms"] = int(_to_text(fields.get("duration_ms")) or 0)
                        except ValueError:
                            pass
                    if not custom_calls[span_id]["sse_messages"] and fields.get("last_seq") is not None:
                        try:
                            custom_calls[span_id]["sse_messages"] = int(fields.get("last_seq") or 0) + 1
                        except ValueError:
                            pass

        added = 0
        handled_resp_logids: set[str] = set()
        for call in custom_calls.values():
            ctx = resp_logid_ctx.get(call["request_id"])
            # Skip standalone connectivity checks (small test payload without chat task context)
            if ctx is None:
                continue
            handled_resp_logids.add(call["request_id"])
            sid, model = ctx
            sid = sid or call["request_id"]
            model = model or "trae-custom-model"
            body_len = int(call["body_len"] or 0)
            sse_msgs = int(call["sse_messages"] or 0)
            dur_ms = int(call["duration_ms"] or 0)
            if body_len <= 0:
                continue

            # Approximate UTF-8 OpenAI JSON payload at ~4 bytes/token, and ~2 tokens per SSE chunk
            prompt_est = max(1, round(body_len / 4))
            output_est = max(1, sse_msgs * 2) if sse_msgs > 0 else max(1, dur_ms // 200)
            full_sid = f"{app_label}:{sid}"
            sessions_seen.add(full_sid)
            events.append(
                UsageEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    session_id=full_sid,
                    timestamp=call["dt"],
                    model=model,
                    uncached_input=prompt_est,
                    output=output_est,
                    title=renderer_titles.get(sid) or f"{app_label} {sid[:10]}",
                    is_estimated=True,
                    duration_ms=dur_ms,
                    source_file=str(log_file),
                )
            )
            added += 1

        # Built-in cloud model turns that do not pass through local CustomModelProxy
        for trace_id, task_list in cloud_tasks.items():
            for task in task_list:
                rlogid = task["resp_logid"]
                if rlogid and rlogid in handled_resp_logids:
                    continue
                if not (task["completed"] or rlogid):
                    continue
                body_len = int(task["body_len"] or 0)
                sid = task["session_id"] or trace_sessions.get(trace_id, "")
                if body_len <= 0 or not sid or sid.startswith("__"):
                    continue
                model = task["model"] or trace_models.get(trace_id, "") or "trae-cloud"
                dur_ms = 0
                if task["end_dt"] is not None and task["end_dt"] >= task["dt"]:
                    dur_ms = int((task["end_dt"] - task["dt"]).total_seconds() * 1000)
                prompt_est = max(1, round(body_len / 4))
                output_est = max(16, int(task["server_replies"]) * 40, dur_ms // 150)
                full_sid = f"{app_label}:{sid}"
                sessions_seen.add(full_sid)
                events.append(
                    UsageEvent(
                        agent_id=self.agent_id,
                        agent_name=self.agent_name,
                        session_id=full_sid,
                        timestamp=task["dt"],
                        model=model,
                        uncached_input=prompt_est,
                        output=output_est,
                        title=renderer_titles.get(sid) or f"{app_label} {sid[:10]}",
                        is_estimated=True,
                        duration_ms=dur_ms,
                        source_file=str(log_file),
                    )
                )
                added += 1
        return added

    def _scan_ai_agent_logs(
        self,
        log_root: Path,
        app_label: str,
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> tuple[int, int]:
        if not log_root.exists():
            return 0, 0
        exact_added = 0
        estimated_added = 0
        title_cache: dict[Path, dict[str, str]] = {}

        for log_file in log_root.rglob("ai-agent*stdout.log"):
            if log_file.is_file():
                exact_added += self._scan_stdout_log(log_file, app_label, events, sessions_seen)

        for alaudalog in log_root.rglob("ai_agent*.alaudalog"):
            if not alaudalog.is_file():
                continue
            session_dir = alaudalog.parent.parent if alaudalog.parent.name == "Modular" else alaudalog.parent
            if session_dir not in title_cache:
                title_cache[session_dir] = _collect_renderer_titles(session_dir)
            estimated_added += self._scan_alaudalog(
                alaudalog,
                app_label,
                title_cache[session_dir],
                events,
                sessions_seen,
            )

        return exact_added, estimated_added

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
        estimated_events = 0

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
                    ex, est = self._scan_ai_agent_logs(log_dir, app_name, events, sessions_seen)
                    exact_log_events += ex
                    estimated_events += est

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
                                        estimated_events += 1
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
            ex, est = self._scan_ai_agent_logs(mc_logs, "MarsCode", events, sessions_seen)
            exact_log_events += ex
            estimated_events += est

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

        if exact_log_events and estimated_events:
            status = "estimated"
            label = "日志精确 + 估算 (部分库已加密)"
        elif exact_log_events:
            status = "exact"
            label = "日志精确记录 (部分库已加密)"
        elif estimated_events:
            status = "estimated"
            label = "隧道/文本估算 (原生库已加密)"
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
                "已解析旧版 logs TokenUsageEvent、新版 3.4+ ai_agent*.alaudalog 中的云端内置模型"
                "与自定义模型请求载荷估算（create_agent_task/body_len/SSE）以及明文 state.vscdb。"
            ),
        )
