from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..models import AgentSourceInfo, UsageEvent
from .base import (
    BaseAdapter,
    close_sqlite_snapshot,
    estimate_tokens_from_text,
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
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path
    return raw


def _decode_json_value(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore").strip()
    else:
        text = str(raw or "").strip()
    if not text:
        return None
    # Cursor Agent store.db meta values are sometimes hex-encoded UTF-8 JSON
    if all(ch in "0123456789abcdefABCDEF" for ch in text) and len(text) % 2 == 0 and len(text) >= 4:
        try:
            decoded = bytes.fromhex(text).decode("utf-8", errors="strict").strip()
            if decoded.startswith(("{", "[")):
                return json.loads(decoded)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_cursor_model(raw_model: str | None, fallback: str = "cursor-agent") -> str:
    m = str(raw_model or "").strip()
    if not m or m.lower() in ("default", "auto"):
        return fallback
    return m


_CURSOR_SYS_OVERHEAD = 1000
_CURSOR_MAX_CONTEXT = 80_000
_CURSOR_MAX_TURN_EXTRA = 16_000
_MICRO_BUBBLE_MERGE_SECS = 15.0


def _estimate_turn_tokens(
    rolling_ctx: int,
    extra_ctx: int,
    out_est: int,
    sys_overhead: int = _CURSOR_SYS_OVERHEAD,
) -> tuple[int, int, int]:
    """Return (uncached_input, cache_read, new_rolling_ctx) without O(N^2) blow-up."""
    hist = min(max(0, rolling_ctx), _CURSOR_MAX_CONTEXT)
    extra = min(max(0, extra_ctx), _CURSOR_MAX_TURN_EXTRA)
    if hist > 0:
        cache_read = int(hist * 0.75)
        uncached_in = (hist - cache_read) + extra + sys_overhead
    else:
        cache_read = 0
        uncached_in = extra + sys_overhead
    # Never compound constant sys_overhead into rolling history
    new_rolling = min(hist + extra + out_est, _CURSOR_MAX_CONTEXT)
    return uncached_in, cache_read, new_rolling


def _bubble_sort_key(bubble: dict[str, Any]) -> float:
    dt = parse_iso_dt(bubble.get("createdAt")) or parse_epoch_dt(bubble.get("createdAt"))
    return dt.timestamp() if dt else 0.0


def _extract_bubble_text(bubble: dict[str, Any]) -> tuple[str, str]:
    """Extract (prompt_context_text, assistant_reply_text) from a Cursor bubble object."""
    prompt_parts: list[str] = []
    reply_parts: list[str] = []

    raw_text = bubble.get("text") or bubble.get("rawText") or ""
    if isinstance(raw_text, str) and raw_text.strip():
        if int(bubble.get("type") or 0) == 1:
            prompt_parts.append(raw_text)
        else:
            reply_parts.append(raw_text)

    thinking = bubble.get("thinking")
    if isinstance(thinking, dict) and thinking.get("text"):
        reply_parts.append(str(thinking["text"]))
    elif isinstance(thinking, str) and thinking.strip():
        reply_parts.append(thinking)

    for blk in bubble.get("codeBlocks") or []:
        if isinstance(blk, dict) and blk.get("content"):
            reply_parts.append(str(blk["content"]))

    tool_data = bubble.get("toolFormerData")
    if isinstance(tool_data, dict):
        for k in ("rawArgs", "params"):
            if tool_data.get(k):
                reply_parts.append(str(tool_data[k]))
        if tool_data.get("result"):
            prompt_parts.append(str(tool_data["result"]))

    ctx = bubble.get("context")
    if isinstance(ctx, dict):
        for sel_key in ("selections", "terminalSelections", "fileSelections"):
            items = ctx.get(sel_key)
            if isinstance(items, list) and items:
                prompt_parts.append(json.dumps(items, ensure_ascii=False))

    return "\n".join(prompt_parts), "\n".join(reply_parts)


def _extract_transcript_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "thinking", "content", "input"):
                    val = item.get(key)
                    if isinstance(val, str) and val:
                        parts.append(val)
                    elif isinstance(val, dict):
                        parts.append(json.dumps(val, ensure_ascii=False))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return ""


class CursorAdapter(BaseAdapter):
    agent_id = "cursor"
    agent_name = "Cursor"
    color = "#f43f5e"
    description = "Cursor IDE / Composer / Agent CLI (~/.cursor + state.vscdb)"

    def __init__(
        self,
        app_dirs: list[Path] | None = None,
        cursor_dir: Path | None = None,
    ) -> None:
        self._app_dirs = app_dirs
        self._cursor_dir = cursor_dir

    def _scan_workspace_vscdbs(
        self,
        ws_root: Path,
        composer_meta: dict[str, dict[str, Any]],
        legacy_tabs: list[tuple[Path, str, dict[str, Any]]],
    ) -> None:
        for db in ws_root.glob("*/state.vscdb"):
            if not db.is_file():
                continue
            ws_dir = db.parent
            project_path = ""
            ws_json = ws_dir / "workspace.json"
            if ws_json.is_file():
                try:
                    wobj = json.loads(ws_json.read_text(encoding="utf-8", errors="ignore"))
                    if isinstance(wobj, dict):
                        project_path = _decode_file_uri(wobj.get("folder") or wobj.get("workspace"))
                except Exception:
                    pass

            conn = open_sqlite_snapshot(db)
            if conn is None:
                continue
            try:
                tables = {
                    r[0]
                    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                if "ItemTable" not in tables:
                    continue
                rows = conn.execute(
                    "SELECT key, value FROM ItemTable WHERE key IN "
                    "('composer.composerData', 'workbench.panel.aichat.view.aichat.chatdata')"
                ).fetchall()
                for key, raw_val in rows:
                    data = _decode_json_value(raw_val)
                    if not isinstance(data, dict):
                        continue
                    if key == "composer.composerData":
                        for comp in data.get("allComposers") or []:
                            if not isinstance(comp, dict):
                                continue
                            cid = str(comp.get("composerId") or "").strip()
                            if not cid:
                                continue
                            meta = composer_meta.setdefault(cid, {})
                            if not meta.get("project_path") and project_path:
                                meta["project_path"] = project_path
                            if not meta.get("title") and comp.get("name"):
                                meta["title"] = str(comp["name"]).strip()[:120]
                            meta.setdefault("created_dt", parse_epoch_dt(comp.get("createdAt")))
                            meta.setdefault("updated_dt", parse_epoch_dt(comp.get("lastUpdatedAt")))
                    elif key == "workbench.panel.aichat.view.aichat.chatdata":
                        for tab in data.get("tabs") or []:
                            if isinstance(tab, dict):
                                legacy_tabs.append((db, project_path, tab))
            except Exception:
                continue
            finally:
                close_sqlite_snapshot(conn)

    def _emit_composer_bubbles(
        self,
        comp_id: str,
        bubbles: list[dict[str, Any]],
        meta: dict[str, Any],
        default_model: str,
        fallback_dt: datetime,
        source_file: Path,
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> None:
        rolling_ctx = 0
        title = str(meta.get("title") or "").strip()[:120]
        project_path = str(meta.get("project_path") or "")
        user_pending = True
        last_est_event: UsageEvent | None = None

        for bubble in bubbles:
            btype = int(bubble.get("type") or 0)
            prompt_ctx, reply_txt = _extract_bubble_text(bubble)
            if not title and btype == 1 and prompt_ctx.strip():
                title = prompt_ctx.strip().replace("\n", " ")[:120]

            tc = bubble.get("tokenCount") if isinstance(bubble.get("tokenCount"), dict) else {}
            in_tok = int(tc.get("inputTokens") or 0)
            out_tok = int(tc.get("outputTokens") or 0)
            cr_tok = int(tc.get("cacheReadTokens") or tc.get("cachedTokens") or 0)
            cw_tok = int(tc.get("cacheWriteTokens") or 0)

            if btype == 1 and in_tok + out_tok + cr_tok + cw_tok == 0:
                user_tok = min(estimate_tokens_from_text(prompt_ctx), _CURSOR_MAX_TURN_EXTRA)
                rolling_ctx = min(rolling_ctx + user_tok + 100, _CURSOR_MAX_CONTEXT)
                user_pending = True
                last_est_event = None
                continue

            # Process assistant turns (type == 2) or any bubble carrying explicit tokenCount
            if btype != 2 and in_tok + out_tok + cr_tok + cw_tok <= 0:
                continue

            dt = (
                parse_iso_dt(bubble.get("createdAt"))
                or parse_epoch_dt(bubble.get("createdAt"))
                or meta.get("updated_dt")
                or meta.get("created_dt")
                or fallback_dt
            )
            m_info = bubble.get("modelInfo") if isinstance(bubble.get("modelInfo"), dict) else {}
            model = _normalize_cursor_model(
                m_info.get("modelName") or bubble.get("model") or default_model,
                fallback=default_model,
            )

            if in_tok + out_tok + cr_tok + cw_tok > 0:
                rolling_ctx = min(max(rolling_ctx, in_tok + cr_tok + out_tok), _CURSOR_MAX_CONTEXT)
                user_pending = False
                last_est_event = None
                sessions_seen.add(comp_id)
                events.append(
                    UsageEvent(
                        agent_id=self.agent_id,
                        agent_name=self.agent_name,
                        session_id=comp_id,
                        timestamp=dt,
                        model=model,
                        uncached_input=in_tok,
                        cache_read=cr_tok,
                        cache_write=cw_tok,
                        output=out_tok,
                        title=title or f"Cursor {comp_id[:8]}",
                        project_path=project_path,
                        is_estimated=False,
                        source_file=str(source_file),
                    )
                )
            else:
                out_est = estimate_tokens_from_text(reply_txt)
                if out_est <= 0:
                    continue
                extra_ctx = estimate_tokens_from_text(prompt_ctx)

                # Cursor splits single agent turns into many adjacent micro-bubbles
                # (thinking/text/code/tool stubs); merge adjacent pure-output fragments
                # within the same turn instead of re-charging full prompt context.
                if (
                    last_est_event is not None
                    and not user_pending
                    and extra_ctx == 0
                    and last_est_event.model == model
                    and 0.0 <= (dt - last_est_event.timestamp).total_seconds() <= _MICRO_BUBBLE_MERGE_SECS
                ):
                    last_est_event.output += out_est
                    last_est_event.total_tokens += out_est
                    rolling_ctx = min(rolling_ctx + out_est, _CURSOR_MAX_CONTEXT)
                    continue

                uncached_in, cr_est, rolling_ctx = _estimate_turn_tokens(rolling_ctx, extra_ctx, out_est)
                user_pending = False
                sessions_seen.add(comp_id)
                ev = UsageEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    session_id=comp_id,
                    timestamp=dt,
                    model=model,
                    uncached_input=uncached_in,
                    cache_read=cr_est,
                    output=out_est,
                    title=title or f"Cursor {comp_id[:8]}",
                    project_path=project_path,
                    is_estimated=True,
                    source_file=str(source_file),
                )
                events.append(ev)
                last_est_event = ev

    def _scan_global_vscdb(
        self,
        global_db: Path,
        composer_meta: dict[str, dict[str, Any]],
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> None:
        conn = open_sqlite_snapshot(global_db)
        if conn is None:
            return
        try:
            tables = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "composerHeaders" in tables:
                for cid, created_at, updated_at, raw_val in conn.execute(
                    "SELECT composerId, createdAt, lastUpdatedAt, value FROM composerHeaders"
                ).fetchall():
                    comp_id = str(cid or "").strip()
                    if not comp_id:
                        continue
                    meta = composer_meta.setdefault(comp_id, {})
                    if not meta.get("created_dt"):
                        meta["created_dt"] = parse_epoch_dt(created_at)
                    if not meta.get("updated_dt"):
                        meta["updated_dt"] = parse_epoch_dt(updated_at)
                    hobj = _decode_json_value(raw_val)
                    if isinstance(hobj, dict) and not meta.get("title") and hobj.get("name"):
                        meta["title"] = str(hobj["name"]).strip()[:120]

            if "cursorDiskKV" not in tables:
                return

            composers: dict[str, dict[str, Any]] = {}
            for key, raw_val in conn.execute(
                "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"
            ).fetchall():
                comp_id = str(key).split(":", 1)[1]
                cobj = _decode_json_value(raw_val)
                if isinstance(cobj, dict):
                    composers[comp_id] = cobj

            bubbles_by_comp: dict[str, dict[str, dict[str, Any]]] = {}
            for key, raw_val in conn.execute(
                "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'"
            ).fetchall():
                parts = str(key).split(":", 2)
                if len(parts) < 3:
                    continue
                comp_id, bid = parts[1], parts[2]
                bobj = _decode_json_value(raw_val)
                if isinstance(bobj, dict):
                    bubbles_by_comp.setdefault(comp_id, {})[bid] = bobj

            file_dt = datetime.fromtimestamp(global_db.stat().st_mtime, tz=local_tz())
            all_comp_ids = list(dict.fromkeys(list(composers.keys()) + list(bubbles_by_comp.keys())))

            for comp_id in all_comp_ids:
                cobj = composers.get(comp_id) or {}
                meta = composer_meta.setdefault(comp_id, {})
                if not meta.get("title") and cobj.get("name"):
                    meta["title"] = str(cobj["name"]).strip()[:120]
                if not meta.get("created_dt"):
                    meta["created_dt"] = parse_epoch_dt(cobj.get("createdAt"))
                if not meta.get("updated_dt"):
                    meta["updated_dt"] = parse_epoch_dt(cobj.get("lastUpdatedAt"))

                mcfg = cobj.get("modelConfig") if isinstance(cobj.get("modelConfig"), dict) else {}
                default_model = _normalize_cursor_model(mcfg.get("modelName"), fallback="cursor-agent")

                bmap = bubbles_by_comp.get(comp_id) or {}
                ordered_bubbles: list[dict[str, Any]] = []
                headers = cobj.get("fullConversationHeadersOnly")
                if isinstance(headers, list) and headers and bmap:
                    seen_bids: set[str] = set()
                    for h in headers:
                        if not isinstance(h, dict):
                            continue
                        bid = str(h.get("bubbleId") or "")
                        if bid and bid in bmap:
                            ordered_bubbles.append(bmap[bid])
                            seen_bids.add(bid)
                    for bid, bobj in bmap.items():
                        if bid not in seen_bids:
                            ordered_bubbles.append(bobj)
                elif bmap:
                    ordered_bubbles = sorted(bmap.values(), key=_bubble_sort_key)
                elif isinstance(cobj.get("conversation"), list):
                    ordered_bubbles = [b for b in cobj["conversation"] if isinstance(b, dict)]

                if not ordered_bubbles:
                    continue

                self._emit_composer_bubbles(
                    comp_id,
                    ordered_bubbles,
                    meta,
                    default_model,
                    file_dt,
                    global_db,
                    events,
                    sessions_seen,
                )
        except Exception:
            pass
        finally:
            close_sqlite_snapshot(conn)

    def _scan_legacy_chat_tabs(
        self,
        legacy_tabs: list[tuple[Path, str, dict[str, Any]]],
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> None:
        for db, project_path, tab in legacy_tabs:
            tab_id = str(tab.get("tabId") or "").strip()
            if not tab_id or tab_id in sessions_seen:
                continue
            bubbles = tab.get("bubbles")
            if not isinstance(bubbles, list) or not bubbles:
                continue

            tab_title = str(tab.get("chatTitle") or "").strip()[:120]
            fallback_dt = parse_epoch_dt(tab.get("lastSendTime")) or datetime.fromtimestamp(
                db.stat().st_mtime, tz=local_tz()
            )
            rolling_ctx = 0
            for b in bubbles:
                if not isinstance(b, dict):
                    continue
                btype = str(b.get("type") or "").lower()
                text = str(b.get("text") or b.get("rawText") or "")
                if btype == "user":
                    if not tab_title and text.strip():
                        tab_title = text.strip().replace("\n", " ")[:120]
                    user_tok = min(estimate_tokens_from_text(text), _CURSOR_MAX_TURN_EXTRA)
                    rolling_ctx = min(rolling_ctx + user_tok + 100, _CURSOR_MAX_CONTEXT)
                    continue
                if btype != "ai":
                    continue
                tc = b.get("tokenCount") if isinstance(b.get("tokenCount"), dict) else {}
                in_tok = int(tc.get("inputTokens") or 0)
                out_tok = int(tc.get("outputTokens") or 0)
                cr_tok = int(tc.get("cacheReadTokens") or tc.get("cachedTokens") or 0)
                model = _normalize_cursor_model(b.get("modelType") or b.get("modelName"), "cursor-chat")
                dt = parse_epoch_dt(b.get("timestamp")) or fallback_dt

                if in_tok + out_tok + cr_tok > 0:
                    rolling_ctx = min(max(rolling_ctx, in_tok + cr_tok + out_tok), _CURSOR_MAX_CONTEXT)
                    is_est = False
                else:
                    out_tok = estimate_tokens_from_text(text)
                    if out_tok <= 0:
                        continue
                    in_tok, cr_tok, rolling_ctx = _estimate_turn_tokens(rolling_ctx, 0, out_tok)
                    is_est = True

                sessions_seen.add(tab_id)
                events.append(
                    UsageEvent(
                        agent_id=self.agent_id,
                        agent_name=self.agent_name,
                        session_id=tab_id,
                        timestamp=dt,
                        model=model,
                        uncached_input=in_tok,
                        cache_read=cr_tok,
                        output=out_tok,
                        title=tab_title or f"Cursor {tab_id[:8]}",
                        project_path=project_path,
                        is_estimated=is_est,
                        source_file=str(db),
                    )
                )

    def _scan_cli_store_dbs(
        self,
        cursor_dir: Path,
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> list[str]:
        found: list[str] = []
        chats_dir = cursor_dir / "chats"
        if not chats_dir.exists():
            return found
        found.append(str(chats_dir))

        for db in chats_dir.rglob("store.db"):
            if not db.is_file():
                continue
            conn = open_sqlite_snapshot(db)
            if conn is None:
                continue
            try:
                tables = {
                    r[0]
                    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                meta_obj: dict[str, Any] = {}
                if "meta" in tables:
                    for _, val in conn.execute("SELECT key, value FROM meta").fetchall():
                        parsed = _decode_json_value(val)
                        if isinstance(parsed, dict):
                            meta_obj.update(parsed)

                sid = str(meta_obj.get("agentId") or db.parent.name).strip()
                if not sid or sid in sessions_seen or "blobs" not in tables:
                    continue

                title = str(meta_obj.get("name") or "").strip()[:120]
                default_model = _normalize_cursor_model(meta_obj.get("lastUsedModel"), "cursor-cli")
                fallback_dt = parse_epoch_dt(meta_obj.get("createdAt")) or datetime.fromtimestamp(
                    db.stat().st_mtime, tz=local_tz()
                )

                rolling_ctx = 0
                for _, raw_blob in conn.execute("SELECT id, data FROM blobs").fetchall():
                    bobj = _decode_json_value(raw_blob)
                    if not isinstance(bobj, dict):
                        continue
                    role = str(bobj.get("role") or "").lower()
                    content = _extract_transcript_content(bobj.get("content"))
                    if role == "user":
                        if not title and content.strip():
                            title = content.strip().replace("\n", " ")[:120]
                        user_tok = min(estimate_tokens_from_text(content), _CURSOR_MAX_TURN_EXTRA)
                        rolling_ctx = min(rolling_ctx + user_tok + 100, _CURSOR_MAX_CONTEXT)
                    elif role == "assistant":
                        usage = bobj.get("usage") if isinstance(bobj.get("usage"), dict) else {}
                        in_tok = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
                        out_tok = int(usage.get("outputTokens") or usage.get("output_tokens") or 0)
                        cr_tok = int(usage.get("cacheReadTokens") or usage.get("cache_read_input_tokens") or 0)
                        if in_tok + out_tok + cr_tok > 0:
                            rolling_ctx = min(max(rolling_ctx, in_tok + cr_tok + out_tok), _CURSOR_MAX_CONTEXT)
                            is_est = False
                        else:
                            out_tok = estimate_tokens_from_text(content)
                            if out_tok <= 0:
                                continue
                            in_tok, cr_tok, rolling_ctx = _estimate_turn_tokens(rolling_ctx, 0, out_tok)
                            is_est = True

                        sessions_seen.add(sid)
                        events.append(
                            UsageEvent(
                                agent_id=self.agent_id,
                                agent_name=self.agent_name,
                                session_id=sid,
                                timestamp=parse_epoch_dt(bobj.get("createdAt")) or fallback_dt,
                                model=_normalize_cursor_model(bobj.get("model"), default_model),
                                uncached_input=in_tok,
                                cache_read=cr_tok,
                                output=out_tok,
                                title=title or f"Cursor CLI {sid[:8]}",
                                is_estimated=is_est,
                                source_file=str(db),
                            )
                        )
            except Exception:
                continue
            finally:
                close_sqlite_snapshot(conn)
        return found

    def _scan_agent_transcripts(
        self,
        cursor_dir: Path,
        events: list[UsageEvent],
        sessions_seen: set[str],
    ) -> list[str]:
        found: list[str] = []
        proj_dir = cursor_dir / "projects"
        if not proj_dir.exists():
            return found

        transcripts = list(proj_dir.glob("*/agent-transcripts/*.jsonl")) + list(
            proj_dir.glob("*/agent-transcripts/*/*.jsonl")
        )
        if transcripts:
            found.append(str(proj_dir))

        for fp in transcripts:
            if not fp.is_file():
                continue
            sid = fp.stem
            if not sid or sid in sessions_seen:
                continue
            try:
                lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            fallback_dt = datetime.fromtimestamp(fp.stat().st_mtime, tz=local_tz())
            title = ""
            rolling_ctx = 0
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                role = str(rec.get("role") or "").lower()
                msg = rec.get("message") if isinstance(rec.get("message"), dict) else rec
                text = _extract_transcript_content(msg.get("content") or rec.get("text"))
                if role == "user":
                    if not title and text.strip():
                        title = text.strip().replace("\n", " ")[:120]
                    user_tok = min(estimate_tokens_from_text(text), _CURSOR_MAX_TURN_EXTRA)
                    rolling_ctx = min(rolling_ctx + user_tok + 100, _CURSOR_MAX_CONTEXT)
                elif role == "assistant":
                    usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else (
                        rec.get("usage") if isinstance(rec.get("usage"), dict) else {}
                    )
                    in_tok = int(usage.get("input_tokens") or usage.get("inputTokens") or 0)
                    out_tok = int(usage.get("output_tokens") or usage.get("outputTokens") or 0)
                    cr_tok = int(usage.get("cache_read_input_tokens") or usage.get("cacheReadTokens") or 0)
                    cw_tok = int(usage.get("cache_creation_input_tokens") or usage.get("cacheWriteTokens") or 0)

                    if in_tok + out_tok + cr_tok + cw_tok > 0:
                        rolling_ctx = min(max(rolling_ctx, in_tok + cr_tok + out_tok), _CURSOR_MAX_CONTEXT)
                        is_est = False
                    else:
                        out_tok = estimate_tokens_from_text(text)
                        if out_tok <= 0:
                            continue
                        in_tok, cr_tok, rolling_ctx = _estimate_turn_tokens(rolling_ctx, 0, out_tok)
                        is_est = True

                    dt = parse_iso_dt(rec.get("timestamp")) or parse_epoch_dt(rec.get("timestamp")) or fallback_dt
                    model = _normalize_cursor_model(msg.get("model") or rec.get("model"), "cursor-agent")
                    sessions_seen.add(sid)
                    events.append(
                        UsageEvent(
                            agent_id=self.agent_id,
                            agent_name=self.agent_name,
                            session_id=sid,
                            timestamp=dt,
                            model=model,
                            uncached_input=in_tok,
                            cache_read=cr_tok,
                            cache_write=cw_tok,
                            output=out_tok,
                            title=title or f"Cursor {sid[:8]}",
                            is_estimated=is_est,
                            source_file=str(fp),
                        )
                    )
        return found

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        app_dirs = self._app_dirs if self._app_dirs is not None else get_app_data_dirs()
        cursor_dir = self._cursor_dir if self._cursor_dir is not None else (Path.home() / ".cursor")

        events: list[UsageEvent] = []
        paths_checked: list[str] = [str(cursor_dir)]
        paths_found: list[str] = []
        sessions_seen: set[str] = set()
        composer_meta: dict[str, dict[str, Any]] = {}
        legacy_tabs: list[tuple[Path, str, dict[str, Any]]] = []
        global_dbs: list[Path] = []
        cursor_dirs_found = False

        if cursor_dir.exists():
            cursor_dirs_found = True
            paths_found.append(str(cursor_dir))

        for base in app_dirs:
            cursor_app = base / "Cursor"
            user_dir = cursor_app / "User"
            global_db = user_dir / "globalStorage" / "state.vscdb"
            ws_dir = user_dir / "workspaceStorage"
            paths_checked.append(str(global_db))

            if cursor_app.exists():
                cursor_dirs_found = True
            if ws_dir.exists():
                paths_found.append(str(ws_dir))
                self._scan_workspace_vscdbs(ws_dir, composer_meta, legacy_tabs)
            if global_db.is_file():
                paths_found.append(str(global_db))
                global_dbs.append(global_db)

        for global_db in global_dbs:
            self._scan_global_vscdb(global_db, composer_meta, events, sessions_seen)

        self._scan_legacy_chat_tabs(legacy_tabs, events, sessions_seen)

        if cursor_dir.exists():
            paths_found.extend(self._scan_cli_store_dbs(cursor_dir, events, sessions_seen))
            paths_found.extend(self._scan_agent_transcripts(cursor_dir, events, sessions_seen))

        if not paths_found and not cursor_dirs_found:
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到 Cursor 数据目录",
                paths_checked=paths_checked,
            )

        exact_cnt = sum(1 for e in events if not e.is_estimated)
        est_cnt = len(events) - exact_cnt
        if exact_cnt and est_cnt:
            status = "estimated"
            label = "精确记录 + 文本估算"
        elif exact_cnt:
            status = "exact"
            label = "精确记录 (state.vscdb)"
        elif est_cnt:
            status = "estimated"
            label = "对话文本估算 (无原生 Token 字段)"
        else:
            status = "unavailable"
            label = "暂无可解析对话记录"

        return events, AgentSourceInfo(
            agent_id=self.agent_id,
            name=self.agent_name,
            color=self.color,
            description=self.description,
            status=status,
            status_label=label,
            paths_checked=paths_checked,
            paths_found=list(dict.fromkeys(paths_found)),
            events_count=len(events),
            sessions_count=len(sessions_seen),
            total_tokens=sum(e.total_tokens for e in events),
            notes=(
                "读取 Cursor User/globalStorage/state.vscdb（cursorDiskKV/composerHeaders）、"
                "workspaceStorage、~/.cursor/chats/*/store.db 与 agent-transcripts/*.jsonl。"
            ),
        )
