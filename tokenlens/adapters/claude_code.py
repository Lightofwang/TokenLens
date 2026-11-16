from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import date, datetime, time
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import (
    BaseAdapter,
    close_sqlite_snapshot,
    local_tz,
    open_sqlite_snapshot,
    parse_epoch_dt,
    parse_iso_dt,
)


def _extract_user_title(obj: dict) -> str:
    msg = obj.get("message")
    if obj.get("type") != "user" or not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip().replace("\n", " ")[:120]
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text", "")
        if text:
            return str(text).strip().replace("\n", " ")[:120]
    return ""


def _extract_cache_write(usage: dict) -> int:
    cw = int(usage.get("cache_creation_input_tokens") or 0)
    if cw > 0:
        return cw
    nested = usage.get("cache_creation")
    if isinstance(nested, dict):
        return int(nested.get("ephemeral_5m_input_tokens") or 0) + int(
            nested.get("ephemeral_1h_input_tokens") or 0
        )
    return 0


class ClaudeCodeAdapter(BaseAdapter):
    agent_id = "claude_code"
    agent_name = "Claude Code"
    color = "#38bdf8"
    description = "Anthropic Claude Code CLI (~/.claude)"

    def __init__(
        self,
        projects_dirs: list[Path] | None = None,
        cc_switch_db: Path | None = None,
    ) -> None:
        self._projects_dirs = projects_dirs
        self._cc_switch_db = cc_switch_db

    def _scan_jsonl_file(self, fp: Path) -> list[UsageEvent]:
        project_name = fp.parent.name
        first_title = ""
        req_events: dict[str, UsageEvent] = {}
        anon_events: list[UsageEvent] = []

        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue

                    if not first_title:
                        first_title = _extract_user_title(obj)

                    msg = obj.get("message")
                    if not isinstance(msg, dict) or "usage" not in msg:
                        continue
                    usage = msg.get("usage")
                    if not isinstance(usage, dict):
                        continue

                    inp = int(usage.get("input_tokens") or 0)
                    out = int(usage.get("output_tokens") or 0)
                    cr = int(usage.get("cache_read_input_tokens") or 0)
                    cw = _extract_cache_write(usage)
                    if inp + out + cr + cw <= 0:
                        continue

                    dt = parse_iso_dt(obj.get("timestamp"))
                    if not dt:
                        continue

                    session_id = str(obj.get("sessionId") or fp.stem)
                    default_title = f"Claude {session_id[:8]}"
                    event = UsageEvent(
                        agent_id=self.agent_id,
                        agent_name=self.agent_name,
                        session_id=session_id,
                        timestamp=dt,
                        model=str(msg.get("model") or "claude-code"),
                        uncached_input=inp,
                        cache_read=cr,
                        cache_write=cw,
                        output=out,
                        title=first_title or default_title,
                        project_path=str(obj.get("cwd") or project_name),
                        source_file=str(fp),
                    )

                    req_id = obj.get("requestId") or msg.get("id")
                    if req_id:
                        req_key = str(req_id)
                        prev = req_events.get(req_key)
                        # Keep the final/largest completion chunk for streamed messages
                        if prev is None or event.total_tokens >= prev.total_tokens:
                            req_events[req_key] = event
                    else:
                        anon_events.append(event)
        except Exception:
            return []

        file_events = [*req_events.values(), *anon_events]
        if first_title:
            for ev in file_events:
                if ev.title.startswith("Claude "):
                    ev.title = first_title
        return file_events

    def _scan_cc_switch_rollups(
        self, db_path: Path
    ) -> list[UsageEvent]:
        if not db_path.is_file():
            return []
        conn = open_sqlite_snapshot(db_path)
        if conn is None:
            return []
        events: list[UsageEvent] = []
        tz = local_tz()
        try:
            rows = conn.execute(
                """
                SELECT date, model, request_model, pricing_model, request_count,
                       input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
                FROM usage_daily_rollups
                WHERE app_type = 'claude'
                """
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            close_sqlite_snapshot(conn)

        for (
            date_str,
            model,
            request_model,
            pricing_model,
            request_count,
            inp,
            out,
            cr,
            cw,
        ) in rows:
            if not date_str:
                continue
            try:
                row_date = date.fromisoformat(str(date_str))
            except ValueError:
                continue

            u_inp = int(inp or 0)
            u_out = int(out or 0)
            u_cr = int(cr or 0)
            u_cw = int(cw or 0)
            if u_inp + u_out + u_cr + u_cw <= 0:
                continue

            req_cnt = int(request_count or 0)
            model_name = str(model or request_model or pricing_model or "claude-code")
            events.append(
                UsageEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    session_id=f"ccswitch-{date_str}",
                    timestamp=datetime.combine(row_date, time(12, 0), tzinfo=tz),
                    model=model_name,
                    uncached_input=u_inp,
                    cache_read=u_cr,
                    cache_write=u_cw,
                    output=u_out,
                    title=f"CC-Switch 历史日汇总 ({date_str}, {req_cnt} reqs)",
                    project_path="cc-switch-rollup",
                    source_file=str(db_path),
                )
            )
        return events

    def _scan_cc_switch_proxy_logs(self, db_path: Path) -> list[UsageEvent]:
        """Read live per-request logs from CC-Switch's proxy_request_logs.

        On installs where usage_daily_rollups stopped updating, the
        authoritative live usage lives in proxy_request_logs (one row per
        request; request_id is the primary key, so no dedup is needed).
        """
        if not db_path.is_file():
            return []
        conn = open_sqlite_snapshot(db_path)
        if conn is None:
            return []
        events: list[UsageEvent] = []
        try:
            rows = conn.execute(
                """
                SELECT request_id, model, request_model, pricing_model,
                       input_tokens, output_tokens, cache_read_tokens,
                       cache_creation_tokens, session_id, created_at
                FROM proxy_request_logs
                WHERE app_type = 'claude'
                """
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            close_sqlite_snapshot(conn)

        for (
            req_id,
            model,
            request_model,
            pricing_model,
            inp,
            out,
            cr,
            cw,
            session_id,
            created_at,
        ) in rows:
            u_inp = int(inp or 0)
            u_out = int(out or 0)
            u_cr = int(cr or 0)
            u_cw = int(cw or 0)
            if u_inp + u_out + u_cr + u_cw <= 0:
                continue
            dt = parse_epoch_dt(created_at)
            if not dt:
                continue
            model_name = str(
                model or request_model or pricing_model or "claude-code"
            )
            events.append(
                UsageEvent(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    session_id=str(session_id or req_id or "ccswitch-proxy"),
                    timestamp=dt,
                    model=model_name,
                    uncached_input=u_inp,
                    cache_read=u_cr,
                    cache_write=u_cw,
                    output=u_out,
                    title=f"CC-Switch 代理请求 ({dt.strftime('%Y-%m-%d')})",
                    project_path="cc-switch-proxy",
                    source_file=str(db_path),
                )
            )
        return events

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        projects_dirs = (
            self._projects_dirs
            if self._projects_dirs is not None
            else [
                Path.home() / ".claude" / "projects",
                Path.home() / ".config" / "claude" / "projects",
            ]
        )
        cc_switch_db = (
            self._cc_switch_db
            if self._cc_switch_db is not None
            else Path.home() / ".cc-switch" / "cc-switch.db"
        )

        paths_checked = [str(p) for p in projects_dirs] + [str(cc_switch_db)]
        paths_found: list[str] = []
        jsonl_events: list[UsageEvent] = []
        seen_files: set[Path] = set()

        for projects_dir in projects_dirs:
            if not projects_dir.exists():
                continue
            paths_found.append(str(projects_dir))
            for fp in projects_dir.rglob("*.jsonl"):
                try:
                    resolved = fp.resolve()
                except OSError:
                    resolved = fp
                if resolved in seen_files:
                    continue
                seen_files.add(resolved)
                jsonl_events.extend(self._scan_jsonl_file(fp))

        tz = local_tz()
        if cc_switch_db.is_file():
            paths_found.append(str(cc_switch_db))
        rollup_events = self._scan_cc_switch_rollups(cc_switch_db)
        proxy_events = self._scan_cc_switch_proxy_logs(cc_switch_db)

        # Day-level "take the largest" merge across three sources.
        # CC-Switch sits as a proxy between Claude Code and the upstream API.
        # usage_daily_rollups is the older aggregate (may be frozen on some
        # installs); proxy_request_logs is the live per-request table. The
        # local ~/.claude/projects JSONL only records what the CLI persisted.
        # For each date, pick whichever source has the largest day-total and
        # emit only that source's events for the day — this never under-counts
        # vs CC-Switch while keeping per-request/per-session detail from JSONL
        # on days where JSONL already captured everything.
        jsonl_by_date: dict[date, list[UsageEvent]] = defaultdict(list)
        for e in jsonl_events:
            jsonl_by_date[e.timestamp.astimezone(tz).date()].append(e)
        rollup_by_date: dict[date, list[UsageEvent]] = defaultdict(list)
        for e in rollup_events:
            rollup_by_date[e.timestamp.astimezone(tz).date()].append(e)
        proxy_by_date: dict[date, list[UsageEvent]] = defaultdict(list)
        for e in proxy_events:
            proxy_by_date[e.timestamp.astimezone(tz).date()].append(e)

        events: list[UsageEvent] = []
        for d in set(jsonl_by_date) | set(rollup_by_date) | set(proxy_by_date):
            candidates = (
                jsonl_by_date.get(d, []),
                rollup_by_date.get(d, []),
                proxy_by_date.get(d, []),
            )
            best = max(candidates, key=lambda lst: sum(e.total_tokens for e in lst))
            events.extend(best)
        if not paths_found:
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地目录",
                paths_checked=paths_checked,
            )

        sessions_seen = {e.session_id for e in events}
        if events:
            status = "exact"
            src_tags = []
            if jsonl_events:
                src_tags.append("JSONL")
            if rollup_events or proxy_events:
                src_tags.append("CC-Switch")
            label = "精确记录 (" + " + ".join(src_tags) + ")" if src_tags else "精确记录"
        else:
            status = "unavailable"
            label = "目录存在但暂无记录"

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
            notes="读取 ~/.claude/projects/**/*.jsonl（取流式末帧）+ CC-Switch usage_daily_rollups + proxy_request_logs，按天取三者最大值，避免本地转录/汇总表冻结导致的漏计。",
        )
