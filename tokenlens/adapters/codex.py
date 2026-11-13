from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..models import AgentSourceInfo, UsageEvent
from .base import BaseAdapter, open_sqlite_ro, parse_epoch_dt, parse_iso_dt


def _decompress_zst(fp: Path) -> str:
    raw = fp.read_bytes()
    # Python 3.14+ stdlib
    try:
        from compression import zstd  # type: ignore

        return zstd.decompress(raw).decode("utf-8", errors="ignore")
    except Exception:
        pass
    # Optional zstandard package
    try:
        import zstandard  # type: ignore

        dctx = zstandard.ZstdDecompressor()
        return dctx.decompress(raw).decode("utf-8", errors="ignore")
    except Exception:
        pass
    # Fallback to zstd CLI if present (including macOS Apple Silicon Homebrew path)
    zstd_candidates = [
        shutil.which("zstd"),
        "/opt/homebrew/bin/zstd",
        "/usr/local/bin/zstd",
        "/usr/bin/zstd",
    ]
    for zstd_bin in zstd_candidates:
        if zstd_bin and Path(zstd_bin).exists():
            proc = subprocess.run([zstd_bin, "-dc", str(fp)], capture_output=True, check=False)
            if proc.returncode == 0:
                return proc.stdout.decode("utf-8", errors="ignore")
    raise RuntimeError("No zstd decoder available")


class CodexAdapter(BaseAdapter):
    agent_id = "codex"
    agent_name = "Codex"
    color = "#10b981"
    description = "OpenAI Codex CLI / Desktop (.codex)"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        codex_dir = Path.home() / ".codex"
        events: list[UsageEvent] = []
        paths_checked = [str(codex_dir)]
        paths_found: list[str] = []

        if not codex_dir.exists():
            return events, AgentSourceInfo(
                agent_id=self.agent_id,
                name=self.agent_name,
                color=self.color,
                description=self.description,
                status="not_installed",
                status_label="未检测到本地目录",
                paths_checked=paths_checked,
            )

        paths_found.append(str(codex_dir))

        # Build thread metadata map from state_5.sqlite if available
        thread_meta: dict[str, dict] = {}
        state_db = codex_dir / "state_5.sqlite"
        if state_db.exists():
            try:
                with open_sqlite_ro(state_db) as conn:
                    rows = conn.execute(
                        "SELECT id, title, cwd, model, tokens_used, updated_at, created_at FROM threads"
                    ).fetchall()
                    for tid, title, cwd, model, tokens_used, updated_at, created_at in rows:
                        thread_meta[str(tid)] = {
                            "title": (title or "").strip().replace("\n", " ")[:120],
                            "cwd": cwd or "",
                            "model": model or "",
                            "tokens_used": int(tokens_used or 0),
                            "updated_at": updated_at or created_at,
                        }
            except Exception:
                pass

        sessions_from_files: set[str] = set()
        rollout_files: list[Path] = []
        for sub in ("sessions", "archived_sessions"):
            sub_dir = codex_dir / sub
            if sub_dir.exists():
                rollout_files.extend(sub_dir.rglob("*.jsonl"))
                rollout_files.extend(sub_dir.rglob("*.jsonl.zst"))

        for fp in rollout_files:
            try:
                if fp.name.endswith(".zst"):
                    text = _decompress_zst(fp)
                else:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Extract thread id from filename rollout-YYYY-MM-DDTHH-MM-SS-<uuid>.jsonl(.zst)
            stem = fp.name.replace(".jsonl.zst", "").replace(".jsonl", "")
            session_id = stem[-36:] if len(stem) >= 36 else stem
            meta = thread_meta.get(session_id, {})
            cur_model = meta.get("model") or "gpt-codex"
            cur_cwd = meta.get("cwd") or ""
            title = meta.get("title") or ""
            prev_total_sig = None

            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                otype = obj.get("type")
                payload = obj.get("payload")
                if otype == "session_meta" and isinstance(payload, dict):
                    session_id = payload.get("id") or session_id
                    cur_cwd = payload.get("cwd") or cur_cwd
                elif otype == "turn_context" and isinstance(payload, dict):
                    cur_model = payload.get("model") or cur_model
                    cur_cwd = payload.get("cwd") or cur_cwd
                elif otype == "event_msg" and isinstance(payload, dict) and payload.get("type") == "token_count":
                    info = payload.get("info")
                    if not isinstance(info, dict):
                        continue
                    tot = info.get("total_token_usage") or {}
                    last = info.get("last_token_usage") or {}
                    sig = (
                        tot.get("total_tokens"),
                        tot.get("input_tokens"),
                        tot.get("output_tokens"),
                    )
                    if sig == prev_total_sig:
                        continue
                    prev_total_sig = sig

                    inp_raw = int(last.get("input_tokens") or 0)
                    cr = int(last.get("cached_input_tokens") or 0)
                    cw = int(last.get("cache_write_input_tokens") or 0)
                    out = int(last.get("output_tokens") or 0)
                    reasoning = int(last.get("reasoning_output_tokens") or 0)
                    uncached = max(0, inp_raw - cr)
                    if uncached + cr + cw + out <= 0:
                        continue

                    dt = parse_iso_dt(obj.get("timestamp"))
                    if not dt:
                        continue
                    sessions_from_files.add(session_id)
                    events.append(
                        UsageEvent(
                            agent_id=self.agent_id,
                            agent_name=self.agent_name,
                            session_id=session_id,
                            timestamp=dt,
                            model=cur_model,
                            uncached_input=uncached,
                            cache_read=cr,
                            cache_write=cw,
                            output=out,
                            reasoning=reasoning,
                            title=title or f"Codex {session_id[:8]}",
                            project_path=cur_cwd,
                            source_file=str(fp),
                        )
                    )

        # Fallback: threads present in state_5.sqlite with tokens_used > 0 but missing rollout file events
        for tid, meta in thread_meta.items():
            if tid not in sessions_from_files and meta["tokens_used"] > 0:
                dt = parse_epoch_dt(meta["updated_at"])
                if dt:
                    sessions_from_files.add(tid)
                    events.append(
                        UsageEvent(
                            agent_id=self.agent_id,
                            agent_name=self.agent_name,
                            session_id=tid,
                            timestamp=dt,
                            model=meta["model"] or "gpt-codex",
                            uncached_input=meta["tokens_used"],
                            title=meta["title"] or f"Codex {tid[:8]}",
                            project_path=meta["cwd"],
                            is_estimated=False,
                            source_file=str(state_db),
                        )
                    )

        status = "exact" if events else "unavailable"
        label = "精确记录 (JSONL.ZST / SQLite)" if events else "目录存在但暂无记录"
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
            sessions_count=len(sessions_from_files),
            total_tokens=sum(e.total_tokens for e in events),
            notes="读取 ~/.codex/sessions 与 archived_sessions 的 .jsonl/.jsonl.zst 增量 token_count 事件，并结合 state_5.sqlite 元数据。",
        )
