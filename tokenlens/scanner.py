from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Any

from .adapters import ALL_ADAPTERS
from .config import load_config
from .models import AgentSourceInfo, UsageEvent
from .pricing import estimate_cost_usd


class TokenScanner:
    """Thread-safe local agent scanner and analytics aggregator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[UsageEvent] = []
        self._sources: list[AgentSourceInfo] = []
        self._last_scan_at: datetime | None = None
        self._scan_duration_ms: int = 0

    def ensure_scanned(self, force: bool = False) -> None:
        with self._lock:
            if self._last_scan_at is not None and not force:
                return
            t0 = time.perf_counter()
            all_events: list[UsageEvent] = []
            all_sources: list[AgentSourceInfo] = []
            for adapter in ALL_ADAPTERS:
                try:
                    evts, src = adapter.scan()
                    all_events.extend(evts)
                    all_sources.append(src)
                except Exception as exc:
                    all_sources.append(
                        AgentSourceInfo(
                            agent_id=adapter.agent_id,
                            name=adapter.agent_name,
                            color=adapter.color,
                            description=adapter.description,
                            status="unavailable",
                            status_label="扫描异常",
                            notes=str(exc),
                        )
                    )
            all_events.sort(key=lambda e: e.timestamp, reverse=True)
            self._events = all_events
            self._sources = all_sources
            self._last_scan_at = datetime.now().astimezone()
            self._scan_duration_ms = int((time.perf_counter() - t0) * 1000)

    def get_sources(self) -> list[dict[str, Any]]:
        self.ensure_scanned()
        return [asdict(s) for s in self._sources]

    def resolve_date_range(
        self,
        range_type: str,
        date_str: str | None = None,
        start_str: str | None = None,
        end_str: str | None = None,
    ) -> tuple[date | None, date | None, str]:
        today = datetime.now().astimezone().date()
        rt = (range_type or "7d").lower().strip()
        if rt == "today":
            return today, today, f"今天 ({today.isoformat()})"
        if rt == "yesterday":
            y = today - timedelta(days=1)
            return y, y, f"昨天 ({y.isoformat()})"
        if rt == "7d":
            s = today - timedelta(days=6)
            return s, today, f"近 7 天 ({s.isoformat()} ~ {today.isoformat()})"
        if rt == "30d":
            s = today - timedelta(days=29)
            return s, today, f"近 30 天 ({s.isoformat()} ~ {today.isoformat()})"
        if rt == "date" and date_str:
            try:
                d = date.fromisoformat(date_str.strip())
                return d, d, f"指定日期 ({d.isoformat()})"
            except Exception:
                return today, today, f"今天 ({today.isoformat()})"
        if rt == "custom":
            try:
                s = date.fromisoformat(start_str.strip()) if start_str else today - timedelta(days=6)
                e = date.fromisoformat(end_str.strip()) if end_str else today
                if s > e:
                    s, e = e, s
                return s, e, f"{s.isoformat()} ~ {e.isoformat()}"
            except Exception:
                pass
        if rt == "all":
            return None, None, "全部本地历史"
        s = today - timedelta(days=6)
        return s, today, f"近 7 天 ({s.isoformat()} ~ {today.isoformat()})"

    def _filter_by_date(self, start_d: date | None, end_d: date | None) -> list[UsageEvent]:
        if start_d is None and end_d is None:
            return list(self._events)
        out: list[UsageEvent] = []
        for e in self._events:
            d = e.timestamp.date()
            if start_d and d < start_d:
                continue
            if end_d and d > end_d:
                continue
            out.append(e)
        return out

    def build_overview(
        self,
        selected_agents: list[str] | None,
        range_type: str = "7d",
        date_str: str | None = None,
        start_str: str | None = None,
        end_str: str | None = None,
        granularity: str = "auto",
        query: str = "",
    ) -> dict[str, Any]:
        self.ensure_scanned()
        cfg = load_config()
        custom_pricing = cfg.get("custom_pricing") or {}
        usd_to_cny = float(cfg.get("usd_to_cny") or 7.25)
        monthly_budget = float(cfg.get("monthly_budget_usd") or 300.0)

        start_d, end_d, range_label = self.resolve_date_range(range_type, date_str, start_str, end_str)
        date_events = self._filter_by_date(start_d, end_d)

        q = (query or "").strip().lower()
        if q:
            date_events = [
                e
                for e in date_events
                if q in e.agent_name.lower()
                or q in e.model.lower()
                or q in e.title.lower()
                or q in e.project_path.lower()
                or q in e.session_id.lower()
            ]

        # Sidebar counts across ALL agents for current date window
        sidebar_totals: dict[str, int] = defaultdict(int)
        sidebar_calls: dict[str, int] = defaultdict(int)
        for e in date_events:
            sidebar_totals[e.agent_id] += e.total_tokens
            sidebar_calls[e.agent_id] += 1

        all_agent_ids = [s.agent_id for s in self._sources]
        if selected_agents is None:
            active_set = set(all_agent_ids)
        else:
            active_set = set(selected_agents)

        filtered = [e for e in date_events if e.agent_id in active_set]

        # Compute KPI & breakdowns
        tot_tokens = 0
        tot_in = 0
        tot_cr = 0
        tot_cw = 0
        tot_out = 0
        tot_cost_usd = 0.0
        tot_savings_usd = 0.0
        sessions_set: set[tuple[str, str]] = set()

        agent_agg: dict[str, dict[str, Any]] = {}
        model_agg: dict[str, dict[str, Any]] = {}
        session_agg: dict[tuple[str, str], dict[str, Any]] = {}

        src_meta = {s.agent_id: s for s in self._sources}

        for e in filtered:
            cost, save = estimate_cost_usd(
                e.model, e.uncached_input, e.cache_read, e.cache_write, e.output, custom_pricing
            )
            tot_tokens += e.total_tokens
            tot_in += e.uncached_input
            tot_cr += e.cache_read
            tot_cw += e.cache_write
            tot_out += e.output
            tot_cost_usd += cost
            tot_savings_usd += save
            skey = (e.agent_id, e.session_id)
            sessions_set.add(skey)

            # Agent agg
            a = agent_agg.get(e.agent_id)
            if a is None:
                sm = src_meta.get(e.agent_id)
                a = {
                    "agent_id": e.agent_id,
                    "agent_name": e.agent_name,
                    "color": sm.color if sm else "#6366f1",
                    "status": sm.status if sm else "exact",
                    "total_tokens": 0,
                    "uncached_input": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "output": 0,
                    "calls": 0,
                    "sessions_set": set(),
                    "models": Counter(),
                    "cost_usd": 0.0,
                    "has_estimated": False,
                }
                agent_agg[e.agent_id] = a
            a["total_tokens"] += e.total_tokens
            a["uncached_input"] += e.uncached_input
            a["cache_read"] += e.cache_read
            a["cache_write"] += e.cache_write
            a["output"] += e.output
            a["calls"] += 1
            a["sessions_set"].add(e.session_id)
            a["models"][e.model] += e.total_tokens
            a["cost_usd"] += cost
            if e.is_estimated:
                a["has_estimated"] = True

            # Model agg
            m = model_agg.get(e.model)
            if m is None:
                m = {
                    "model": e.model,
                    "total_tokens": 0,
                    "uncached_input": 0,
                    "cache_read": 0,
                    "output": 0,
                    "calls": 0,
                    "cost_usd": 0.0,
                    "agents": set(),
                }
                model_agg[e.model] = m
            m["total_tokens"] += e.total_tokens
            m["uncached_input"] += e.uncached_input
            m["cache_read"] += e.cache_read
            m["output"] += e.output
            m["calls"] += 1
            m["cost_usd"] += cost
            m["agents"].add(e.agent_name)

            # Session agg
            sa = session_agg.get(skey)
            if sa is None:
                sm = src_meta.get(e.agent_id)
                sa = {
                    "agent_id": e.agent_id,
                    "agent_name": e.agent_name,
                    "color": sm.color if sm else "#6366f1",
                    "session_id": e.session_id,
                    "title": e.title,
                    "project_path": e.project_path,
                    "models": Counter(),
                    "total_tokens": 0,
                    "uncached_input": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "output": 0,
                    "calls": 0,
                    "cost_usd": 0.0,
                    "first_ts": e.timestamp,
                    "last_ts": e.timestamp,
                    "is_estimated": e.is_estimated,
                }
                session_agg[skey] = sa
            sa["total_tokens"] += e.total_tokens
            sa["uncached_input"] += e.uncached_input
            sa["cache_read"] += e.cache_read
            sa["cache_write"] += e.cache_write
            sa["output"] += e.output
            sa["calls"] += 1
            sa["cost_usd"] += cost
            sa["models"][e.model] += e.total_tokens
            if e.timestamp > sa["last_ts"]:
                sa["last_ts"] = e.timestamp
            if e.timestamp < sa["first_ts"]:
                sa["first_ts"] = e.timestamp
            if e.title and sa["title"].startswith(("Session ", "Claude ", "Codex ", "OpenCode ", "VSCode ")):
                sa["title"] = e.title

        # Determine effective granularity
        is_single_day = start_d is not None and end_d is not None and start_d == end_d
        eff_gran = granularity.lower().strip()
        if eff_gran not in ("hour", "day"):
            eff_gran = "hour" if is_single_day else "day"

        series_agents = sorted(
            agent_agg.values(), key=lambda x: x["total_tokens"], reverse=True
        )
        chart_agent_ids = [a["agent_id"] for a in series_agents]

        buckets: list[dict[str, Any]] = []
        if eff_gran == "hour":
            if is_single_day and start_d is not None:
                bucket_keys = [f"{h:02d}:00" for h in range(24)]
                bucket_map: dict[str, dict[str, int]] = {
                    k: defaultdict(int) for k in bucket_keys
                }
                for e in filtered:
                    hk = f"{e.timestamp.hour:02d}:00"
                    bucket_map[hk][e.agent_id] += e.total_tokens
                for k in bucket_keys:
                    row = {"label": k, "total": sum(bucket_map[k].values()), "by_agent": dict(bucket_map[k])}
                    buckets.append(row)
            else:
                # Multi-day hourly aggregation (by hour of day 00..23 or chronological hour)
                bucket_keys = [f"{h:02d}:00" for h in range(24)]
                bucket_map = {k: defaultdict(int) for k in bucket_keys}
                for e in filtered:
                    hk = f"{e.timestamp.hour:02d}:00"
                    bucket_map[hk][e.agent_id] += e.total_tokens
                for k in bucket_keys:
                    row = {"label": k, "total": sum(bucket_map[k].values()), "by_agent": dict(bucket_map[k])}
                    buckets.append(row)
        else:
            # Daily buckets
            if start_d is not None and end_d is not None:
                cur = start_d
                day_keys: list[str] = []
                while cur <= end_d:
                    day_keys.append(cur.isoformat())
                    cur += timedelta(days=1)
                day_map: dict[str, dict[str, int]] = {k: defaultdict(int) for k in day_keys}
                for e in filtered:
                    dk = e.timestamp.date().isoformat()
                    if dk in day_map:
                        day_map[dk][e.agent_id] += e.total_tokens
                for k in day_keys:
                    buckets.append(
                        {"label": k, "total": sum(day_map[k].values()), "by_agent": dict(day_map[k])}
                    )
            else:
                # "all": from min date to max date if <= 90 days, otherwise only active days
                if filtered:
                    dates = sorted({e.timestamp.date() for e in filtered})
                    span = (dates[-1] - dates[0]).days
                    day_map_all: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
                    for e in filtered:
                        day_map_all[e.timestamp.date().isoformat()][e.agent_id] += e.total_tokens
                    if span <= 90:
                        cur = dates[0]
                        while cur <= dates[-1]:
                            k = cur.isoformat()
                            buckets.append(
                                {"label": k, "total": sum(day_map_all[k].values()), "by_agent": dict(day_map_all[k])}
                            )
                            cur += timedelta(days=1)
                    else:
                        for k in sorted(day_map_all.keys()):
                            buckets.append(
                                {"label": k, "total": sum(day_map_all[k].values()), "by_agent": dict(day_map_all[k])}
                            )

        # Format agent breakdown
        agent_breakdown = []
        for a in series_agents:
            top_m = a["models"].most_common(1)[0][0] if a["models"] else "-"
            share = (a["total_tokens"] / tot_tokens * 100.0) if tot_tokens > 0 else 0.0
            agent_breakdown.append(
                {
                    "agent_id": a["agent_id"],
                    "agent_name": a["agent_name"],
                    "color": a["color"],
                    "status": a["status"],
                    "total_tokens": a["total_tokens"],
                    "uncached_input": a["uncached_input"],
                    "cache_read": a["cache_read"],
                    "cache_write": a["cache_write"],
                    "output": a["output"],
                    "calls": a["calls"],
                    "sessions": len(a["sessions_set"]),
                    "top_model": top_m,
                    "share_pct": round(share, 2),
                    "cost_usd": round(a["cost_usd"], 4),
                    "has_estimated": a["has_estimated"],
                }
            )

        # Model breakdown
        model_list = sorted(model_agg.values(), key=lambda x: x["total_tokens"], reverse=True)[:12]
        model_breakdown = [
            {
                "model": m["model"],
                "total_tokens": m["total_tokens"],
                "uncached_input": m["uncached_input"],
                "cache_read": m["cache_read"],
                "output": m["output"],
                "calls": m["calls"],
                "cost_usd": round(m["cost_usd"], 4),
                "share_pct": round((m["total_tokens"] / tot_tokens * 100.0) if tot_tokens > 0 else 0.0, 2),
                "agents": sorted(m["agents"]),
            }
            for m in model_list
        ]

        # Sessions sorted by latest timestamp
        sessions_sorted = sorted(session_agg.values(), key=lambda s: s["last_ts"], reverse=True)
        recent_sessions = []
        for s in sessions_sorted[:150]:
            top_m = s["models"].most_common(1)[0][0] if s["models"] else "-"
            hit_denom = s["uncached_input"] + s["cache_read"]
            hit_pct = (s["cache_read"] / hit_denom * 100.0) if hit_denom > 0 else 0.0
            recent_sessions.append(
                {
                    "agent_id": s["agent_id"],
                    "agent_name": s["agent_name"],
                    "color": s["color"],
                    "session_id": s["session_id"],
                    "title": s["title"] or f"{s['agent_name']} {s['session_id'][:8]}",
                    "project_path": s["project_path"],
                    "model": top_m,
                    "total_tokens": s["total_tokens"],
                    "uncached_input": s["uncached_input"],
                    "cache_read": s["cache_read"],
                    "cache_write": s["cache_write"],
                    "output": s["output"],
                    "calls": s["calls"],
                    "cache_hit_pct": round(hit_pct, 1),
                    "cost_usd": round(s["cost_usd"], 4),
                    "last_ts": s["last_ts"].isoformat(),
                    "last_ts_display": s["last_ts"].strftime("%Y-%m-%d %H:%M:%S"),
                    "is_estimated": s["is_estimated"],
                }
            )

        prompt_total = tot_in + tot_cr + tot_cw
        cache_hit_rate = (tot_cr / prompt_total * 100.0) if prompt_total > 0 else 0.0
        avg_per_session = int(round(tot_tokens / len(sessions_set))) if sessions_set else 0

        sidebar_agents = []
        for s in self._sources:
            sidebar_agents.append(
                {
                    "agent_id": s.agent_id,
                    "name": s.name,
                    "color": s.color,
                    "status": s.status,
                    "status_label": s.status_label,
                    "selected": s.agent_id in active_set,
                    "window_tokens": sidebar_totals.get(s.agent_id, 0),
                    "window_calls": sidebar_calls.get(s.agent_id, 0),
                    "all_time_tokens": s.total_tokens,
                    "notes": s.notes,
                }
            )

        return {
            "meta": {
                "range_type": range_type,
                "range_label": range_label,
                "start_date": start_d.isoformat() if start_d else None,
                "end_date": end_d.isoformat() if end_d else None,
                "granularity": eff_gran,
                "last_scan_at": self._last_scan_at.isoformat() if self._last_scan_at else None,
                "scan_duration_ms": self._scan_duration_ms,
                "total_indexed_events": len(self._events),
            },
            "kpis": {
                "total_tokens": tot_tokens,
                "uncached_input": tot_in,
                "cache_read": tot_cr,
                "cache_write": tot_cw,
                "output": tot_out,
                "prompt_total": prompt_total,
                "estimated_cost_usd": round(tot_cost_usd, 2),
                "estimated_cost_cny": round(tot_cost_usd * usd_to_cny, 2),
                "monthly_budget_usd": monthly_budget,
                "budget_used_pct": round((tot_cost_usd / monthly_budget * 100.0) if monthly_budget > 0 else 0.0, 1),
                "cache_hit_rate": round(cache_hit_rate, 1),
                "cache_savings_usd": round(tot_savings_usd, 2),
                "calls_count": len(filtered),
                "sessions_count": len(sessions_set),
                "avg_tokens_per_session": avg_per_session,
            },
            "sidebar_agents": sidebar_agents,
            "timeseries": {
                "granularity": eff_gran,
                "agent_ids": chart_agent_ids,
                "buckets": buckets,
            },
            "agent_breakdown": agent_breakdown,
            "model_breakdown": model_breakdown,
            "recent_sessions": recent_sessions,
        }

    def get_session_detail(self, agent_id: str, session_id: str) -> dict[str, Any] | None:
        self.ensure_scanned()
        cfg = load_config()
        custom_pricing = cfg.get("custom_pricing") or {}
        turns = [
            e
            for e in self._events
            if e.agent_id == agent_id and e.session_id == session_id
        ]
        if not turns:
            return None
        turns.sort(key=lambda e: e.timestamp)
        rows = []
        tot = 0
        cost_sum = 0.0
        for idx, e in enumerate(turns, start=1):
            cost, _ = estimate_cost_usd(
                e.model, e.uncached_input, e.cache_read, e.cache_write, e.output, custom_pricing
            )
            tot += e.total_tokens
            cost_sum += cost
            rows.append(
                {
                    "index": idx,
                    "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": e.model,
                    "uncached_input": e.uncached_input,
                    "cache_read": e.cache_read,
                    "cache_write": e.cache_write,
                    "output": e.output,
                    "total_tokens": e.total_tokens,
                    "cost_usd": round(cost, 5),
                    "is_estimated": e.is_estimated,
                }
            )
        first = turns[0]
        return {
            "agent_id": agent_id,
            "agent_name": first.agent_name,
            "session_id": session_id,
            "title": first.title,
            "project_path": first.project_path,
            "source_file": first.source_file,
            "total_tokens": tot,
            "cost_usd": round(cost_sum, 4),
            "turns": rows,
        }

    def export_events(
        self,
        selected_agents: list[str] | None,
        range_type: str = "7d",
        date_str: str | None = None,
        start_str: str | None = None,
        end_str: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_scanned()
        start_d, end_d, _ = self.resolve_date_range(range_type, date_str, start_str, end_str)
        evs = self._filter_by_date(start_d, end_d)
        if selected_agents is not None:
            aset = set(selected_agents)
            evs = [e for e in evs if e.agent_id in aset]
        return [
            {
                "agent_id": e.agent_id,
                "agent_name": e.agent_name,
                "session_id": e.session_id,
                "timestamp": e.timestamp.isoformat(),
                "model": e.model,
                "uncached_input": e.uncached_input,
                "cache_read": e.cache_read,
                "cache_write": e.cache_write,
                "output": e.output,
                "total_tokens": e.total_tokens,
                "title": e.title,
                "project_path": e.project_path,
                "is_estimated": e.is_estimated,
            }
            for e in evs
        ]
