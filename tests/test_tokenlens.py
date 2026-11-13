from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from tokenlens.adapters.base import local_tz
from tokenlens.models import AgentSourceInfo, UsageEvent
from tokenlens.pricing import estimate_cost_usd
from tokenlens.scanner import TokenScanner


class TokenLensCoreTest(unittest.TestCase):
    def test_pricing_and_savings(self) -> None:
        cost, savings = estimate_cost_usd(
            model="claude-sonnet-4-5",
            uncached_input=1_000_000,
            cache_read=1_000_000,
            cache_write=0,
            output=100_000,
        )
        # 1M * $3 + 1M * $0.3 + 0.1M * $15 = $4.80
        self.assertAlmostEqual(cost, 4.80, places=4)
        # savings = 1M * (3.0 - 0.3) = $2.70
        self.assertAlmostEqual(savings, 2.70, places=4)

    def test_scanner_filter_by_agent_and_date(self) -> None:
        scanner = TokenScanner()
        now = datetime.now(tz=local_tz())
        yesterday = now - timedelta(days=1)

        scanner._events = [
            UsageEvent(
                agent_id="codex",
                agent_name="Codex",
                session_id="s1",
                timestamp=now,
                model="gpt-5.6-sol",
                uncached_input=1000,
                cache_read=500,
                output=200,
                title="Fix bug",
            ),
            UsageEvent(
                agent_id="antigravity",
                agent_name="Antigravity",
                session_id="s2",
                timestamp=yesterday,
                model="gemini-3-pro",
                uncached_input=2000,
                cache_read=0,
                output=300,
                title="Plan task",
            ),
        ]
        scanner._sources = [
            AgentSourceInfo("codex", "Codex", "#10b981", "", "exact", "exact"),
            AgentSourceInfo("antigravity", "Antigravity", "#f97316", "", "exact", "exact"),
        ]
        scanner._last_scan_at = now

        # Today + all agents -> only codex event
        ov_today = scanner.build_overview(selected_agents=None, range_type="today")
        self.assertEqual(ov_today["kpis"]["total_tokens"], 1700)
        self.assertEqual(len(ov_today["agent_breakdown"]), 1)
        self.assertEqual(ov_today["agent_breakdown"][0]["agent_id"], "codex")

        # 7d + only antigravity -> only antigravity event
        ov_agy = scanner.build_overview(selected_agents=["antigravity"], range_type="7d")
        self.assertEqual(ov_agy["kpis"]["total_tokens"], 2300)
        self.assertEqual(ov_agy["agent_breakdown"][0]["agent_id"], "antigravity")

        # Empty agent selection -> 0 tokens
        ov_none = scanner.build_overview(selected_agents=[], range_type="7d")
        self.assertEqual(ov_none["kpis"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
