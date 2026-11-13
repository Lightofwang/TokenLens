from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from tokenlens.adapters.base import get_app_data_dirs, local_tz
from tokenlens.adapters.claude_code import ClaudeCodeAdapter
from tokenlens.adapters.trae import TraeAdapter
from tokenlens.adapters.vscode import VSCodeAdapter
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

    def test_macos_apple_silicon_paths(self) -> None:
        fake_home = PurePosixPath("/Users/apple_silicon")
        with patch("tokenlens.adapters.base.os.name", "posix"), patch(
            "tokenlens.adapters.base.sys.platform", "darwin"
        ), patch("tokenlens.adapters.base.Path.home", return_value=fake_home):
            dirs = get_app_data_dirs()
            self.assertEqual(dirs[0], fake_home / "Library" / "Application Support")
            self.assertIn(fake_home / ".local" / "share", dirs)

    def test_vscode_custom_model_events_session_db_dedup_and_chatsessions_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            copilot_ss = root / ".copilot" / "session-state"
            app_dir = root / "appdata"
            sdk_sid = "sdk-1111-2222"
            host_sid = "host-3333-4444"

            # 1) Copilot session-state events.jsonl + workspace.yaml with custom endpoint model
            ss_dir = copilot_ss / sdk_sid
            ss_dir.mkdir(parents=True)
            (ss_dir / "workspace.yaml").write_text(
                f"id: {sdk_sid}\ncwd: C:\\repo\\demo\nname: Custom model test\ncreated_at: 2026-11-13T09:17:47Z\n",
                encoding="utf-8",
            )
            ev_lines = [
                {
                    "type": "session.start",
                    "timestamp": "2026-11-13T09:17:48Z",
                    "data": {"sessionId": sdk_sid, "selectedModel": "customendpoint/dft/qwen3.8-27b"},
                },
                {
                    "id": "ev-dup-1",
                    "type": "session.usage_record",
                    "timestamp": "2026-11-13T09:22:13Z",
                    "data": {
                        "usage": {
                            "model": "dft/qwen3.8-27b",
                            "inputTokens": 39867,
                            "outputTokens": 76,
                            "cacheReadTokens": 0,
                        }
                    },
                },
            ]
            (ss_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(x) for x in ev_lines) + "\n",
                encoding="utf-8",
            )

            # 2) agentSessionData session.db containing BOTH a duplicate and a session.db-only turn
            sess_dir = app_dir / "Code" / "agentSessionData" / host_sid
            sess_dir.mkdir(parents=True)
            sdb = sess_dir / "session.db"
            conn = sqlite3.connect(str(sdb))
            conn.execute("CREATE TABLE session_metadata (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("CREATE TABLE turn_usage (turn_id TEXT PRIMARY KEY, usage TEXT)")
            conn.execute(
                "INSERT INTO session_metadata VALUES (?, ?)",
                ("customTitle", "Custom model test"),
            )
            conn.execute(
                "INSERT INTO session_metadata VALUES (?, ?)",
                ("defaultChatProviderData", json.dumps({"sdkSessionId": sdk_sid})),
            )
            conn.execute(
                "INSERT INTO turn_usage VALUES (?, ?)",
                (
                    "req_dup",
                    json.dumps(
                        {
                            "inputTokens": 39867,
                            "outputTokens": 76,
                            "model": "dft/qwen3.8-27b",
                            "_meta": {"vscode.modelCall": {"sdkSessionId": sdk_sid, "eventId": "ev-dup-1"}},
                        }
                    ),
                ),
            )
            conn.execute(
                "INSERT INTO turn_usage VALUES (?, ?)",
                (
                    "req_extra",
                    json.dumps(
                        {
                            "inputTokens": 1200,
                            "outputTokens": 300,
                            "model": "dft/qwen3.8-27b",
                            "_meta": {"vscode.modelCall": {"sdkSessionId": sdk_sid, "eventId": "ev-only-db"}},
                        }
                    ),
                ),
            )
            conn.commit()
            conn.close()

            # 3) workspaceStorage/*/chatSessions/*.jsonl with kind=2 k=["requests"] appends
            cs_dir = app_dir / "Code" / "User" / "workspaceStorage" / "ws1" / "chatSessions"
            cs_dir.mkdir(parents=True)
            patch_lines = [
                {"kind": 0, "v": {"creationDate": 1791531583322, "customTitle": "Patch test", "requests": []}},
                {
                    "kind": 2,
                    "k": ["requests"],
                    "v": [
                        {
                            "requestId": "r1",
                            "timestamp": 1791533582532,
                            "modelId": "copilot/auto",
                            "promptTokens": 1000,
                            "completionTokens": 100,
                        }
                    ],
                },
                {
                    "kind": 2,
                    "k": ["requests"],
                    "v": [{"requestId": "r2", "timestamp": 1791533682532, "modelId": "customendpoint/local/my-model"}],
                },
                {"kind": 1, "k": ["requests", 1, "promptTokens"], "v": 2000},
                {"kind": 1, "k": ["requests", 1, "outputTokens"], "v": 250},
            ]
            (cs_dir / "chat-1.jsonl").write_text(
                "\n".join(json.dumps(x) for x in patch_lines) + "\n",
                encoding="utf-8",
            )

            events, info = VSCodeAdapter(app_dirs=[app_dir], copilot_dir=copilot_ss).scan()
            self.assertEqual(info.status, "exact")
            # 1 from events.jsonl + 1 non-duplicate from session.db + 2 from chatSessions jsonl = 4
            self.assertEqual(len(events), 4)
            models = sorted(e.model for e in events)
            self.assertEqual(models, ["copilot/auto", "dft/qwen3.8-27b", "dft/qwen3.8-27b", "local/my-model"])
            self.assertEqual(sum(e.total_tokens for e in events), 39867 + 76 + 1500 + 1100 + 2250)

    def test_trae_log_token_usage_event_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "appdata"
            log_dir = app_dir / "Trae CN" / "logs" / "20260507T162624" / "Modular"
            log_dir.mkdir(parents=True)
            sample_log = "\n".join(
                [
                    '2026-05-07T16:57:13.252491+08:00  INFO process_ipc_request:route: ai_agent::handler::chat::handler: model_info: CustomModel { provider: Some("siliconflow"), is_preset: false, config_name: "siliconflow//Pro/moonshotai/Kimi-K2.6", config_source: Personal, model_name: "siliconflow//Pro/moonshotai/Kimi-K2.6" } trace_id="t1"',
                    '2026-05-07T16:57:14.637240+08:00  INFO generate_session_title_and_icon: ai_agent::domain::chat::service: rust: generate_session_title_and_icon: session_id: "sess_abc123", title: "测试自定义模型"',
                    '2026-05-07T16:57:14.900000+08:00  INFO ai_agent::domain::model::llm_stream: token usage: TokenUsageEvent { name: "", prompt_tokens: 1500, completion_tokens: 120, total_tokens: 1620, reasoning_tokens: Some(0), cache_creation_input_tokens: Some(0), cache_read_input_tokens: Some(500) } trace_id="t1"',
                    '2026-05-07T16:57:15.000000+08:00  INFO generate_session_title: ai_agent::domain::chat::service: rust: generate_session_title: model_info: CustomModel { provider: None, is_preset: true, config_name: "title_generation", config_source: Trae, model_name: "no_thinking_model" } trace_id="t2"',
                    '2026-05-07T16:57:15.100000+08:00  INFO generate_session_title: ai_agent::domain::model::llm_stream: token usage: TokenUsageEvent { name: "", prompt_tokens: 400, completion_tokens: 10, total_tokens: 410, reasoning_tokens: Some(0), cache_creation_input_tokens: Some(0), cache_read_input_tokens: Some(0) } trace_id="t2"',
                ]
            )
            (log_dir / "ai-agent_0_123_stdout.log").write_text(sample_log, encoding="utf-8")

            events, info = TraeAdapter(app_dirs=[app_dir], marscode_dir=root / ".marscode").scan()
            self.assertEqual(info.status, "exact")
            self.assertEqual(len(events), 2)
            self.assertFalse(events[0].is_estimated)
            self.assertEqual(events[0].model, "siliconflow//Pro/moonshotai/Kimi-K2.6")
            self.assertEqual(events[0].uncached_input, 1000)
            self.assertEqual(events[0].cache_read, 500)
            self.assertEqual(events[0].output, 120)
            self.assertEqual(events[1].model, "trae-utility")

    def test_claude_code_stream_dedup_and_ccswitch_rollup_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj_dir = root / "projects" / "demo-proj"
            sub_dir = proj_dir / "sess-main" / "subagents"
            sub_dir.mkdir(parents=True)

            main_lines = [
                {
                    "type": "user",
                    "sessionId": "sess-main",
                    "timestamp": "2026-07-16T08:00:00Z",
                    "message": {"role": "user", "content": "修复统计逻辑"},
                },
                # 同一 requestId 先出现中间流式块（out=0），随后出现完整块（out=813）
                {
                    "type": "assistant",
                    "sessionId": "sess-main",
                    "requestId": "req_stream_1",
                    "timestamp": "2026-07-16T08:00:02Z",
                    "message": {
                        "id": "msg_1",
                        "model": "gpt-5.6-sol",
                        "usage": {
                            "input_tokens": 5000,
                            "cache_read_input_tokens": 844,
                            "output_tokens": 0,
                        },
                    },
                },
                {
                    "type": "assistant",
                    "sessionId": "sess-main",
                    "requestId": "req_stream_1",
                    "timestamp": "2026-07-16T08:00:05Z",
                    "message": {
                        "id": "msg_1",
                        "model": "gpt-5.6-sol",
                        "usage": {
                            "input_tokens": 40000,
                            "cache_read_input_tokens": 4392,
                            "cache_creation_input_tokens": 0,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": 100,
                                "ephemeral_1h_input_tokens": 50,
                            },
                            "output_tokens": 813,
                        },
                    },
                },
            ]
            (proj_dir / "sess-main.jsonl").write_text(
                "\n".join(json.dumps(x) for x in main_lines) + "\n",
                encoding="utf-8",
            )

            sub_line = {
                "type": "assistant",
                "sessionId": "sess-main",
                "requestId": "req_sub_1",
                "timestamp": "2026-07-16T08:01:00Z",
                "message": {
                    "id": "msg_sub_1",
                    "model": "gpt-5.6-sol",
                    "usage": {
                        "input_tokens": 100,
                        "cache_read_input_tokens": 200,
                        "output_tokens": 50,
                    },
                },
            }
            (sub_dir / "agent-a1.jsonl").write_text(json.dumps(sub_line) + "\n", encoding="utf-8")

            db_path = root / "cc-switch.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE usage_daily_rollups (
                    date TEXT,
                    app_type TEXT,
                    provider_id TEXT,
                    model TEXT,
                    request_model TEXT,
                    pricing_model TEXT,
                    request_count INTEGER,
                    success_count INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cache_read_tokens INTEGER,
                    cache_creation_tokens INTEGER,
                    total_cost_usd TEXT,
                    avg_latency_ms INTEGER,
                    input_token_semantics TEXT
                )
                """
            )
            # 2026-07-16 已存在于本地 JSONL，不应重复统计
            conn.execute(
                "INSERT INTO usage_daily_rollups VALUES (?, 'claude', 'p1', 'gpt-5.6-sol', 'gpt-5.6-sol', 'gpt-5.6-sol', 2, 2, 40100, 863, 4592, 150, '0.1', 1000, 'raw')"
                , ("2026-07-16",)
            )
            # 2026-05-10 本地 JSONL 已被清理，应从 CC-Switch 回填
            conn.execute(
                "INSERT INTO usage_daily_rollups VALUES (?, 'claude', 'p1', 'kimi-k2.6', 'kimi-k2.6', 'kimi-k2.6', 5, 5, 10000, 500, 2000, 0, '0.02', 500, 'raw')"
                , ("2026-05-10",)
            )
            # 非 claude app_type 不应计入 ClaudeCodeAdapter
            conn.execute(
                "INSERT INTO usage_daily_rollups VALUES (?, 'codex', 'p2', 'gpt-5', 'gpt-5', 'gpt-5', 3, 3, 99999, 999, 0, 0, '0.5', 500, 'raw')"
                , ("2026-05-11",)
            )
            conn.commit()
            conn.close()

            events, info = ClaudeCodeAdapter(
                projects_dirs=[root / "projects"], cc_switch_db=db_path
            ).scan()
            self.assertEqual(info.status, "exact")
            self.assertEqual(len(events), 3)
            # 主会话 JSONL + subagent 均归属 sess-main，外加 2026-05-10 的 rollup 会话
            self.assertEqual(info.sessions_count, 2)
            stream_ev = [e for e in events if e.output == 813][0]
            self.assertEqual(stream_ev.uncached_input, 40000)
            self.assertEqual(stream_ev.cache_read, 4392)
            self.assertEqual(stream_ev.cache_write, 150)
            self.assertEqual(stream_ev.total_tokens, 45355)
            backfill_ev = [e for e in events if e.model == "kimi-k2.6"][0]
            self.assertEqual(backfill_ev.total_tokens, 12500)


if __name__ == "__main__":
    unittest.main()
