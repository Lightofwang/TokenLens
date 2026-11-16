from __future__ import annotations

import json
import sqlite3
import struct
import tempfile
import unittest
import zlib
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from tokenlens.adapters.base import get_app_data_dirs, local_tz
from tokenlens.adapters.claude_code import ClaudeCodeAdapter
from tokenlens.adapters.cursor import CursorAdapter
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

    def test_trae_alaudalog_custom_model_proxy_estimation(self) -> None:
        def enc_hdr(major: int, val: int) -> bytes:
            if val < 24:
                return bytes([(major << 5) | val])
            if val < 256:
                return bytes([(major << 5) | 24, val])
            if val < 65536:
                return struct.pack(">BH", (major << 5) | 25, val)
            return struct.pack(">BI", (major << 5) | 26, val)

        def encode_cbor(obj) -> bytes:
            if isinstance(obj, int):
                return enc_hdr(0, obj)
            if isinstance(obj, str):
                b = obj.encode("utf-8")
                return enc_hdr(3, len(b)) + b
            if isinstance(obj, list):
                return enc_hdr(4, len(obj)) + b"".join(encode_cbor(x) for x in obj)
            if isinstance(obj, dict):
                return enc_hdr(5, len(obj)) + b"".join(
                    encode_cbor(k) + encode_cbor(v) for k, v in obj.items()
                )
            raise TypeError(type(obj))

        records = [
            # 1) 连通性测试探针（无 create_agent_task resp_logid，但共用同一个 trace_id）
            {
                "name": "custom_model_request",
                "sid": 10,
                "psid": 1,
                "tags": [["request_id", 'Some("probe_req_1")']],
            },
            {
                "span": 10,
                "time": [2026, 317, 10, 0, 13, 0],
                "fields": {
                    "trace_id": "shared_ws_trace",
                    "message": "[aha_net] send: calling Fetch method=POST, url=http://10.10.1.168:8000/v1/chat/completions, body_len=348",
                },
            },
            # 2) 真实自定义模型对话（外层 create_agent_task 与内层 CustomModelProxy 同 resp_logid，应去重只保留内层）
            {
                "span": 2,
                "time": [2026, 317, 10, 0, 22, 0],
                "fields": {
                    "trace_id": "shared_ws_trace",
                    "session_id": "6af6dfb685ec8ac0a3df5628",
                    "provider": 'Some("custom_openai_compatible")',
                    "config_name": 'Some("qwen3.8-27b")',
                    "message": "[aha_net] send: calling Fetch id=req_outer, method=POST, url=https://trae-api-cn.mchost.guru/api/agent/v3/create_agent_task, body_len=86076",
                },
            },
            {
                "span": 2,
                "time": [2026, 317, 10, 0, 22, 100000000],
                "fields": {
                    "trace_id": "shared_ws_trace",
                    "url": "https://api.trae.com.cn/create_agent_task",
                    "resp_logid": "real_req_1",
                    "message": "cloud stream response headers",
                },
            },
            {
                "name": "custom_model_request",
                "sid": 20,
                "psid": 1,
                "tags": [["request_id", 'Some("real_req_1")']],
            },
            {
                "name": "fetch_child",
                "sid": 21,
                "psid": 20,
            },
            {
                "span": 21,
                "time": [2026, 317, 10, 0, 23, 385285000],
                "fields": {
                    "trace_id": "shared_ws_trace",
                    "message": "[aha_net] send: calling Fetch method=POST, url=http://10.10.1.168:8000/v1/chat/completions, body_len=110178",
                },
            },
            {
                "span": 20,
                "time": [2026, 317, 10, 0, 31, 0],
                "fields": {
                    "trace_id": "shared_ws_trace",
                    "message": "SSE stream ended normally after 7 messages in 8.45s",
                },
            },
            {
                "span": 20,
                "time": [2026, 317, 10, 0, 31, 100000000],
                "fields": {
                    "trace_id": "shared_ws_trace",
                    "duration_ms": "7780",
                    "message": "[CustomModelProxy] sse.end dispatched",
                },
            },
            # 3) 官方自带内置云端模型对话（不走 CustomModelProxy，从 create_agent_task 估算）
            {
                "span": 30,
                "time": [2026, 317, 10, 1, 0, 0],
                "fields": {
                    "trace_id": "builtin_trace_1",
                    "chat_session_id": "sess_builtin_1",
                    "model_name": "DeepSeek-V3.1",
                    "message": "create_task",
                },
            },
            {
                "span": 31,
                "time": [2026, 317, 10, 1, 0, 50000000],
                "fields": {
                    "trace_id": "builtin_trace_1",
                    "message": "[aha_net] send: calling Fetch id=req_builtin, method=POST, url=https://trae-api-cn.mchost.guru/api/agent/v3/create_agent_task, body_len=64000",
                },
            },
            {
                "span": 30,
                "time": [2026, 317, 10, 1, 0, 150000000],
                "fields": {
                    "trace_id": "builtin_trace_1",
                    "url": "https://trae-api-cn.mchost.guru/api/agent/v3/create_agent_task",
                    "resp_logid": "builtin_req_1",
                    "message": "cloud stream response headers",
                },
            },
            {
                "span": 30,
                "time": [2026, 317, 10, 1, 5, 0],
                "fields": {
                    "trace_id": "builtin_trace_1",
                    "history_source": "Server",
                    "content_source": 'Other("llm_default")',
                    "message": "recording history entry",
                },
            },
            {
                "span": 30,
                "time": [2026, 317, 10, 1, 6, 50000000],
                "fields": {
                    "trace_id": "builtin_trace_1",
                    "status": "Ok(Completed)",
                    "message": "chat: runner completed",
                },
            },
        ]

        cbor_stream = b"".join(encode_cbor(r) for r in records)
        comp = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        deflate_payload = comp.compress(cbor_stream) + comp.flush()
        ver = b"3.4.18"
        alauda_bytes = (
            b"AalG"
            + b"\x00" * 32
            + b"AALG"
            + b"\x00" * 12
            + struct.pack("<I", len(ver))
            + ver
            + deflate_payload
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sess_dir = root / "Trae CN" / "logs" / "20261113T175549"
            mod_dir = sess_dir / "Modular"
            win_dir = sess_dir / "window1"
            mod_dir.mkdir(parents=True)
            win_dir.mkdir(parents=True)

            (mod_dir / "ai_agent-0-2026-11-13.alaudalog").write_bytes(alauda_bytes)
            (win_dir / "renderer.log").write_text(
                '2026-11-13 18:00:32 [info] update session {"chat_session_id":"6af6dfb685ec8ac0a3df5628","title":"你好"}\n'
                '2026-11-13 18:01:06 [info] update session {"chat_session_id":"sess_builtin_1","title":"内置模型测试"}\n',
                encoding="utf-8",
            )

            events, info = TraeAdapter(
                app_dirs=[root], marscode_dir=root / ".marscode"
            ).scan()

            self.assertEqual(info.status, "estimated")
            self.assertEqual(len(events), 2)
            ev_custom = [e for e in events if "qwen3.8" in e.model][0]
            self.assertTrue(ev_custom.is_estimated)
            self.assertEqual(ev_custom.session_id, "Trae CN:6af6dfb685ec8ac0a3df5628")
            self.assertEqual(ev_custom.model, "custom_openai_compatible//qwen3.8-27b")
            self.assertEqual(ev_custom.uncached_input, 27544)
            self.assertEqual(ev_custom.output, 14)
            self.assertEqual(ev_custom.total_tokens, 27558)
            self.assertEqual(ev_custom.duration_ms, 7780)
            self.assertEqual(ev_custom.title, "你好")

            ev_builtin = [e for e in events if e.model == "DeepSeek-V3.1"][0]
            self.assertTrue(ev_builtin.is_estimated)
            self.assertEqual(ev_builtin.session_id, "Trae CN:sess_builtin_1")
            self.assertEqual(ev_builtin.uncached_input, 16000)
            self.assertEqual(ev_builtin.output, 40)
            self.assertEqual(ev_builtin.duration_ms, 6000)
            self.assertEqual(ev_builtin.title, "内置模型测试")

    def test_vscode_text_estimation_wal_session_store_and_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_user = root / "Code" / "User"
            ws_chat = code_user / "workspaceStorage" / "ws_plain" / "chatSessions"
            empty_chat = code_user / "globalStorage" / "emptyWindowChatSessions"
            copilot_gs = code_user / "globalStorage" / "github.copilot-chat"
            ws_chat.mkdir(parents=True)
            empty_chat.mkdir(parents=True)
            copilot_gs.mkdir(parents=True)

            # 1) Plain Copilot Chat .json without native promptTokens/completionTokens
            (ws_chat / "no-tok-1.json").write_text(
                json.dumps(
                    {
                        "creationDate": 1791531583322,
                        "customTitle": "Plain Copilot Chat",
                        "requests": [
                            {
                                "timestamp": 1791531585000,
                                "isCanceled": False,
                                "message": {"text": "帮我检查一下登录鉴权逻辑"},
                                "result": {
                                    "metadata": {
                                        "renderedUserMessage": [{"type": 1, "text": "<userPrompt>帮我检查一下登录鉴权逻辑</userPrompt>"}],
                                        "toolCallRounds": [
                                            {"response": "我来帮你检查 src/auth.py 的鉴权流程并确认 token 校验是否正确。", "toolCalls": []}
                                        ],
                                    }
                                },
                            },
                            {
                                "timestamp": 1791531595000,
                                "isCanceled": True,
                                "message": {"text": "空取消请求"},
                                "response": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            # 2) emptyWindowChatSessions .jsonl with completionTokens only (missing promptTokens)
            empty_lines = [
                {"kind": 0, "v": {"creationDate": 1791532000000, "customTitle": "Empty window patch"}},
                {"kind": 2, "k": ["requests"], "v": [{"requestId": "r0", "timestamp": 1791532010000, "message": {"text": "总结变更"}}]},
                {"kind": 1, "k": ["requests", 0, "modelId"], "v": "copilot/claude-haiku-4.5"},
                {"kind": 1, "k": ["requests", 0, "completionTokens"], "v": 120},
                {"kind": 1, "k": ["requests", 0, "result", "metadata", "renderedUserMessage"], "v": [{"type": 1, "text": "git diff summary prompt content here"}]},
            ]
            (empty_chat / "empty-1.jsonl").write_text(
                "\n".join(json.dumps(x) for x in empty_lines) + "\n",
                encoding="utf-8",
            )

            # 3) WAL-active session-store.db containing both duplicate (no-tok-1) and store-only-1
            store_db = copilot_gs / "session-store.db"
            conn = sqlite3.connect(str(store_db))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, summary TEXT, created_at TEXT)")
            conn.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY, session_id TEXT, turn_index INTEGER, user_message TEXT, assistant_response TEXT, timestamp TEXT)")
            conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", ("no-tok-1", "/repo", "Plain Copilot Chat", "2026-11-12T10:00:00Z"))
            conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", ("store-only-1", "/repo", "Store Fallback Session", "2026-11-12T11:00:00Z"))
            conn.execute("INSERT INTO turns (session_id, turn_index, user_message, assistant_response, timestamp) VALUES (?, ?, ?, ?, ?)", ("no-tok-1", 0, "u", "should be deduped", "2026-11-12T10:00:01Z"))
            conn.execute("INSERT INTO turns (session_id, turn_index, user_message, assistant_response, timestamp) VALUES (?, ?, ?, ?, ?)", ("store-only-1", 0, "如何优化数据库索引", "可以对频繁查询的 session_id 列建立复合索引。", "2026-11-12T11:00:02Z"))
            conn.commit()
            # Keep connection open briefly while scanning so -wal exists and is non-empty!
            try:
                events, info = VSCodeAdapter(app_dirs=[root], copilot_dir=root / ".no_copilot").scan()
            finally:
                conn.close()

            self.assertEqual(info.status, "estimated")
            self.assertEqual(len(events), 3)
            self.assertEqual(info.sessions_count, 3)
            by_sid = {e.session_id: e for e in events}
            self.assertIn("no-tok-1", by_sid)
            self.assertGreater(by_sid["no-tok-1"].uncached_input, 800)
            self.assertGreater(by_sid["no-tok-1"].output, 10)
            self.assertEqual(by_sid["empty-1"].model, "copilot/claude-haiku-4.5")
            self.assertEqual(by_sid["empty-1"].output, 120)
            self.assertGreater(by_sid["empty-1"].uncached_input, 800)
            self.assertIn("store-only-1", by_sid)
            self.assertEqual(by_sid["store-only-1"].title, "Store Fallback Session")

    def test_cursor_global_vscdb_legacy_cli_and_transcript_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cursor_app = root / "Cursor" / "User"
            ws_dir = cursor_app / "workspaceStorage" / "ws_cursor_1"
            gs_dir = cursor_app / "globalStorage"
            dot_cursor = root / ".cursor"
            cli_dir = dot_cursor / "chats" / "projhash" / "cli_sess_1"
            tr_dir = dot_cursor / "projects" / "proj1" / "agent-transcripts" / "tr_sess_1"

            ws_dir.mkdir(parents=True)
            gs_dir.mkdir(parents=True)
            cli_dir.mkdir(parents=True)
            tr_dir.mkdir(parents=True)

            (ws_dir / "workspace.json").write_text(
                json.dumps({"folder": "file:///c:/repo/my-cursor-app"}),
                encoding="utf-8",
            )
            ws_conn = sqlite3.connect(str(ws_dir / "state.vscdb"))
            ws_conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
            ws_conn.execute(
                "INSERT INTO ItemTable VALUES (?, ?)",
                (
                    "composer.composerData",
                    json.dumps(
                        {
                            "allComposers": [
                                {
                                    "composerId": "comp_exact",
                                    "name": "Refactor auth module",
                                    "createdAt": 1794795000000,
                                    "lastUpdatedAt": 1794795100000,
                                }
                            ]
                        }
                    ),
                ),
            )
            ws_conn.execute(
                "INSERT INTO ItemTable VALUES (?, ?)",
                (
                    "workbench.panel.aichat.view.aichat.chatdata",
                    json.dumps(
                        {
                            "tabs": [
                                {
                                    "tabId": "tab_legacy_1",
                                    "chatTitle": "Legacy Chat Panel",
                                    "lastSendTime": 1794794000000,
                                    "bubbles": [
                                        {"type": "user", "text": "Explain decorators"},
                                        {
                                            "type": "ai",
                                            "text": "Decorators wrap functions.",
                                            "modelType": "gpt-4o",
                                            "tokenCount": {"inputTokens": 400, "outputTokens": 80, "cacheReadTokens": 100},
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                ),
            )
            ws_conn.commit()
            ws_conn.close()

            g_conn = sqlite3.connect(str(gs_dir / "state.vscdb"))
            g_conn.execute(
                "CREATE TABLE composerHeaders (composerId TEXT PRIMARY KEY, workspaceId TEXT, createdAt INTEGER, lastUpdatedAt INTEGER, value TEXT)"
            )
            g_conn.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
            g_conn.execute(
                "INSERT INTO composerHeaders VALUES (?, ?, ?, ?, ?)",
                ("comp_est", "ws_cursor_1", 1794796000000, 1794796100000, json.dumps({"name": "Fix CSS layout"})),
            )
            g_conn.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                (
                    "composerData:comp_exact",
                    json.dumps(
                        {
                            "composerId": "comp_exact",
                            "modelConfig": {"modelName": "claude-4-sonnet"},
                            "fullConversationHeadersOnly": [
                                {"bubbleId": "b1", "type": 1},
                                {"bubbleId": "b2", "type": 2},
                            ],
                        }
                    ),
                ),
            )
            g_conn.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                ("bubbleId:comp_exact:b1", json.dumps({"type": 1, "text": "重构鉴权中间件", "createdAt": "2026-11-16T02:10:00Z"})),
            )
            g_conn.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                (
                    "bubbleId:comp_exact:b2",
                    json.dumps(
                        {
                            "type": 2,
                            "text": "已完成中间件重构。",
                            "createdAt": "2026-11-16T02:10:15Z",
                            "modelInfo": {"modelName": "claude-4.5-sonnet"},
                            "tokenCount": {"inputTokens": 1500, "outputTokens": 250, "cacheReadTokens": 300},
                        }
                    ),
                ),
            )
            g_conn.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                (
                    "composerData:comp_est",
                    json.dumps(
                        {
                            "composerId": "comp_est",
                            "modelConfig": {"modelName": "default"},
                            "fullConversationHeadersOnly": [
                                {"bubbleId": "e1", "type": 1},
                                {"bubbleId": "e2", "type": 2},
                            ],
                        }
                    ),
                ),
            )
            g_conn.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                ("bubbleId:comp_est:e1", json.dumps({"type": 1, "text": "修复 Flex 布局溢出的问题"})),
            )
            g_conn.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                ("bubbleId:comp_est:e2", json.dumps({"type": 2, "text": "请给容器添加 min-width: 0; 与 overflow: hidden;"})),
            )
            g_conn.commit()
            g_conn.close()

            # Cursor Agent CLI store.db with hex-encoded meta JSON
            s_conn = sqlite3.connect(str(cli_dir / "store.db"))
            s_conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            s_conn.execute("CREATE TABLE blobs (id TEXT PRIMARY KEY, data TEXT)")
            meta_hex = json.dumps(
                {"agentId": "cli_sess_1", "name": "CLI Store Task", "createdAt": 1794797000000, "lastUsedModel": "gpt-5"}
            ).encode("utf-8").hex()
            s_conn.execute("INSERT INTO meta VALUES ('0', ?)", (meta_hex,))
            s_conn.execute("INSERT INTO blobs VALUES ('m1', ?)", (json.dumps({"role": "user", "content": "Inspect build.py"}),))
            s_conn.execute(
                "INSERT INTO blobs VALUES ('m2', ?)",
                (json.dumps({"role": "assistant", "content": "Checked build configuration.", "usage": {"inputTokens": 900, "outputTokens": 120}}),),
            )
            s_conn.commit()
            s_conn.close()

            # Agent transcripts jsonl (plus a duplicate of comp_exact that must be deduped)
            (tr_dir / "tr_sess_1.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"role": "user", "message": {"content": [{"type": "text", "text": "Write unit tests"}]}}),
                        json.dumps(
                            {
                                "role": "assistant",
                                "message": {
                                    "model": "grok-4",
                                    "content": [{"type": "text", "text": "Added comprehensive unit tests in test_app.py."}],
                                    "usage": {"input_tokens": 600, "output_tokens": 90},
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (tr_dir.parent / "comp_exact.jsonl").write_text(
                json.dumps({"role": "assistant", "message": {"content": "dup", "usage": {"input_tokens": 9999, "output_tokens": 999}}})
                + "\n",
                encoding="utf-8",
            )

            events, info = CursorAdapter(app_dirs=[root], cursor_dir=dot_cursor).scan()
            self.assertEqual(info.status, "estimated")
            self.assertEqual(info.sessions_count, 5)
            self.assertEqual(len(events), 5)

            by_sid = {e.session_id: e for e in events}
            self.assertFalse(by_sid["comp_exact"].is_estimated)
            self.assertEqual(by_sid["comp_exact"].model, "claude-4.5-sonnet")
            self.assertEqual(by_sid["comp_exact"].total_tokens, 2050)
            self.assertEqual(by_sid["comp_exact"].project_path, "c:/repo/my-cursor-app")
            self.assertEqual(by_sid["comp_exact"].title, "Refactor auth module")

            self.assertTrue(by_sid["comp_est"].is_estimated)
            self.assertEqual(by_sid["comp_est"].model, "cursor-agent")
            self.assertEqual(by_sid["comp_est"].title, "Fix CSS layout")

            self.assertFalse(by_sid["tab_legacy_1"].is_estimated)
            self.assertEqual(by_sid["tab_legacy_1"].model, "gpt-4o")
            self.assertEqual(by_sid["tab_legacy_1"].total_tokens, 580)

            self.assertFalse(by_sid["cli_sess_1"].is_estimated)
            self.assertEqual(by_sid["cli_sess_1"].model, "gpt-5")
            self.assertEqual(by_sid["cli_sess_1"].total_tokens, 1020)

            self.assertFalse(by_sid["tr_sess_1"].is_estimated)
            self.assertEqual(by_sid["tr_sess_1"].model, "grok-4")
            self.assertEqual(by_sid["tr_sess_1"].total_tokens, 690)


if __name__ == "__main__":
    unittest.main()
