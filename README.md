# TokenLens

> **Local-first, zero-dependency multi-agent AI coding token analytics & visualization software.**
> 一款开源、纯本地运行、零第三方硬依赖的多 AI Agent Token 消耗统计与图表分析桌面软件。

---

## 核心特性 (Features)

- **多 Agent 统一聚合**：一次扫描本机多个 AI 编程工具 / Agent 的本地数据：
  - **Antigravity** (`~/.gemini/antigravity/brain/**/transcript.jsonl`)
  - **OpenAI Codex** (`~/.codex/sessions` & `archived_sessions/*.jsonl(.zst)` + `state_5.sqlite`)
  - **Claude Code** (`~/.claude/projects/**/*.jsonl`)
  - **OpenCode** (`~/.local/share/opencode/opencode.db`)
  - **Qoder / QoderCN / QoderWork CN** (`SharedClientCache/cache/db/local.db` & `agents.db`)
  - **VS Code (GitHub Copilot Chat)** (`Code/User/workspaceStorage/*/chatSessions/*.json(l)`)
  - **Trae / Trae CN** (检测加密 `database.db` 并从明文 `state.vscdb` 进行文本长度估算，带明确“估算”标记)
  - **WorkBuddy / CodeBuddy** (本地目录状态探测与透明提示)
- **自由选择 Agent 工具**：左侧面板支持复选框自由勾选任意 Agent 组合，所有 KPI、图表与明细实时重算。
- **灵活时间切片**：内置 **今天 (Today)**、**昨天 (Yesterday)**、**近 7 天**、**近 30 天**、**指定单日**、**自定义区间**、**全部历史**。
- **交互式可视化图表**：
  - 堆叠面积图 / 多线折线图 / 堆叠柱状图一键切换
  - 按小时 (`Hourly`) / 按天 (`Daily`) 粒度切换 + 悬浮数据明细 Tooltip
  - 各 Agent 消耗占比进度条 & 排行 + Top 模型分布条 + 会话逐轮次下钻抽屉
- **隐私安全 & 开源可扩展**：100% 本地只读扫描，绝不上传任何对话内容；`tokenlens/adapters/` 采用标准适配器接口，欢迎 PR 新增更多工具（Cursor、Windsurf、Cline 等）。

---

## 快速开始 (Quick Start)

### 环境要求
- Python **3.10+**（若需解析 Codex `.jsonl.zst` 压缩历史，Python 3.14+ 已原生内置 `compression.zstd`；Python 3.10~3.13 可选安装 `pip install zstandard`）

### 运行方式

1. **直接运行（免安装依赖）**：
   ```bash
   git clone https://github.com/Lightofwang/TokenLens.git
   cd TokenLens
   python -m tokenlens
   ```
   Windows 用户也可直接双击 [`start_tokenlens.bat`](./start_tokenlens.bat)，软件会自动优先通过 Edge / Chrome `--app` 独立桌面窗口模式打开（或回退到默认浏览器）。

2. **安装为全局命令**：
   ```bash
   pip install -e .
   tokenlens
   ```

3. **常用启动参数**：
   ```bash
   python -m tokenlens --port 7890        # 指定端口
   python -m tokenlens --browser          # 在浏览器标签页打开
   python -m tokenlens --no-open          # 仅启动本地 HTTP 服务
   ```

---

## 开源扩展新 Agent 适配器

只需在 `tokenlens/adapters/` 中继承 `BaseAdapter` 并注册到 `ALL_ADAPTERS`：

```python
from tokenlens.adapters.base import BaseAdapter
from tokenlens.models import AgentSourceInfo, UsageEvent

class MyAgentAdapter(BaseAdapter):
    agent_id = "my_agent"
    agent_name = "My Agent"
    color = "#22d3ee"
    description = "My custom coding agent"

    def scan(self) -> tuple[list[UsageEvent], AgentSourceInfo]:
        ...
```

---

## License

[MIT](./LICENSE)
