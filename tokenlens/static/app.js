const state = {
  range: "30d",
  date: new Date().toISOString().slice(0, 10),
  start: "",
  end: "",
  granularity: "auto",
  chartType: "area",
  query: "",
  selectedAgents: null, // null = all
  lastData: null,
  chartHits: [],
};

const $ = (id) => document.getElementById(id);

function fmtTokens(n) {
  const num = Number(n || 0);
  if (num >= 1_000_000_000) return (num / 1_000_000_000).toFixed(2) + "B";
  if (num >= 1_000_000) return (num / 1_000_000).toFixed(2) + "M";
  if (num >= 10_000) return (num / 1_000).toFixed(1) + "k";
  return num.toLocaleString("en-US");
}

function fmtNum(n) {
  return Number(n || 0).toLocaleString("en-US");
}

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildQueryParams() {
  const params = new URLSearchParams();
  params.set("range", state.range);
  if (state.range === "date" && state.date) params.set("date", state.date);
  if (state.range === "custom") {
    if (state.start) params.set("start", state.start);
    if (state.end) params.set("end", state.end);
  }
  params.set("granularity", state.granularity);
  if (state.query) params.set("q", state.query);
  if (state.selectedAgents !== null) {
    params.set("agents", state.selectedAgents.length ? state.selectedAgents.join(",") : "__none__");
  }
  return params.toString();
}

async function fetchOverview() {
  const res = await fetch(`/api/overview?${buildQueryParams()}`);
  const data = await res.json();
  state.lastData = data;
  if (state.selectedAgents === null) {
    state.selectedAgents = data.sidebar_agents
      .filter((a) => a.selected)
      .map((a) => a.agent_id);
  }
  renderAll(data);
}

function renderAll(data) {
  const { meta, kpis, sidebar_agents, timeseries, agent_breakdown, model_breakdown, recent_sessions } = data;

  $("daemonEndpoint").textContent = window.location.host || "127.0.0.1";
  $("metaScanMs").textContent = `${meta.scan_duration_ms} ms`;
  $("metaTotalEvents").textContent = fmtNum(meta.total_indexed_events);
  $("metaWindowShort").textContent = meta.start_date
    ? meta.start_date === meta.end_date
      ? meta.start_date
      : `${meta.start_date.slice(5)} ~ ${meta.end_date.slice(5)}`
    : "All Time";

  // KPIs
  $("kpiRangeTag").textContent = meta.range_label;
  $("kpiTotalTokens").textContent = fmtTokens(kpis.total_tokens);
  $("kpiInputTokens").textContent = fmtTokens(kpis.uncached_input);
  $("kpiOutputTokens").textContent = fmtTokens(kpis.output);

  $("kpiCostUsd").innerHTML = `$${kpis.estimated_cost_usd.toFixed(2)} <small>≈ ¥${kpis.estimated_cost_cny.toFixed(1)}</small>`;
  $("kpiBudgetLimit").textContent = `$${kpis.monthly_budget_usd}`;
  $("kpiBudgetPct").textContent = `${kpis.budget_used_pct}%`;
  $("kpiBudgetBar").style.width = `${Math.min(100, kpis.budget_used_pct)}%`;

  $("kpiCacheRate").innerHTML = `${kpis.cache_hit_rate.toFixed(1)}% <small>省减 $${kpis.cache_savings_usd.toFixed(2)}</small>`;
  $("kpiCacheRead").textContent = fmtTokens(kpis.cache_read);
  $("kpiCacheWrite").textContent = fmtTokens(kpis.cache_write);

  $("kpiAgentsCount").textContent = `${(state.selectedAgents || []).length} 款已选`;
  $("kpiSessionsCount").innerHTML = `${fmtNum(kpis.sessions_count)} <small>${fmtNum(kpis.calls_count)} 次请求</small>`;
  $("kpiAvgSession").textContent = `${fmtTokens(kpis.avg_tokens_per_session)} tok`;
  $("kpiTotalCredits").textContent = kpis.total_credits > 0 ? `${kpis.total_credits.toFixed(1)} 积分` : "0";

  renderSidebarAgents(sidebar_agents);
  renderRank(agent_breakdown);
  renderModelBars(model_breakdown);
  renderSessionsTable(recent_sessions);
  drawTrendChart(timeseries, sidebar_agents);
}

function renderSidebarAgents(agents) {
  const wrap = $("agentFilterList");
  const selectedSet = new Set(state.selectedAgents || []);
  $("selectedCount").textContent = selectedSet.size;
  wrap.innerHTML = "";

  for (const a of agents) {
    const checked = selectedSet.has(a.agent_id);
    const row = document.createElement("label");
    row.className = `agent-check-row ${checked ? "checked" : ""}`;
    row.title = a.notes || a.status_label;

    let miniBadge = "";
    if (a.status === "estimated") miniBadge = `<span class="tag-mini">估算</span>`;
    else if (a.status === "unavailable" || a.status === "not_installed" || a.status === "encrypted") {
      miniBadge = `<span class="tag-mini unavail">无记录</span>`;
    }

    const allSub = a.all_time_tokens > 0 && a.all_time_tokens !== a.window_tokens
      ? `<span class="agent-tok-all">历史 ${fmtTokens(a.all_time_tokens)}</span>`
      : "";

    row.innerHTML = `
      <div class="agent-check-left">
        <input type="checkbox" ${checked ? "checked" : ""} data-agent="${esc(a.agent_id)}" />
        <span class="agent-dot" style="background:${esc(a.color)}"></span>
        <span class="agent-label">${esc(a.name)}</span>
        ${miniBadge}
      </div>
      <div class="agent-tok-col">
        <span class="agent-tok ${a.window_tokens === 0 ? "zero" : ""}">${fmtTokens(a.window_tokens)}</span>
        ${allSub}
      </div>
    `;

    row.querySelector("input").addEventListener("change", (ev) => {
      const id = ev.target.dataset.agent;
      const cur = new Set(state.selectedAgents || []);
      if (ev.target.checked) cur.add(id);
      else cur.delete(id);
      state.selectedAgents = Array.from(cur);
      fetchOverview();
    });
    wrap.appendChild(row);
  }
}

function renderRank(breakdown) {
  const bar = $("agentStackedBar");
  const list = $("agentRankList");
  bar.innerHTML = "";
  list.innerHTML = "";

  if (!breakdown.length) {
    $("topAgentBadge").textContent = "TOP 1: -";
    list.innerHTML = `<div class="muted" style="padding:20px;text-align:center;">所选条件无数据</div>`;
    return;
  }

  $("topAgentBadge").textContent = `TOP 1: ${breakdown[0].agent_name}`;

  for (const item of breakdown) {
    const seg = document.createElement("div");
    seg.className = "stacked-seg";
    seg.style.width = `${Math.max(item.share_pct, 1.5)}%`;
    seg.style.background = item.color;
    seg.title = `${item.agent_name}: ${item.share_pct}%`;
    bar.appendChild(seg);

    const row = document.createElement("div");
    row.className = "rank-item";
    row.title = "点击仅统计此 Agent（再次点击全选恢复）";
    row.innerHTML = `
      <div class="rank-left">
        <div class="rank-title">
          <span class="agent-dot" style="background:${esc(item.color)}"></span>
          <span>${esc(item.agent_name)}</span>
          ${item.has_estimated ? '<span class="tag-mini">含估算</span>' : ""}
        </div>
        <div class="rank-meta">${item.sessions} 会话 · ${item.calls} 请求 · ${esc(item.top_model)}</div>
      </div>
      <div class="rank-right">
        <div class="rank-tok">${fmtTokens(item.total_tokens)} <span class="muted">(${item.share_pct}%)</span></div>
        <div class="rank-cost">${item.credits > 0 ? `${item.credits.toFixed(1)} 积分 · ` : ""}估算 $${item.cost_usd.toFixed(2)}</div>
      </div>
    `;
    row.addEventListener("click", () => {
      if (state.selectedAgents && state.selectedAgents.length === 1 && state.selectedAgents[0] === item.agent_id) {
        state.selectedAgents = state.lastData.sidebar_agents.map((a) => a.agent_id);
      } else {
        state.selectedAgents = [item.agent_id];
      }
      fetchOverview();
    });
    list.appendChild(row);
  }
}

function renderModelBars(models) {
  const wrap = $("modelBars");
  wrap.innerHTML = "";
  if (!models.length) {
    wrap.innerHTML = `<div class="muted">暂无模型调用记录</div>`;
    return;
  }
  const maxVal = Math.max(...models.map((m) => m.total_tokens), 1);
  for (const m of models) {
    const tot = Math.max(m.total_tokens, 1);
    const wTotal = (m.total_tokens / maxVal) * 100;
    const pIn = (m.uncached_input / tot) * wTotal;
    const pCr = (m.cache_read / tot) * wTotal;
    const pOut = Math.max(0, wTotal - pIn - pCr);

    const row = document.createElement("div");
    row.className = "model-row";
    row.innerHTML = `
      <div class="model-row-top">
        <span class="model-name">${esc(m.model)} <span class="muted">(${esc(m.agents.join(", "))})</span></span>
        <span class="model-val">${fmtTokens(m.total_tokens)} · ${m.share_pct}% · $${m.cost_usd.toFixed(2)}</span>
      </div>
      <div class="model-track" title="蓝:新增输入 / 绿:缓存读取 / 紫:输出">
        <div class="model-fill-in" style="width:${pIn}%"></div>
        <div class="model-fill-cr" style="width:${pCr}%"></div>
        <div class="model-fill-out" style="width:${pOut}%"></div>
      </div>
    `;
    wrap.appendChild(row);
  }
}

function renderSessionsTable(sessions) {
  const tbody = $("sessionTableBody");
  const fullBody = $("fullSessionBody");
  $("sessionTableMeta").textContent = `展示 ${Math.min(sessions.length, 30)} / ${sessions.length} 条会话`;
  $("fullSessionMeta").textContent = `共 ${sessions.length} 条会话`;

  const renderRows = (list, isFull) => {
    if (!list.length) {
      return `<tr><td colspan="10" style="text-align:center;padding:24px;color:var(--text-dim);">当前条件暂无会话流水</td></tr>`;
    }
    return list
      .map((s) => {
        const titleTxt = s.project_path ? `${s.title} · ${s.project_path}` : s.title;
        if (isFull) {
          return `
            <tr data-agent="${esc(s.agent_id)}" data-sid="${esc(s.session_id)}" style="cursor:pointer">
              <td class="mono muted">${esc(s.last_ts_display)}</td>
              <td><span class="agent-pill"><span class="agent-dot" style="background:${esc(s.color)}"></span>${esc(s.agent_name)}</span></td>
              <td class="title-cell" title="${esc(titleTxt)}">${esc(titleTxt)}</td>
              <td><span class="model-pill">${esc(s.model)}</span></td>
              <td class="num">${s.calls}</td>
              <td class="num green-text">${s.cache_hit_pct}%</td>
              <td class="num"><b>${fmtTokens(s.total_tokens)}</b></td>
              <td class="num green-text">$${s.cost_usd.toFixed(3)}</td>
              <td><button class="btn-mini">明细</button></td>
            </tr>
          `;
        }
        return `
          <tr data-agent="${esc(s.agent_id)}" data-sid="${esc(s.session_id)}" style="cursor:pointer">
            <td class="mono muted">${esc(s.last_ts_display)}</td>
            <td><span class="agent-pill"><span class="agent-dot" style="background:${esc(s.color)}"></span>${esc(s.agent_name)}</span></td>
            <td class="title-cell" title="${esc(titleTxt)}">${esc(s.title)}</td>
            <td><span class="model-pill">${esc(s.model)}</span></td>
            <td class="num">${fmtTokens(s.uncached_input)}</td>
            <td class="num green-text">${fmtTokens(s.cache_read)}</td>
            <td class="num">${fmtTokens(s.output)}</td>
            <td class="num"><b>${fmtTokens(s.total_tokens)}</b></td>
            <td class="num green-text">${s.credits > 0 ? `${s.credits.toFixed(1)} 积分` : `$${s.cost_usd.toFixed(3)}`}</td>
            <td><button class="btn-mini">明细</button></td>
          </tr>
        `;
      })
      .join("");
  };

  tbody.innerHTML = renderRows(sessions.slice(0, 30), false);
  fullBody.innerHTML = renderRows(sessions, true);

  for (const tb of [tbody, fullBody]) {
    tb.querySelectorAll("tr[data-sid]").forEach((tr) => {
      tr.addEventListener("click", () => openSessionDetail(tr.dataset.agent, tr.dataset.sid));
    });
  }
}

async function openSessionDetail(agentId, sessionId) {
  const res = await fetch(`/api/session?agent_id=${encodeURIComponent(agentId)}&session_id=${encodeURIComponent(sessionId)}`);
  if (!res.ok) return;
  const d = await res.json();
  $("modalAgentBadge").textContent = d.agent_name;
  $("modalTitle").textContent = d.title || d.session_id;
  $("modalPath").textContent = `${d.project_path || ""}  [${d.source_file}]`;
  $("modalTurnsBody").innerHTML = d.turns
    .map(
      (t) => `
      <tr>
        <td class="mono muted">${t.index}</td>
        <td class="mono muted">${esc(t.timestamp)}</td>
        <td><span class="model-pill">${esc(t.model)}</span></td>
        <td class="num">${fmtNum(t.uncached_input)}</td>
        <td class="num green-text">${fmtNum(t.cache_read)}</td>
        <td class="num">${fmtNum(t.output)}</td>
        <td class="num"><b>${fmtNum(t.total_tokens)}</b></td>
        <td class="num green-text">$${t.cost_usd.toFixed(4)}</td>
      </tr>
    `
    )
    .join("");
  $("sessionModal").classList.remove("hidden");
}

/* Custom HD Canvas Chart */
function drawTrendChart(timeseries, sidebarAgents) {
  const canvas = $("trendCanvas");
  const wrap = $("trendCanvasWrap");
  const emptyEl = $("chartEmpty");
  const legendEl = $("chartLegend");
  const buckets = timeseries.buckets || [];
  const agentIds = timeseries.agent_ids || [];

  const metaMap = {};
  for (const a of sidebarAgents) metaMap[a.agent_id] = a;

  $("chartSubtitle").textContent =
    timeseries.granularity === "hour"
      ? "按小时分布趋势（00:00 ~ 23:00）"
      : `按天分布趋势（共 ${buckets.length} 天）`;

  legendEl.innerHTML = agentIds
    .map((id) => {
      const m = metaMap[id] || { name: id, color: "#38bdf8" };
      return `<span class="legend-item"><span class="agent-dot" style="background:${esc(m.color)}"></span>${esc(m.name)}</span>`;
    })
    .join("");

  const hasAny = buckets.some((b) => b.total > 0) && agentIds.length > 0;
  if (!hasAny) {
    emptyEl.classList.remove("hidden");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    state.chartHits = [];
    return;
  }
  emptyEl.classList.add("hidden");

  const dpr = window.devicePixelRatio || 1;
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const pad = { top: 18, right: 18, bottom: 30, left: 56 };
  const plotW = Math.max(10, width - pad.left - pad.right);
  const plotH = Math.max(10, height - pad.top - pad.bottom);

  const isArea = state.chartType === "area";
  const isBar = state.chartType === "bar";

  // Max value
  let maxY = 0;
  for (const b of buckets) {
    if (isArea || isBar) {
      maxY = Math.max(maxY, b.total);
    } else {
      for (const id of agentIds) {
        maxY = Math.max(maxY, (b.by_agent && b.by_agent[id]) || 0);
      }
    }
  }
  if (maxY <= 0) maxY = 100;
  maxY = maxY * 1.12;

  // Grid lines
  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.fillStyle = "#64748b";
  ctx.font = "11px Consolas, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const val = (maxY / ticks) * i;
    const y = pad.top + plotH - (plotH * i) / ticks;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillText(fmtTokens(Math.round(val)), pad.left - 8, y);
  }

  const n = buckets.length;
  const stepX = n > 1 ? plotW / (isBar ? n : n - 1) : plotW;

  const getX = (i) => {
    if (isBar) return pad.left + i * stepX + stepX * 0.5;
    if (n === 1) return pad.left + plotW * 0.5;
    return pad.left + i * stepX;
  };
  const getY = (val) => pad.top + plotH - (val / maxY) * plotH;

  // X-axis labels
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const maxLabels = Math.max(2, Math.floor(plotW / 76));
  const labelStride = Math.max(1, Math.ceil(n / maxLabels));
  for (let i = 0; i < n; i++) {
    if (i % labelStride === 0 || i === n - 1) {
      const lbl = buckets[i].label.length === 10 ? buckets[i].label.slice(5) : buckets[i].label;
      ctx.fillText(lbl, getX(i), pad.top + plotH + 9);
    }
  }

  state.chartHits = buckets.map((b, i) => ({ x: getX(i), bucket: b }));

  if (isBar) {
    const barW = Math.max(4, Math.min(28, stepX * 0.62));
    for (let i = 0; i < n; i++) {
      let acc = 0;
      const x = getX(i) - barW / 2;
      for (const id of agentIds) {
        const v = (buckets[i].by_agent && buckets[i].by_agent[id]) || 0;
        if (v <= 0) continue;
        const yTop = getY(acc + v);
        const yBot = getY(acc);
        ctx.fillStyle = (metaMap[id] && metaMap[id].color) || "#38bdf8";
        ctx.fillRect(x, yTop, barW, Math.max(1, yBot - yTop));
        acc += v;
      }
    }
    return;
  }

  // Area or Line
  const cumulative = new Array(n).fill(0);
  const layers = [];
  for (const id of agentIds) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const v = (buckets[i].by_agent && buckets[i].by_agent[id]) || 0;
      const base = isArea ? cumulative[i] : 0;
      const top = base + v;
      if (isArea) cumulative[i] = top;
      pts.push({ x: getX(i), yTop: getY(top), yBase: getY(base), val: v });
    }
    layers.push({ id, color: (metaMap[id] && metaMap[id].color) || "#38bdf8", pts });
  }

  const drawSmooth = (pts, key) => {
    ctx.moveTo(pts[0].x, pts[0][key]);
    for (let i = 0; i < pts.length - 1; i++) {
      const mx = (pts[i].x + pts[i + 1].x) / 2;
      ctx.bezierCurveTo(mx, pts[i][key], mx, pts[i + 1][key], pts[i + 1].x, pts[i + 1][key]);
    }
  };

  for (const layer of layers) {
    const pts = layer.pts;
    if (isArea) {
      ctx.save();
      ctx.beginPath();
      drawSmooth(pts, "yTop");
      for (let i = pts.length - 1; i >= 0; i--) {
        if (i === pts.length - 1) ctx.lineTo(pts[i].x, pts[i].yBase);
        else {
          const mx = (pts[i + 1].x + pts[i].x) / 2;
          ctx.bezierCurveTo(mx, pts[i + 1].yBase, mx, pts[i].yBase, pts[i].x, pts[i].yBase);
        }
      }
      ctx.closePath();
      ctx.globalAlpha = 0.24;
      ctx.fillStyle = layer.color;
      ctx.fill();
      ctx.restore();
    }

    ctx.save();
    ctx.beginPath();
    drawSmooth(pts, "yTop");
    ctx.strokeStyle = layer.color;
    ctx.lineWidth = 2.2;
    ctx.stroke();
    ctx.restore();
  }
}

function setupChartHover() {
  const wrap = $("trendCanvasWrap");
  const tooltip = $("chartTooltip");
  const canvas = $("trendCanvas");

  wrap.addEventListener("mousemove", (ev) => {
    if (!state.chartHits.length || !state.lastData) {
      tooltip.classList.add("hidden");
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    let best = state.chartHits[0];
    let bestDist = Infinity;
    for (const h of state.chartHits) {
      const d = Math.abs(h.x - mx);
      if (d < bestDist) {
        bestDist = d;
        best = h;
      }
    }
    if (!best || bestDist > 60) {
      tooltip.classList.add("hidden");
      return;
    }

    const metaMap = {};
    for (const a of state.lastData.sidebar_agents) metaMap[a.agent_id] = a;
    const entries = Object.entries(best.bucket.by_agent || {})
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1]);

    tooltip.innerHTML = `
      <div class="tt-head">
        <span>${esc(best.bucket.label)}</span>
        <span class="mono">${fmtTokens(best.bucket.total)}</span>
      </div>
      ${
        entries.length
          ? entries
              .map(([id, val]) => {
                const m = metaMap[id] || { name: id, color: "#38bdf8" };
                return `<div class="tt-row"><span class="tt-name"><span class="agent-dot" style="background:${esc(m.color)}"></span>${esc(m.name)}</span><span class="mono">${fmtTokens(val)}</span></div>`;
              })
              .join("")
          : `<div class="muted">该时段无调用</div>`
      }
    `;
    tooltip.classList.remove("hidden");
    const left = Math.min(wrap.clientWidth - tooltip.offsetWidth - 10, Math.max(10, best.x + 12));
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `14px`;
  });

  wrap.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));
}

async function loadSourcesView() {
  const res = await fetch("/api/sources");
  const { sources } = await res.json();
  const grid = $("sourcesGrid");
  grid.innerHTML = sources
    .map(
      (s) => `
      <div class="source-card">
        <div class="source-head">
          <div style="display:flex;align-items:center;gap:8px;">
            <span class="agent-dot" style="background:${esc(s.color)}"></span>
            <b>${esc(s.name)}</b>
          </div>
          <span class="source-status ${esc(s.status)}">${esc(s.status_label)}</span>
        </div>
        <div class="muted">${esc(s.description)}</div>
        <div class="kpi-sub">
          <span>会话: <b class="mono">${fmtNum(s.sessions_count)}</b></span>
          <span>事件: <b class="mono">${fmtNum(s.events_count)}</b></span>
          <span>历史总量: <b class="mono">${fmtTokens(s.total_tokens)}</b></span>
        </div>
        <div class="source-path">${s.paths_found.length ? esc(s.paths_found.join("\n")) : "未检测到：" + esc(s.paths_checked[0] || "")}</div>
        <div class="muted" style="font-size:11.5px;">${esc(s.notes)}</div>
      </div>
    `
    )
    .join("");
}

async function loadConfigView() {
  const res = await fetch("/api/config");
  const { config, default_pricing } = await res.json();
  $("cfgBudget").value = config.monthly_budget_usd;
  $("cfgRate").value = config.usd_to_cny;
  $("pricingTableBody").innerHTML = Object.entries(default_pricing)
    .map(
      ([model, arr]) => `
      <tr>
        <td class="mono">${esc(model)}</td>
        <td class="num">$${arr[0]}</td>
        <td class="num">$${arr[1]}</td>
        <td class="num">$${arr[2]}</td>
        <td class="num">$${arr[3]}</td>
      </tr>
    `
    )
    .join("");
}

function bindEvents() {
  // Range buttons
  document.querySelectorAll(".range-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".range-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.range = btn.dataset.range;
      $("singleDatePicker").classList.toggle("hidden", state.range !== "date");
      $("customRangePicker").classList.toggle("hidden", state.range !== "custom");
      if (state.range === "date") {
        $("inputSingleDate").value = state.date;
      }
      fetchOverview();
    });
  });

  $("inputSingleDate").addEventListener("change", (ev) => {
    state.date = ev.target.value;
    if (state.range === "date") fetchOverview();
  });

  $("btnApplyCustom").addEventListener("click", () => {
    state.start = $("inputStartDate").value;
    state.end = $("inputEndDate").value;
    state.range = "custom";
    $("customRangePicker").classList.add("hidden");
    fetchOverview();
  });

  // Granularity & chart type
  $("granularityGroup").querySelectorAll(".seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      $("granularityGroup").querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.granularity = b.dataset.gran;
      fetchOverview();
    });
  });
  $("chartTypeGroup").querySelectorAll(".seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      $("chartTypeGroup").querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.chartType = b.dataset.type;
      if (state.lastData) drawTrendChart(state.lastData.timeseries, state.lastData.sidebar_agents);
    });
  });

  // Sidebar nav
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const v = btn.dataset.view;
      document.querySelectorAll(".view-pane").forEach((p) => p.classList.remove("active"));
      $(`view-${v}`).classList.add("active");
      if (v === "sources") loadSourcesView();
      if (v === "settings") loadConfigView();
    });
  });

  // Agent select all / none
  $("btnSelectAll").addEventListener("click", () => {
    if (!state.lastData) return;
    state.selectedAgents = state.lastData.sidebar_agents.map((a) => a.agent_id);
    fetchOverview();
  });
  $("btnSelectNone").addEventListener("click", () => {
    state.selectedAgents = [];
    fetchOverview();
  });

  // Search debounce
  let timer = null;
  $("globalSearch").addEventListener("input", (ev) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      state.query = ev.target.value;
      fetchOverview();
    }, 220);
  });

  // Rescan
  const doRescan = async () => {
    $("rescanIcon").textContent = "⏳";
    await fetch("/api/rescan", { method: "POST" });
    $("rescanIcon").textContent = "↻";
    await fetchOverview();
    loadSourcesView();
  };
  $("btnRescan").addEventListener("click", doRescan);
  $("btnRescanSources").addEventListener("click", doRescan);

  // Export
  $("btnExportJson").addEventListener("click", () => {
    window.location.href = `/api/export?format=json&${buildQueryParams()}`;
  });
  $("btnExportCsv").addEventListener("click", () => {
    window.location.href = `/api/export?format=csv&${buildQueryParams()}`;
  });

  // Config save
  $("configForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        monthly_budget_usd: Number($("cfgBudget").value || 300),
        usd_to_cny: Number($("cfgRate").value || 7.25),
      }),
    });
    $("cfgSaveMsg").textContent = "已保存";
    setTimeout(() => ($("cfgSaveMsg").textContent = ""), 2000);
    fetchOverview();
  });

  // Modal close
  $("btnCloseModal").addEventListener("click", () => $("sessionModal").classList.add("hidden"));
  $("sessionModal").addEventListener("click", (ev) => {
    if (ev.target === $("sessionModal")) $("sessionModal").classList.add("hidden");
  });

  window.addEventListener("resize", () => {
    if (state.lastData) drawTrendChart(state.lastData.timeseries, state.lastData.sidebar_agents);
  });
}

(function init() {
  const today = new Date();
  const d7 = new Date(today.getTime() - 6 * 86400 * 1000);
  $("inputSingleDate").value = today.toISOString().slice(0, 10);
  $("inputStartDate").value = d7.toISOString().slice(0, 10);
  $("inputEndDate").value = today.toISOString().slice(0, 10);
  bindEvents();
  setupChartHover();
  fetchOverview();
})();
