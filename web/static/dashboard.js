import { createVisionCapture } from "/static/vision-capture.js";

const appState = {
  sessions: [],
  selectedSessionId: null,
  livePayload: null,
  visionCapture: null,
};

const stageMeta = {
  steady: {
    label: "steady",
    title: "Quiet Monitoring",
    reason: "Silent by default. Evidence first.",
  },
  silent_log: {
    label: "silent log",
    title: "Silent Logging",
    reason: "状态开始下滑，先静默记录，不打断工作。",
  },
  light_nudge: {
    label: "light nudge",
    title: "Gentle Nudge",
    reason: "状态持续走低，已到最轻提醒阈值。",
  },
  escalation: {
    label: "escalation",
    title: "Escalation",
    reason: "出现强风险信号，建议暂停并进入复盘。",
  },
  recovered: {
    label: "recovered",
    title: "Recovered",
    reason: "状态重新回稳，继续观察即可。",
  },
};

const protocolSteps = [
  {
    key: "silent_log",
    title: "1. Silent Log",
    desc: "第一次恶化只记录，不打断。适合无感工作场景。",
  },
  {
    key: "light_nudge",
    title: "2. Light Nudge",
    desc: "只有持续恶化或二次恶化时，才提示一次。",
  },
  {
    key: "escalation",
    title: "3. Escalation",
    desc: "强风险或严重信号时，引导暂停并进入复盘。",
  },
];

function $(id) {
  return document.getElementById(id);
}

function clamp(num, min, max) {
  return Math.max(min, Math.min(max, num));
}

function stageClass(stage) {
  return ["steady", "silent_log", "light_nudge", "escalation", "recovered"].includes(stage)
    ? stage
    : "steady";
}

function formatSessionTime(iso) {
  if (!iso) return "unknown";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function formatMinutes(sec) {
  const total = Number(sec) || 0;
  if (total < 60) {
    return `${Math.round(total)} sec`;
  }
  return `${Math.round(total / 60)} min`;
}

function renderProtocolRail(stage) {
  $("protocolRail").innerHTML = protocolSteps
    .map((step) => {
      let cls = "rail-step";
      if (step.key === stage) {
        cls += " active";
        if (stage === "light_nudge") cls += " warn";
        if (stage === "escalation") cls += " danger";
      }
      return `<div class="${cls}">
        <strong>${step.title}</strong>
        <p>${step.desc}</p>
      </div>`;
    })
    .join("");
}

function renderMetricCards(stamina) {
  const consistency = Math.round(((stamina?.consistency ?? 0) * 100));
  const tension = Math.round(((stamina?.tension ?? 0) * 100));
  const fatigue = Math.round(((stamina?.fatigue ?? 0) * 100));
  $("metricGrid").innerHTML = `
    <div class="metric-card"><span>Consistency</span><strong>${consistency}%</strong></div>
    <div class="metric-card"><span>Tension</span><strong>${tension}%</strong></div>
    <div class="metric-card"><span>Fatigue</span><strong>${fatigue}%</strong></div>
  `;
}

function formatSignedDegrees(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}°`;
}

function renderVisionPanel(vision) {
  const hasVision = Boolean(vision);
  const stale = Boolean(vision?.stale);
  const quality = Math.round((vision?.quality ?? 0) * 100);
  const blinkRate = hasVision ? `${Number(vision?.blink_rate ?? 0).toFixed(1)}/min` : "--";
  const perclos = hasVision ? `${Math.round((vision?.perclos ?? 0) * 100)}%` : "--";
  const yawns = hasVision ? String(vision?.yawn_count ?? 0) : "--";

  let alertness = "waiting";
  if (hasVision) {
    alertness = stale ? "stale" : (vision?.alertness || "tracking");
  }

  let faceState = "等待摄像头授权";
  if (hasVision) {
    if (!vision?.face_present) {
      faceState = stale ? "视觉流已暂停" : "未检测到人脸";
    } else {
      faceState = `${stale ? "最近一帧已过期" : "人脸已锁定"} · 质量 ${quality}%`;
    }
  }

  let poseState = "pitch -- · yaw --";
  if (hasVision) {
    const states = [
      `pitch ${formatSignedDegrees(vision?.head_pitch_mean)}`,
      `yaw ${formatSignedDegrees(vision?.head_yaw_mean)}`,
    ];
    if (vision?.head_nod) states.push("低头风险");
    if (vision?.head_distracted) states.push("注意分散");
    if (vision?.yawn_active) states.push("哈欠进行中");
    poseState = states.join(" · ");
  }

  $("visionBlinkRate").textContent = blinkRate;
  $("visionPerclos").textContent = perclos;
  $("visionYawns").textContent = yawns;
  $("visionAlertness").textContent = alertness;
  $("visionFaceState").textContent = faceState;
  $("visionPoseState").textContent = poseState;
  $("visionFrameMeta").textContent = hasVision ? (stale ? "stale" : "live") : "等待摄像头";
}

function renderSignalSummary(payload) {
  const source = payload?.fusion?.source || "emg_only";
  const activity = payload?.activity || "idle";
  const confidence = Math.round((payload?.confidence || 0) * 100);
  const chips = [
    `source: ${source}`,
    `activity: ${activity}`,
    `confidence: ${confidence}%`,
  ];
  if (payload?.fusion?.alerts?.length) {
    chips.push(`alerts: ${payload.fusion.alerts.join(", ")}`);
  }
  $("heroChips").innerHTML = chips.map((text) => `<span class="chip">${text}</span>`).join("");
}

async function refreshLiveState() {
  try {
    const stateResp = await fetch("/api/v1/state");
    const stateBody = await stateResp.json();
    renderLive(stateBody?.ok ? stateBody.data : {});
  } catch (_error) {
    renderLive({});
  }
}

function renderLive(payload) {
  appState.livePayload = payload;
  const intervention = payload?.intervention || { stage: "steady", evidence: [] };
  const stage = stageClass(intervention.stage);
  const stageInfo = stageMeta[stage] || stageMeta.steady;
  const score = payload?.fusion?.stamina ?? payload?.stamina?.value;
  const reasons = intervention.evidence?.length
    ? intervention.evidence.join(" · ")
    : (payload?.decision?.reasons || []).join(" · ");

  $("liveStage").textContent = stageInfo.label;
  $("liveStage").className = `stage-pill ${stage}`;
  $("heroScore").textContent = typeof score === "number" ? Math.round(score) : "--";
  $("heroState").textContent = stageInfo.title;
  $("heroReason").textContent = reasons || stageInfo.reason;
  $("liveSource").textContent = payload?.fusion?.source || "none";
  $("liveWorkMin").textContent = Math.round(payload?.decision?.continuous_work_min || 0);
  $("liveNextMin").textContent = Math.round(
    payload?.decision?.suggested_break_min || payload?.decision?.suggested_work_min || 0,
  );
  $("liveStageMeta").textContent = intervention.should_surface ? "visible intervention" : "silent by default";

  renderMetricCards(payload?.stamina);
  renderProtocolRail(stage);
  renderSignalSummary(payload);
  renderVisionPanel(payload?.vision);
}

function eventMessage(event) {
  if (event?.message) return event.message;
  if (Array.isArray(event?.evidence) && event.evidence.length) return event.evidence.join(" · ");
  const fallback = {
    silent_log: "第一次恶化，已静默记录。",
    light_nudge: "持续恶化，建议轻提醒一次。",
    escalation: "强风险阶段，建议暂停并复盘。",
    recovered: "状态回稳，结束当前干预阶段。",
  };
  return fallback[event?.type] || "No detail";
}

async function loadTimeline() {
  const resp = await fetch("/api/v1/timeline");
  const body = await resp.json();
  const events = body?.ok ? body.data.events.slice().reverse() : [];
  const feed = $("eventFeed");
  if (!events.length) {
    feed.innerHTML = `<div class="placeholder">还没有 intervention 事件。开始 demo 或真实数据后，这里会出现 silent log、light nudge、escalation 的轨迹。</div>`;
    return;
  }
  feed.innerHTML = events
    .map((event) => {
      const d = new Date((event.time || 0) * 1000);
      const stamp = Number.isNaN(d.getTime()) ? "--:--" : `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
      return `<div class="event-item ${stageClass(event.type)}">
        <strong>${event.type.replace("_", " ")} · ${stamp}</strong>
        <p>${eventMessage(event)}</p>
      </div>`;
    })
    .join("");
}

async function loadEvidence() {
  const [sessionsResp, flywheelResp] = await Promise.all([
    fetch("/api/v1/sessions"),
    fetch("/api/v1/flywheel/stats"),
  ]);
  const sessionsBody = await sessionsResp.json();
  const flywheelBody = await flywheelResp.json();
  const sessions = sessionsBody?.ok ? sessionsBody.data.sessions : [];
  const stats = flywheelBody?.ok ? flywheelBody.data : {};
  $("evidenceCards").innerHTML = `
    <div class="evidence-card"><span>Sessions</span><strong>${sessions.length}</strong></div>
    <div class="evidence-card"><span>Labels</span><strong>${stats.total_labels ?? 0}</strong></div>
    <div class="evidence-card"><span>Labeled Samples</span><strong>${stats.labeled_samples ?? 0}</strong></div>
  `;
}

function reviewNarrative(review) {
  const summary = review.summary || {};
  const parts = [
    `平均耐力 ${summary.average_stamina ?? 0}。`,
    `最大下滑 ${summary.largest_drop ?? 0}。`,
    summary.recovered ? "后段出现恢复。" : "后段恢复不明显。",
  ];
  if ((review.flags || []).some((flag) => flag.type === "high_tension")) {
    parts.push("张力段偏高，值得在打字姿势和微休息上做实验。");
  }
  return parts.join(" ");
}

function makeSparkline(series) {
  if (!series.length) {
    return `<div class="placeholder">No series data.</div>`;
  }
  const width = 720;
  const height = 120;
  const xs = series.map((point) => point.minute);
  const ys = series.map((point) => point.stamina);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs, minX + 1);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 100);
  const points = series
    .map((point) => {
      const x = ((point.minute - minX) / (maxX - minX)) * width;
      const y = height - ((point.stamina - minY) / (maxY - minY || 1)) * height;
      return `${x.toFixed(1)},${clamp(y, 0, height).toFixed(1)}`;
    })
    .join(" ");
  return `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="Session stamina trend">
      <polyline points="${points}" fill="none" stroke="#8cf0b5" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>
    </svg>
  `;
}

function renderReviewBoard(review, sessionMeta) {
  if (!review) {
    $("reviewBoard").innerHTML = `<div class="placeholder">还没有可展示的复盘数据。先录一段 session，结束后这里会自动出现。</div>`;
    return;
  }
  const summary = review.summary || {};
  const moments = review.moments || {};
  const flags = review.flags || [];
  $("reviewBoard").innerHTML = `
    <div class="review-stat-grid">
      <div class="review-stat"><span>Average</span><strong>${summary.average_stamina ?? 0}</strong></div>
      <div class="review-stat"><span>Largest Drop</span><strong>${summary.largest_drop ?? 0}</strong></div>
      <div class="review-stat"><span>Net Change</span><strong>${summary.net_change ?? 0}</strong></div>
    </div>
    <p class="review-copy">${reviewNarrative(review)}</p>
    <div class="review-moments">
      <div class="review-moment"><span>Session</span><strong>${sessionMeta ? `${formatSessionTime(sessionMeta.startedAt)} · ${formatMinutes(sessionMeta.durationSec)}` : "current"}</strong></div>
      <div class="review-moment"><span>Lowest</span><strong>${moments.lowest ? `${moments.lowest.stamina} @ ${moments.lowest.minute}m` : "—"}</strong></div>
      <div class="review-moment"><span>Steepest drop</span><strong>${moments.steepest_drop ? `${moments.steepest_drop.delta} @ ${moments.steepest_drop.minute}m` : "—"}</strong></div>
    </div>
    <div class="sparkline-wrap">${makeSparkline(review.series || [])}</div>
    <div class="flag-list">
      ${flags.length
        ? flags
            .map((flag) => `<div class="flag ${flag.type}"><strong>${flag.type}</strong><div>${flag.message}</div></div>`)
            .join("")
        : `<div class="placeholder">No strong flags. This session was comparatively stable.</div>`}
    </div>
  `;
}

function renderSessionList() {
  const list = $("sessionList");
  if (!appState.sessions.length) {
    list.innerHTML = `<div class="placeholder">还没有归档 session。点 “开始记录” 再 “结束并保存”，这里会生成 review 列表。</div>`;
    return;
  }
  list.innerHTML = appState.sessions
    .map((session) => {
      const active = session.id === appState.selectedSessionId ? " active" : "";
      return `<div class="session-item${active}" data-session-id="${session.id}">
        <strong>${session.title || "Web 记录"}</strong>
        <p>${formatSessionTime(session.startedAt)}</p>
        <div class="session-meta">
          <span>${formatMinutes(session.durationSec)}</span>
          <span>${session.snapshotCount || 0} snapshots</span>
        </div>
      </div>`;
    })
    .join("");

  list.querySelectorAll("[data-session-id]").forEach((node) => {
    node.addEventListener("click", async () => {
      appState.selectedSessionId = node.getAttribute("data-session-id");
      renderSessionList();
      await loadReview(appState.selectedSessionId);
    });
  });
}

async function loadSessions() {
  const resp = await fetch("/api/v1/sessions");
  const body = await resp.json();
  appState.sessions = body?.ok ? body.data.sessions : [];
  if (!appState.selectedSessionId && appState.sessions.length) {
    appState.selectedSessionId = appState.sessions[0].id;
  } else if (
    appState.selectedSessionId &&
    !appState.sessions.some((session) => session.id === appState.selectedSessionId)
  ) {
    appState.selectedSessionId = appState.sessions[0]?.id || null;
  }
  renderSessionList();
  if (appState.selectedSessionId) {
    await loadReview(appState.selectedSessionId);
  } else {
    renderReviewBoard(null, null);
  }
}

async function loadReview(sessionId) {
  const resp = await fetch(`/api/v1/sessions/${sessionId}/review`);
  const body = await resp.json();
  const sessionMeta = appState.sessions.find((session) => session.id === sessionId) || null;
  renderReviewBoard(body?.ok ? body.data : null, sessionMeta);
}

function syncTransportFields(mode) {
  $("serialPortSelect").style.display = mode === "serial" ? "" : "none";
  $("serialPortManual").style.display = mode === "serial" ? "" : "none";
  $("bleAddrInput").style.display = mode === "ble" ? "" : "none";
}

async function refreshTransport() {
  const resp = await fetch("/api/v1/transport?include_all=1");
  const body = await resp.json();
  const data = body?.ok ? body.data : {};
  const mode = data.mode || "idle";
  $("transportMode").value = mode;
  syncTransportFields(mode);

  const select = $("serialPortSelect");
  const ports = (data.serial_ports || []).concat(data.serial_ports_other || []);
  select.innerHTML = `<option value="">Auto detect</option>` + ports
    .map((item) => `<option value="${item.port}">${item.label || item.port}</option>`)
    .join("");
  if (data.serial_port) {
    select.value = data.serial_port;
    $("serialPortManual").value = data.serial_port;
  }
  if (data.ble_address) {
    $("bleAddrInput").value = data.ble_address;
  }
  $("transportMsg").textContent = data.stream_stats
    ? `EMG ${data.stream_stats.emg_frames} frames · dropped ${data.stream_stats.dropped_frames}`
    : (data.signal_hint || "ready");
}

async function applyTransport() {
  const mode = $("transportMode").value;
  const body = { mode };
  if (mode === "serial") {
    const port = $("serialPortManual").value.trim() || $("serialPortSelect").value;
    if (port) body.port = port;
  }
  if (mode === "ble") {
    const address = $("bleAddrInput").value.trim();
    if (address) body.address = address;
  }
  const resp = await fetch("/api/v1/transport/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  $("transportMsg").textContent = data?.ok ? (data.data?.message || "applied") : (data.message || data.error || "failed");
  await refreshTransport();
}

async function startSession() {
  const resp = await fetch("/api/v1/sessions/web/start", { method: "POST" });
  const body = await resp.json();
  $("sessionMsg").textContent = body?.ok ? "记录中，结束后将自动进入复盘。" : (body.message || body.error || "failed");
}

async function stopSession() {
  const resp = await fetch("/api/v1/sessions/web/stop", { method: "POST" });
  const body = await resp.json();
  if (!body?.ok) {
    $("sessionMsg").textContent = body.message || body.error || "failed";
    return;
  }
  $("sessionMsg").textContent = body.data?.insight || "已保存";
  if (body.data?.session_id) {
    appState.selectedSessionId = body.data.session_id;
  }
  await loadSessions();
  await loadEvidence();
}

async function sendLabel(label) {
  const resp = await fetch("/api/v1/flywheel/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, window_sec: 300 }),
  });
  const body = await resp.json();
  $("labelMsg").textContent = body?.ok ? `已标注：${label}` : (body.message || body.error || "failed");
  await loadEvidence();
}

async function initVision() {
  const video = $("dashCam");
  const canvas = $("dashProbe");
  const status = $("visionMsg");
  appState.visionCapture = createVisionCapture({
    video,
    canvas,
    onStatus: (text) => {
      status.textContent = text;
    },
    onWsState: (text) => {
      $("visionWs").textContent = `Vision WS: ${text}`;
    },
  });

  $("dashVisionStart").addEventListener("click", async () => {
    try {
      await appState.visionCapture.start();
      video.classList.add("on");
      $("visionMsg").textContent = "摄像头已开启。";
      await refreshLiveState();
    } catch (error) {
      $("visionMsg").textContent = String(error?.message || error);
    }
  });

  $("dashVisionStop").addEventListener("click", () => {
    appState.visionCapture.stop();
    video.classList.remove("on");
    $("visionMsg").textContent = "摄像头已停止。";
    renderVisionPanel(null);
  });

  $("openVisionBtn").addEventListener("click", () => {
    window.open("/static/vision.html", "_blank", "noopener,noreferrer");
  });
}

function connectLive() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.addEventListener("open", () => {
    $("connPill").textContent = "live";
    $("connPill").className = "conn-pill live";
  });
  ws.addEventListener("close", () => {
    $("connPill").textContent = "offline";
    $("connPill").className = "conn-pill";
    setTimeout(connectLive, 2000);
  });
  ws.addEventListener("message", ({ data }) => {
    const payload = JSON.parse(data);
    if (payload.type === "state_update") {
      renderLive(payload);
    }
  });
}

async function boot() {
  $("transportMode").addEventListener("change", (event) => syncTransportFields(event.target.value));
  $("transportApplyBtn").addEventListener("click", applyTransport);
  $("sessionStartBtn").addEventListener("click", startSession);
  $("sessionStopBtn").addEventListener("click", stopSession);
  $("labelFatiguedBtn").addEventListener("click", () => sendLabel("fatigued"));
  $("labelAlertBtn").addEventListener("click", () => sendLabel("alert"));

  await refreshTransport();
  await Promise.all([loadTimeline(), loadSessions(), loadEvidence(), initVision()]);
  await refreshLiveState();

  connectLive();
  window.setInterval(refreshLiveState, 2000);
  window.setInterval(loadTimeline, 5000);
  window.setInterval(loadEvidence, 10000);
  window.setInterval(refreshTransport, 5000);
}

boot();
