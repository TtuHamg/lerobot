"use strict";

const COLORS = [
  "#5ad4df", "#e4b653", "#79d59c", "#ef7479",
  "#9b8afb", "#f19862", "#d77bd8"
];
const state = {
  connected: false,
  chunks: new Map(),
  actual: [],
  cameras: { camera1: [], camera2: [] },
  cameraEnabled: { camera1: false, camera2: false },
  status: null,
  live: true,
  cursorT: Date.now() / 1000,
  windowSeconds: 30,
  maxHistory: 60,
  poseFrame: "eef",
  link8Available: false,
};
let socket = null;

const el = id => document.getElementById(id);
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
const keyOf = chunk => chunk.key;

function newestTime() {
  let value = Date.now() / 1000;
  for (const chunk of state.chunks.values()) {
    if (chunk.times && chunk.times.length) value = Math.max(value, chunk.times.at(-1));
  }
  if (state.actual.length) value = Math.max(value, state.actual.at(-1).t);
  return value;
}

function oldestTime() {
  let value = newestTime() - state.maxHistory;
  const candidates = [];
  for (const chunk of state.chunks.values()) {
    if (chunk.times && chunk.times.length) candidates.push(chunk.times[0]);
  }
  if (state.actual.length) candidates.push(state.actual[0].t);
  return candidates.length ? Math.max(value, Math.min(...candidates)) : value;
}

function prune() {
  const cutoff = Date.now() / 1000 - state.maxHistory - 2;
  for (const [key, chunk] of state.chunks) {
    if (chunk.received_t < cutoff) state.chunks.delete(key);
  }
  state.actual = state.actual.filter(sample => sample.t >= cutoff);
  for (const name of ["camera1", "camera2"]) {
    state.cameras[name] = state.cameras[name].filter(frame => frame.t >= cutoff);
  }
}

function applyEvent(event) {
  switch (event.type) {
    case "snapshot":
      state.maxHistory = event.history_seconds || 60;
      state.poseFrame = event.default_pose_frame || "eef";
      state.link8Available = Boolean(event.link8_available);
      state.chunks.clear();
      for (const chunk of event.chunks || []) state.chunks.set(keyOf(chunk), chunk);
      state.actual = event.actual || [];
      state.cameras = event.cameras || { camera1: [], camera2: [] };
      state.cameraEnabled = event.camera_enabled || { camera1: false, camera2: false };
      state.status = event.status;
      break;
    case "chunk":
      state.chunks.set(keyOf(event.chunk), event.chunk);
      break;
    case "ik": {
      const chunk = state.chunks.get(event.key);
      if (chunk) chunk.ik_joints = event.joints;
      break;
    }
    case "ack": {
      const chunk = state.chunks.get(event.key);
      if (chunk) chunk.ack = event.ack;
      break;
    }
    case "status":
      state.status = event.status;
      break;
    case "actual":
      state.actual.push(event.sample);
      break;
    case "camera_state":
      state.cameraEnabled[event.camera] = event.enabled;
      if (!event.enabled) state.cameras[event.camera] = [];
      break;
    case "camera":
      state.cameras[event.camera].push(event.frame);
      break;
    case "frame_transform":
      state.link8Available = Boolean(event.available);
      for (const update of event.chunks || []) {
        const chunk = state.chunks.get(update.key);
        if (chunk) chunk.link8 = update.link8;
      }
      for (const update of event.actual || []) {
        const sample = state.actual.find(value => Math.abs(value.t - update.t) < 1e-6);
        if (sample) sample.link8 = update.link8;
      }
      break;
  }
  prune();
  syncControls();
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.onopen = () => {
    state.connected = true;
    syncControls();
  };
  socket.onmessage = message => {
    try { applyEvent(JSON.parse(message.data)); }
    catch (error) { console.error("dashboard event error", error); }
  };
  socket.onclose = () => {
    state.connected = false;
    syncControls();
    setTimeout(connect, 1000);
  };
}

function requestCamera(name, enabled) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "camera", camera: name, enabled }));
  }
}

function syncControls() {
  const connection = el("connection");
  connection.textContent = state.connected ? "已连接" : "断线重连中";
  connection.className = `badge ${state.connected ? "ok" : "warn"}`;
  const status = state.status;
  const gateway = el("gateway");
  gateway.textContent = status?.armed ? "ARMED" : "HOLD";
  gateway.className = `badge ${status?.armed ? "ok" : "warn"}`;
  for (const name of ["camera1", "camera2"]) {
    el(`toggle-${name}`).checked = Boolean(state.cameraEnabled[name]);
  }
  el("pose-frame").value = state.poseFrame;
  el("pose-frame").querySelector('option[value="link8"]').disabled = !state.link8Available;
}

function currentChunk(cursorT) {
  const chunks = [...state.chunks.values()].sort((a, b) => a.chunk_idx - b.chunk_idx);
  let candidate = null;
  for (const chunk of chunks) {
    const first = chunk.times?.[0] ?? chunk.t0;
    if (first <= cursorT) candidate = chunk;
  }
  return candidate || chunks.at(-1) || null;
}

function nearestCameraFrame(name, cursorT) {
  const frames = state.cameras[name] || [];
  if (!frames.length) return null;
  let best = frames[0], distance = Math.abs(frames[0].t - cursorT);
  for (const frame of frames) {
    const nextDistance = Math.abs(frame.t - cursorT);
    if (nextDistance < distance) { best = frame; distance = nextDistance; }
  }
  return best;
}

function updateCameras(cursorT) {
  for (const name of ["camera1", "camera2"]) {
    const image = el(name);
    const placeholder = el(`${name}-placeholder`);
    const frame = nearestCameraFrame(name, cursorT);
    if (state.cameraEnabled[name] && frame) {
      image.src = `data:image/jpeg;base64,${frame.jpeg}`;
      image.style.display = "block";
      placeholder.style.display = "none";
    } else {
      image.removeAttribute("src");
      image.style.display = "none";
      placeholder.style.display = "block";
      placeholder.textContent = state.cameraEnabled[name] ? "等待缩略图…" : `未订阅 ${name}`;
    }
  }
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(100, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(80, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return ratio;
}

function drawChart(canvas, config) {
  const ratio = resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height;
  const margin = { left: 52 * ratio, right: 12 * ratio, top: 22 * ratio, bottom: 25 * ratio };
  const x0 = margin.left, x1 = width - margin.right;
  const y0 = margin.top, y1 = height - margin.bottom;
  ctx.clearRect(0, 0, width, height);

  const allValues = [];
  for (const segment of config.segments) {
    for (const row of segment.values) {
      for (const value of row) if (Number.isFinite(value)) allValues.push(value);
    }
  }
  if (config.actual) {
    for (const row of config.actual.values) {
      for (const value of row) if (Number.isFinite(value)) allValues.push(value);
    }
  }
  let minY = allValues.length ? Math.min(...allValues) : -1;
  let maxY = allValues.length ? Math.max(...allValues) : 1;
  if (config.fixedRange) [minY, maxY] = config.fixedRange;
  if (Math.abs(maxY - minY) < 1e-6) { minY -= 0.5; maxY += 0.5; }
  const pad = (maxY - minY) * 0.08;
  minY -= pad; maxY += pad;
  const px = t => x0 + (t - config.startT) / (config.endT - config.startT) * (x1 - x0);
  const py = v => y1 - (v - minY) / (maxY - minY) * (y1 - y0);

  ctx.lineWidth = ratio;
  ctx.strokeStyle = "#263241";
  ctx.fillStyle = "#718192";
  ctx.font = `${10 * ratio}px ui-monospace, monospace`;
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const value = minY + (maxY - minY) * i / 4;
    const y = py(value);
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillText(value.toFixed(config.precision ?? 3), x0 - 5 * ratio, y + 3 * ratio);
  }
  ctx.textAlign = "center";
  for (let i = 0; i <= 3; i++) {
    const t = config.startT + (config.endT - config.startT) * i / 3;
    const x = px(t);
    ctx.fillText(`${(t - config.endT).toFixed(1)}s`, x, height - 7 * ratio);
  }

  // Chunk boundaries and explicit idx labels make replacement cadence visible.
  for (const chunk of state.chunks.values()) {
    const t = chunk.times?.[0] ?? chunk.t0;
    if (t < config.startT || t > config.endT) continue;
    const x = px(t);
    ctx.strokeStyle = "rgba(230,184,92,.35)";
    ctx.setLineDash([3 * ratio, 3 * ratio]);
    ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#e6b85c";
    ctx.textAlign = "left";
    ctx.fillText(`#${chunk.chunk_idx}`, x + 2 * ratio, y0 + 10 * ratio);
  }

  function drawSegments(segments, dashed = false, alpha = 1) {
    for (const segment of segments) {
      const dimensions = segment.values[0]?.length || 0;
      for (let dim = 0; dim < dimensions; dim++) {
        ctx.strokeStyle = config.colors[dim % config.colors.length];
        ctx.globalAlpha = alpha;
        ctx.lineWidth = (dashed ? 1.2 : 1.6) * ratio;
        ctx.setLineDash(dashed ? [5 * ratio, 4 * ratio] : []);
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < segment.times.length; i++) {
          const t = segment.times[i], value = segment.values[i][dim];
          if (t < config.startT || t > config.endT || !Number.isFinite(value)) continue;
          const x = px(t), y = py(value);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        if (started) ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }
  drawSegments(config.segments);
  if (config.actual) drawSegments([config.actual], true, .9);

  const cursorX = px(clamp(config.cursorT, config.startT, config.endT));
  ctx.strokeStyle = "rgba(255,255,255,.55)";
  ctx.lineWidth = ratio;
  ctx.beginPath(); ctx.moveTo(cursorX, y0); ctx.lineTo(cursorX, y1); ctx.stroke();

  ctx.textAlign = "left";
  let legendX = x0 + 5 * ratio;
  for (let dim = 0; dim < config.labels.length; dim++) {
    ctx.fillStyle = config.colors[dim % config.colors.length];
    ctx.fillRect(legendX, 5 * ratio, 9 * ratio, 3 * ratio);
    ctx.fillStyle = "#aebdca";
    ctx.fillText(config.labels[dim], legendX + 12 * ratio, 10 * ratio);
    legendX += (config.labels[dim].length * 7 + 26) * ratio;
  }
}

function chunkSegments(kind, startT, endT) {
  const segments = [];
  const chunks = [...state.chunks.values()].sort((a, b) => a.chunk_idx - b.chunk_idx);
  for (const chunk of chunks) {
    if (!chunk.times?.length || chunk.times.at(-1) < startT || chunk.times[0] > endT) continue;
    let values = null;
    if (kind === "joints") values = chunk.ik_joints;
    if (kind === "gripper") values = chunk.gripper?.map(value => [value]);
    const poseRows = chunk[state.poseFrame] || chunk.eef;
    if (kind === "xyz") values = poseRows?.map(row => row.slice(0, 3));
    if (kind === "quat") values = poseRows?.map(row => row.slice(3, 7));
    if (values?.length) segments.push({ times: chunk.times.slice(0, values.length), values });
  }
  return segments;
}

function nearestPoint(chunk, cursorT, source) {
  if (!chunk?.times?.length || !source?.length) return null;
  let index = 0, distance = Infinity;
  chunk.times.forEach((time, candidate) => {
    const d = Math.abs(time - cursorT);
    if (d < distance) { distance = d; index = candidate; }
  });
  return source[Math.min(index, source.length - 1)];
}

function render() {
  const latest = newestTime();
  if (state.live) state.cursorT = latest;
  const endT = state.cursorT;
  const startT = endT - state.windowSeconds;
  const chunk = currentChunk(state.cursorT);
  el("chunk-label").textContent = chunk ? `chunk #${chunk.chunk_idx}` : "chunk --";
  el("plan-label").textContent = chunk
    ? `plan ${chunk.session_id.slice(0, 8)}/${chunk.plan_id}`
    : "plan --";
  const poseRows = chunk?.[state.poseFrame] || chunk?.eef;
  const first = poseRows?.[0], last = poseRows?.at(-1);
  el("timeline-summary").textContent = chunk
    ? `chunk #${chunk.chunk_idx} · ${poseRows.length} wp · ${state.poseFrame} z ${first[2].toFixed(3)} → ${last[2].toFixed(3)} m · ${chunk.ack?.detail || "awaiting ACK"}`
    : "等待 action chunk…";

  const earliest = oldestTime();
  const slider = el("timeline");
  slider.min = String(earliest);
  slider.max = String(latest);
  slider.value = String(clamp(state.cursorT, earliest, latest));
  el("time-label").textContent = `${new Date(state.cursorT * 1000).toLocaleTimeString()} · window ${state.windowSeconds}s`;
  el("live-button").className = state.live ? "active" : "";

  const joints = chunkSegments("joints", startT, endT);
  const gripper = chunkSegments("gripper", startT, endT);
  const xyz = chunkSegments("xyz", startT, endT);
  const quat = chunkSegments("quat", startT, endT);
  const actual = {
    times: state.actual.map(sample => sample.t),
    values: state.actual.map(sample => (sample[state.poseFrame] || sample.eef).slice(0, 3)),
  };
  const common = { startT, endT, cursorT: state.cursorT, colors: COLORS };
  drawChart(el("joints-chart"), {
    ...common, segments: joints, labels: ["j1","j2","j3","j4","j5","j6","j7"],
  });
  drawChart(el("gripper-chart"), {
    ...common, segments: gripper, labels: ["closed"], fixedRange: [0, 1],
  });
  drawChart(el("xyz-chart"), {
    ...common, segments: xyz, actual, labels: ["x","y","z"],
  });
  drawChart(el("quat-chart"), {
    ...common, segments: quat, labels: ["qx","qy","qz","qw"], fixedRange: [-1, 1],
  });

  const jointValue = nearestPoint(chunk, state.cursorT, chunk?.ik_joints);
  const gripperValue = nearestPoint(chunk, state.cursorT, chunk?.gripper?.map(v => [v]));
  const eefValue = nearestPoint(chunk, state.cursorT, poseRows);
  el("joints-value").textContent = jointValue ? jointValue.map((v,i) => `j${i+1}=${v.toFixed(3)}`).join(" ") : "IK pending";
  el("gripper-value").textContent = gripperValue ? gripperValue[0].toFixed(3) : "--";
  el("xyz-value").textContent = eefValue ? eefValue.slice(0,3).map(v => v.toFixed(3)).join(", ") : "--";
  el("quat-value").textContent = eefValue ? eefValue.slice(3,7).map(v => v.toFixed(3)).join(", ") : "--";
  const frameLabel = state.poseFrame === "link8" ? "Link8 / flange" : "EEF / TCP";
  el("xyz-title").textContent = `${frameLabel} xyz`;
  el("quat-title").textContent = `${frameLabel} quaternion xyzw`;
  updateCameras(state.cursorT);
}

el("live-button").addEventListener("click", () => { state.live = true; });
el("window-seconds").addEventListener("change", event => {
  state.windowSeconds = Number(event.target.value);
});
el("pose-frame").addEventListener("change", event => {
  if (event.target.value === "link8" && !state.link8Available) {
    event.target.value = "eef";
    return;
  }
  state.poseFrame = event.target.value;
});
el("timeline").addEventListener("input", event => {
  state.live = false;
  state.cursorT = Number(event.target.value);
});
for (const name of ["camera1", "camera2"]) {
  el(`toggle-${name}`).addEventListener("change", event => requestCamera(name, event.target.checked));
}
window.addEventListener("resize", render);

connect();
syncControls();
setInterval(render, 200);
