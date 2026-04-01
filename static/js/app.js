/* ═══════════════════════════════════════════════════════
   MUSCOTT — Dashboard JS
   Features:
   - Real-time status / alerts polling
   - IP-only camera add (auto-builds RTSP URL)
   - Remove camera with confirmation
   - AI toggle, snapshot, fullscreen
   - Stale feed watchdog (auto-reconnect)
   - Live FPS display
═══════════════════════════════════════════════════════ */
'use strict';

// ── State ────────────────────────────────────────────
let lastLevel    = 'safe';
let alertsCache  = [];
let aiEnabled    = true;
let maxP = 1, maxV = 1, maxW = 1;

// ── Clock ────────────────────────────────────────────
(function clock() {
  const months = ['JAN','FEB','MAR','APR','MAY','JUN',
                  'JUL','AUG','SEP','OCT','NOV','DEC'];
  function tick() {
    const d   = new Date();
    const pad = n => String(n).padStart(2,'0');
    document.getElementById('clock').textContent =
      `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    document.getElementById('clockDate').textContent =
      `${pad(d.getDate())} ${months[d.getMonth()]} ${d.getFullYear()}`;
  }
  setInterval(tick, 1000); tick();
})();

// ── Threat level ─────────────────────────────────────
const LEVEL_META = {
  safe:   { label:'SECURE',   sub:'All Systems Normal',           icon:'●' },
  level1: { label:'ELEVATED', sub:'Motion Violence Detected',     icon:'▲' },
  level2: { label:'HIGH',     sub:'Armed Violence Detected',      icon:'⚠' },
  level3: { label:'CRITICAL', sub:'⚠ ACTIVE ARMED THREAT',       icon:'☣' },
};

function setThreatLevel(level) {
  if (level === lastLevel) return;
  lastLevel = level;
  const badge = document.getElementById('threatBadge');
  const m     = LEVEL_META[level] || LEVEL_META.safe;
  badge.dataset.level = level;
  document.getElementById('threatIcon').textContent  = m.icon;
  document.getElementById('threatLabel').textContent = m.label;
  document.getElementById('threatSub').textContent   = m.sub;

  ['safe','level1','level2','level3'].forEach(l => {
    document.getElementById(`tl-${l}`)?.classList.toggle('active-level', l === level);
    const ind = document.getElementById(`ind-${l}`);
    if (ind) ind.classList.toggle('active', l === level);
  });

  const overlay = document.getElementById('feedAlertOverlay');
  overlay.style.display = (level === 'level2' || level === 'level3') ? 'flex' : 'none';
  document.getElementById('globalFlash').style.display =
    level === 'level3' ? 'block' : 'none';
}

// ── Status poll ──────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch('/status');
    if (!res.ok) throw new Error(res.status);
    const d    = await res.json();

    setThreatLevel(d.overall_level);

    setText('personCount',  d.total_persons  ?? 0);
    setText('weaponCount',  d.total_weapons  ?? 0);
    setText('vehicleCount', d.total_vehicles ?? 0);
    setText('camCount',     d.camera_count   ?? 0);

    const fps = d.stream_fps ?? 0;
    setText('streamFps',  fps.toFixed(1));
    setText('statusFps',  fps.toFixed(1));
    const fpsEl = document.getElementById('statusFps');
    if (fpsEl) fpsEl.className = 'fsb-val ' + (fps >= 15 ? 'green' : 'yellow');

    // AI status
    aiEnabled = d.ai_enabled ?? true;
    const aiBtn = document.getElementById('aiToggleBtn');
    if (aiBtn) {
      aiBtn.classList.toggle('ai-off', !aiEnabled);
      aiBtn.innerHTML = aiEnabled
        ? `<svg width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5" fill="none"/><rect x="5" y="4" width="1.5" height="6" fill="currentColor"/><rect x="7.5" y="4" width="1.5" height="6" fill="currentColor"/></svg> PAUSE AI`
        : `<svg width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5" fill="none"/><polygon points="5.5,4 11,7 5.5,10" fill="currentColor"/></svg> RESUME AI`;
    }
    const aiStatusEl = document.getElementById('aiStatus');
    if (aiStatusEl) {
      aiStatusEl.textContent = aiEnabled ? 'ACTIVE' : 'PAUSED';
      aiStatusEl.className   = 'fsb-val ' + (aiEnabled ? 'green' : 'yellow');
    }

    // Stat bars
    maxP = Math.max(maxP, d.total_persons,  1);
    maxV = Math.max(maxV, d.total_vehicles, 1);
    maxW = Math.max(maxW, d.total_weapons,  1);
    setBar('hbPersons',  'hbPersonsVal',  d.total_persons,  maxP);
    setBar('hbVehicles', 'hbVehiclesVal', d.total_vehicles, maxV);
    setBar('hbWeapons',  'hbWeaponsVal',  d.total_weapons,  maxW);

    // Camera rows
    updateCamRows(d.cameras ?? {});
    updateTrackList(d.cameras ?? {});

    setStreamStatus('STREAMING', 'green');

  } catch(e) {
    setStreamStatus('ERROR', 'red');
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBar(barId, valId, val, max) {
  const pct = Math.min(100, Math.round((val / max) * 100));
  const bar = document.getElementById(barId);
  if (bar) bar.style.width = pct + '%';
  setText(valId, val);
}

function setStreamStatus(label, color) {
  const el  = document.getElementById('streamStatus');
  const dot = document.querySelector('#streamStatusPill .fsb-dot');
  if (el)  el.textContent = label;
  if (dot) dot.className  = `fsb-dot ${color}`;
}

function updateCamRows(cameras) {
  Object.entries(cameras).forEach(([id, cs]) => {
    const dot   = document.getElementById(`camDot${id}`);
    const meta  = document.getElementById(`camMeta${id}`);
    const badge = document.getElementById(`camBadge${id}`);
    if (!dot) return;
    const online = cs.level !== 'offline';
    dot.className  = 'cam-live-dot' + (online ? '' : ' offline');
    if (meta)  meta.textContent  =
      `${cs.persons}p  ${cs.vehicles}v  ${(cs.fps_ai||0).toFixed(1)} ai-fps`;
    if (badge) {
      badge.textContent = cs.level.toUpperCase();
      badge.className   = 'cam-badge ' + cs.level;
    }
  });
}

function updateTrackList(cameras) {
  const list = document.getElementById('trackList');
  if (!list) return;
  const items = [];
  Object.entries(cameras).forEach(([camId, cs]) => {
    (cs.tracks || []).forEach(gid => items.push({ gid, camId }));
  });
  if (!items.length) {
    list.innerHTML = '<div class="empty-state">No active tracks</div>';
    return;
  }
  list.innerHTML = '';
  items.forEach(({gid, camId}) => {
    const c = document.createElement('div');
    c.className = 'track-chip';
    c.innerHTML = `<span>ID #${gid}</span><span>CAM ${camId}</span>`;
    list.appendChild(c);
  });
}

// ── Alerts ───────────────────────────────────────────
async function pollAlerts() {
  try {
    const res  = await fetch('/alerts?n=40');
    const data = await res.json();
    if (JSON.stringify(data) === JSON.stringify(alertsCache)) return;
    alertsCache = data;
    const feed = document.getElementById('alertFeed');
    if (!feed) return;
    if (!data.length) {
      feed.innerHTML = '<div class="empty-state">System secure — no alerts</div>';
      return;
    }
    feed.innerHTML = '';
    data.forEach(a => {
      const card = document.createElement('div');
      card.className = `alert-card ${a.level}`;
      const reasons = (a.reasons||[]).join(', ') || '—';
      const wpns    = (a.weapons||[]).join(', ')  || 'none';
      card.innerHTML = `
        <div class="alert-ts">${a.time}</div>
        <div class="alert-row">
          <span class="alert-lv ${a.level}">${a.level.toUpperCase()}</span>
          <span class="alert-body">Motion: ${reasons} · Weapon: ${wpns}</span>
          <span class="alert-cam">CAM ${a.cam}</span>
        </div>`;
      feed.appendChild(card);
    });
  } catch(e) {}
}

function clearAlerts() {
  alertsCache = [];
  const f = document.getElementById('alertFeed');
  if (f) f.innerHTML = '<div class="empty-state">System secure — no alerts</div>';
}

// ── Log ──────────────────────────────────────────────
async function refreshLog() {
  try {
    const res  = await fetch('/log_file');
    const text = await res.text();
    const el   = document.getElementById('rawLog');
    if (el) { el.textContent = text || 'No events logged yet.'; el.scrollTop = el.scrollHeight; }
  } catch(e) {}
}

// ── IP Camera Add ────────────────────────────────────
// User types only the IP address (e.g. 169.254.47.240)
// We auto-build: rtsp://<ip>/live/0/MAIN

const IP_RE = /^(\d{1,3}\.){3}\d{1,3}$/;

function validateIP(ip) {
  if (!IP_RE.test(ip)) return false;
  return ip.split('.').every(n => parseInt(n) <= 255);
}

function showHint(msg, type) {
  const el = document.getElementById('ipHint');
  if (!el) return;
  el.textContent = msg;
  el.className   = `ip-hint ${type}`;
}

async function addCamera() {
  const input = document.getElementById('ipInput');
  if (!input) return;
  const ip = input.value.trim();

  if (!ip) { showHint('Enter an IP address', 'err'); return; }
  if (!validateIP(ip)) { showHint('Invalid IP — use format: 169.254.47.240', 'err'); return; }

  const rtsp = `rtsp://${ip}/live/0/MAIN`;
  showHint(`Connecting to ${rtsp}…`, '');

  try {
    const res  = await fetch('/add_camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: rtsp }),
    });
    const data = await res.json();
    showHint(`✓ CAM ${data.cam_id} added`, 'ok');
    input.value = '';
    addCamRowToSidebar(data.cam_id, ip);
    setTimeout(() => showHint('', ''), 3000);
  } catch(e) {
    showHint('Connection failed — check IP', 'err');
  }
}

function addCamRowToSidebar(id, ip) {
  const list = document.getElementById('camStatusList');
  if (!list) return;
  const row = document.createElement('div');
  row.className = 'cam-item'; row.id = `camRow${id}`;
  row.innerHTML = `
    <div class="cam-item-left">
      <div class="cam-live-dot" id="camDot${id}"></div>
      <div class="cam-item-info">
        <div class="cam-item-name">CAM ${id}</div>
        <div class="cam-item-meta" id="camMeta${id}">${ip}</div>
      </div>
    </div>
    <div class="cam-item-right">
      <div class="cam-badge safe" id="camBadge${id}">SAFE</div>
      <button class="cam-remove-btn" onclick="removeCamera(${id})" title="Remove">✕</button>
    </div>`;
  list.appendChild(row);
}

// ── Remove Camera ────────────────────────────────────
async function removeCamera(id) {
  const confirmed = window.confirm(`Remove CAM ${id}?`);
  if (!confirmed) return;
  try {
    await fetch(`/remove_camera/${id}`, { method: 'DELETE' });
    const row = document.getElementById(`camRow${id}`);
    if (row) {
      row.style.opacity    = '0';
      row.style.transition = 'opacity 0.3s';
      setTimeout(() => row.remove(), 300);
    }
  } catch(e) { alert(`Failed to remove CAM ${id}`); }
}

// ── AI Toggle ────────────────────────────────────────
async function toggleAI() {
  aiEnabled = !aiEnabled;
  try {
    await fetch('/toggle_ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: aiEnabled }),
    });
  } catch(e) { aiEnabled = !aiEnabled; }
}

// ── Fullscreen / Snapshot ────────────────────────────
function toggleFullscreen() {
  const el = document.getElementById('feedWrapper');
  if (!document.fullscreenElement) el.requestFullscreen().catch(()=>{});
  else document.exitFullscreen();
}

function captureSnapshot() {
  const img    = document.getElementById('videoFeed');
  const c      = document.createElement('canvas');
  c.width      = img.naturalWidth  || 640;
  c.height     = img.naturalHeight || 360;
  c.getContext('2d').drawImage(img, 0, 0);
  const a      = document.createElement('a');
  a.download   = `sentinel_${Date.now()}.jpg`;
  a.href       = c.toDataURL('image/jpeg', 0.93);
  a.click();
}

// ── Stale feed watchdog ──────────────────────────────
// If the MJPEG img hasn't refreshed in 4 s, force a reconnect
function monitorFeed() {
  const img = document.getElementById('videoFeed');
  let lastW = 0, stale = 0;
  setInterval(() => {
    if (img.naturalWidth === lastW) {
      stale++;
      if (stale >= 2) {
        setStreamStatus('RECONNECTING…', 'yellow');
        const src = img.src;
        img.src = '';
        setTimeout(() => { img.src = src; stale = 0; }, 300);
      }
    } else {
      stale  = 0;
      lastW  = img.naturalWidth;
    }
  }, 4000);
}

// ── IP input — allow only digits and dots ────────────
function initIPInput() {
  const input = document.getElementById('ipInput');
  if (!input) return;
  input.addEventListener('keypress', e => {
    if (!/[\d.]/.test(e.key)) e.preventDefault();
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') addCamera();
  });
}

// ── Boot ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initIPInput();
  monitorFeed();
  pollStatus();
  pollAlerts();
  refreshLog();
  setInterval(pollStatus,  1000);
  setInterval(pollAlerts,  2000);
  setInterval(refreshLog, 15000);
});
