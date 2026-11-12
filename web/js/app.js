// Lightsail Propulsion Simulator: UI, real-time animation and live analysis.
import * as ph from './physics.js';
import { Plot, Tornado, cssVar, setupCanvas, withAlpha } from './charts.js';

const { C, G_N, AU, LY, YEAR, CLASSICAL: CL, RELATIVISTIC: RE, MODELS } = ph;
const NAME = { classical: 'Classical', relativistic: 'Relativistic' };
const COLOR = { classical: '--classical', relativistic: '--relativistic' };
const SPACE = { classical: '--space-classical', relativistic: '--space-relativistic' };

import { num, fmtDist, fmtTime, pctc, TIME_TICKS, DIST_TICKS } from './format.js';
import { initLive } from './live.js';

// ============================================================================ state
const ui = { illumStop: 10, emissivity: 1 };
let params = { ...ph.DEFAULTS };
let D = null;                      // derived quantities
const traj = {}, summ = {};
let tEnd = 1, valid = true;
let simT = 0, playing = true, speed = 'auto';
const T_START = 0.05, AUTO_SECONDS = 40;
let dirty = { explore: true, unc: true };
let activeTab = 'sim';
let live = null;

const PRESETS = {
  'UCSB calculator defaults': {},
  'Ultralight sail (0.1 g/m²)': { arealDensity: 0.1e-3 },
  'Thick sail (10 g/m²)': { arealDensity: 10e-3 },
  'Lossy sail (R = 0.6)': { reflectivity: 0.6 },
  'Laser switched off at L₀': { illuminationStop: 1 },
  '1 TW beam': { power: 1e12 },
  '10 TW beam (classical model breaks)': { power: 1e13 },
  'Kulkarni et al. 2018, Table 1 (1 g)': { circularArray: false, wavelength: 1064e-9, arealDensity: 1.4e-3 },
};

// ============================================================================ controls
const CTL = [
  { g: 'sail', key: 'arealDensity', label: 'Sail areal density σ', unit: 'g/m²', min: 0.01, max: 100, log: true, k: 1e3,
    hint: 'mass per area = thickness × density (1 g/m² ≈ 1 µm at 1 g/cm³)' },
  { g: 'sail', key: 'reflectivity', label: 'Reflectivity R', unit: '', min: 0, max: 1, step: 0.001, k: 1,
    hint: 'fraction of the laser light reflected' },
  { g: 'sail', key: 'payloadMass', label: 'Payload mass m₀', unit: 'g', min: 0.01, max: 1000, log: true, k: 1e3 },
  { g: 'sail', type: 'check', key: 'optimalSail', label: 'Optimal sail size (sail mass = payload)' },
  { g: 'sail', key: 'sailSize', label: 'Sail size D', unit: 'm', min: 0.05, max: 100, log: true, k: 1,
    off: (p) => p.optimalSail, shown: () => D && D.D },
  { g: 'sail', type: 'seg', key: 'sailShape', label: 'Sail shape', options: [['square', 'Square'], ['circular', 'Circular']] },
  { g: 'beam', key: 'power', label: 'Laser power P', unit: 'GW', min: 0.01, max: 1e5, log: true, k: 1e-9 },
  { g: 'beam', key: 'arraySize', label: 'Array size d', unit: 'km', min: 0.1, max: 100, log: true, k: 1e-3 },
  { g: 'beam', key: 'wavelength', label: 'Wavelength λ', unit: 'nm', min: 300, max: 2000, step: 1, k: 1e9 },
  { g: 'beam', type: 'seg', key: 'circularArray', label: 'Array', options: [[true, 'Circular, α = 1.22'], [false, 'Square, α = 1']] },
  { g: 'mission', type: 'select', key: 'targetDistance', label: 'Target', options: Object.entries(ph.TARGETS).map(([n, v]) => [v, `${n} (${num(v / LY)} ly)`]) },
  { g: 'mission', type: 'check', key: '_illumInf', label: 'Laser stays on (illuminate to ∞)' },
  { g: 'mission', key: '_illumStop', label: 'Laser switched off at', unit: '× L₀', min: 0.1, max: 1000, log: true, k: 1,
    off: (p) => !Number.isFinite(p.illuminationStop) },
  { g: 'more', key: 'absorption', label: 'Absorbed fraction of non-reflected light A', unit: '', min: 0, max: 1, step: 0.01, k: 1,
    hint: '1 = opaque sail (UCSB default); 0 = the rest passes through' },
  { g: 'more', key: 'beamQuality', label: 'Beam quality factor (× α)', unit: '', min: 1, max: 3, step: 0.01, k: 1,
    hint: '1 = diffraction-limited beam' },
  { g: 'more', key: '_emissivity', label: 'Sail emissivity ε (each side)', unit: '', min: 0.01, max: 1, step: 0.01, k: 1,
    hint: 'only affects the reported sail temperature' },
];

function getVal(c) {
  if (c.key === '_illumInf') return !Number.isFinite(params.illuminationStop);
  if (c.key === '_illumStop') return Number.isFinite(params.illuminationStop) ? params.illuminationStop : ui.illumStop;
  if (c.key === '_emissivity') return params.emissivityFront;
  return params[c.key];
}
function setVal(c, v) {
  if (c.key === '_illumInf') params.illuminationStop = v ? Infinity : ui.illumStop;
  else if (c.key === '_illumStop') { ui.illumStop = v; if (Number.isFinite(params.illuminationStop)) params.illuminationStop = v; }
  else if (c.key === '_emissivity') { params.emissivityFront = v; params.emissivityBack = v; }
  else params[c.key] = v;
}

const pos2val = (c, pos) => (c.log ? c.min * (c.max / c.min) ** (pos / 1000) : c.min + ((c.max - c.min) * pos) / 1000);
const val2pos = (c, v) => (c.log ? (1000 * Math.log(v / c.min)) / Math.log(c.max / c.min) : (1000 * (v - c.min)) / (c.max - c.min));
const fmtInput = (v) => (Number.isFinite(v) ? String(Number(v.toPrecision(4))) : '');

function buildControls() {
  for (const c of CTL) {
    const host = document.getElementById(`ctl-${c.g}`);
    const wrap = document.createElement('div');
    wrap.className = 'ctl';
    const id = `c-${c.key}`;
    if (c.type === 'check') {
      wrap.innerHTML = `<label class="check"><input type="checkbox" id="${id}"> ${c.label}</label>`;
      wrap.querySelector('input').addEventListener('change', (e) => { setVal(c, e.target.checked); onParams(); });
    } else if (c.type === 'seg') {
      wrap.innerHTML = `<div class="ctl-row"><label>${c.label}</label><div class="seg" role="group" aria-label="${c.label}">${
        c.options.map(([v, l], i) => `<button type="button" data-i="${i}">${l}</button>`).join('')}</div></div>`;
      wrap.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
        setVal(c, c.options[+b.dataset.i][0]); onParams();
      }));
    } else if (c.type === 'select') {
      wrap.innerHTML = `<div class="ctl-row"><label for="${id}">${c.label}</label></div><select id="${id}">${
        c.options.map(([v, l], i) => `<option value="${i}">${l}</option>`).join('')}</select>`;
      wrap.querySelector('select').addEventListener('change', (e) => { setVal(c, c.options[+e.target.value][0]); onParams(); });
    } else {
      wrap.innerHTML = `<div class="ctl-row"><label for="${id}-n">${c.label}</label>
        <span class="ctl-val"><input type="number" id="${id}-n" step="any"><span class="unit">${c.unit}</span></span></div>
        <input type="range" id="${id}" min="0" max="1000" step="${c.step && !c.log ? Math.max(1, (1000 * c.step) / (c.max - c.min)) : 1}" aria-label="${c.label}">
        ${c.hint ? `<div class="hint">${c.hint}</div>` : ''}`;
      const range = wrap.querySelector('input[type=range]');
      const box = wrap.querySelector('input[type=number]');
      range.addEventListener('input', () => {
        let v = pos2val(c, +range.value);
        if (c.step) v = Math.round(v / c.step) * c.step;
        setVal(c, v / c.k);
        box.value = fmtInput(v);
        onParams(true);
      });
      box.addEventListener('change', () => {
        const v = parseFloat(box.value);
        if (!(v >= (c.log ? c.min / 100 : c.min) && v <= (c.log ? c.max * 100 : c.max))) { syncControls(); return; }
        setVal(c, v / c.k);
        onParams();
      });
    }
    c.el = wrap;
    host.append(wrap);
  }
}

function syncControls() {
  for (const c of CTL) {
    const v = getVal(c);
    const off = c.off ? c.off(params) : false;
    c.el.classList.toggle('disabled', off);
    if (c.type === 'check') c.el.querySelector('input').checked = !!v;
    else if (c.type === 'seg') c.el.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(c.options[+b.dataset.i][0] === v)));
    else if (c.type === 'select') {
      const i = c.options.findIndex(([ov]) => Math.abs(ov - v) < 1);
      c.el.querySelector('select').value = String(Math.max(0, i));
    } else {
      const shown = off && c.shown ? c.shown() : v;
      const disp = shown * c.k;
      const range = c.el.querySelector('input[type=range]'), box = c.el.querySelector('input[type=number]');
      range.value = String(Math.max(0, Math.min(1000, val2pos(c, disp))));
      if (document.activeElement !== box) box.value = fmtInput(disp);
      range.disabled = off; box.disabled = off;
    }
  }
}

// ============================================================================ physics update
let recomputeQueued = false;
function onParams(fromSlider = false) {
  document.getElementById('preset').value = '';
  if (fromSlider) {
    if (!recomputeQueued) { recomputeQueued = true; requestAnimationFrame(() => { recomputeQueued = false; recompute(); }); }
  } else recompute();
}

function recompute() {
  D = ph.derive(params);
  valid = D.a0 > 0 && Number.isFinite(D.a0) && D.L0 > 0;
  syncControls();
  if (!valid) { document.getElementById('phase').textContent = 'No momentum transfer: increase R or A'; return; }
  for (const m of MODELS) {
    traj[m] = ph.buildTrajectory(params, m);
    summ[m] = ph.missionSummary(params, m);
  }
  tEnd = Math.max(summ[CL].tTarget, summ[RE].tTarget);
  simT = Math.min(simT, tEnd);
  renderTiles();
  renderSimCharts();
  dirty = { explore: true, unc: true };
  if (activeTab === 'explore') renderExplore();
  if (activeTab === 'unc') scheduleMC();
  saveHash();
}

// ============================================================================ tiles
function tile(label, val, sub) {
  return `<div class="tile"><div class="lbl">${label}</div><div class="val">${val}</div><div class="sub">${sub}</div></div>`;
}
function renderTiles() {
  const r = summ[RE], c = summ[CL];
  const tgtName = Object.entries(ph.TARGETS).find(([, v]) => Math.abs(v - params.targetDistance) < 1)?.[0] ?? 'target';
  const finite = Number.isFinite(params.illuminationStop);
  const corr = 100 * (r.betaFinal / c.betaFinal - 1);
  const absorbedGW = (D.beamPower * (1 - params.reflectivity) * params.absorption) / 1e9;
  document.getElementById('tiles').innerHTML = [
    tile(finite ? 'Final speed (relativistic)' : 'Limiting speed v∞ (relativistic)', pctc(r.betaFinal),
      `<b>${num(r.vFinal / 1e3)} km/s</b> · classical ${pctc(c.betaFinal)} (${corr.toFixed(1)}% relativistic correction)`),
    tile(`Travel time to ${tgtName.replace(' A/B', '')}`, fmtTime(r.tTarget),
      `ship clock <b>${fmtTime(r.tauTarget)}</b> · classical ${fmtTime(c.tTarget)}`),
    tile('Peak acceleration a₀', `${num(D.a0 / G_N)} g`,
      `${num(D.a0)} m/s² · force ${num(D.force)} N · momentum factor η = ${num(D.eta)}`),
    tile('L₀ (beam spot = sail size)', fmtDist(D.L0),
      `reached after <b>${fmtTime(r.tL0)}</b> (classical ${fmtTime(c.tL0)}) · β₀ = ${num(D.beta0)}`),
    tile('Spacecraft mass', `${num(D.totalMass * 1e3)} g`,
      `sail ${num(D.sailMass * 1e3)} g (D = ${num(D.D)} m, ${num(D.area)} m²) + payload ${num(params.payloadMass * 1e3)} g`),
    tile('Peak sail temperature', absorbedGW > 0 ? `${num(D.Teq)} K` : 'n/a',
      absorbedGW > 0 ? `absorbs ${num(absorbedGW)} GW of the beam (${num(D.absorbedFlux / 1e9)} GW/m²)` : 'perfect reflector absorbs nothing'),
  ].join('');
}

// ============================================================================ simulation charts
let chartBeta, chartAcc;
function timeTip(v) { return `t = ${fmtTime(v)}`; }
function renderSimCharts() {
  const series = (key, scale = 1) => MODELS.map((m) => {
    const tr = traj[m];
    const x = [], y = [];
    for (let i = 1; i < tr.t.length; i++) { x.push(tr.t[i]); y.push(tr[key][i] * scale); }
    return { name: NAME[m], color: COLOR[m], x, y };
  });
  const bmax = Math.max(summ[CL].betaFinal, summ[RE].betaFinal);
  const tMin = Math.max(1e-3, Math.min(traj[RE].t[1] * 10, 0.1));
  const markers = MODELS.map((m) => ({ x: summ[m].tL0, y: summ[m].betaL0, color: COLOR[m] }));
  chartBeta.setData({
    series: series('beta').map((s) => ({ ...s, fmt: (v) => `${num(v)} (${num(v * C / 1e3)} km/s)` })),
    xDomain: [tMin, tEnd], yDomain: [0, Math.max(bmax * 1.12, 1e-9)], markers,
    hlines: bmax > 0.9 ? [{ y: 1, label: 'speed of light', color: '--ink' }] : [],
    vlines: [{ x: summ[RE].tL0, label: 'L₀ reached', color: '--beam' }],
  });
  const a0g = D.a0 / G_N;
  chartAcc.setData({
    series: series('accel', 1 / G_N).map((s) => ({ ...s, fmt: (v) => `${num(v)} g` })),
    xDomain: [tMin, tEnd], yDomain: [a0g * 1e-9, a0g * 3],
    vlines: [{ x: summ[RE].tL0, label: 'L₀', color: '--beam' }],
  });
}

// ============================================================================ animation: space track
const track = document.getElementById('track');
const stars = (() => {
  let s = 12345;
  const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  return Array.from({ length: 140 }, () => [r(), r(), r() * 0.9 + 0.3, r()]);
})();

function drawTrack(state) {
  const { ctx, w, h } = setupCanvas(track, track.clientHeight);
  const ink = cssVar('--space-ink'), ink2 = cssVar('--space-ink-2'), gridc = cssVar('--space-grid');
  const beam = cssVar('--space-beam');
  ctx.fillStyle = cssVar('--space');
  ctx.fillRect(0, 0, w, h);
  for (const [sx, sy, sr, tw] of stars) {
    ctx.globalAlpha = 0.25 + 0.35 * tw;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(sx * w, sy * h, sr, sr);
  }
  ctx.globalAlpha = 1;
  if (!valid) return;
  const narrow = w < 560;
  const left = narrow ? 44 : 64, right = narrow ? 34 : 50, top = 34, bottom = 34;
  const xMin = 1e3, xMax = params.targetDistance * 1.25;
  const lx = (x) => Math.log10(Math.max(x, xMin));
  const X = (x) => left + ((lx(x) - lx(xMin)) / (lx(xMax) - lx(xMin))) * (w - left - right);
  const laneH = (h - top - bottom) / 2;
  const laneY = { [CL]: top + laneH * 0.5, [RE]: top + laneH * 1.5 };
  ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';

  // distance grid
  ctx.textBaseline = 'top'; ctx.textAlign = 'center';
  let lastR = -1e9;
  for (const [v, lab] of DIST_TICKS) {
    if (v < xMin || v > xMax) continue;
    const x = Math.round(X(v)) + 0.5;
    ctx.strokeStyle = gridc; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, top - 4); ctx.lineTo(x, h - bottom + 4); ctx.stroke();
    const tw = ctx.measureText(lab).width;
    if (x - tw / 2 > lastR + 8) { ctx.fillStyle = ink2; ctx.fillText(lab, x, h - bottom + 9); lastR = x + tw / 2; }
  }
  // landmarks (top)
  ctx.textBaseline = 'bottom';
  let lastL = -1e9;
  for (const [name, v] of ph.LANDMARKS) {
    const x = X(v);
    ctx.fillStyle = ink2;
    ctx.fillRect(Math.round(x), top - 9, 1, 5);
    const tw = ctx.measureText(name).width;
    if (!narrow && x - tw / 2 > lastL + 6 && x + tw / 2 < w - right - 30) { ctx.fillText(name, x, top - 10); lastL = x + tw / 2; }
  }
  // target star
  const xt = X(params.targetDistance);
  ctx.fillStyle = '#ffd98a';
  starGlyph(ctx, xt, top - 16, 6);
  ctx.fillStyle = ink; ctx.textAlign = 'right';
  ctx.fillText(narrow ? 'target' : 'target star', xt - 9, top - 10);
  ctx.strokeStyle = 'rgba(255,217,138,0.35)';
  ctx.beginPath(); ctx.moveTo(Math.round(xt) + 0.5, top - 4); ctx.lineTo(Math.round(xt) + 0.5, h - bottom + 4); ctx.stroke();
  // L0 and laser-off markers
  const marks = [[D.L0, 'L₀']];
  if (Number.isFinite(D.xStop)) marks.push([D.xStop, 'laser off']);
  for (const [v, lab] of marks) {
    if (v < xMin) continue;
    const x = Math.round(X(v)) + 0.5;
    ctx.strokeStyle = withAlpha(beam, 0.8); ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(x, top - 2); ctx.lineTo(x, h - bottom + 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = beam; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(lab, x + 4, lab === 'L₀' ? top : top + 13);
  }
  // Earth + array
  ctx.fillStyle = '#3d7bd9';
  ctx.beginPath(); ctx.arc(left - 26, h / 2, 9, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = '#5bb07a';
  ctx.beginPath(); ctx.arc(left - 28, h / 2 - 2, 4, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText('Earth', left - 26, h / 2 + 12);

  for (const m of MODELS) {
    const st = state[m];
    const y = laneY[m];
    const col = cssVar(SPACE[m]);
    const xs = X(st.x);
    // lane label
    ctx.fillStyle = ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillStyle = col; ctx.fillRect(left, y - laneH / 2 + 8, 8, 8);
    ctx.fillStyle = ink; ctx.fillText(NAME[m], left + 13, y - laneH / 2 + 12);
    // array
    ctx.fillStyle = '#c9c7bf';
    ctx.fillRect(left - 6, y - 9, 3, 18);
    // trail
    ctx.strokeStyle = withAlpha(col, 0.35); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(xs, y); ctx.stroke();
    // beam cone: its height grows like the diffraction spot, D * x / L0
    const sailH = Math.min(24, laneH * 0.42);
    if (st.flux > 0 && !st.arrived) {
      const spot = (sailH / 2) * (st.x / D.L0);
      const hh = Math.max(1.2, Math.min(spot, laneH * 0.4));
      const g = ctx.createLinearGradient(left, 0, xs, 0);
      const a = 0.08 + 0.45 * st.flux;
      g.addColorStop(0, withAlpha(beam, 0.75)); g.addColorStop(1, withAlpha(beam, a));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.moveTo(left - 3, y - 3); ctx.lineTo(xs, y - hh); ctx.lineTo(xs, y + hh); ctx.lineTo(left - 3, y + 3); ctx.closePath();
      ctx.fill();
    }
    // sail + payload
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.roundRect ? ctx.roundRect(xs - 1, y - sailH / 2, 4, sailH, 2) : ctx.rect(xs - 1, y - sailH / 2, 4, sailH); ctx.fill();
    ctx.fillStyle = '#e9e8e4';
    ctx.beginPath(); ctx.arc(xs + 6, y, 2.6, 0, 2 * Math.PI); ctx.fill();
    // speed tag
    ctx.fillStyle = ink; ctx.textAlign = xs > w - right - 90 ? 'right' : 'left'; ctx.textBaseline = 'bottom';
    const tagX = ctx.textAlign === 'right' ? xs - 6 : xs + 10;
    ctx.fillText(st.arrived ? `arrived · ${fmtTime(summ[m].tTarget)}` : `${pctc(st.beta)}`, tagX, y - sailH / 2 - 2);
  }
}

function starGlyph(ctx, x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = (Math.PI / 5) * i - Math.PI / 2, rr = i % 2 ? r * 0.45 : r;
    ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
  }
  ctx.closePath(); ctx.fill();
}

// beam-spot inset: diffraction-limited spot vs sail at the relativistic sail's distance
const spotCanvas = document.getElementById('spot');
function drawSpot(st) {
  const { ctx, w, h } = setupCanvas(spotCanvas, spotCanvas.clientHeight);
  ctx.fillStyle = cssVar('--surface'); ctx.fillRect(0, 0, w, h);
  if (!valid) return;
  const ink2 = cssVar('--ink-2'), beam = cssVar('--beam'), col = cssVar('--relativistic');
  const cx = w / 2, cy = h / 2, S = Math.min(70, h * 0.42);
  const ratio = st.x / D.L0;                      // spot diameter / sail size
  const on = st.flux > 0 && !st.arrived;
  if (on) {
    const r = (S / 2) * ratio;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(r, 1));
    g.addColorStop(0, withAlpha(beam, 0.55)); g.addColorStop(1, withAlpha(beam, 0.12));
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx, cy, Math.min(r, Math.hypot(w, h)), 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = beam; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cx, cy, Math.min(r, Math.hypot(w, h)), 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.strokeStyle = col; ctx.lineWidth = 2.5;
  if (params.sailShape === 'circular') { ctx.beginPath(); ctx.arc(cx, cy, S / 2, 0, 2 * Math.PI); ctx.stroke(); }
  else ctx.strokeRect(cx - S / 2, cy - S / 2, S, S);
  ctx.fillStyle = ink2; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText(`sail ${num(D.D)} m`, 8, 8);
  const spotM = D.D * ratio;
  const txt = document.getElementById('spot-text');
  if (!on) txt.textContent = st.arrived ? 'Arrived at the target.' : 'Laser switched off: the sail coasts.';
  else if (ratio <= 1) txt.textContent = `Spot ${num(spotM)} m is smaller than the sail: the sail captures 100 % of the beam.`;
  else txt.textContent = `Spot ${fmtDist(spotM)} is ${num(ratio)}× the sail size: only ${num(100 * st.flux)} % of the beam hits the sail.`;
}

// ============================================================================ live readout table
const readoutRows = [
  ['Distance from Earth', (s) => fmtDist(s.x)],
  ['Distance / L₀', (s) => num(s.x / D.L0)],
  ['Speed', (s) => `${num(s.beta * C / 1e3)} km/s`],
  ['Speed / c', (s) => pctc(s.beta)],
  ['Lorentz factor γ', (s, m) => (m === CL ? '—' : num(s.gamma, 6))],
  ['Acceleration felt', (s) => `${num(s.accel / G_N)} g`],
  ['Beam power on sail', (s) => `${num(100 * s.flux)} %`],
  ['Laser wavelength seen by sail', (s, m) => `${num(params.wavelength * 1e9 * (m === CL ? 1 + s.beta : s.doppler), 4)} nm`],
  ['Ship clock', (s, m) => (m === CL ? 'same as Earth' : fmtTime(s.tau))],
];
function renderReadout(state) {
  const tb = document.querySelector('#readout tbody');
  if (tb.children.length !== readoutRows.length) tb.innerHTML = readoutRows.map(([l]) => `<tr><td>${l}</td><td></td><td></td></tr>`).join('');
  readoutRows.forEach(([, f], i) => {
    const tds = tb.children[i].children;
    tds[1].textContent = f(state[CL], CL);
    tds[2].textContent = f(state[RE], RE);
  });
}

function phaseOf(st) {
  if (st.arrived) return ['Arrived at the target', '#ffd98a'];
  if (st.flux === 0) return ['Laser off: coasting at constant speed', '#a9a8a1'];
  if (st.x <= D.L0) return ['Near field: the sail captures the whole beam', cssVar('--space-beam')];
  return ['Far field: the spot outgrows the sail, power ∝ (L₀/x)²', cssVar('--space-beam')];
}

// ============================================================================ main loop
const playBtn = document.getElementById('play');
const scrub = document.getElementById('scrub');
let lastFrame = null, lastDom = 0;
function setPlaying(p) {
  playing = p;
  playBtn.textContent = p ? '❚❚ Pause' : (simT >= tEnd ? '↺ Replay' : '▶ Play');
  playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
}
function warp() {
  if (speed !== 'auto') return Number(speed);
  const rate = Math.log10(tEnd / T_START) / AUTO_SECONDS;
  return Math.LN10 * rate * Math.max(simT, T_START);
}
function frame(now) {
  const dt = lastFrame == null ? 0 : Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  if (playing && valid) {
    if (speed === 'auto') {
      const rate = Math.log10(tEnd / T_START) / AUTO_SECONDS;
      simT = Math.max(simT, T_START) * 10 ** (rate * dt);
    } else simT += dt * Number(speed);
    if (simT >= tEnd) { simT = tEnd; setPlaying(false); }
  }
  if (valid && activeTab === 'sim') {
    const state = { [CL]: ph.stateAt(traj[CL], simT), [RE]: ph.stateAt(traj[RE], simT) };
    drawTrack(state);
    chartBeta.setCursor(simT);
    chartAcc.setCursor(simT);
    if (now - lastDom > 90) {
      lastDom = now;
      drawSpot(state[RE]);
      renderReadout(state);
      const [ptxt, pcol] = phaseOf(state[RE]);
      document.getElementById('phase').textContent = ptxt;
      document.getElementById('phase-dot').style.background = pcol;
      document.getElementById('clock-earth').textContent = fmtTime(simT);
      document.getElementById('clock-ship').textContent = fmtTime(state[RE].tau);
      const wv = warp();
      document.getElementById('clock-warp').textContent = playing ? `×${num(wv, 2)}` : 'paused';
      if (document.activeElement !== scrub) scrub.value = String(Math.round((1000 * Math.log(Math.max(simT, T_START) / T_START)) / Math.log(tEnd / T_START)));
    }
  }
  requestAnimationFrame(frame);
}

// ============================================================================ explore tab
let chSigma, chRefl, chPower, chDist;
function renderExplore() {
  dirty.explore = false;
  const xs = params.illuminationStop;
  // areal density sweep
  const sig = Array.from({ length: 121 }, (_, i) => 1e-5 * 10 ** (i / 30));   // 0.01 - 100 g/m^2 in kg/m^2
  const sweep = (vals, mk, out) => MODELS.map((m) => ({
    name: NAME[m], color: COLOR[m], x: vals.map((v) => v.x), y: vals.map((v) => out(mk(v.v), m)),
  }));
  const vinf = (q, m) => { const d = ph.derive(q); return 100 * ph.betaTerminal(d.a0, d.L0, d.xStop, m); };
  chSigma.setData({
    series: sweep(sig.map((s) => ({ x: s * 1e3, v: s })), (s) => ({ ...params, arealDensity: s }), vinf).map((s) => ({ ...s, fmt: (v) => `${num(v)}% c` })),
    xDomain: [0.01, 100], yDomain: [0, Math.min(100, Math.max(...MODELS.map((m) => vinf({ ...params, arealDensity: 1e-5 }, m))) * 1.1)],
    markers: MODELS.map((m) => ({ x: params.arealDensity * 1e3, y: vinf(params, m), color: COLOR[m] })),
  });
  document.getElementById('exp-sigma-note').textContent = params.optimalSail
    ? 'Optimal sail on: the sail size changes with σ so that the sail mass equals the payload.'
    : `Sail size fixed at D = ${num(params.sailSize)} m.`;
  // reflectivity sweep -> travel time
  const Rs = Array.from({ length: 101 }, (_, i) => i / 100);
  const ttime = (q, m) => { const d = ph.derive(q); return ph.timeToDistance(q.targetDistance, d.a0, d.L0, d.xStop, m)[0] / YEAR; };
  const reflSeries = sweep(Rs.map((r) => ({ x: r, v: r })), (r) => ({ ...params, reflectivity: r }), ttime);
  const tmax = Math.max(...reflSeries.flatMap((s) => s.y.filter(Number.isFinite)));
  const tmin = Math.min(...reflSeries.flatMap((s) => s.y.filter(Number.isFinite)));
  chRefl.setData({
    series: reflSeries.map((s) => ({ ...s, fmt: (v) => fmtTime(v * YEAR) })),
    xDomain: [0, 1], yDomain: [Math.max(0, tmin * 0.85), tmax * 1.05],
    markers: MODELS.map((m) => ({ x: params.reflectivity, y: ttime(params, m), color: COLOR[m] })),
  });
  // power sweep
  const Ps = Array.from({ length: 141 }, (_, i) => 1e7 * 10 ** (i / 20));
  const bf = (q, m) => { const d = ph.derive(q); return ph.betaTerminal(d.a0, d.L0, d.xStop, m); };
  const pSeries = sweep(Ps.map((p) => ({ x: p / 1e9, v: p })), (p) => ({ ...params, power: p }), bf);
  chPower.setData({
    series: pSeries.map((s) => ({ ...s, fmt: (v) => `β = ${num(v)}` })),
    xDomain: [0.01, 1e5], yDomain: [0, 2.2], hlines: [{ y: 1, label: 'speed of light', color: '--ink' }],
    markers: MODELS.map((m) => ({ x: params.power / 1e9, y: bf(params, m), color: COLOR[m] })),
  });
  // speed vs distance
  chDist.setData({
    series: MODELS.map((m) => ({ name: NAME[m], color: COLOR[m], x: Array.from(traj[m].x).slice(1), y: Array.from(traj[m].beta).slice(1), fmt: (v) => pctc(v) })),
    xDomain: [D.L0 * 1e-4, params.targetDistance], yDomain: [0, Math.max(summ[CL].betaFinal, summ[RE].betaFinal) * 1.12],
    vlines: [{ x: D.L0, label: 'L₀', color: '--beam' }, ...(Number.isFinite(xs) ? [{ x: D.xStop, label: 'laser off', color: '--beam' }] : [])],
  });
}

// ============================================================================ uncertainty tab
const SPEC_UI = [
  ['arealDensity', 'normal', 'σ, ±', 10, '% (1σ)'],
  ['payloadMass', 'normal', '±', 10, '% (1σ)'],
  ['sailSize', 'normal', '±', 2, '% (1σ)'],
  ['reflectivity', 'uniform', '', [0.95, 1.0], ''],
  ['power', 'normal', '±', 5, '% (1σ)'],
  ['beamQuality', 'uniform', '', [1.0, 1.5], ''],
];
let chMC, tornadoChart, mcTimer = null;
function buildSpec() {
  const t = document.getElementById('spec');
  t.innerHTML = SPEC_UI.map(([k, kind, , v, unit]) => `<tr><td>${ph.UNCERTAIN[k].label}${ph.UNCERTAIN[k].group === 'beam' ? ' <span class="muted small">(beam)</span>' : ''}</td>
    <td class="dist">${kind === 'normal' ? 'normal' : 'uniform'}</td>
    <td>${kind === 'normal'
      ? `± <input type="number" step="any" min="0" data-k="${k}" data-f="s" value="${v}"> ${unit}`
      : `<input type="number" step="any" data-k="${k}" data-f="a" value="${v[0]}"> – <input type="number" step="any" data-k="${k}" data-f="b" value="${v[1]}">`}</td></tr>`).join('');
  t.querySelectorAll('input').forEach((i) => i.addEventListener('change', scheduleMC));
}
function readSpec() {
  const spec = {};
  for (const [k, kind] of SPEC_UI) {
    const get = (f) => parseFloat(document.querySelector(`#spec input[data-k="${k}"][data-f="${f}"]`).value);
    if (kind === 'normal') spec[k] = { kind: 'normal_rel', s: Math.max(0, get('s')) / 100 };
    else { const a = get('a'), b = get('b'); spec[k] = { kind: 'uniform', a: Math.min(a, b), b: Math.max(a, b) }; }
  }
  if (spec.reflectivity) { spec.reflectivity.a = Math.max(0, spec.reflectivity.a); spec.reflectivity.b = Math.min(1, spec.reflectivity.b); }
  if (spec.beamQuality) spec.beamQuality.a = Math.max(0.1, spec.beamQuality.a);
  return spec;
}
function scheduleMC() { clearTimeout(mcTimer); mcTimer = setTimeout(runMC, 120); }
function runMC() {
  if (!valid) return;
  dirty.unc = false;
  const spec = readSpec();
  const n = +document.getElementById('mc-n').value, seed = +document.getElementById('mc-seed').value || 1;
  const t0 = performance.now();
  const mc = ph.monteCarlo(params, spec, n, seed);
  const lin = Object.fromEntries(MODELS.map((m) => [m, ph.linearPropagation(params, spec, 'tTarget', m)]));
  const dtms = performance.now() - t0;
  // histogram
  let lo = Infinity, hi = -Infinity;
  for (const m of MODELS) for (const v of mc[m].tTarget) { if (v < lo) lo = v; if (v > hi) hi = v; }
  lo /= YEAR; hi /= YEAR;
  const pad = (hi - lo) * 0.04 || 1, nb = 70;
  const edges = Array.from({ length: nb + 1 }, (_, i) => lo - pad + ((hi - lo + 2 * pad) * i) / nb);
  const width = edges[1] - edges[0];
  let ymax = 0;
  const series = MODELS.map((m) => {
    const cnt = new Array(nb).fill(0);
    for (const v of mc[m].tTarget) cnt[Math.min(nb - 1, Math.max(0, Math.floor((v / YEAR - edges[0]) / width)))]++;
    const y = cnt.map((c) => c / (n * width));
    ymax = Math.max(ymax, ...y);
    return { name: NAME[m], color: COLOR[m], kind: 'step', x: edges, y, fmt: (v) => `${num(v)} /yr` };
  });
  const stats = Object.fromEntries(MODELS.map((m) => {
    const [p5, p50, p95] = ph.quantiles(mc[m].tTarget, [0.05, 0.5, 0.95]).map((v) => v / YEAR);
    const [mean, sd] = ph.meanStd(mc[m].tTarget).map((v) => v / YEAR);
    const [b50] = ph.quantiles(mc[m].betaFinal, [0.5]);
    return [m, { p5, p50, p95, mean, sd, b50 }];
  }));
  chMC.opts.xFmt = (v) => `travel time ${num(v)} yr`;
  chMC.setData({
    series, xDomain: [edges[0], edges[nb]], yDomain: [0, ymax * 1.15],
    markers: MODELS.map((m) => ({ x: stats[m].p50, y: ymax * (m === CL ? 1.04 : 1.1), color: COLOR[m] })),
  });
  const yr = (v) => `${v.toFixed(1)}`;
  const row = (label, f) => `<tr><td>${label}</td>${MODELS.map((m) => `<td>${f(m)}</td>`).join('')}</tr>`;
  document.getElementById('mc-table').innerHTML = `<thead><tr><th></th><th><span class="key classical"></span>Classical</th><th><span class="key relativistic"></span>Relativistic</th></tr></thead><tbody>${
    row('Median travel time', (m) => `${yr(stats[m].p50)} yr`)
  }${row('90 % interval', (m) => `${yr(stats[m].p5)} – ${yr(stats[m].p95)} yr`)
  }${row('Monte Carlo mean ± σ', (m) => `${yr(stats[m].mean)} ± ${stats[m].sd.toFixed(2)} yr`)
  }${row('Linear propagation mean ± σ', (m) => `${yr(lin[m].y / YEAR)} ± ${(lin[m].std / YEAR).toFixed(2)} yr`)
  }${row('Median limiting speed', (m) => pctc(stats[m].b50))}</tbody>`;
  // tornado (relativistic travel time)
  const fmtIn = { arealDensity: (v) => `${num(v * 1e3)} g/m²`, payloadMass: (v) => `${num(v * 1e3)} g`, sailSize: (v) => `${num(v)} m`,
    reflectivity: (v) => `R ${num(v, 3)}`, power: (v) => `${num(v / 1e9)} GW`, beamQuality: (v) => `×${num(v)}` };
  const rows = ph.tornado(params, spec, 'tTarget', RE).map((r) => ({
    ...r, yLo: r.yLo / YEAR, yHi: r.yHi / YEAR, y0: r.y0 / YEAR, loText: fmtIn[r.key](r.lo), hiText: fmtIn[r.key](r.hi),
  }));
  tornadoChart.setRows(rows);
  // variance shares
  const el = (v) => (Math.abs(v) < 1e-4 ? '≈ 0' : v.toFixed(3));
  document.getElementById('var-table').innerHTML = `<thead><tr><th>Input</th><th>S (classical)</th><th>S (relativistic)</th><th>Variance share</th></tr></thead><tbody>${
    Object.keys(spec).map((k) => `<tr><td>${ph.UNCERTAIN[k].label}</td><td>${el(lin[CL].elasticities[k])}</td><td>${el(lin[RE].elasticities[k])}</td><td>${lin[RE].share[k] < 1e-4 ? '≈ 0' : `${(100 * lin[RE].share[k]).toFixed(1)} %`}</td></tr>`).join('')}</tbody>`;
  document.getElementById('mc-time').textContent = `${n.toLocaleString('en-US')} samples × 2 models in ${dtms.toFixed(0)} ms (closed-form speeds, quadrature for travel time).`;
}

// ============================================================================ tabs, theme, sharing
function showTab(name) {
  activeTab = name;
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  document.querySelectorAll('.tab-panel').forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (name === 'explore' && dirty.explore) requestAnimationFrame(renderExplore);
  if (name === 'unc' && dirty.unc) scheduleMC();
  if (name === 'live') live.activate(); else live.deactivate();
}

function redrawAll() {
  for (const ch of [chartBeta, chartAcc, chSigma, chRefl, chPower, chDist, chMC]) ch.draw();
  tornadoChart.draw();
  live.redraw();
}

function saveHash() {
  const q = new URLSearchParams();
  for (const [k, def] of Object.entries(ph.DEFAULTS)) {
    const v = params[k];
    if (v !== def) q.set(k, v === Infinity ? 'inf' : String(v));
  }
  const s = q.toString();
  try { history.replaceState(null, '', s ? `#${s}` : location.pathname + location.search); } catch { /* sandboxed */ }
}
function loadHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const [k, def] of Object.entries(ph.DEFAULTS)) {
    if (!q.has(k)) continue;
    const raw = q.get(k);
    if (typeof def === 'boolean') params[k] = raw === 'true';
    else if (typeof def === 'string') params[k] = raw;
    else { const v = raw === 'inf' ? Infinity : parseFloat(raw); if (!Number.isNaN(v)) params[k] = v; }
  }
}

function downloadCSV() {
  const lines = ['model,t_s,x_m,beta,gamma,tau_s,accel_m_s2'];
  for (const m of MODELS) {
    const tr = traj[m];
    for (let i = 0; i < tr.t.length; i++) lines.push([m, tr.t[i], tr.x[i], tr.beta[i], tr.gamma[i], tr.tau[i], tr.accel[i]].map((v) => (typeof v === 'number' ? v.toPrecision(10) : v)).join(','));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'lightsail_trajectory.csv';
  document.body.append(a); a.click(); a.remove();
}

function legendHTML() {
  return MODELS.map((m) => `<span><i class="key ${m}"></i>${NAME[m]}</span>`).join('');
}

// ============================================================================ init
function init() {
  try {
    const th = localStorage.getItem('lightsail-theme');
    if (th) document.documentElement.dataset.theme = th;
  } catch { /* storage unavailable */ }
  document.querySelectorAll('[data-legend]').forEach((el) => { el.innerHTML = legendHTML(); });
  buildControls();
  const pre = document.getElementById('preset');
  pre.innerHTML = `<option value="" hidden>Custom</option>${Object.keys(PRESETS).map((n) => `<option>${n}</option>`).join('')}`;
  pre.addEventListener('change', () => {
    const v = pre.value;
    params = { ...ph.DEFAULTS, ...PRESETS[v] };
    if (Number.isFinite(params.illuminationStop)) ui.illumStop = params.illuminationStop;
    recompute();
    pre.value = v;
  });
  document.getElementById('reset').addEventListener('click', () => { params = { ...ph.DEFAULTS }; recompute(); pre.value = Object.keys(PRESETS)[0]; });

  const timeAxis = { xLog: true, xTicks: TIME_TICKS, xFmt: timeTip, xLabel: 'time since launch (Earth frame, log scale)' };
  chartBeta = new Plot(document.getElementById('chart-beta'), { ...timeAxis, yLabel: 'speed β = v/c', yFmt: (v) => num(v), ariaLabel: 'Speed versus time for both models' });
  chartAcc = new Plot(document.getElementById('chart-acc'), { ...timeAxis, yLog: true, yLabel: 'acceleration [g]', ariaLabel: 'Acceleration felt by the sail versus time' });
  chSigma = new Plot(document.getElementById('chart-sigma'), { xLog: true, xLabel: 'sail areal density σ [g/m²]', yLabel: 'limiting speed [% c]', xFmt: (v) => `σ = ${num(v)} g/m²`, yTickFmt: (v) => `${num(v)}%` });
  chRefl = new Plot(document.getElementById('chart-refl'), { xLabel: 'reflectivity R', yLabel: 'travel time [yr]', xFmt: (v) => `R = ${v.toFixed(2)}` });
  chPower = new Plot(document.getElementById('chart-power'), { xLog: true, xLabel: 'laser power P [GW]', yLabel: 'limiting speed β∞', xFmt: (v) => `P = ${num(v)} GW`, xTickFmt: (v) => num(v) });
  chDist = new Plot(document.getElementById('chart-dist'), { xLog: true, xTicks: DIST_TICKS.map(([v, l]) => [v, l]), xLabel: 'distance from Earth (log scale)', yLabel: 'speed β', xFmt: (v) => fmtDist(v), yFmt: (v) => num(v) });
  chMC = new Plot(document.getElementById('chart-mc'), { xLabel: 'travel time to target [yr] (Earth frame)', yLabel: 'probability density [1/yr]', height: 240 });
  tornadoChart = new Tornado(document.getElementById('chart-tornado'), { xLabel: 'relativistic travel time [yr]', xFmt: (v) => `${v.toFixed(1)} yr` });
  buildSpec();
  live = initLive({ getSidebarParams: () => ({ ...params }) });
  document.getElementById('mc-go').addEventListener('click', runMC);
  document.getElementById('mc-n').addEventListener('change', scheduleMC);
  document.getElementById('mc-seed').addEventListener('change', scheduleMC);

  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
  document.getElementById('theme').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('lightsail-theme', document.documentElement.dataset.theme); } catch { /* ignore */ }
    redrawAll();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', redrawAll);

  playBtn.addEventListener('click', () => {
    if (!playing && simT >= tEnd) simT = 0;
    setPlaying(!playing);
  });
  document.getElementById('restart').addEventListener('click', () => { simT = 0; setPlaying(true); });
  document.getElementById('speed').addEventListener('change', (e) => { speed = e.target.value; });
  scrub.addEventListener('input', () => {
    simT = T_START * (tEnd / T_START) ** (+scrub.value / 1000);
    if (+scrub.value === 0) simT = 0;
    if (simT < tEnd && !playing) setPlaying(false);
  });
  document.getElementById('share').addEventListener('click', async (e) => {
    saveHash();
    try { await navigator.clipboard.writeText(location.href); e.target.textContent = 'Link copied'; }
    catch { e.target.textContent = 'Copy the address bar URL'; }
    setTimeout(() => { e.target.textContent = 'Copy link to this setup'; }, 1800);
  });
  document.getElementById('csv').addEventListener('click', downloadCSV);

  loadHash();
  if (Number.isFinite(params.illuminationStop)) ui.illumStop = params.illuminationStop;
  pre.value = location.hash.length > 1 ? '' : Object.keys(PRESETS)[0];
  recompute();
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) setPlaying(false); else setPlaying(true);
  requestAnimationFrame(frame);
  window.__live = live.state;
  window.__lightsail = { get params() { return params; }, get simT() { return simT; }, get summ() { return summ; }, set: (p) => { Object.assign(params, p); recompute(); } };
}

init();
