// Live data tab: a stream of tracking measurements is assimilated in real time.
// The built-in stream is SIMULATED (no laser-driven lightsail has flown); real measurements can be
// loaded from CSV or a Server-Sent-Events stream in the same format.
import * as ph from './physics.js';
import * as asm from './assimilation.js';
import { Plot, cssVar, setupCanvas, withAlpha } from './charts.js';
import { num, fmtDist, fmtTime, pctc, TIME_TICKS, DIST_TICKS } from './format.js';

const { C, G_N, YEAR, RELATIVISTIC: RE } = ph;
const SPEC = ph.DEFAULT_SPEC;
const KULKARNI = { ...ph.DEFAULTS, circularArray: false, wavelength: 1064e-9, arealDensity: 1.4e-3 };
const DESIGNS = {
  'Kulkarni et al. 2018, Table 1 (1 g)': () => KULKARNI,
  'UCSB calculator defaults': () => ({ ...ph.DEFAULTS }),
  'Current sidebar setup': null,
};
const AUTO_SECONDS = 45;
const $ = (id) => document.getElementById(id);

export function initLive({ getSidebarParams }) {
  const S = {
    active: false, source: 'sim', designName: Object.keys(DESIGNS)[0], seed: 1, noise: 0.002,
    reveal: false, playing: false, speed: 'auto', simT: 0, tStart: 0.5, tEnd: 1e9,
    records: [], next: 0, received: [], A: null, truth: null, truthTraj: null,
    history: [], summary: null, band: null, priorBand: null, est: null, pings: [],
    dirty: true, lastSummaryAt: 0, sse: null, sseQueue: [], meta: null, status: '', tMaxPlot: 1e9, zoom: false,
  };
  let chBeta, chFunnel, lastFrame = null, lastDom = 0;

  const design = () => {
    const f = DESIGNS[S.designName];
    return { ...(f ? f() : getSidebarParams()), illuminationStop: Infinity };
  };
  const displayTimes = () => Array.from({ length: 140 }, (_, i) => Math.pow(10, -0.3 + (i / 139) * (Math.log10(S.tMaxPlot) + 0.3)));

  // -------------------------------------------------------------------- setup / reset
  function resetEstimator() {
    const d = design();
    S.A = new asm.Assimilator(d, SPEC);
    S.received = [];
    S.history = [];
    S.summary = S.A.prior;
    S.tMaxPlot = Math.max(S.A.prior.t_target.p95 * 1.4, 1e9);
    S.priorBand = S.A.predictiveBand(displayTimes());
    S.band = S.priorBand;
    S.est = estimateTrajectories(S.A.prior);
    S.pings = [];
    S.dirty = true;
  }

  function newSimulatedStream() {
    S.truth = asm.sampleTruth(design(), SPEC, S.seed);
    S.truthTraj = ph.buildTrajectory(S.truth, RE, 1500);
    S.records = asm.simulatedStream(S.truth, { noise: S.noise, seed: S.seed });
    S.meta = { simulated: true, n: S.records.length };
  }

  function restart() {
    S.next = 0;
    S.simT = 0;
    resetEstimator();
    if (S.records.length) {
      S.tStart = Math.max(0.2, S.records[0].received_at * 0.5);
      S.tEnd = S.records[S.records.length - 1].received_at * 1.02;
    }
    S.playing = S.source !== 'sse';
    updatePlayButton();
    renderStatic();
  }

  function rebuild() {
    if (S.source === 'sim') newSimulatedStream();
    restart();
  }

  // -------------------------------------------------------------------- estimates
  function estimateTrajectories(sm) {
    const X = design().targetDistance;
    const mk = (a0, L0) => ph.buildTrajectoryFromDerived({ a0, L0, xStop: Infinity }, X, RE, 500);
    return { mid: mk(sm.a0.p50, sm.L0.p50), lo: mk(sm.a0.p5, sm.L0.p5), hi: mk(sm.a0.p95, sm.L0.p95) };
  }

  function ingest(batch, now) {
    if (!batch.length) return;
    S.A.update(batch);
    for (const r of batch) { S.received.push(r); S.pings.push({ at: now, r }); }
    S.dirty = true;
  }

  function refreshSummary(now) {
    if (!S.dirty || now - S.lastSummaryAt < 180) return;
    S.lastSummaryAt = now;
    S.dirty = false;
    S.summary = S.A.summary();
    S.band = S.A.predictiveBand(displayTimes());
    S.est = estimateTrajectories(S.summary);
    const last = S.received[S.received.length - 1];
    if (last) S.history.push({ t: last.received_at, ...S.summary.t_target });
    renderStatic();
  }

  // -------------------------------------------------------------------- rendering: charts, tiles, table
  function renderStatic() {
    if (!chBeta) return;
    const T = displayTimes();
    const rec = S.received;
    const series = [
      { name: 'Prior 90 %', color: '--muted', kind: 'band', x: T, lo: S.priorBand.map((b) => b[0]), hi: S.priorBand.map((b) => b[2]), alpha: 0.14, fmt: (v) => num(v) },
      { name: 'Fit 90 %', color: '--relativistic', kind: 'band', x: T, lo: S.band.map((b) => b[0]), hi: S.band.map((b) => b[2]), alpha: 0.22, fmt: (v) => num(v) },
      { name: 'Fit median', color: '--relativistic', x: T, y: S.band.map((b) => b[1]), fmt: (v) => num(v) },
      { name: 'Measured', color: '--measured', kind: 'points', x: rec.map((r) => r.t), y: rec.map((r) => r.beta), err: rec.map((r) => r.sigma), fmt: (v) => num(v, 4) },
    ];
    if (S.reveal && S.truthTraj) {
      const tr = S.truthTraj;
      series.push({ name: 'True sail', color: '--ink', width: 1.25, x: Array.from(tr.t).slice(1), y: Array.from(tr.beta).slice(1), fmt: (v) => num(v, 4) });
    }
    const ymax = Math.max(0.05, ...S.priorBand.map((b) => b[2]), ...rec.map((r) => r.beta + r.sigma)) * 1.1;
    chBeta.setData({ series, xDomain: [0.5, S.tMaxPlot], yDomain: [0, ymax], vlines: [] });

    const pr = S.A.prior.t_target;
    const hx = S.history.map((h) => h.t);
    const tmin = Math.max(0.3, S.records[0]?.received_at * 0.5 || 0.5), tmaxF = Math.max(S.tEnd, 10);
    const fseries = [
      { name: 'Prior 90 %', color: '--muted', kind: 'band', x: [tmin, tmaxF], lo: [pr.p5 / YEAR, pr.p5 / YEAR], hi: [pr.p95 / YEAR, pr.p95 / YEAR], alpha: 0.14, fmt: (v) => `${v.toFixed(2)} yr` },
    ];
    if (hx.length) {
      fseries.push({ name: 'Prediction 90 %', color: '--relativistic', kind: 'band', x: hx, lo: S.history.map((h) => h.p5 / YEAR), hi: S.history.map((h) => h.p95 / YEAR), alpha: 0.22, fmt: (v) => `${v.toFixed(2)} yr` });
      fseries.push({ name: 'Prediction median', color: '--relativistic', x: hx, y: S.history.map((h) => h.p50 / YEAR), fmt: (v) => `${v.toFixed(2)} yr` });
    }
    const hl = [];
    if (S.reveal && S.truth) {
      const d = ph.derive(S.truth);
      hl.push({ y: ph.timeToDistance(S.truth.targetDistance, d.a0, d.L0, Infinity, RE)[0] / YEAR, label: 'true travel time', color: '--ink', align: 'right' });
    }
    const lo = Math.min(pr.p5 / YEAR, ...S.history.map((h) => h.p5 / YEAR)), hi = Math.max(pr.p95 / YEAR, ...S.history.map((h) => h.p95 / YEAR));
    chFunnel.setData({ series: fseries, xDomain: [tmin, tmaxF], yDomain: [lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.15], hlines: hl });
    renderTiles();
    renderTable();
    drawHeat();
  }

  function renderTiles() {
    const sm = S.summary, pr = S.A.prior;
    const rel = (q) => q.sd / q.mean;
    const truth = S.reveal && S.truth ? ph.derive(S.truth) : null;
    const tT = truth ? ph.timeToDistance(S.truth.targetDistance, truth.a0, truth.L0, Infinity, RE)[0] : null;
    const bT = truth ? ph.betaTerminal(truth.a0, truth.L0, Infinity, RE) : null;
    const last = S.received[S.received.length - 1];
    const wPrior = pr.t_target.p95 - pr.t_target.p5, wNow = sm.t_target.p95 - sm.t_target.p5;
    const tile = (l, v, sub) => `<div class="tile"><div class="lbl">${l}</div><div class="val">${v}</div><div class="sub">${sub}</div></div>`;
    $('live-tiles').innerHTML = [
      tile('Records received', `${S.received.length}${S.records.length ? ` / ${S.records.length}` : ''}`,
        last ? `latest light-time lag <b>${fmtTime(last.received_at - last.t)}</b>` : 'waiting for the first record'),
      tile('Peak acceleration a₀', `${num(sm.a0.p50 / G_N)} g`,
        `± ${num(100 * rel(sm.a0), 2)} % (prior ± ${num(100 * rel(pr.a0), 2)} %)${truth ? ` · true ${num(truth.a0 / G_N)} g` : ''}`),
      tile('L₀ (beam spot = sail)', fmtDist(sm.L0.p50),
        `± ${num(100 * rel(sm.L0), 2)} % (prior ± ${num(100 * rel(pr.L0), 2)} %)${truth ? ` · true ${fmtDist(truth.L0)}` : ''}`),
      tile('Predicted top speed', pctc(sm.beta_inf.p50),
        `90 %: ${pctc(sm.beta_inf.p5)} – ${pctc(sm.beta_inf.p95)}${truth ? ` · true ${pctc(bT)}` : ''}`),
      tile('Predicted travel time', `${(sm.t_target.p50 / YEAR).toFixed(2)} yr`,
        `90 %: ${(sm.t_target.p5 / YEAR).toFixed(2)} – ${(sm.t_target.p95 / YEAR).toFixed(2)} yr${truth ? ` · true ${(tT / YEAR).toFixed(2)} yr` : ''}`),
      tile('Uncertainty vs prior', wNow < wPrior * 0.999 ? `${num(wPrior / wNow, 3)}× narrower` : 'prior only',
        `travel-time 90 % width: ${(wPrior / YEAR).toFixed(2)} → <b>${(wNow / YEAR).toFixed(3)} yr</b>`),
    ].join('');
  }

  function renderTable() {
    const rows = S.received.slice(-8).reverse();
    $('live-records').innerHTML = `<thead><tr><th>#</th><th>Event time</th><th>Received</th><th>β measured</th><th>λ<sub>r</sub>/λ</th></tr></thead><tbody>${
      rows.length ? rows.map((r) => `<tr><td>${r.seq}</td><td>${fmtTime(r.t)}</td><td>${fmtTime(r.received_at)}</td><td>${r.beta.toFixed(5)} ± ${r.sigma.toFixed(4)}</td><td>${((1 + r.beta) / (1 - r.beta)).toFixed(5)}</td></tr>`).join('')
        : '<tr><td colspan="5">No records yet. Press Play.</td></tr>'}</tbody>`;
  }

  // -------------------------------------------------------------------- posterior heatmap
  const heat = $('live-heat');
  function heatColor(f) {
    const stops = ['--heat-0', '--heat-1', '--heat-2', '--heat-3', '--heat-4'].map((v) => hexRgb(cssVar(v)));
    const x = Math.min(0.9999, Math.max(0, f)) * (stops.length - 1), i = Math.floor(x), u = x - i;
    const c = stops[i].map((a, j) => Math.round(a + (stops[i + 1][j] - a) * u));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  function drawHeat() {
    const { ctx, w, h } = setupCanvas(heat, heat.clientHeight);
    ctx.fillStyle = cssVar('--surface'); ctx.fillRect(0, 0, w, h);
    if (!S.A) return;
    const A = S.A;
    let mu = A.mu, sd = [Math.sqrt(A.cov[0][0]), Math.sqrt(A.cov[1][1])];
    if (S.zoom) {
      const { m, S: P } = A.posteriorMoments(A.weights());
      mu = m; sd = [Math.sqrt(P[0][0]), Math.sqrt(P[1][1])];
    }
    const pad = { l: 62, r: 12, t: 12, b: 40 }, K = 4.2;
    const X = (la) => pad.l + ((la - (mu[0] - K * sd[0])) / (2 * K * sd[0])) * (w - pad.l - pad.r);
    const Y = (ll) => h - pad.b - ((ll - (mu[1] - K * sd[1])) / (2 * K * sd[1])) * (h - pad.t - pad.b);
    const ink2 = cssVar('--ink-2'), grid = cssVar('--grid'), muted = cssVar('--muted');
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    // ticks: a0 in g, L0 in million km
    ctx.strokeStyle = grid; ctx.fillStyle = ink2; ctx.lineWidth = 1;
    for (let k = -4; k <= 4; k += 2) {
      const la = mu[0] + k * sd[0], ll = mu[1] + k * sd[1];
      const x = Math.round(X(la)) + 0.5, y = Math.round(Y(ll)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(num(Math.exp(la) / G_N, S.zoom ? 4 : 3), x, h - pad.b + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(num(Math.exp(ll) / 1e9, S.zoom ? 4 : 3), pad.l - 6, y);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('peak acceleration a₀ [g]', pad.l + (w - pad.l - pad.r) / 2, h - 6);
    ctx.save(); ctx.translate(13, pad.t + (h - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('L₀ [million km]', 0, 0); ctx.restore();
    ctx.save();
    ctx.beginPath(); ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b); ctx.clip();
    // prior 90 % ellipse
    const Lp = A.Lp, r90 = Math.sqrt(4.605), m0 = A.mu;
    ctx.strokeStyle = muted; ctx.setLineDash([3, 3]); ctx.beginPath();
    for (let i = 0; i <= 96; i++) {
      const th = (i / 96) * 2 * Math.PI, z1 = r90 * Math.cos(th), z2 = r90 * Math.sin(th);
      const la = m0[0] + Lp[0][0] * z1, ll = m0[1] + Lp[1][0] * z1 + Lp[1][1] * z2;
      i ? ctx.lineTo(X(la), Y(ll)) : ctx.moveTo(X(la), Y(ll));
    }
    ctx.stroke(); ctx.setLineDash([]);
    if (!S.zoom) { ctx.fillStyle = ink2; ctx.textAlign = 'left'; ctx.fillText('prior 90 %', X(m0[0] + Lp[0][0] * r90) + 4, Y(m0[1] + Lp[1][0] * r90)); }
    // posterior cells
    const wts = A.weights();
    let mx = 0;
    for (const v of wts) mx = Math.max(mx, v);
    const step = (2 * A.span) / (A.G - 1);
    const cw = Math.max(2, Math.abs(X(mu[0] + A.M[0][0] * step) - X(mu[0]))), chh = Math.max(2, Math.abs(Y(mu[1] + A.M[1][1] * step) - Y(mu[1])));
    for (let k = 0; k < wts.length; k++) {
      const f = wts[k] / mx;
      if (f < 2e-3) continue;
      ctx.fillStyle = heatColor(Math.pow(f, 0.6));
      ctx.fillRect(X(A.lnA[k]) - cw / 2, Y(A.lnL[k]) - chh / 2, cw + 0.6, chh + 0.6);
    }
    if (S.reveal && S.truth) {
      const d = ph.derive(S.truth), x = X(Math.log(d.a0)), y = Y(Math.log(d.L0));
      ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = cssVar('--ink'); ctx.textAlign = 'left'; ctx.fillText('true sail', x + 9, y - 8);
    }
    ctx.restore();
  }

  // -------------------------------------------------------------------- live track (space scene)
  const track = $('live-track');
  function drawTrack(now) {
    const { ctx, w, h } = setupCanvas(track, track.clientHeight);
    ctx.fillStyle = cssVar('--space'); ctx.fillRect(0, 0, w, h);
    if (!S.A) return;
    const ink = cssVar('--space-ink'), ink2 = cssVar('--space-ink-2'), gridc = cssVar('--space-grid'), pink = cssVar('--space-beam');
    const X0 = design().targetDistance, left = w < 560 ? 44 : 64, right = 40, y = h / 2 + 6;
    const xMin = 1e3, xMax = X0 * 1.25, lx = (x) => Math.log10(Math.max(x, xMin));
    const XP = (x) => left + ((lx(x) - lx(xMin)) / (lx(xMax) - lx(xMin))) * (w - left - right);
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    let lastR = -1e9;
    for (const [v, lab] of DIST_TICKS) {
      if (v < xMin || v > xMax) continue;
      const x = Math.round(XP(v)) + 0.5;
      ctx.strokeStyle = gridc; ctx.beginPath(); ctx.moveTo(x, 22); ctx.lineTo(x, h - 26); ctx.stroke();
      const tw = ctx.measureText(lab).width;
      if (x - tw / 2 > lastR + 8) { ctx.fillStyle = ink2; ctx.fillText(lab, x, h - 20); lastR = x + tw / 2; }
    }
    // Earth + ground station
    ctx.fillStyle = '#3d7bd9'; ctx.beginPath(); ctx.arc(left - 26, y, 9, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = '#5bb07a'; ctx.beginPath(); ctx.arc(left - 28, y - 2, 4, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = '#c9c7bf'; ctx.fillRect(left - 6, y - 10, 3, 20);
    // pings: expanding rings at the station when a record arrives, and a marker where it came from
    S.pings = S.pings.filter((p) => now - p.at < 1600);
    for (const p of S.pings) {
      const age = (now - p.at) / 1600;
      ctx.strokeStyle = withAlpha(pink, 0.8 * (1 - age)); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(left - 6, y, 6 + 26 * age, 0, 2 * Math.PI); ctx.stroke();
    }
    for (const r of S.received.slice(-14)) {
      const xe = ph.stateAt(S.est.mid, r.t).x;
      ctx.fillStyle = withAlpha(pink, 0.65);
      ctx.beginPath(); ctx.moveTo(XP(xe), y - 22); ctx.lineTo(XP(xe) + 4, y - 18); ctx.lineTo(XP(xe), y - 14); ctx.lineTo(XP(xe) - 4, y - 18); ctx.closePath(); ctx.fill();
    }
    // estimated sail (pink) with approximate 90 % range, true sail if revealed
    const tNow = Math.max(S.simT, 1e-3);
    const xm = ph.stateAt(S.est.mid, tNow).x, xl = ph.stateAt(S.est.lo, tNow).x, xh = ph.stateAt(S.est.hi, tNow).x;
    const g = ctx.createLinearGradient(left, 0, XP(xm), 0);
    g.addColorStop(0, withAlpha(pink, 0.7)); g.addColorStop(1, withAlpha(pink, 0.12));
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.moveTo(left - 3, y - 3); ctx.lineTo(XP(xm), y - 12); ctx.lineTo(XP(xm), y + 12); ctx.lineTo(left - 3, y + 3); ctx.closePath(); ctx.fill();
    ctx.fillStyle = withAlpha(pink, 0.28);
    ctx.fillRect(Math.min(XP(xl), XP(xh)) - 2, y - 15, Math.abs(XP(xh) - XP(xl)) + 4, 30);
    ctx.fillStyle = pink; ctx.fillRect(XP(xm) - 1, y - 13, 4, 26);
    const est = ph.stateAt(S.est.mid, tNow);
    const nearEnd = XP(xm) > w - 170;
    ctx.fillStyle = ink; ctx.textBaseline = 'bottom'; ctx.textAlign = nearEnd ? 'right' : 'left';
    ctx.fillText(`estimate ${pctc(est.beta)}`, nearEnd ? Math.min(XP(xl), XP(xh)) - 8 : Math.max(XP(xl), XP(xh)) + 8, y - 16);
    if (S.reveal && S.truthTraj) {
      const st = ph.stateAt(S.truthTraj, tNow);
      ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.strokeRect(XP(st.x) - 2, y + 18, 4, 16);
      ctx.fillStyle = ink2; ctx.textBaseline = 'top'; ctx.textAlign = nearEnd ? 'right' : 'left';
      ctx.fillText(`true ${pctc(st.beta)}`, nearEnd ? XP(st.x) - 8 : XP(st.x) + 8, y + 20);
    }
    const tx = XP(X0);
    ctx.fillStyle = '#ffd98a'; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('target ★', tx + 4, 6);
  }

  // -------------------------------------------------------------------- clock and loop
  function warp() {
    if (S.speed !== 'auto') return Number(S.speed);
    const rate = Math.log10(S.tEnd / S.tStart) / AUTO_SECONDS;
    return Math.LN10 * rate * Math.max(S.simT, S.tStart);
  }
  function frame(now) {
    requestAnimationFrame(frame);
    if (!S.active) { lastFrame = null; return; }
    const dt = lastFrame == null ? 0 : Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;
    if (S.source === 'sse') {
      if (S.sseQueue.length) {
        const batch = S.sseQueue.splice(0);
        ingest(batch, now);
        S.simT = Math.max(S.simT, ...batch.map((r) => r.received_at));
      }
    } else if (S.playing) {
      if (S.speed === 'auto') {
        const rate = Math.log10(S.tEnd / S.tStart) / AUTO_SECONDS;
        S.simT = Math.max(S.simT, S.tStart) * 10 ** (rate * dt);
      } else S.simT += dt * Number(S.speed);
      if (S.simT >= S.tEnd) { S.simT = S.tEnd; S.playing = false; updatePlayButton(); }
      const batch = [];
      while (S.next < S.records.length && S.records[S.next].received_at <= S.simT) batch.push(S.records[S.next++]);
      ingest(batch, now);
    }
    refreshSummary(now);
    drawTrack(now);
    chBeta.setCursor(Math.max(S.simT, 0.5));
    if (now - lastDom > 120) {
      lastDom = now;
      $('live-clock').textContent = fmtTime(S.simT);
      $('live-warp').textContent = S.source === 'sse' ? 'streaming' : (S.playing ? `×${num(warp(), 2)}` : 'paused');
      $('live-status').textContent = S.status || (S.source === 'sim' ? 'SIMULATED tracking stream' : S.source === 'csv' ? 'Replaying CSV records' : 'Live stream');
    }
  }

  function updatePlayButton() {
    const b = $('live-play');
    b.textContent = S.playing ? '❚❚ Pause' : (S.simT >= S.tEnd && S.records.length ? '↺ Replay' : '▶ Play');
    b.disabled = S.source === 'sse';
  }

  // -------------------------------------------------------------------- sources
  function setSource(src) {
    S.source = src;
    disconnect();
    document.querySelectorAll('[data-live-src]').forEach((el) => { el.hidden = el.dataset.liveSrc !== src; });
    $('live-reveal').disabled = src !== 'sim';
    if (src !== 'sim') { S.reveal = false; $('live-reveal').checked = false; S.truth = null; S.truthTraj = null; }
    if (src === 'sim') rebuild();
    else { S.records = []; S.status = src === 'csv' ? 'Choose a CSV file to replay' : 'Not connected'; restart(); }
  }

  function connect() {
    disconnect();
    const url = $('live-url').value.trim();
    S.records = [];
    restart();
    try {
      const es = new EventSource(url);
      S.sse = es;
      S.status = `Connecting to ${url} …`;
      es.addEventListener('meta', (e) => { try { S.meta = JSON.parse(e.data); S.status = `Connected${S.meta.simulated ? ' (SIMULATED source)' : ''}`; } catch { /* ignore */ } });
      es.onopen = () => { if (!S.meta) S.status = 'Connected'; };
      es.onmessage = (e) => {
        try {
          const r = JSON.parse(e.data);
          if (r.t > 0 && r.sigma > 0 && Number.isFinite(r.beta)) S.sseQueue.push({ seq: r.seq ?? S.received.length, t: r.t, received_at: r.received_at ?? r.t, beta: r.beta, sigma: r.sigma });
        } catch { /* ignore malformed lines */ }
      };
      es.addEventListener('end', () => { S.status = `Stream finished (${S.received.length + S.sseQueue.length} records)`; disconnect(); });
      es.onerror = () => { if (S.sse) S.status = 'Connection error (is the stream server running?)'; };
    } catch (err) { S.status = `Cannot connect: ${err.message}`; }
    $('live-connect').textContent = 'Disconnect';
  }
  function disconnect() {
    if (S.sse) { S.sse.close(); S.sse = null; }
    const b = $('live-connect');
    if (b) b.textContent = 'Connect';
  }

  // -------------------------------------------------------------------- wire up
  function build() {
    const ds = $('live-design');
    ds.innerHTML = Object.keys(DESIGNS).map((n) => `<option>${n}</option>`).join('');
    ds.addEventListener('change', () => { S.designName = ds.value; S.source === 'sim' ? rebuild() : restart(); });
    $('live-source').addEventListener('change', (e) => setSource(e.target.value));
    const noise = $('live-noise');
    const setNoise = () => { S.noise = 10 ** (+noise.value / 1000 * Math.log10(50) + Math.log10(2e-4)); $('live-noise-val').textContent = S.noise.toFixed(4); };
    noise.addEventListener('input', setNoise);
    noise.addEventListener('change', () => { setNoise(); if (S.source === 'sim') rebuild(); });
    $('live-newsail').addEventListener('click', () => { S.seed += 1; if (S.source === 'sim') rebuild(); });
    $('live-reveal').addEventListener('change', (e) => { S.reveal = e.target.checked; renderStatic(); });
    $('live-play').addEventListener('click', () => {
      if (!S.playing && S.simT >= S.tEnd) restart();
      else { S.playing = !S.playing; updatePlayButton(); }
    });
    $('live-restart').addEventListener('click', restart);
    $('live-speed').addEventListener('change', (e) => { S.speed = e.target.value; });
    $('live-file').addEventListener('change', async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try {
        S.records = asm.parseCSV(await f.text());
        S.status = `${S.records.length} records loaded from ${f.name}`;
        S.meta = { simulated: false };
        restart();
      } catch (err) { S.status = `CSV error: ${err.message}`; }
    });
    $('live-sample').addEventListener('click', () => {
      const truth = asm.sampleTruth(design(), SPEC, S.seed);
      const recs = asm.simulatedStream(truth, { noise: S.noise, seed: S.seed });
      const csv = ['# SIMULATED lightsail tracking records (no real flight data exist)', 'seq,t_s,received_s,beta,sigma',
        ...recs.map((r) => [r.seq, r.t, r.received_at, r.beta, r.sigma].map((v) => (typeof v === 'number' ? +v.toPrecision(12) : v)).join(','))].join('\n');
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
      a.download = 'simulated_tracking.csv';
      document.body.append(a); a.click(); a.remove();
    });
    $('live-connect').addEventListener('click', () => (S.sse ? disconnect() : connect()));
    $('live-zoom').addEventListener('click', (e) => {
      S.zoom = !S.zoom;
      e.target.setAttribute('aria-pressed', String(S.zoom));
      e.target.textContent = S.zoom ? 'Show prior range' : 'Zoom to posterior';
      drawHeat();
    });

    const timeAxis = { xLog: true, xTicks: TIME_TICKS, xFmt: (v) => `t = ${fmtTime(v)}` };
    chBeta = new Plot($('chart-live-beta'), { ...timeAxis, xLabel: 'event time at the sail (Earth frame, log scale)', yLabel: 'speed β = v/c', yFmt: (v) => num(v, 4), ariaLabel: 'Measured speed with the model fit' });
    chFunnel = new Plot($('chart-live-funnel'), { ...timeAxis, xLabel: 'time the data were received (log scale)', yLabel: 'travel time [yr]', yFmt: (v) => `${v.toFixed(2)} yr`, xFmt: (v) => `received by t = ${fmtTime(v)}`, ariaLabel: 'Predicted travel time interval narrowing as data arrive' });
    new ResizeObserver(() => drawHeat()).observe(heat);
    setNoise();
    requestAnimationFrame(frame);
  }

  let built = false;
  return {
    activate() {
      if (!built) {
        built = true;
        build();
        $('live-status').textContent = 'Preparing the estimator…';
        setTimeout(() => { rebuild(); S.active = true; }, 30);
        return;
      }
      if (S.designName === 'Current sidebar setup' && S.source === 'sim') rebuild();
      S.active = true;
    },
    deactivate() { S.active = false; },
    redraw() { if (built) renderStatic(); },
    state: S,
  };
}

function hexRgb(h) {
  h = h.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
