// Live Bayesian assimilation of lightsail tracking measurements (relativistic model).
// A port of lightsail/telemetry.py and lightsail/assimilation.py; tests/test_js_crosscheck.py
// checks that both give the same posterior for the same records.
//
// The measurement stream generated here is SIMULATED: no laser-driven lightsail has flown, so no
// real tracking data exist. Real data in the same format can be loaded from CSV or a stream.
import {
  C, RELATIVISTIC, betaFromU, dopplerMinusOne, timeToDistance, derive, resolved,
  specMoments, buildTrajectory, stateAt,
} from './physics.js';

const GL_X = [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
  0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
const GL_W = [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
  0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];

function geomspace(a, b, n) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = a * Math.pow(b / a, i / (n - 1));
  out[0] = a; out[n - 1] = b;
  return out;
}
function linspace(a, b, n) {
  const out = new Float64Array(n), st = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + i * st;
  out[n - 1] = b;
  return out;
}
/** numpy.interp: linear, constant beyond the ends. xp increasing. */
function interp(x, xp, fp) {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  return fp[lo] + ((fp[hi] - fp[lo]) * (x - xp[lo])) / (xp[hi] - xp[lo]);
}

// ============================================================================ universal table
export class UniversalTable {
  constructor() {
    this.NB = 128; this.NT = 2400; this.NNEAR = 400; this.NFAR = 1600;
    this.lnb0 = linspace(Math.log(1e-3), Math.log(2.0), this.NB);
    this.lntau = linspace(Math.log(1e-4), Math.log(1e9), this.NT);
    this.table = new Float64Array(this.NB * this.NT);
    for (let i = 0; i < this.NB; i++) this.table.set(this.row(Math.exp(this.lnb0[i])), i * this.NT);
    this.db = this.lnb0[1] - this.lnb0[0];
    this.dt = this.lntau[1] - this.lntau[0];
  }

  row(b0) {
    const a0 = b0 * C, L0 = (b0 * C) / 2;
    const uL = dopplerMinusOne(3 * b0 * b0);
    const u = geomspace(uL * 1e-9, uL, this.NNEAR);
    const N = this.NNEAR + this.NFAR;
    const lt = new Float64Array(N), lb = new Float64Array(N);
    for (let i = 0; i < this.NNEAR; i++) {
      const ui = u[i];
      lt[i] = Math.log((ui * ui * ui + 3 * ui * ui + 6 * ui) / (6 * b0));
      lb[i] = Math.log(betaFromU(ui));
    }
    const tNear = Math.exp(lt[this.NNEAR - 1]);
    const SMAX = Math.log(1e14), h = SMAX / this.NFAR;
    let t = tNear;
    for (let k = 0; k < this.NFAR; k++) {
      const s0 = k * h;
      let dt = 0;
      for (let j = 0; j < 8; j++) {
        const x = L0 * Math.exp(s0 + ((GL_X[j] + 1) / 2) * h);
        const beta = betaFromU(dopplerMinusOne((6 * a0 * (2 * L0 - (L0 * L0) / x)) / (C * C)));
        dt += (x / (C * beta)) * (GL_W[j] / 2) * h;
      }
      t += dt;
      const xe = L0 * Math.exp((k + 1) * h);
      lt[this.NNEAR + k] = Math.log(t);
      lb[this.NNEAR + k] = Math.log(betaFromU(dopplerMinusOne((6 * a0 * (2 * L0 - (L0 * L0) / xe)) / (C * C))));
    }
    const row = new Float64Array(this.NT);
    for (let i = 0; i < this.NT; i++) {
      const x = this.lntau[i];
      row[i] = x < lt[0] ? lb[0] + (x - lt[0]) : interp(x, lt, lb);
    }
    return row;
  }

  lnBeta(lnb0, lntau) {
    let fb = (lnb0 - this.lnb0[0]) / this.db, ft = (lntau - this.lntau[0]) / this.dt;
    fb = Math.min(Math.max(fb, 0), this.NB - 1 - 1e-9);
    ft = Math.min(Math.max(ft, 0), this.NT - 1 - 1e-9);
    const ib = Math.floor(fb), it = Math.floor(ft), wb = fb - ib, wt = ft - it;
    const T = this.table, r0 = ib * this.NT, r1 = r0 + this.NT;
    return (1 - wb) * ((1 - wt) * T[r0 + it] + wt * T[r0 + it + 1])
      + wb * ((1 - wt) * T[r1 + it] + wt * T[r1 + it + 1]);
  }

  beta(t, a0, L0) {
    const b0 = Math.sqrt(2 * a0 * L0) / C, t0 = Math.sqrt((2 * L0) / a0);
    return Math.exp(this.lnBeta(Math.log(b0), Math.log(t) - Math.log(t0)));
  }
}

let SHARED = null;
export function sharedTable() { if (!SHARED) SHARED = new UniversalTable(); return SHARED; }

// ============================================================================ prior
export function logPriorMoments(design, spec, h = 1e-6) {
  const p = resolved(design);
  const mom = specMoments(p, spec);
  const pm = { ...p };
  for (const [k, [mu]] of Object.entries(mom)) pm[k] = mu;
  const theta = (q) => { const d = derive(q); return [Math.log(d.a0), Math.log(d.L0)]; };
  const mu = theta(pm);
  const cov = [[0, 0], [0, 0]];
  for (const [k, [m, sd]] of Object.entries(mom)) {
    const dv = m * h;
    const up = theta({ ...pm, [k]: m + dv }), dn = theta({ ...pm, [k]: m - dv });
    const j0 = (up[0] - dn[0]) / (2 * dv), j1 = (up[1] - dn[1]) / (2 * dv), v = sd * sd;
    cov[0][0] += j0 * j0 * v; cov[0][1] += j0 * j1 * v; cov[1][1] += j1 * j1 * v;
  }
  cov[1][0] = cov[0][1];
  return { mu, cov };
}

export function chol2(S) {
  const l11 = Math.sqrt(S[0][0]), l21 = S[1][0] / l11;
  const l22 = Math.sqrt(Math.max(S[1][1] - l21 * l21, 1e-300));
  return [[l11, 0], [l21, l22]];
}
const inv2 = (M) => {
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  return [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]];
};
const mul2 = (A, B) => [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
  [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]];
const tr2 = (A) => [[A[0][0], A[1][0]], [A[0][1], A[1][1]]];

export function weightedQuantile(v, w, qs) {
  const n = v.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (v[a] - v[b]) || (a - b));
  const xs = new Float64Array(n), cs = new Float64Array(n);
  let c = 0;
  for (let i = 0; i < n; i++) { c += w[idx[i]]; cs[i] = c - 0.5 * w[idx[i]]; xs[i] = v[idx[i]]; }
  for (let i = 0; i < n; i++) cs[i] /= c;
  return qs.map((q) => interp(q, cs, xs));
}

// ============================================================================ grid posterior
export class Assimilator {
  constructor(design, spec, { grid = 96, span = 5, table = null } = {}) {
    this.design = { ...resolved(design), illuminationStop: Infinity };
    this.table = table || sharedTable();
    this.G = grid; this.span = span;
    const { mu, cov } = logPriorMoments(this.design, spec);
    this.mu = mu; this.cov = cov;
    this.Lp = chol2(cov); this.LpInv = inv2(this.Lp);
    this.records = [];
    this.regrids = 0;
    this.setGrid(mu, this.Lp);
    this.prior = this.summary();
  }

  setGrid(center, M) {
    const G = this.G, N = G * G, z = linspace(-this.span, this.span, G);
    this.center = center; this.M = M;
    this.lnA = new Float64Array(N); this.lnL = new Float64Array(N);
    this.logPrior = new Float64Array(N); this.lnb0 = new Float64Array(N); this.lnt0 = new Float64Array(N);
    this.loglik = new Float64Array(N);
    const Li = this.LpInv, ln2 = Math.log(2), lnc = Math.log(C);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const k = i * G + j, z1 = z[i], z2 = z[j];
      const a = center[0] + M[0][0] * z1 + M[0][1] * z2;
      const l = center[1] + M[1][0] * z1 + M[1][1] * z2;
      this.lnA[k] = a; this.lnL[k] = l;
      const d0 = Li[0][0] * (a - this.mu[0]) + Li[0][1] * (l - this.mu[1]);
      const d1 = Li[1][0] * (a - this.mu[0]) + Li[1][1] * (l - this.mu[1]);
      this.logPrior[k] = -0.5 * (d0 * d0 + d1 * d1);
      this.lnb0[k] = 0.5 * (ln2 + a + l) - lnc;
      this.lnt0[k] = 0.5 * (ln2 + l - a);
    }
    for (const r of this.records) this.addLL(r);
    this.tt = null;   // travel time per cell, computed lazily
  }

  /** Travel time per cell: exact on a 24 x 24 coarse lattice, bilinear (in ln T) elsewhere. */
  travelTimes() {
    if (this.tt) return this.tt;
    const G = this.G, GC = 24, zc = linspace(-this.span, this.span, GC), M = this.M, c = this.center;
    const lnT = new Float64Array(GC * GC);
    for (let i = 0; i < GC; i++) for (let j = 0; j < GC; j++) {
      const a = c[0] + M[0][0] * zc[i] + M[0][1] * zc[j], l = c[1] + M[1][0] * zc[i] + M[1][1] * zc[j];
      lnT[i * GC + j] = Math.log(timeToDistance(this.design.targetDistance, Math.exp(a), Math.exp(l), Infinity, RELATIVISTIC)[0]);
    }
    const f = linspace(0, GC - 1, G), idx = new Int32Array(G), fr = new Float64Array(G);
    for (let i = 0; i < G; i++) { idx[i] = Math.min(Math.floor(f[i]), GC - 2); fr[i] = f[i] - idx[i]; }
    const tt = new Float64Array(G * G);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const a = idx[i], b = idx[j], wa = fr[i], wb = fr[j];
      tt[i * G + j] = Math.exp((1 - wa) * ((1 - wb) * lnT[a * GC + b] + wb * lnT[a * GC + b + 1])
        + wa * ((1 - wb) * lnT[(a + 1) * GC + b] + wb * lnT[(a + 1) * GC + b + 1]));
    }
    this.tt = tt;
    return tt;
  }

  pred(k, t) { return Math.exp(this.table.lnBeta(this.lnb0[k], Math.log(t) - this.lnt0[k])); }

  addLL(r) {
    const lt = Math.log(r.t);
    for (let k = 0; k < this.loglik.length; k++) {
      const e = (r.beta - Math.exp(this.table.lnBeta(this.lnb0[k], lt - this.lnt0[k]))) / r.sigma;
      this.loglik[k] -= 0.5 * e * e;
    }
  }

  weights() {
    const N = this.loglik.length, w = new Float64Array(N);
    let mx = -Infinity;
    for (let k = 0; k < N; k++) mx = Math.max(mx, this.logPrior[k] + this.loglik[k]);
    let s = 0;
    for (let k = 0; k < N; k++) { w[k] = Math.exp(this.logPrior[k] + this.loglik[k] - mx); s += w[k]; }
    for (let k = 0; k < N; k++) w[k] /= s;
    return w;
  }

  posteriorMoments(w) {
    let m0 = 0, m1 = 0;
    for (let k = 0; k < w.length; k++) { m0 += w[k] * this.lnA[k]; m1 += w[k] * this.lnL[k]; }
    let s00 = 0, s01 = 0, s11 = 0;
    for (let k = 0; k < w.length; k++) {
      const d0 = this.lnA[k] - m0, d1 = this.lnL[k] - m1;
      s00 += w[k] * d0 * d0; s01 += w[k] * d0 * d1; s11 += w[k] * d1 * d1;
    }
    const h = (2 * this.span) / (this.G - 1), c = (h * h) / 12, M = this.M;
    const add = mul2(mul2(M, [[c, 0], [0, c]]), tr2(M));
    return { m: [m0, m1], S: [[s00 + add[0][0], s01 + add[0][1]], [s01 + add[1][0], s11 + add[1][1]]] };
  }

  maybeRegrid() {
    for (let it = 0; it < 6; it++) {
      const { m, S } = this.posteriorMoments(this.weights());
      const Mi = inv2(this.M), Sz = mul2(mul2(Mi, S), tr2(Mi));
      const sd = [Math.sqrt(Sz[0][0]), Math.sqrt(Sz[1][1])];
      const dm = [m[0] - this.center[0], m[1] - this.center[1]];
      const zm = [Mi[0][0] * dm[0] + Mi[0][1] * dm[1], Mi[1][0] * dm[0] + Mi[1][1] * dm[1]];
      const h = (2 * this.span) / (this.G - 1);
      if (Math.min(sd[0], sd[1]) > 4 * h && Math.abs(zm[0]) + 3 * sd[0] < this.span && Math.abs(zm[1]) + 3 * sd[1] < this.span) return;
      const Lc = chol2(S);
      this.setGrid(m, [[Lc[0][0] * 1.2, 0], [Lc[1][0] * 1.2, Lc[1][1] * 1.2]]);
      this.regrids += 1;
    }
  }

  update(records) {
    for (const r of records) { this.records.push(r); this.addLL(r); }
    this.maybeRegrid();
  }

  /** Cells that carry posterior weight (for plots): indices with w > frac * max. */
  liveCells(w, frac = 1e-6) {
    let mx = 0;
    for (const x of w) mx = Math.max(mx, x);
    const out = [];
    for (let k = 0; k < w.length; k++) if (w[k] > frac * mx) out.push(k);
    return out;
  }

  predictiveBand(times, qs = [0.05, 0.5, 0.95], maxCells = 1500) {
    const w = this.weights();
    let cells = this.liveCells(w, 1e-4);
    if (cells.length > maxCells) { const st = Math.ceil(cells.length / maxCells); cells = cells.filter((_, i) => i % st === 0); }
    const ww = cells.map((k) => w[k]);
    return times.map((t) => weightedQuantile(cells.map((k) => this.pred(k, t)), ww, qs));
  }

  summary(travelTime = true) {
    const w = this.weights(), N = w.length;
    const a0 = new Float64Array(N), L0 = new Float64Array(N), binf = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      a0[k] = Math.exp(this.lnA[k]); L0[k] = Math.exp(this.lnL[k]);
      const b0 = Math.exp(this.lnb0[k]);
      binf[k] = betaFromU(dopplerMinusOne(6 * b0 * b0));
    }
    const out = { n: this.records.length, regrids: this.regrids };
    for (const [name, v] of [['a0', a0], ['L0', L0], ['beta_inf', binf]]) {
      let mean = 0;
      for (let k = 0; k < N; k++) mean += v[k] * w[k];
      let vv = 0;
      for (let k = 0; k < N; k++) vv += (v[k] - mean) ** 2 * w[k];
      const [p5, p50, p95] = weightedQuantile(v, w, [0.05, 0.5, 0.95]);
      out[name] = { mean, sd: Math.sqrt(vv), p5, p50, p95 };
    }
    if (travelTime) {
      const [p5, p50, p95] = weightedQuantile(this.travelTimes(), w, [0.05, 0.5, 0.95]);
      out.t_target = { p5, p50, p95 };
    }
    return out;
  }
}

// ============================================================================ SIMULATED stream
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** Draw one 'true' sail from the uncertainty prior around a design (sail size held fixed). */
export function sampleTruth(design, spec, seed = 0) {
  const p = { ...resolved(design), illuminationStop: Infinity };
  const rand = mulberry32(seed * 2654435761 + 12345);
  for (const [k, s] of Object.entries(spec)) {
    p[k] = s.kind === 'normal_rel' ? p[k] * (1 + s.s * Math.max(-4, Math.min(4, gauss(rand)))) : s.a + (s.b - s.a) * rand();
  }
  return p;
}

/** SIMULATED Doppler tracking records of a 'true' sail, ordered by reception time. */
export function simulatedStream(truth, { noise = 0.002, perDecade = 30, tStart = 1, seed = 0 } = {}) {
  const tr = buildTrajectory(truth, RELATIVISTIC, 6000);
  const tEnd = tr.t[tr.t.length - 1];
  const n = Math.max(2, Math.ceil(Math.log10(tEnd / tStart) * perDecade) + 1);
  const rand = mulberry32(seed * 7919 + 1);
  const recs = [];
  for (let i = 0; i < n; i++) {
    const t = tStart * Math.pow(tEnd / tStart, i / (n - 1));
    const st = stateAt(tr, Math.min(t, tEnd * (1 - 1e-12)));
    recs.push({ seq: i, t, received_at: t + st.x / C, beta: st.beta + noise * gauss(rand), sigma: noise, truthBeta: st.beta });
  }
  return recs.sort((a, b) => a.received_at - b.received_at);
}

/** Parse measurement CSV: seq,t_s,received_s,beta,sigma (or v_m_s instead of beta, sigma in m/s). */
export function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim() && !l.trim().startsWith('#'));
  const head = lines[0].split(',').map((h) => h.trim());
  const col = (name) => head.indexOf(name);
  if (col('t_s') < 0 || col('sigma') < 0 || (col('beta') < 0 && col('v_m_s') < 0)) {
    throw new Error('CSV needs columns t_s, sigma and beta (or v_m_s)');
  }
  const recs = lines.slice(1).map((l, i) => {
    const f = l.split(',').map((x) => x.trim());
    const t = parseFloat(f[col('t_s')]);
    const useBeta = col('beta') >= 0 && f[col('beta')] !== '';
    const beta = useBeta ? parseFloat(f[col('beta')]) : parseFloat(f[col('v_m_s')]) / C;
    const sigma = useBeta ? parseFloat(f[col('sigma')]) : parseFloat(f[col('sigma')]) / C;
    const rc = col('received_s') >= 0 && f[col('received_s')] !== '' ? parseFloat(f[col('received_s')]) : t;
    const seq = col('seq') >= 0 && f[col('seq')] !== '' ? parseInt(f[col('seq')], 10) : i;
    return { seq, t, received_at: rc, beta, sigma };
  }).filter((r) => Number.isFinite(r.t) && r.t > 0 && Number.isFinite(r.beta) && r.sigma > 0);
  return recs.sort((a, b) => a.received_at - b.received_at);
}
