// Lightsail physics engine (browser + Node). A line-by-line port of the Python reference
// implementation in lightsail/physics.py; tests/test_js_crosscheck.py requires the two to agree.
//
// Model (Lubin 2016 "A Roadmap to Interstellar Flight"; Kulkarni, Lubin & Zhang 2018):
//   beam fraction on sail  f(x) = 1 (x <= L0), (L0/x)^2 (L0 < x <= x_stop), 0 beyond
//   classical              dv/dt = a0 f(x)
//   relativistic           c d(gamma beta)/dt = a0 f(x) (1 - beta)/(1 + beta)
//   a0 = eta P0 / (m c),  eta = 2R + (1-R)A,  L0 = d D / (2 lambda alpha)
// Closed forms with W(x) = integral_0^x f:  classical v = sqrt(2 a0 W);
//   relativistic Doppler factor Dop solves (Dop-1)^2 (Dop+2) = 6 a0 W / c^2.

export const C = 299792458;
export const G_N = 9.80665;
export const SIGMA_SB = 5.670374419e-8;
export const AU = 149597870700;
export const LY = 9.4607e15;
export const YEAR = 365.25 * 86400;
export const DAY = 86400;
export const GW = 1e9;
export const GRAM = 1e-3;
export const CLASSICAL = 'classical';
export const RELATIVISTIC = 'relativistic';
export const MODELS = [CLASSICAL, RELATIVISTIC];

export const TARGETS = {
  'Alpha Centauri A/B': 4.37 * LY,
  'Proxima Centauri': 4.2465 * LY,
  "Barnard's Star": 5.96 * LY,
};

export const LANDMARKS = [
  ['Moon', 3.844e8],
  ['1 AU', AU],
  ['Neptune', 30.07 * AU],
  ['Heliopause', 120 * AU],
  ['Oort cloud', 2000 * AU],
];

// Defaults of the UCSB 1-D calculators (SI units).
export const DEFAULTS = Object.freeze({
  power: 100e9, arraySize: 1e4, wavelength: 1060e-9, circularArray: true,
  beamQuality: 1, beamEfficiency: 1,
  optimalSail: true, sailSize: 1, arealDensity: 1e-3, sailShape: 'square',
  reflectivity: 1, absorption: 1, emissivityFront: 1, emissivityBack: 1,
  payloadMass: 1e-3, targetDistance: 4.37 * LY, illuminationStop: Infinity,
});

// ------------------------------------------------------------------------- derived quantities
export function derive(p) {
  const xi = p.sailShape === 'circular' ? Math.PI / 4 : 1;
  const sigma = p.arealDensity;
  const D = p.optimalSail ? Math.sqrt(p.payloadMass / (xi * sigma)) : p.sailSize;
  const area = xi * D * D;
  const sailMass = sigma * area;
  const totalMass = sailMass + p.payloadMass;
  const alpha = (p.circularArray ? 1.22 : 1) * p.beamQuality;
  const R = p.reflectivity, A = p.absorption;
  const eta = 2 * R + (1 - R) * A;
  const beamPower = p.beamEfficiency * p.power;
  const force = eta * beamPower / C;
  const a0 = force / totalMass;
  const L0 = p.arraySize * D / (2 * p.wavelength * alpha);
  const xStop = p.illuminationStop * L0;
  const beta0 = Math.sqrt(2 * a0 * L0) / C;
  const t0 = Math.sqrt(2 * L0 / a0);
  const flux = beamPower / area;
  const absorbedFlux = flux * (1 - R) * A;
  const Teq = Math.pow(absorbedFlux / (SIGMA_SB * (p.emissivityFront + p.emissivityBack)), 0.25);
  return { xi, sigma, D, area, sailMass, totalMass, alpha, eta, beamPower, force, a0, L0, xStop,
           beta0, t0, flux, absorbedFlux, Teq };
}

/** Copy of p with the sail size frozen at its current value (used for sensitivity / uncertainty). */
export function resolved(p) {
  return { ...p, optimalSail: false, sailSize: derive(p).D };
}

// ------------------------------------------------------------------------- beam geometry
export function fluxFraction(x, L0, xStop = Infinity) {
  if (x > xStop) return 0;
  return x <= L0 ? 1 : (L0 / x) * (L0 / x);
}

export function workIntegral(x, L0, xStop = Infinity) {
  const xe = Math.min(x, xStop);
  return xe <= L0 ? xe : 2 * L0 - (L0 * L0) / xe;
}

// ------------------------------------------------------------------------- Doppler cubic
/** u = Dop - 1 >= 0 solving u^2 (u + 3) = q (stable for small q). */
export function dopplerMinusOne(q) {
  if (q <= 4) {
    const psi = (2 / 3) * Math.asin(Math.sqrt(Math.max(q, 0)) / 2);
    const s = Math.sin(psi / 2);
    return -2 * s * s + Math.sqrt(3) * Math.sin(psi);
  }
  return 2 * Math.cosh(Math.acosh(q / 2 - 1) / 3) - 1;
}
export const betaFromU = (u) => (u * (u + 2)) / (u * u + 2 * u + 2);
export const gammaFromU = (u) => (u * u + 2 * u + 2) / (2 * (1 + u));
export const gammaBetaFromU = (u) => (u * (u + 2)) / (2 * (1 + u));

// ------------------------------------------------------------------------- speed vs distance
export function betaOfX(x, a0, L0, xStop, model) {
  const W = workIntegral(x, L0, xStop);
  if (model === CLASSICAL) return Math.sqrt(2 * a0 * W) / C;
  return betaFromU(dopplerMinusOne((6 * a0 * W) / (C * C)));
}

export function gammaOfX(x, a0, L0, xStop, model) {
  if (model === CLASSICAL) return 1;
  return gammaFromU(dopplerMinusOne((6 * a0 * workIntegral(x, L0, xStop)) / (C * C)));
}

export function betaTerminal(a0, L0, xStop, model) {
  return betaOfX(Infinity, a0, L0, xStop, model);
}

/** Acceleration felt by the sail [m/s^2]. */
export function properAcceleration(x, a0, L0, xStop, model) {
  const f = a0 * fluxFraction(x, L0, xStop);
  if (model === CLASSICAL) return f;
  const u = dopplerMinusOne((6 * a0 * workIntegral(x, L0, xStop)) / (C * C));
  return f / ((1 + u) * (1 + u));
}

/** Doppler factor of the laser light received by the sail, lambda_obs / lambda = sqrt((1+b)/(1-b)). */
export function dopplerFactor(beta) {
  return Math.sqrt((1 + beta) / (1 - beta));
}

// ------------------------------------------------------------------------- time vs distance
const GL_X = [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
               0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
const GL_W = [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
              0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];

function nearFieldTime(x, a0, model) {
  if (model === CLASSICAL) {
    const t = Math.sqrt((2 * x) / a0);
    return [t, t];
  }
  const u = dopplerMinusOne((6 * a0 * x) / (C * C));
  return [((C / a0) * (u * u * u + 3 * u * u + 6 * u)) / 6, ((C / a0) * u * (u + 2)) / 2];
}

function farFieldClassical(y, L0, a0) {
  const v0 = Math.sqrt(2 * a0 * L0);
  const F = (w) => w * Math.sqrt(w * w + 1) + Math.asinh(w);
  const w = Math.sqrt(2 * y - 1);
  return ((L0 / v0) * (F(w) - F(1))) / (2 * Math.SQRT2);
}

/** Integrals of dx/v and dx/(gamma v) over x = L0 e^s, s in [s0, s1] (Gauss-Legendre, nPanels). */
function farFieldPanels(s0, s1, L0, a0, model, nPanels) {
  let t = 0, tau = 0;
  const h = (s1 - s0) / nPanels;
  for (let m = 0; m < nPanels; m++) {
    const a = s0 + m * h;
    for (let k = 0; k < 8; k++) {
      const s = a + ((GL_X[k] + 1) / 2) * h;
      const x = L0 * Math.exp(s);
      const W = 2 * L0 - (L0 * L0) / x;
      let beta, gb;
      if (model === CLASSICAL) {
        beta = Math.sqrt(2 * a0 * W) / C;
        gb = beta;
      } else {
        const u = dopplerMinusOne((6 * a0 * W) / (C * C));
        beta = betaFromU(u);
        gb = gammaBetaFromU(u);
      }
      const w = (GL_W[k] / 2) * h * x;
      t += w / (C * beta);
      tau += w / (C * gb);
    }
  }
  return [t, tau];
}

/** Earth time t and ship proper time tau [s] to travel from rest to x. */
export function timeToDistance(x, a0, L0, xStop, model) {
  const x1 = Math.min(x, L0, xStop);
  let [t, tau] = nearFieldTime(x1, a0, model);
  const x2 = Math.min(x, xStop);
  if (x2 > L0) {
    const y = x2 / L0;
    if (model === CLASSICAL) {
      const tf = farFieldClassical(y, L0, a0);
      t += tf;
      tau += tf;
    } else {
      const S = Math.log(y);
      const [tf, tauf] = farFieldPanels(0, S, L0, a0, model, Math.max(1, Math.ceil(S * 2)));
      t += tf;
      tau += tauf;
    }
  }
  if (x > xStop) {
    const b = betaOfX(xStop, a0, L0, xStop, model);
    const g = gammaOfX(xStop, a0, L0, xStop, model);
    t += (x - xStop) / (C * b);
    tau += (x - xStop) / (C * b * g);
  }
  return [t, tau];
}

// ------------------------------------------------------------------------- mission outputs
export function missionSummary(p, model) {
  const d = derive(p);
  const { a0, L0, xStop } = d;
  const X = p.targetDistance;
  const betaL0 = betaOfX(L0, a0, L0, xStop, model);
  const betaFinal = betaTerminal(a0, L0, xStop, model);
  const [tL0] = timeToDistance(L0, a0, L0, xStop, model);
  const [tTarget, tauTarget] = timeToDistance(X, a0, L0, xStop, model);
  return { model, a0, L0, betaL0, vL0: betaL0 * C, tL0, betaFinal, vFinal: betaFinal * C,
           tTarget, tauTarget, tTargetCoastOnly: X / (betaFinal * C) };
}

/**
 * Trajectory on a log-spaced distance grid, from rest to the target. Times are accumulated
 * panel by panel (Gauss-Legendre in ln x), so the whole trajectory costs ~10^4 evaluations.
 */
export function buildTrajectory(p, model, n = 1400) {
  const d = derive(p);
  const { a0, L0, xStop } = d;
  const X = p.targetDistance;
  const xmin = Math.min(L0, X) * 1e-9;
  const grid = new Set();
  for (let i = 0; i < n; i++) grid.add(xmin * Math.pow(X / xmin, i / (n - 1)));
  if (L0 < X) grid.add(L0);
  if (xStop < X) grid.add(xStop);
  const xs = [0, ...[...grid].sort((a, b) => a - b)];
  const N = xs.length;
  const t = new Float64Array(N), tau = new Float64Array(N), beta = new Float64Array(N),
        gamma = new Float64Array(N), accel = new Float64Array(N), x = Float64Array.from(xs);
  const nearEnd = Math.min(L0, xStop);
  let farT = 0, farTau = 0, sPrev = 0;
  const [tNear, tauNear] = nearFieldTime(nearEnd, a0, model);
  for (let i = 0; i < N; i++) {
    const xi = x[i];
    beta[i] = betaOfX(xi, a0, L0, xStop, model);
    gamma[i] = gammaOfX(xi, a0, L0, xStop, model);
    accel[i] = properAcceleration(xi, a0, L0, xStop, model);
    if (xi <= nearEnd) {
      [t[i], tau[i]] = nearFieldTime(xi, a0, model);
      continue;
    }
    const x2 = Math.min(xi, xStop);
    if (x2 > L0) {
      const s = Math.log(x2 / L0);
      if (s > sPrev) {
        const [dt, dtau] = model === CLASSICAL
          ? (() => { const v = farFieldClassical(x2 / L0, L0, a0) - farT; return [v, v]; })()
          : farFieldPanels(sPrev, s, L0, a0, model, 1);
        farT += dt;
        farTau += dtau;
        sPrev = s;
      }
    }
    let ti = tNear + farT, taui = tauNear + farTau;
    if (xi > xStop) {
      const b = betaOfX(xStop, a0, L0, xStop, model), g = gammaOfX(xStop, a0, L0, xStop, model);
      ti += (xi - xStop) / (C * b);
      taui += (xi - xStop) / (C * b * g);
    }
    t[i] = ti;
    tau[i] = taui;
  }
  return { model, x, t, tau, beta, gamma, accel, derived: d, target: X };
}

/** State of the sail at Earth time t (log-log interpolation of x(t), closed-form speed at that x). */
export function stateAt(traj, time) {
  const { x, t, tau, derived: d } = traj;
  const N = t.length;
  if (time <= 0) return pack(traj, 0, 0, false);
  if (time >= t[N - 1]) return pack(traj, x[N - 1], tau[N - 1] + 0, true, time);
  let lo = 0, hi = N - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= time) lo = mid; else hi = mid;
  }
  let xi, taui;
  if (lo === 0 || x[lo] <= 0) {
    // from rest: x ~ t^2 initially
    const r = time / t[hi];
    xi = x[hi] * r * r;
    taui = tau[hi] * r;
  } else {
    const f = Math.log(time / t[lo]) / Math.log(t[hi] / t[lo]);
    xi = x[lo] * Math.pow(x[hi] / x[lo], f);
    taui = tau[lo] + (tau[hi] - tau[lo]) * ((time - t[lo]) / (t[hi] - t[lo]));
  }
  return pack(traj, xi, taui, false);

  function pack(tr, xx, tt, arrived) {
    const { a0, L0, xStop } = d;
    const b = betaOfX(xx, a0, L0, xStop, tr.model);
    return {
      x: xx, tau: tt, arrived,
      beta: b,
      gamma: gammaOfX(xx, a0, L0, xStop, tr.model),
      accel: arrived ? 0 : properAcceleration(xx, a0, L0, xStop, tr.model),
      flux: arrived ? 0 : fluxFraction(xx, L0, xStop),
      doppler: tr.model === CLASSICAL ? 1 + b : dopplerFactor(b),
    };
  }
}

// ------------------------------------------------------------------------- sensitivity & uncertainty
export const OUTPUTS = {
  betaFinal: (p, m) => { const d = derive(p); return betaTerminal(d.a0, d.L0, d.xStop, m); },
  tTarget: (p, m) => { const d = derive(p); return timeToDistance(p.targetDistance, d.a0, d.L0, d.xStop, m)[0]; },
};

export const UNCERTAIN = {
  arealDensity: { label: 'Sail areal density σ', group: 'sail' },
  payloadMass: { label: 'Payload mass m₀', group: 'sail' },
  sailSize: { label: 'Sail size D', group: 'sail' },
  reflectivity: { label: 'Reflectivity R', group: 'sail' },
  power: { label: 'Laser power P', group: 'beam' },
  beamQuality: { label: 'Beam quality factor', group: 'beam' },
};

// Same budget as lightsail/uncertainty.py DEFAULT_SPEC
export const DEFAULT_SPEC = {
  arealDensity: { kind: 'normal_rel', s: 0.10 },
  payloadMass: { kind: 'normal_rel', s: 0.10 },
  sailSize: { kind: 'normal_rel', s: 0.02 },
  reflectivity: { kind: 'uniform', a: 0.95, b: 1.0 },
  power: { kind: 'normal_rel', s: 0.05 },
  beamQuality: { kind: 'uniform', a: 1.0, b: 1.5 },
};

/** d ln Y / d ln p by central differences (backward when R = 1). */
export function elasticities(p0, out, model, params = Object.keys(UNCERTAIN), h = 1e-4) {
  const p = resolved(p0);
  const f = OUTPUTS[out];
  const res = {};
  for (const k of params) {
    const v = p[k];
    const up = v * (1 + h), dn = v * (1 - h);
    if (k === 'reflectivity' && up > 1) {
      res[k] = (Math.log(f(p, model)) - Math.log(f({ ...p, [k]: dn }, model))) / (Math.log(v) - Math.log(dn));
    } else {
      res[k] = (Math.log(f({ ...p, [k]: up }, model)) - Math.log(f({ ...p, [k]: dn }, model)))
        / (Math.log(up) - Math.log(dn));
    }
  }
  return res;
}

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

function normalSampler(rand) {
  let spare = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0;
    while (u === 0) u = rand();
    v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

export function specMoments(p, spec) {
  const out = {};
  for (const [k, s] of Object.entries(spec)) {
    if (s.kind === 'normal_rel') out[k] = [p[k], p[k] * s.s];
    else out[k] = [(s.a + s.b) / 2, (s.b - s.a) / Math.sqrt(12)];
  }
  return out;
}

export function specPercentiles(p, spec, q = [0.05, 0.95]) {
  const z = { 0.05: -1.6448536269514722, 0.95: 1.6448536269514722 };
  const out = {};
  for (const [k, s] of Object.entries(spec)) {
    out[k] = q.map((qq) => (s.kind === 'normal_rel' ? p[k] * (1 + s.s * z[qq]) : s.a + (s.b - s.a) * qq));
  }
  return out;
}

/** Monte Carlo propagation. Returns per-model Float64Arrays of travel time and final speed. */
export function monteCarlo(p0, spec, n = 5000, seed = 20261112) {
  const p = resolved(p0);
  const rand = mulberry32(seed);
  const normal = normalSampler(rand);
  const out = { n, [CLASSICAL]: { tTarget: new Float64Array(n), betaFinal: new Float64Array(n) },
                [RELATIVISTIC]: { tTarget: new Float64Array(n), betaFinal: new Float64Array(n) } };
  const q = { ...p };
  for (let i = 0; i < n; i++) {
    for (const [k, s] of Object.entries(spec)) {
      q[k] = s.kind === 'normal_rel'
        ? p[k] * (1 + s.s * Math.max(-4, Math.min(4, normal())))
        : s.a + (s.b - s.a) * rand();
    }
    const d = derive(q);
    for (const m of MODELS) {
      out[m].betaFinal[i] = betaTerminal(d.a0, d.L0, d.xStop, m);
      out[m].tTarget[i] = timeToDistance(q.targetDistance, d.a0, d.L0, d.xStop, m)[0];
    }
  }
  return out;
}

/** First-order propagation (sigma_Y/Y)^2 = sum (S_i sigma_i / mu_i)^2 at the input means. */
export function linearPropagation(p0, spec, out, model) {
  const p = resolved(p0);
  const mom = specMoments(p, spec);
  const pm = { ...p };
  for (const [k, [mu]] of Object.entries(mom)) pm[k] = mu;
  const S = elasticities(pm, out, model, Object.keys(spec));
  const y = OUTPUTS[out](pm, model);
  let tot = 0;
  const contrib = {};
  for (const [k, [mu, sd]] of Object.entries(mom)) { contrib[k] = (S[k] * sd / mu) ** 2; tot += contrib[k]; }
  const share = {};
  for (const k of Object.keys(contrib)) share[k] = tot > 0 ? contrib[k] / tot : 0;
  return { y, relStd: Math.sqrt(tot), std: Math.sqrt(tot) * y, elasticities: S, share, centre: pm };
}

/** One-at-a-time swings of an output around the centre (mean inputs), sorted by size. */
export function tornado(p0, spec, out, model) {
  const p = resolved(p0);
  const mom = specMoments(p, spec);
  const centre = { ...p };
  for (const [k, [mu]] of Object.entries(mom)) centre[k] = mu;
  const pr = specPercentiles(p, spec);
  const f = OUTPUTS[out];
  const y0 = f(centre, model);
  const rows = Object.keys(spec).map((k) => ({
    key: k, label: UNCERTAIN[k].label, group: UNCERTAIN[k].group, lo: pr[k][0], hi: pr[k][1],
    yLo: f({ ...centre, [k]: pr[k][0] }, model), yHi: f({ ...centre, [k]: pr[k][1] }, model), y0,
  }));
  rows.sort((a, b) => Math.abs(b.yHi - b.yLo) - Math.abs(a.yHi - a.yLo));
  return rows;
}

export function quantiles(arr, qs) {
  const s = Float64Array.from(arr).sort();
  return qs.map((q) => {
    const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  });
}

export function meanStd(arr) {
  let m = 0;
  for (const v of arr) m += v;
  m /= arr.length;
  let s = 0;
  for (const v of arr) s += (v - m) * (v - m);
  return [m, Math.sqrt(s / (arr.length - 1))];
}
