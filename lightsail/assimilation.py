"""Live Bayesian assimilation of tracking measurements (relativistic model).

The sail's motion (with the laser left on, x_stop = inf) depends only on two quantities: the
peak acceleration a0 and the diffraction distance L0. Doppler tracking therefore constrains
(a0, L0), and through them the predicted limiting speed and arrival time, but not the
individual inputs (areal density, payload, reflectivity, power, beam quality) separately.

Method
------
* Prior: Gaussian in theta = (ln a0, ln L0), from the input uncertainty spec by first-order
  (delta-method) propagation around the design. Deterministic, so the Python and JavaScript
  implementations agree exactly.
* Posterior on a G x G grid in prior-whitened coordinates (+-5 sigma). Each record adds a
  Gaussian log-likelihood. When the posterior becomes narrower than a few grid cells, the grid is
  re-centred and re-scaled on the posterior and all stored records are re-applied.
* Predicted speed: the dimensionless solution beta(tau; beta0), tau = t/t0, t0 = sqrt(2 L0/a0),
  beta0 = sqrt(2 a0 L0)/c, is universal. It is tabulated once (`UniversalTable`) from the closed
  forms and interpolated bilinearly in (ln beta0, ln tau).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .constants import C
from .params import SailParams
from .physics import (RELATIVISTIC, beta_from_u, doppler_minus_one, time_to_distance)
from .uncertainty import DEFAULT_SPEC, input_moments

GL_X, GL_W = np.polynomial.legendre.leggauss(8)


class UniversalTable:
    """ln beta on a grid of (ln beta0, ln tau) for the relativistic sail with the laser left on."""
    NB, NT, NNEAR, NFAR = 128, 2400, 400, 1600
    B0_MIN, B0_MAX = 1e-3, 2.0
    TAU_MIN, TAU_MAX = 1e-4, 1e9
    S_MAX = np.log(1e14)
    _shared = None

    def __init__(self):
        self.lnb0 = np.linspace(np.log(self.B0_MIN), np.log(self.B0_MAX), self.NB)
        self.lntau = np.linspace(np.log(self.TAU_MIN), np.log(self.TAU_MAX), self.NT)
        self.table = np.vstack([self._row(np.exp(lb)) for lb in self.lnb0])

    @classmethod
    def shared(cls) -> "UniversalTable":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def _row(self, b0: float) -> np.ndarray:
        # units with t0 = 1 s: a0 = beta0 c, L0 = beta0 c / 2 (so c/a0 = 1/beta0)
        a0, L0 = b0 * C, b0 * C / 2
        uL = float(doppler_minus_one(3 * b0 * b0))
        u = np.geomspace(uL * 1e-9, uL, self.NNEAR)
        t_near = (u ** 3 + 3 * u ** 2 + 6 * u) / (6 * b0)
        beta_near = beta_from_u(u)
        s = np.linspace(0.0, self.S_MAX, self.NFAR + 1)
        h = s[1] - s[0]
        nodes = s[:-1, None] + (GL_X[None, :] + 1) / 2 * h
        x = L0 * np.exp(nodes)
        beta = beta_from_u(doppler_minus_one(6 * a0 * (2 * L0 - L0 * L0 / x) / C ** 2))
        dt = (x / (C * beta)) @ (GL_W / 2 * h)
        t_far = t_near[-1] + np.cumsum(dt)
        xe = L0 * np.exp(s[1:])
        beta_far = beta_from_u(doppler_minus_one(6 * a0 * (2 * L0 - L0 * L0 / xe) / C ** 2))
        lt = np.log(np.concatenate([t_near, t_far]))
        lb = np.log(np.concatenate([beta_near, beta_far]))
        row = np.interp(self.lntau, lt, lb)
        below = self.lntau < lt[0]
        row[below] = lb[0] + (self.lntau[below] - lt[0])     # beta proportional to tau at early times
        return row

    def ln_beta(self, ln_b0, ln_tau):
        """Bilinear interpolation (clamped to the table)."""
        fb = (np.asarray(ln_b0) - self.lnb0[0]) / (self.lnb0[1] - self.lnb0[0])
        ft = (np.asarray(ln_tau) - self.lntau[0]) / (self.lntau[1] - self.lntau[0])
        fb = np.clip(fb, 0, self.NB - 1 - 1e-9)
        ft = np.clip(ft, 0, self.NT - 1 - 1e-9)
        ib, it = np.floor(fb).astype(int), np.floor(ft).astype(int)
        wb, wt = fb - ib, ft - it
        T = self.table
        return ((1 - wb) * ((1 - wt) * T[ib, it] + wt * T[ib, it + 1])
                + wb * ((1 - wt) * T[ib + 1, it] + wt * T[ib + 1, it + 1]))

    def beta(self, t, a0, L0):
        """Predicted relativistic beta at Earth time t for peak acceleration a0 and distance L0."""
        a0, L0 = np.asarray(a0, float), np.asarray(L0, float)
        b0 = np.sqrt(2 * a0 * L0) / C
        t0 = np.sqrt(2 * L0 / a0)
        return np.exp(self.ln_beta(np.log(b0), np.log(t) - np.log(t0)))


def log_prior_moments(design: SailParams, spec=DEFAULT_SPEC, h: float = 1e-6):
    """Mean and covariance of theta = (ln a0, ln L0) by first-order propagation of the spec."""
    p = design.resolved()
    mom = input_moments(p, spec)
    pm = replace(p, **{k: mu for k, (mu, sd) in mom.items()})

    def theta(q):
        return np.array([np.log(q.a0), np.log(q.L0)])

    mu = theta(pm)
    J = np.zeros((2, len(mom)))
    var = np.zeros(len(mom))
    for j, (k, (m, sd)) in enumerate(mom.items()):
        dv = m * h
        J[:, j] = (theta(replace(pm, **{k: m + dv})) - theta(replace(pm, **{k: m - dv}))) / (2 * dv)
        var[j] = sd * sd
    return mu, (J * var) @ J.T


def chol2(S):
    """Cholesky factor of a 2x2 SPD matrix (explicit, identical in the JS port)."""
    l11 = np.sqrt(S[0, 0])
    l21 = S[1, 0] / l11
    l22 = np.sqrt(max(S[1, 1] - l21 * l21, 1e-300))
    return np.array([[l11, 0.0], [l21, l22]])


def weighted_quantile(v, w, qs):
    idx = np.argsort(v, kind="stable")
    v, w = np.asarray(v)[idx], np.asarray(w)[idx]
    c = np.cumsum(w)
    c = (c - 0.5 * w) / c[-1]
    return np.interp(qs, c, v)


class Assimilator:
    """Grid posterior over (ln a0, ln L0), updated record by record."""

    def __init__(self, design: SailParams, spec=DEFAULT_SPEC, grid: int = 96, span: float = 5.0,
                 table: UniversalTable | None = None):
        self.design = replace(design.resolved(), illumination_stop=np.inf)
        self.table = table or UniversalTable.shared()
        self.G, self.span = grid, span
        self.mu, self.cov = log_prior_moments(self.design, spec)
        self.Lp = chol2(self.cov)
        self.Lp_inv = np.linalg.inv(self.Lp)
        self.records = []
        self.regrids = 0
        self._set_grid(self.mu, self.Lp)
        self.prior = self.summary()

    # ------------------------------------------------------------------ grid
    def _set_grid(self, center, M):
        z = np.linspace(-self.span, self.span, self.G)
        Z1, Z2 = np.meshgrid(z, z, indexing="ij")
        Z = np.vstack([Z1.ravel(), Z2.ravel()])
        TH = center[:, None] + M @ Z
        self.center, self.M = center, M
        self.ln_a0, self.ln_L0 = TH
        d = self.Lp_inv @ (TH - self.mu[:, None])
        self.log_prior = -0.5 * (d * d).sum(axis=0)
        self.ln_b0 = 0.5 * (np.log(2.0) + self.ln_a0 + self.ln_L0) - np.log(C)
        self.ln_t0 = 0.5 * (np.log(2.0) + self.ln_L0 - self.ln_a0)
        self.loglik = np.zeros_like(self.log_prior)
        for r in self.records:
            self.loglik += self._ll(r)
        self._ttarget = None

    GC = 24   # coarse lattice for travel times (exact there, bilinear in grid coordinates elsewhere)

    def _travel_times(self):
        if self._ttarget is None:
            zc = np.linspace(-self.span, self.span, self.GC)
            Z1, Z2 = np.meshgrid(zc, zc, indexing="ij")
            TH = self.center[:, None] + self.M @ np.vstack([Z1.ravel(), Z2.ravel()])
            tt, _ = time_to_distance(self.design.target_distance, np.exp(TH[0]), np.exp(TH[1]),
                                     np.inf, RELATIVISTIC)
            lnT = np.log(tt).reshape(self.GC, self.GC)
            f = np.linspace(0, self.GC - 1, self.G)              # fine grid index -> coarse coordinate
            i = np.minimum(np.floor(f).astype(int), self.GC - 2)
            u = f - i
            I, J = np.meshgrid(np.arange(self.G), np.arange(self.G), indexing="ij")
            a, b = i[I.ravel()], i[J.ravel()]
            wa, wb = u[I.ravel()], u[J.ravel()]
            self._ttarget = np.exp((1 - wa) * ((1 - wb) * lnT[a, b] + wb * lnT[a, b + 1])
                                   + wa * ((1 - wb) * lnT[a + 1, b] + wb * lnT[a + 1, b + 1]))
        return self._ttarget

    def _pred(self, t):
        return np.exp(self.table.ln_beta(self.ln_b0, np.log(t) - self.ln_t0))

    def _ll(self, r):
        e = (r.beta - self._pred(r.t)) / r.sigma
        return -0.5 * e * e

    def weights(self):
        lp = self.log_prior + self.loglik
        w = np.exp(lp - lp.max())
        return w / w.sum()

    def _posterior_moments(self, w):
        th = np.vstack([self.ln_a0, self.ln_L0])
        m = th @ w
        d = th - m[:, None]
        S = (d * w) @ d.T
        # never narrower than the grid resolution (cell variance h^2/12 in grid units)
        h = 2 * self.span / (self.G - 1)
        S = S + self.M @ (np.eye(2) * h * h / 12) @ self.M.T
        return m, S

    def _maybe_regrid(self):
        for _ in range(6):
            w = self.weights()
            m, S = self._posterior_moments(w)
            Minv = np.linalg.inv(self.M)
            sd_z = np.sqrt(np.diag(Minv @ S @ Minv.T))
            zm = Minv @ (m - self.center)
            h = 2 * self.span / (self.G - 1)
            if sd_z.min() > 4 * h and np.all(np.abs(zm) + 3 * sd_z < self.span):
                return
            self._set_grid(m, chol2(S) * 1.2)
            self.regrids += 1

    # ------------------------------------------------------------------ public API
    def update(self, records):
        for r in records:
            self.records.append(r)
            self.loglik += self._ll(r)
        self._maybe_regrid()

    def predictive_band(self, times, qs=(0.05, 0.5, 0.95)):
        w = self.weights()
        return np.array([weighted_quantile(self._pred(t), w, qs) for t in np.atleast_1d(times)])

    def summary(self, travel_time: bool = True) -> dict:
        w = self.weights()
        a0, L0 = np.exp(self.ln_a0), np.exp(self.ln_L0)
        b0 = np.exp(self.ln_b0)
        binf = beta_from_u(doppler_minus_one(6 * b0 * b0))
        out = {"n": len(self.records), "regrids": self.regrids}
        for name, v in (("a0", a0), ("L0", L0), ("beta_inf", binf)):
            mean = float(v @ w)
            out[name] = {"mean": mean, "sd": float(np.sqrt(((v - mean) ** 2) @ w)),
                         **dict(zip(("p5", "p50", "p95"), map(float, weighted_quantile(v, w, (0.05, 0.5, 0.95)))))}
        if travel_time:
            q = weighted_quantile(self._travel_times(), w, (0.05, 0.5, 0.95))
            out["t_target"] = dict(zip(("p5", "p50", "p95"), map(float, q)))
        return out
