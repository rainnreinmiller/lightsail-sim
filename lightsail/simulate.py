"""Direct numerical integration of the equations of motion with scipy.integrate.solve_ivp.

This is independent of the closed forms in `physics.py`; the test suite checks that both agree.

State (dimensionless): X = x / L0, U = gamma*beta (relativistic) or beta (classical), and the
ship proper time. Time unit T = L0 / c. With k = a0 L0 / c^2 = beta0^2 / 2:

    dX/dT   = beta
    dU/dT   = k f(X) D(beta)        D = (1-beta)/(1+beta) relativistic, 1 classical
    dtau/dT = 1/gamma

The run is split at the breakpoints X = 1 (L0, where f has a kink), X = x_stop / L0
(laser off) and the target, each located with a terminal event. Integrator: DOP853 (8th-order
Dormand-Prince), rtol = 1e-11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .constants import C
from .physics import CLASSICAL, RELATIVISTIC, flux_fraction
from .params import SailParams


@dataclass
class Trajectory:
    model: str
    t: np.ndarray          # Earth-frame time [s]
    x: np.ndarray          # distance [m]
    beta: np.ndarray       # v / c
    gamma: np.ndarray
    tau: np.ndarray        # ship proper time [s]
    accel: np.ndarray      # acceleration felt by the sail (proper acceleration) [m/s^2]
    events: dict = field(default_factory=dict)   # name -> (t, x, beta, tau)
    nfev: int = 0

    @property
    def v(self):
        return self.beta * C


def _rhs_factory(k, X_stop, model):
    def rhs(T, y):
        X, U, _tau = y
        if model == CLASSICAL:
            beta, gamma, dop = U, 1.0, 1.0
        else:
            gamma = np.sqrt(1.0 + U * U)
            beta = U / gamma
            dop = (1.0 - beta) / (1.0 + beta)
        f = float(flux_fraction(X, 1.0, X_stop))
        return [beta, k * f * dop, 1.0 / gamma]
    return rhs


def simulate(p: SailParams, model: str = RELATIVISTIC, x_end: float | None = None,
             n_samples: int = 600, rtol: float = 1e-11) -> Trajectory:
    """Integrate from rest at x = 0 to x_end (default: the target distance)."""
    L0 = p.L0
    k = p.a0 * L0 / C ** 2
    X_stop = p.x_stop / L0
    X_end = (x_end if x_end is not None else p.target_distance) / L0
    rhs = _rhs_factory(k, X_stop, model)

    breakpoints = sorted({b for b in (1.0, X_stop, X_end) if 0 < b <= X_end})
    names = (("L0", 1.0), ("laser_off", X_stop), ("end", X_end))

    y = np.array([0.0, 0.0, 0.0])
    T = 0.0
    segs_T, segs_Y, events, nfev = [], [], {}, 0
    for b in breakpoints:
        if y[0] >= b:
            continue

        def hit(T_, y_, b=b):
            return y_[0] - b
        hit.terminal, hit.direction = True, 1

        # generous upper bound on the time needed to reach b (the event stops the run)
        beta_lo = max(_beta_of(y, model), 1e-3 * np.sqrt(k))
        T_max = T + 100.0 * (np.sqrt(2.0 * b / k) + (b - y[0]) / beta_lo)
        sol = solve_ivp(rhs, (T, T_max), y, method="DOP853", rtol=rtol,
                        atol=[1e-14, 1e-16, 1e-14], events=hit, dense_output=True)
        if sol.status != 1:
            raise RuntimeError(f"integration did not reach X={b}: {sol.message}")
        nfev += sol.nfev
        T_hit = sol.t_events[0][0]
        y = sol.y_events[0][0].copy()
        y[0] = b  # remove event-location round-off
        # sample the segment on a log-spaced time grid for plotting
        Ts = np.unique(np.concatenate([
            np.geomspace(max(T, T_hit * 1e-9, 1e-12), T_hit, n_samples // len(breakpoints)),
            sol.t]))
        Ts = Ts[(Ts >= T) & (Ts <= T_hit)]
        segs_T.append(Ts)
        segs_Y.append(sol.sol(Ts))
        T = T_hit
        for name, val in names:
            if val == b:
                events[name] = (T * L0 / C, b * L0, _beta_of(y, model), y[2] * L0 / C)

    Tall = np.concatenate(segs_T)
    Yall = np.concatenate(segs_Y, axis=1)
    X, U, TAU = Yall
    if model == CLASSICAL:
        beta, gamma = U, np.ones_like(U)
        dop = np.ones_like(U)
    else:
        gamma = np.sqrt(1.0 + U * U)
        beta = U / gamma
        dop = (1.0 - beta) / (1.0 + beta)
    accel = p.a0 * flux_fraction(X, 1.0, X_stop) * dop
    return Trajectory(model=model, t=Tall * L0 / C, x=X * L0, beta=beta, gamma=gamma,
                      tau=TAU * L0 / C, accel=accel, events=events, nfev=nfev)


def _beta_of(y, model):
    U = y[1]
    return U if model == CLASSICAL else U / np.sqrt(1.0 + U * U)
