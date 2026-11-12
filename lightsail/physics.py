"""Equations of motion of an idealized 1-D laser-driven lightsail, with closed-form solutions.

Every function is NumPy-vectorized: arguments broadcast, so a Monte Carlo sample of 10^5
parameter sets is evaluated in one call.

Model
-----
The fraction of the beam intercepted by the sail is

    f(x) = 1            for x <= L0           (spot smaller than the sail)
         = (L0/x)^2     for L0 < x <= x_stop  (diffraction-limited spot larger than the sail)
         = 0            for x > x_stop        (laser switched off)

Classical (Newtonian) equation of motion, peak acceleration a0 = eta P0 / (m c):

    dv/dt = a0 f(x)

Relativistic equation of motion: in the sail's instantaneous rest frame the received power is
Doppler-reduced by (1-beta)/(1+beta) (photon energy and arrival rate are each reduced by
sqrt((1-beta)/(1+beta))). Absorbed energy is re-radiated isotropically in the sail frame, so the
rest mass is constant, and the longitudinal force is frame-invariant:

    m c d(gamma beta)/dt = (eta P0 / c) f(x) (1 - beta)/(1 + beta)

Closed forms
------------
With the work integral W(x) = integral_0^x f dx' (W = x for x <= L0, 2 L0 - L0^2/x beyond):

* classical:     v = sqrt(2 a0 W)                         -> v_inf = sqrt(2) v0 (Lubin 2016)
* relativistic:  energy theorem m c^2 dgamma = F dx, with gamma = cosh(phi) and
                 (1-beta)/(1+beta) = exp(-2 phi), integrates to
                     (Dop - 1)^2 (Dop + 2) = 6 a0 W / c^2,   Dop = sqrt((1+beta)/(1-beta))
                 a cubic in the Doppler factor Dop, solved here in closed form.

In the near field (x <= L0) the time and proper time are also closed form:
                     t   = (c / a0) (Dop^3 + 3 Dop - 4) / 6
                     tau = (c / a0) (Dop^2 - 1) / 2
In the far field the classical time has a closed form; the relativistic one is a 1-D quadrature
(Gauss-Legendre panels in ln x), done in `time_to_distance`.
"""

from __future__ import annotations

import numpy as np

from .constants import C

CLASSICAL = "classical"
RELATIVISTIC = "relativistic"
MODELS = (CLASSICAL, RELATIVISTIC)

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


# --------------------------------------------------------------------------- beam geometry
def flux_fraction(x, L0, x_stop=np.inf):
    """Fraction of the beam power intercepted by the sail at distance x."""
    x, L0, x_stop = _arr(x, L0, x_stop)
    with np.errstate(divide="ignore", invalid="ignore"):
        far = (L0 / x) ** 2
    f = np.where(x <= L0, 1.0, far)
    return np.where(x > x_stop, 0.0, f)


def work_integral(x, L0, x_stop=np.inf):
    """W(x) = integral of f from 0 to x [m]. Work done on the sail per unit of near-field force."""
    x, L0, x_stop = _arr(x, L0, x_stop)
    xe = np.minimum(x, x_stop)
    with np.errstate(divide="ignore", invalid="ignore"):
        far = 2.0 * L0 - L0 ** 2 / xe
    return np.where(xe <= L0, xe, far)


def terminal_work(L0, x_stop=np.inf):
    """W after the laser switches off (2 L0 for illumination to infinity)."""
    return work_integral(np.inf, L0, x_stop)


# --------------------------------------------------------------------------- Doppler cubic
def doppler_minus_one(q):
    """Solve u^2 (u + 3) = q for u = Dop - 1 >= 0 (equivalently (Dop-1)^2 (Dop+2) = q).

    Trigonometric root for q <= 4, hyperbolic root for q > 4; written so that small q
    (beta << 1) keeps full relative precision.
    """
    q = np.asarray(q, dtype=float)
    qs = np.clip(q, 0.0, 4.0)
    psi = (2.0 / 3.0) * np.arcsin(np.sqrt(qs) / 2.0)
    u_small = -2.0 * np.sin(psi / 2.0) ** 2 + np.sqrt(3.0) * np.sin(psi)
    s = np.maximum(q / 2.0 - 1.0, 1.0)
    u_large = 2.0 * np.cosh(np.arccosh(s) / 3.0) - 1.0
    return np.where(q <= 4.0, u_small, u_large)


def beta_from_u(u):
    return u * (u + 2.0) / (u * u + 2.0 * u + 2.0)


def gamma_from_u(u):
    return (u * u + 2.0 * u + 2.0) / (2.0 * (1.0 + u))


def gammabeta_from_u(u):
    return u * (u + 2.0) / (2.0 * (1.0 + u))


def u_from_beta(beta):
    """Inverse of beta_from_u: Dop - 1 = sqrt((1+beta)/(1-beta)) - 1."""
    beta = np.asarray(beta, dtype=float)
    return np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0


def doppler_integral(beta):
    """G(beta) = (Dop^3 - 3 Dop + 2)/6 = a0 W / c^2 for the relativistic sail."""
    u = u_from_beta(beta)
    return u * u * (u + 3.0) / 6.0


# --------------------------------------------------------------------------- speed vs distance
def beta_of_x(x, a0, L0, x_stop=np.inf, model=RELATIVISTIC):
    """Speed / c at distance x (closed form). The classical result may exceed 1 (unphysical)."""
    W = work_integral(x, L0, x_stop)
    a0 = np.asarray(a0, dtype=float)
    if model == CLASSICAL:
        return np.sqrt(2.0 * a0 * W) / C
    return beta_from_u(doppler_minus_one(6.0 * a0 * W / C ** 2))


def gamma_of_x(x, a0, L0, x_stop=np.inf, model=RELATIVISTIC):
    if model == CLASSICAL:
        return np.ones(np.broadcast(*_arr(x, a0, L0, x_stop)).shape)
    W = work_integral(x, L0, x_stop)
    return gamma_from_u(doppler_minus_one(6.0 * np.asarray(a0, float) * W / C ** 2))


def beta_terminal(a0, L0, x_stop=np.inf, model=RELATIVISTIC):
    """Final coasting speed / c after illumination ends (x_stop = inf: limiting speed)."""
    return beta_of_x(np.inf, a0, L0, x_stop, model)


def beta_classical_to_relativistic(beta_cl):
    """Map a classical speed to the relativistic one for the same work done: the relation used
    (as a lookup table) by the UCSB relativistic calculator. 3 beta_cl^2 = (Dop-1)^2 (Dop+2)."""
    return beta_from_u(doppler_minus_one(3.0 * np.asarray(beta_cl, float) ** 2))


def proper_acceleration(x, a0, L0, x_stop=np.inf, model=RELATIVISTIC):
    """Acceleration felt by the sail [m/s^2]: a0 f(x) (1-beta)/(1+beta) (classical: a0 f(x))."""
    f = flux_fraction(x, L0, x_stop) * np.asarray(a0, dtype=float)
    if model == CLASSICAL:
        return f
    u = doppler_minus_one(6.0 * np.asarray(a0, float) * work_integral(x, L0, x_stop) / C ** 2)
    return f / (1.0 + u) ** 2


# --------------------------------------------------------------------------- time vs distance
def _near_field_time(x, a0, model):
    """(t, tau) to reach x <= L0 from rest."""
    if model == CLASSICAL:
        t = np.sqrt(2.0 * x / a0)
        return t, t
    u = doppler_minus_one(6.0 * a0 * x / C ** 2)
    t = (C / a0) * (u ** 3 + 3.0 * u ** 2 + 6.0 * u) / 6.0
    tau = (C / a0) * u * (u + 2.0) / 2.0
    return t, tau


def _far_field_classical(y, L0, a0):
    """Classical time from L0 to x = y L0 (closed form)."""
    v0 = np.sqrt(2.0 * a0 * L0)

    def F(w):
        return w * np.sqrt(w * w + 1.0) + np.arcsinh(w)

    w = np.sqrt(2.0 * y - 1.0)
    return (L0 / v0) * (F(w) - F(1.0)) / (2.0 * np.sqrt(2.0))


def _far_field_quadrature(y, L0, a0, model, panels_per_unit=2.0):
    """(t, tau) from L0 to x = y L0 by Gauss-Legendre panels in s = ln(x/L0). Vectorized."""
    y, L0, a0 = np.broadcast_arrays(*_arr(y, L0, a0))
    shape = y.shape
    y, L0, a0 = y.ravel(), L0.ravel(), a0.ravel()
    S = np.log(np.maximum(y, 1.0))
    if S.size == 0:
        return np.zeros(shape), np.zeros(shape)
    M = max(1, int(np.ceil(S.max() * panels_per_unit)))
    # node positions as a fraction of S: panel m covers [m/M, (m+1)/M]
    frac = ((np.arange(M)[:, None] + (_GL_NODES[None, :] + 1.0) / 2.0) / M).ravel()
    wts = np.tile(_GL_WEIGHTS / (2.0 * M), M)
    t = np.empty_like(S)
    tau = np.empty_like(S)
    chunk = max(1, 4_000_000 // frac.size)
    for i in range(0, S.size, chunk):
        sl = slice(i, i + chunk)
        s = S[sl, None] * frac[None, :]
        x = L0[sl, None] * np.exp(s)
        jac = x * S[sl, None]                 # dx = x ds,  ds = S dfrac
        W = 2.0 * L0[sl, None] - L0[sl, None] ** 2 / x
        if model == CLASSICAL:
            beta = np.sqrt(2.0 * a0[sl, None] * W) / C
            gb = beta
        else:
            u = doppler_minus_one(6.0 * a0[sl, None] * W / C ** 2)
            beta = beta_from_u(u)
            gb = gammabeta_from_u(u)
        t[sl] = (jac / (C * beta)) @ wts
        tau[sl] = (jac / (C * gb)) @ wts
    return t.reshape(shape), tau.reshape(shape)


def time_to_distance(x, a0, L0, x_stop=np.inf, model=RELATIVISTIC, quadrature_classical=False):
    """Earth-frame time t and ship proper time tau [s] for the sail to travel from rest to x.

    Near field: closed form. Far field: closed form (classical) or Gauss-Legendre quadrature
    (relativistic). After the laser switches off at x_stop the sail coasts at constant speed.
    """
    x, a0, L0, x_stop = np.broadcast_arrays(*_arr(x, a0, L0, x_stop))
    x1 = np.minimum(np.minimum(x, L0), x_stop)
    t, tau = _near_field_time(x1, a0, model)

    x2 = np.minimum(x, x_stop)
    far = x2 > L0
    y = np.where(far, x2 / L0, 1.0)
    if model == CLASSICAL and not quadrature_classical:
        tf = _far_field_classical(y, L0, a0)
        tauf = tf
    else:
        tf, tauf = _far_field_quadrature(y, L0, a0, model)
    t = t + np.where(far, tf, 0.0)
    tau = tau + np.where(far, tauf, 0.0)

    coast = x > x_stop
    if np.any(coast):
        xs = np.where(coast, x_stop, 0.0)
        b = beta_of_x(xs, a0, L0, x_stop, model)
        g = gamma_of_x(xs, a0, L0, x_stop, model)
        dx = np.where(coast, x - x_stop, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = t + np.where(coast, dx / (C * b), 0.0)
            tau = tau + np.where(coast, dx / (C * b * g), 0.0)
    return t, tau


# --------------------------------------------------------------------------- conveniences on SailParams
def mission_summary(p, model=RELATIVISTIC):
    """Key outputs for one parameter set (closed form + quadrature, no ODE solver)."""
    a0, L0, xs, X = p.a0, p.L0, p.x_stop, p.target_distance
    b_L0 = float(beta_of_x(L0, a0, L0, xs, model))
    b_inf = float(beta_terminal(a0, L0, xs, model))
    t_L0, _ = time_to_distance(L0, a0, L0, xs, model)
    t_X, tau_X = time_to_distance(X, a0, L0, xs, model)
    return {
        "model": model,
        "a0": a0, "L0": L0,
        "beta_L0": b_L0, "v_L0": b_L0 * C, "t_L0": float(t_L0),
        "beta_final": b_inf, "v_final": b_inf * C,
        "t_target": float(t_X), "tau_target": float(tau_X),
        "t_target_coast_only": X / (b_inf * C),
    }
