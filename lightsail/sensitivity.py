"""Sensitivity analysis: logarithmic elasticities and one-at-a-time (tornado) swings.

The elasticity S = d ln Y / d ln p is the % change of an output Y for a 1 % change of input p.
Outputs Y: 'beta_final' (coasting speed) or 't_target' (Earth-frame travel time).

Sensitivities are evaluated with the sail size D held fixed at its nominal value: a sail is built
with a given size, and an error in its thickness changes its mass, not its size.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .params import SailParams
from .physics import RELATIVISTIC, beta_terminal, time_to_distance

# name -> (label, group)
PARAMETERS = {
    "thickness":     ("Sail areal density σ", "sail"),
    "reflectivity":  ("Reflectivity R", "sail"),
    "payload_mass":  ("Payload mass m₀", "sail"),
    "sail_size":     ("Sail size D", "sail"),
    "power":         ("Laser power P", "beam"),
    "beam_quality":  ("Beam quality factor", "beam"),
    "array_size":    ("Array size d", "beam"),
    "wavelength":    ("Wavelength λ", "beam"),
}


def output(p: SailParams, name: str, model: str = RELATIVISTIC):
    """Evaluate an output for a (possibly array-valued) parameter set."""
    if name == "beta_final":
        return beta_terminal(p.a0, p.L0, p.x_stop, model)
    if name == "t_target":
        return time_to_distance(p.target_distance, p.a0, p.L0, p.x_stop, model)[0]
    if name == "tau_target":
        return time_to_distance(p.target_distance, p.a0, p.L0, p.x_stop, model)[1]
    raise ValueError(name)


def elasticities(p: SailParams, out: str = "beta_final", model: str = RELATIVISTIC,
                 params=None, h: float = 1e-4) -> dict:
    """d ln Y / d ln p by central differences (backward difference when R = 1)."""
    p = p.resolved()
    params = params or list(PARAMETERS)
    result = {}
    for name in params:
        v = getattr(p, name)
        up, dn = v * (1 + h), v * (1 - h)
        if name == "reflectivity" and up > 1.0:
            y0 = output(p, out, model)
            y1 = output(replace(p, **{name: dn}), out, model)
            result[name] = float((np.log(y0) - np.log(y1)) / (np.log(v) - np.log(dn)))
            continue
        yu = output(replace(p, **{name: up}), out, model)
        yd = output(replace(p, **{name: dn}), out, model)
        result[name] = float((np.log(yu) - np.log(yd)) / (np.log(up) - np.log(dn)))
    return result


def classical_elasticities_analytic(p: SailParams) -> dict:
    """Exact elasticities of the classical limiting speed v_inf ∝ sqrt(eta P d D / (lambda alpha m))."""
    p = p.resolved()
    fs = p.sail_mass / p.total_mass
    R, A, eta = p.reflectivity, p.absorption, p.momentum_factor
    return {
        "thickness": -0.5 * fs,
        "reflectivity": 0.5 * R * (2.0 - A) / eta,
        "payload_mass": -0.5 * (1.0 - fs),
        "sail_size": 0.5 - fs,
        "power": 0.5,
        "beam_quality": -0.5,
        "array_size": 0.5,
        "wavelength": -0.5,
    }


def tornado(p: SailParams, ranges: dict, out: str = "t_target", model: str = RELATIVISTIC):
    """One-at-a-time swings: for each input set to its (low, high) value with all others nominal.

    Returns rows sorted by swing size: (name, label, low, high, Y_low, Y_high, Y_nominal).
    """
    p = p.resolved()
    y0 = float(output(p, out, model))
    rows = []
    for name, (lo, hi) in ranges.items():
        y_lo = float(output(replace(p, **{name: lo}), out, model))
        y_hi = float(output(replace(p, **{name: hi}), out, model))
        rows.append((name, PARAMETERS[name][0], lo, hi, y_lo, y_hi, y0))
    rows.sort(key=lambda r: abs(r[5] - r[4]), reverse=True)
    return rows
