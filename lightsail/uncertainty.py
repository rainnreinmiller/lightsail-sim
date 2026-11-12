"""Uncertainty propagation: Monte Carlo sampling and first-order (linear) propagation.

Each uncertain input is described relative to the nominal design (`DEFAULT_SPEC`). The
budgets below are illustrative assumptions for a gram-scale sail, not measured tolerances:

* sail areal density (thickness)  ±10 % (1σ, normal)   fabrication tolerance of a ~1 µm film
* payload mass                    ±10 % (1σ, normal)
* sail size                       ±2 %  (1σ, normal)
* reflectivity                    uniform 0.95 - 1.00  dielectric-stack performance across the
                                                       Doppler-shifted band
* laser power delivered           ±5 %  (1σ, normal)
* beam quality factor (×alpha)    uniform 1.0 - 1.5   beam not perfectly diffraction-limited

Wavelength and array size are treated as exactly known.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .params import SailParams
from .physics import CLASSICAL, RELATIVISTIC
from .sensitivity import elasticities, output

DEFAULT_SPEC = {
    "thickness":    ("normal_rel", 0.10),
    "payload_mass": ("normal_rel", 0.10),
    "sail_size":    ("normal_rel", 0.02),
    "reflectivity": ("uniform", 0.95, 1.00),
    "power":        ("normal_rel", 0.05),
    "beam_quality": ("uniform", 1.0, 1.5),
}


def _dist(spec_entry, nominal):
    kind = spec_entry[0]
    if kind == "normal_rel":
        return "normal", nominal, nominal * spec_entry[1]
    if kind == "uniform":
        a, b = spec_entry[1], spec_entry[2]
        return "uniform", 0.5 * (a + b), (b - a) / np.sqrt(12.0)
    raise ValueError(kind)


def input_moments(p: SailParams, spec=DEFAULT_SPEC) -> dict:
    """name -> (mean, std) of each uncertain input."""
    p = p.resolved()
    return {k: _dist(v, getattr(p, k))[1:] for k, v in spec.items()}


def percentile_range(p: SailParams, spec=DEFAULT_SPEC, q=(5, 95)) -> dict:
    """name -> (low, high) input values at the given percentiles (used for the tornado chart)."""
    from scipy import stats
    p = p.resolved()
    out = {}
    for k, v in spec.items():
        if v[0] == "normal_rel":
            nom = getattr(p, k)
            z = stats.norm.ppf(np.array(q) / 100.0)
            out[k] = tuple(nom * (1 + v[1] * z))
        else:
            a, b = v[1], v[2]
            out[k] = tuple(a + (b - a) * np.array(q) / 100.0)
    return out


def sample_inputs(p: SailParams, spec=DEFAULT_SPEC, n=100_000, seed=20261112) -> dict:
    p = p.resolved()
    rng = np.random.default_rng(seed)
    samples = {}
    for k, v in spec.items():
        nom = getattr(p, k)
        if v[0] == "normal_rel":
            # truncate at 4 sigma to keep every sample physical (positive masses, powers ...)
            z = np.clip(rng.standard_normal(n), -4, 4)
            samples[k] = nom * (1.0 + v[1] * z)
        else:
            samples[k] = rng.uniform(v[1], v[2], n)
    return samples


def monte_carlo(p: SailParams, spec=DEFAULT_SPEC, n=100_000, seed=20261112,
                models=(CLASSICAL, RELATIVISTIC)) -> dict:
    """Propagate input uncertainty through both models. Returns samples and output arrays."""
    p = p.resolved()
    samples = sample_inputs(p, spec, n, seed)
    pa = replace(p, **samples)              # SailParams with array-valued fields
    res = {"inputs": samples, "n": n}
    for m in models:
        t, tau = output(pa, "t_target", m), output(pa, "tau_target", m)
        res[m] = {"beta_final": output(pa, "beta_final", m), "t_target": t, "tau_target": tau}
    return res


def summarize(x, q=(5, 50, 95)) -> dict:
    x = np.asarray(x)
    pct = np.percentile(x, q)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
            **{f"p{qq}": float(v) for qq, v in zip(q, pct)}}


def linear_propagation(p: SailParams, spec=DEFAULT_SPEC, out="t_target", model=RELATIVISTIC) -> dict:
    """First-order propagation: (sigma_Y / Y)^2 = sum_i (S_i sigma_i / mu_i)^2, evaluated at the input means."""
    p = p.resolved()
    mom = input_moments(p, spec)
    p_mean = replace(p, **{k: mu for k, (mu, sd) in mom.items()})
    S = elasticities(p_mean, out, model, params=list(spec))
    y = float(output(p_mean, out, model))
    contrib = {k: (S[k] * sd / mu) ** 2 for k, (mu, sd) in mom.items()}
    rel = float(np.sqrt(sum(contrib.values())))
    return {"y": y, "rel_std": rel, "std": rel * y, "elasticities": S,
            "variance_share": {k: v / rel ** 2 for k, v in contrib.items()}}
