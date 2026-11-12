"""Tracking measurements of a lightsail: SIMULATED stream generator and CSV input/output.

No laser-propelled lightsail has flown, so no real tracking data exist. This module produces
*simulated* measurements of the kind a ground station would record, and reads real
measurements in the same format if they become available (e.g. from a laboratory experiment).

Measurement model
-----------------
The laser array tracks the sail through the Doppler shift of the light it reflects back:
lambda_received / lambda_emitted = (1 + beta)/(1 - beta). Each record gives the sail speed beta
at the sail's event time t (Earth frame, reconstructed from ranging) and the time the reflected
light reaches Earth, received_at = t + x(t)/c. Records therefore arrive with a light-time lag
that grows from milliseconds to years.

CSV format (header required; lines starting with # are comments): seq,t_s,received_s,beta,sigma
  * `beta` may be replaced by `v_m_s` (speed in m/s); `sigma` is then in m/s too.
  * `seq` and `received_s` are optional (received_s defaults to t_s).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, replace

import numpy as np

from .constants import C
from .params import SailParams
from .physics import RELATIVISTIC, beta_of_x, time_to_distance
from .uncertainty import DEFAULT_SPEC


@dataclass(frozen=True)
class Measurement:
    seq: int
    t: float             # event time at the sail [s] (Earth frame)
    received_at: float   # time the reflected light reaches Earth [s]
    beta: float          # measured speed / c
    sigma: float         # 1-sigma uncertainty of beta

    @property
    def doppler_ratio(self) -> float:
        """lambda_received / lambda_emitted for the reflected laser light."""
        return (1 + self.beta) / (1 - self.beta)


def schedule(t_end: float, per_decade: int = 30, t_start: float = 1.0) -> np.ndarray:
    """Event times log-spaced from t_start to t_end (per_decade samples per factor of 10)."""
    n = max(2, int(np.ceil(np.log10(t_end / t_start) * per_decade)) + 1)
    return np.geomspace(t_start, t_end, n)


def state_at_times(p: SailParams, times, model: str = RELATIVISTIC, n: int = 20000):
    """Distance x(t) and speed beta(t) at Earth times t (log-log interpolation of the exact t(x),
    then the closed-form beta at that x). Accurate to ~1e-7 for n = 20000."""
    times = np.asarray(times, dtype=float)
    X = p.target_distance * 1.5
    xs = np.geomspace(min(p.L0, X) * 1e-12, X, n)
    ts, _ = time_to_distance(xs, p.a0, p.L0, p.x_stop, model)
    x = np.exp(np.interp(np.log(times), np.log(ts), np.log(xs)))
    return x, beta_of_x(x, p.a0, p.L0, p.x_stop, model)


def sample_truth(design: SailParams, spec=DEFAULT_SPEC, seed: int = 0) -> SailParams:
    """Draw one 'true' sail from the uncertainty prior around a design (sail size held fixed)."""
    p = design.resolved()
    rng = np.random.default_rng(seed)
    changes = {}
    for k, v in spec.items():
        if v[0] == "normal_rel":
            changes[k] = getattr(p, k) * (1 + v[1] * float(np.clip(rng.standard_normal(), -4, 4)))
        else:
            changes[k] = float(rng.uniform(v[1], v[2]))
    return replace(p, illumination_stop=np.inf, **changes)


def simulated_stream(truth: SailParams, noise: float = 0.002, per_decade: int = 30,
                     t_start: float = 1.0, t_end: float | None = None, seed: int = 0,
                     model: str = RELATIVISTIC) -> list[Measurement]:
    """SIMULATED Doppler tracking records of a 'true' sail, ordered by reception time."""
    if t_end is None:
        t_end = float(time_to_distance(truth.target_distance, truth.a0, truth.L0, truth.x_stop, model)[0])
    t = schedule(t_end, per_decade, t_start)
    x, beta = state_at_times(truth, t, model)
    rng = np.random.default_rng(seed + 7919)
    meas = beta + noise * rng.standard_normal(t.size)
    rec = [Measurement(i, float(t[i]), float(t[i] + x[i] / C), float(meas[i]), float(noise))
           for i in range(t.size)]
    return sorted(rec, key=lambda r: r.received_at)


def to_csv(records, path=None) -> str:
    buf = io.StringIO()
    buf.write("seq,t_s,received_s,beta,sigma\n")
    for r in records:
        buf.write(f"{r.seq},{r.t!r},{r.received_at!r},{r.beta!r},{r.sigma!r}\n")
    text = buf.getvalue()
    if path is not None:
        with open(path, "w") as f:
            f.write(text)
    return text


def from_csv(text: str) -> list[Measurement]:
    """Parse measurement CSV (see module docstring); returns records sorted by reception time."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    out = []
    for i, row in enumerate(rows):
        row = {k.strip(): v for k, v in row.items() if k is not None}
        t = float(row["t_s"])
        if "beta" in row and row["beta"] not in ("", None):
            beta, sigma = float(row["beta"]), float(row["sigma"])
        else:
            beta, sigma = float(row["v_m_s"]) / C, float(row["sigma"]) / C
        rec_t = float(row["received_s"]) if row.get("received_s") not in (None, "") else t
        seq = int(row["seq"]) if row.get("seq") not in (None, "") else i
        out.append(Measurement(seq, t, rec_t, beta, sigma))
    return sorted(out, key=lambda r: r.received_at)
