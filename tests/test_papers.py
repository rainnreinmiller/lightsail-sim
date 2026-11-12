"""Validation against the published papers the model is based on.

[R16] P. Lubin, "A Roadmap to Interstellar Flight", JBIS 69, 40-72 (2016), arXiv:1604.01356 (v8).
[K18] N. Kulkarni, P. Lubin, Q. Zhang, "Relativistic Spacecraft Propelled by Directed Energy",
      The Astronomical Journal (2018), doi:10.3847/1538-3881/aaafd2, arXiv:1710.10732.
"""

import math

import numpy as np
import pytest

from lightsail import C, GW, SailParams, CLASSICAL, RELATIVISTIC
from lightsail import physics as ph

# ----------------------------------------------------------------------------- [K18] Tables 1 and 2
# 100 GW, 10 km square array (alpha = 1), Nd:YAG 1064 nm, optimal square sail (m_sail = m0),
# sail density 1400 kg/m^3, perfect reflector. Columns as printed in the paper:
# (sail thickness [um], bare spacecraft mass m0 [kg], sail size D [m], t0 [s], L0 [m], beta0, beta_max)
K18_TABLES = [
    # Table 1: 1 um sail, varying spacecraft mass
    (1.0, 1e-3, 0.85, 164, 4.0e9, 0.15, 0.21),
    (1.0, 1e-2, 2.7, 901, 1.3e10, 0.091, 0.13),   # our beta_max 0.1247 prints as 0.12, not 0.13
    (1.0, 1e-1, 8.5, 4.99e3, 4.0e10, 0.053, 0.073),
    (1.0, 1e0, 27, 2.79e4, 1.3e11, 0.030, 0.042),
    (1.0, 1e1, 85, 1.56e5, 4.0e11, 0.017, 0.024),
    (1.0, 1e2, 270, 8.75e5, 1.3e12, 0.0096, 0.014),
    (1.0, 1e3, 850, 4.91e6, 4.0e12, 0.0054, 0.0077),
    (1.0, 1e4, 2700, 2.76e7, 1.3e13, 0.0031, 0.0043),
    (1.0, 1e5, 8500, 1.55e8, 4.0e13, 0.0017, 0.0024),
    # Table 2: 1 g spacecraft, thinner sails
    (0.01, 1e-3, 8.5, 587, 4.0e10, 0.39, 0.48),
    (0.05, 1e-3, 3.8, 370, 1.8e10, 0.29, 0.37),
    (0.1, 1e-3, 2.7, 306, 1.3e10, 0.25, 0.33),
    (0.5, 1e-3, 1.2, 198, 5.7e9, 0.18, 0.24),
]


def k18_params(h_um, m0):
    return SailParams(power=100 * GW, array_size=1e4, wavelength=1064e-9, circular_array=False,
                      sail_shape="square", thickness=h_um * 1e-6, density=1400.0, payload_mass=m0,
                      reflectivity=1.0)


@pytest.mark.parametrize("h,m0,D,t0,L0,b0,bmax", K18_TABLES)
def test_reproduces_kulkarni_2018_tables(h, m0, D, t0, L0, b0, bmax):
    """Tolerances reflect the 2-3 significant figures printed in the tables."""
    p = k18_params(h, m0)
    s = ph.mission_summary(p, RELATIVISTIC)
    assert p.D == pytest.approx(D, rel=0.015)
    assert s["t_L0"] == pytest.approx(t0, rel=0.01)        # relativistic time to L0
    assert p.L0 == pytest.approx(L0, rel=0.035)
    assert s["beta_L0"] == pytest.approx(b0, rel=0.025)
    assert s["beta_final"] == pytest.approx(bmax, rel=0.045)


def test_k18_time_to_L0_contradicts_calculator_formula():
    """[K18] Table 1 gives t0 = 164 s for the 1 g case. Our integration gives 163 s. The UCSB
    relativistic calculator's expression (its t0 x beta_cl / beta_rel) would give about 183 s."""
    p = k18_params(1.0, 1e-3)
    s = ph.mission_summary(p, RELATIVISTIC)
    b_cl, b = p.beta0_classical, s["beta_L0"]
    g = 1 / math.sqrt(1 - b * b)
    t_calc = p.t0_classical / (3 * b) * ((1 + b) * (2 - b) * g / (1 - b) - 2)
    assert abs(s["t_L0"] - 164) < 1.5
    assert t_calc > 180


# ----------------------------------------------------------------------------- [K18] equations
@pytest.mark.parametrize("beta", [1e-4, 0.01, 0.1, 0.2, 0.5, 0.9, 0.99])
def test_k18_eq11_equals_our_near_field_time(beta):
    """[K18] Eq. 11: t = (t_rel/6)[(1+b)(2-b)gamma/(1-b) - 2], t_rel = m c^2 / P (perfect mirror).
    Ours: t = (c/a0)(D^3 + 3D - 4)/6 with a0 = 2P/(mc) and Doppler factor D."""
    g = 1 / math.sqrt(1 - beta * beta)
    k18 = ((1 + beta) * (2 - beta) * g / (1 - beta) - 2) / 6            # in units of t_rel
    u = float(ph.u_from_beta(beta))
    ours = 0.5 * (u ** 3 + 3 * u ** 2 + 6 * u) / 6                       # c/a0 = t_rel/2
    assert ours == pytest.approx(k18, rel=1e-10)


def test_k18_quoted_time_to_0p2c():
    """[K18] section 2.1: reaching 0.2c takes t2 = 0.126 t_rel."""
    u = float(ph.u_from_beta(0.2))
    assert 0.5 * (u ** 3 + 3 * u ** 2 + 6 * u) / 6 == pytest.approx(0.126, abs=5e-4)


@pytest.mark.parametrize("beta0", [0.01, 0.15, 0.4, 0.8])
def test_k18_eq27_speed_at_L0(beta0):
    """[K18] Eq. 27: (2b0 - 1)/(3 gamma0 (1-b0)^2) = 2 P L0/(m c^3) - 1/3."""
    lhs = (2 * beta0 - 1) / (3 * (1 - beta0) ** 2 / math.sqrt(1 - beta0 ** 2))
    a0 = 1e5
    L0 = float(ph.doppler_integral(beta0)) * C ** 2 / a0                 # our cubic: G = a0 L0 / c^2
    assert float(ph.beta_of_x(L0, a0, L0)) == pytest.approx(beta0, rel=1e-12)
    rhs = a0 * L0 / C ** 2 - 1 / 3                                       # 2P/(mc) = a0
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-12)


def test_k18_eq26_classical_terminal_speed():
    """[K18] Eq. 25-26: v(x) = sqrt(4 P x/(m c)) inside L0, v_inf = sqrt(8 P L0/(m c)) = sqrt(2) v(L0)."""
    p = k18_params(1.0, 1e-3)
    m, P = p.total_mass, p.power
    assert float(ph.beta_of_x(0.3 * p.L0, p.a0, p.L0, model=CLASSICAL)) * C == pytest.approx(
        math.sqrt(4 * P * 0.3 * p.L0 / (m * C)), rel=1e-12)
    assert float(ph.beta_terminal(p.a0, p.L0, model=CLASSICAL)) * C == pytest.approx(
        math.sqrt(8 * P * p.L0 / (m * C)), rel=1e-12)


# ----------------------------------------------------------------------------- [R16] equations
def test_roadmap_classical_equations():
    """[R16] Eqs. 2, 5, 7, 12, 13 with F = P0 (1 + eps_r)/c (non-reflected light absorbed)."""
    p = SailParams(power=50 * GW, array_size=7e3, wavelength=1064e-9, circular_array=True,
                   sail_size=1.7, thickness=0.6e-6, density=1400, sail_shape="circular",
                   reflectivity=0.9, absorption=1.0, payload_mass=2e-3)
    eps, P0, d, D, lam, al = 0.9, 50 * GW, 7e3, 1.7, 1064e-9, 1.22
    xi, h, rho, m0 = math.pi / 4, 0.6e-6, 1400, 2e-3
    m = xi * D ** 2 * h * rho + m0                                                   # Eq. 3
    a = P0 * (1 + eps) / (m * C)                                                     # Eq. 2
    v0 = math.sqrt(P0 * (1 + eps) * d * D / (C * lam * al * (xi * D ** 2 * h * rho + m0)))   # Eq. 5
    t0 = math.sqrt(C * d * D * (xi * D ** 2 * h * rho + m0) / (P0 * (1 + eps) * lam * al))  # Eq. 12
    s = ph.mission_summary(p, CLASSICAL)
    assert p.a0 == pytest.approx(a, rel=1e-12)
    assert p.L0 == pytest.approx(d * D / (2 * lam * al), rel=1e-12)                 # Eq. 13
    assert s["v_L0"] == pytest.approx(v0, rel=1e-12)
    assert s["v_final"] == pytest.approx(math.sqrt(2) * v0, rel=1e-12)              # Eq. 7
    assert s["t_L0"] == pytest.approx(t0, rel=1e-12)


def test_roadmap_optimal_speed_scaling():
    """[R16] Eq. 9: with m_sail = m0, v_max = (P0 (1+eps) d / (c lambda alpha))^(1/2) (xi h rho m0)^(-1/4)."""
    for h, m0 in [(1e-6, 1e-3), (0.1e-6, 1e-3), (1e-6, 1.0)]:
        p = SailParams(thickness=h, density=1400, payload_mass=m0)        # optimal sail (sail_size=None)
        v = float(ph.beta_terminal(p.a0, p.L0, model=CLASSICAL)) * C
        ref = math.sqrt(p.power * 2 * p.array_size / (C * p.wavelength * p.alpha)) * (p.area_factor * h * 1400 * m0) ** -0.25
        assert v == pytest.approx(ref, rel=1e-12)


def test_roadmap_first_order_relativistic_formula_is_not_used():
    """[R16] Eqs. 22-23 give a first-order relativistic t(beta) = t_E (gamma^2 beta + atanh beta)/(2(1+eps)),
    noting that a more complete solution is given elsewhere; that solution is [K18] Eq. 11, which this
    model uses. The two agree to first order in beta but differ by ~20 % in time at 0.2c."""
    b = 0.2
    g2 = 1 / (1 - b * b)
    t_r16 = (g2 * b + math.atanh(b)) / 4                   # units of t_E = m c^2 / P, eps = 1
    u = float(ph.u_from_beta(b))
    t_k18 = 0.5 * (u ** 3 + 3 * u ** 2 + 6 * u) / 6
    assert t_r16 == pytest.approx(0.1028, abs=2e-4)
    assert t_k18 == pytest.approx(0.1259, abs=2e-4)
    small = 1e-4                                            # first-order agreement
    u = float(ph.u_from_beta(small))
    assert (1 / (1 - small ** 2) * small + math.atanh(small)) / 4 == pytest.approx(
        0.5 * (u ** 3 + 3 * u ** 2 + 6 * u) / 6, rel=1e-3)
