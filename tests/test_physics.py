"""Validation of the lightsail model.

1. Reproduces the UCSB 1-D laser propulsion calculators (formulas transcribed from
   https://www.deepspace.ucsb.edu/wp-content/uploads/2023/05/Laser-Propulsion-Classical-1D-Standalone.html
   and the relativistic page of the same name).
2. Checks the closed-form solutions against direct ODE integration (solve_ivp).
3. Checks physical limits and identities.
"""

import math

import numpy as np
import pytest

from lightsail import C, GW, GRAM, LY, SailParams, baseline, CLASSICAL, RELATIVISTIC, MODELS
from lightsail.constants import SIGMA_SB
from lightsail import physics as ph
from lightsail.simulate import simulate
from lightsail.sensitivity import elasticities, classical_elasticities_analytic
from lightsail.uncertainty import monte_carlo, linear_propagation, summarize


# ----------------------------------------------------------------------------- UCSB calculators
def calculator_classical(m0, h, rho, D, xi, eps_r, alpha_abs, eps_beam, P, d, lam, alpha,
                         L_target, eps_f=1.0, eps_b=1.0):
    """The classical calculator's output formulas, transcribed verbatim (SI units)."""
    m_sail = xi * D ** 2 * h * rho
    m = m0 + m_sail
    eff = 2 * eps_r + (1 - eps_r) * alpha_abs           # hiddens.total_light_efficiency
    P0 = eps_beam * P
    flux = P0 / (xi * D ** 2)
    out = {
        "m_sail": m_sail, "m": m,
        "force": eff * P0 / C,
        "a": eff * P0 / (m * C),
        "L0": d * D / (2 * lam * alpha),
        "t0": math.sqrt(C * d * D * m / (eff * P0 * lam * alpha)),
        "v0": math.sqrt(eff * P0 * d * D / (C * lam * alpha * m)),
        "T_eq": (flux * (1 - eps_r) * alpha_abs / (SIGMA_SB * (eps_f + eps_b))) ** 0.25,
    }
    out["v_inf"] = math.sqrt(2) * out["v0"]
    out["time_to_target"] = L_target / out["v_inf"]
    return out


CASES = [
    # calculator defaults: optimal square sail, circular 10 km array, 100 GW, 1060 nm
    (baseline(), dict(m0=1e-3, h=1e-6, rho=1000, D=1.0, xi=1.0, eps_r=1, alpha_abs=1, eps_beam=1,
                      P=100 * GW, d=1e4, lam=1060e-9, alpha=1.22, L_target=4.37 * LY)),
    # a non-default case exercising every option
    (SailParams(power=20 * GW, array_size=3_000, wavelength=800e-9, circular_array=False,
                beam_efficiency=0.8, sail_size=2.5, thickness=0.3e-6, density=3100,
                sail_shape="circular", reflectivity=0.9, absorption=0.5, payload_mass=0.5e-3,
                target_distance=4.2465 * LY),
     dict(m0=0.5e-3, h=0.3e-6, rho=3100, D=2.5, xi=math.pi / 4, eps_r=0.9, alpha_abs=0.5,
          eps_beam=0.8, P=20 * GW, d=3000, lam=800e-9, alpha=1.0, L_target=4.2465 * LY)),
]


@pytest.mark.parametrize("p,kw", CASES)
def test_reproduces_classical_calculator(p, kw):
    ref = calculator_classical(**kw)
    s = ph.mission_summary(p, CLASSICAL)
    assert p.sail_mass == pytest.approx(ref["m_sail"], rel=1e-12)
    assert p.total_mass == pytest.approx(ref["m"], rel=1e-12)
    assert p.force == pytest.approx(ref["force"], rel=1e-12)
    assert p.a0 == pytest.approx(ref["a"], rel=1e-12)
    assert p.L0 == pytest.approx(ref["L0"], rel=1e-12)
    assert p.equilibrium_temperature == pytest.approx(ref["T_eq"], rel=1e-12)
    assert s["t_L0"] == pytest.approx(ref["t0"], rel=1e-12)
    assert s["v_L0"] == pytest.approx(ref["v0"], rel=1e-12)
    assert s["v_final"] == pytest.approx(ref["v_inf"], rel=1e-12)
    # the calculator ignores the (short) acceleration phase: coast-only time must match exactly,
    # and the full travel time differs by less than 1e-5
    assert s["t_target_coast_only"] == pytest.approx(ref["time_to_target"], rel=1e-12)
    assert s["t_target"] == pytest.approx(ref["time_to_target"], rel=1e-5)


# (beta_non_relativistic, ratio) pairs from the relativistic calculator's lookup table
# `beta_non_relativistic_to_ratio`; beta_relativistic = beta_nr / ratio, with the ratio tabulated
# to two decimals (so the table itself is only good to ~0.6 %).
CALC_TABLE = [(0.0101, 1.0067), (0.0515, 1.0300), (0.1070, 1.0700), (0.1680, 1.1200),
              (0.2340, 1.1700), (0.3025, 1.2100), (0.3840, 1.2800), (0.5720, 1.4300),
              (0.8150, 1.6300), (1.1520, 1.9200), (2.0475, 2.7300), (3.4000, 4.0000),
              (8.6925, 9.1500), (30.3633, 30.6700)]


@pytest.mark.parametrize("beta_nr,ratio", CALC_TABLE)
def test_relativistic_speed_at_L0_matches_calculator_table(beta_nr, ratio):
    ours = float(ph.beta_classical_to_relativistic(beta_nr))
    assert ours == pytest.approx(beta_nr / ratio, rel=7e-3)


@pytest.mark.parametrize("beta0", [0.01, 0.1, 0.1515, 0.3, 0.6, 0.9, 0.99])
def test_relativistic_limiting_speed_matches_calculator_formula(beta0):
    """Calculator: value = (2-4b0) sqrt(1-b0^2) / (3 (1-b0)^2) - 1/3 is looked up in a table of
    1/3 - G(beta_inf) (e.g. 0.3062 -> 0.2, -3 -> 0.8). Our closed form gives G(b_inf) = 2 G(b0)."""
    value = (2 - 4 * beta0) * math.sqrt(1 - beta0 ** 2) / (3 * (1 - beta0) ** 2) - 1 / 3
    assert value == pytest.approx(1 / 3 - 2 * float(ph.doppler_integral(beta0)), rel=1e-12, abs=1e-14)
    # and the calculator's table entries satisfy value = 1/3 - G(beta_inf)
    for v, b in [(0.3062, 0.2), (-3.0, 0.8), (0.0984, 0.45), (-11.6237, 0.9)]:
        assert 1 / 3 - float(ph.doppler_integral(b)) == pytest.approx(v, abs=2e-4)
    # end to end: a sail that reaches beta0 at L0 coasts at beta_inf with G(beta_inf) = 2 G(beta0)
    a0 = 3.0e5
    L0 = float(ph.doppler_integral(beta0)) * C ** 2 / a0
    b_inf = float(ph.beta_terminal(a0, L0))
    assert float(ph.beta_of_x(L0, a0, L0)) == pytest.approx(beta0, rel=1e-12)
    assert float(ph.doppler_integral(b_inf)) == pytest.approx(2 * float(ph.doppler_integral(beta0)), rel=1e-10)


def test_relativistic_time_to_L0_differs_from_calculator_expression():
    """The calculator's relativistic 'Time to L0' is t0_cl/(3b)((1+b)(2-b)gamma/(1-b) - 2).
    Algebraically this equals our closed form multiplied by beta_classical/beta_relativistic.
    Our closed form is confirmed by direct ODE integration below, so the calculator's value is
    too long by that factor (about 12 % for the default inputs)."""
    p = baseline()
    b_cl = p.beta0_classical
    b = float(ph.beta_of_x(p.L0, p.a0, p.L0))
    g = 1 / math.sqrt(1 - b * b)
    t_calc = p.t0_classical / (3 * b) * ((1 + b) * (2 - b) * g / (1 - b) - 2)
    t_ours = ph.mission_summary(p, RELATIVISTIC)["t_L0"]
    t_ode = simulate(p, RELATIVISTIC, x_end=p.L0).events["L0"][0]
    assert t_ours == pytest.approx(t_ode, rel=1e-9)
    assert t_calc == pytest.approx(t_ours * b_cl / b, rel=1e-12)


# ----------------------------------------------------------------------------- closed form vs ODE
ODE_CASES = [
    baseline(),
    baseline().with_(power=2000 * GW),                       # strongly relativistic
    baseline().with_(illumination_stop=10.0),                # laser off at 10 L0
    baseline().with_(illumination_stop=0.4),                 # laser off before L0
    baseline().with_(power=1 * GW, reflectivity=0.5, sail_shape="circular", circular_array=False),
]


@pytest.mark.parametrize("p", ODE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_closed_form_matches_ode(p, model):
    tr = simulate(p, model)
    m = tr.x > 0
    beta = ph.beta_of_x(tr.x[m], p.a0, p.L0, p.x_stop, model)
    t, tau = ph.time_to_distance(tr.x[m], p.a0, p.L0, p.x_stop, model)
    assert np.max(np.abs(beta / tr.beta[m] - 1)) < 1e-8
    assert np.max(np.abs(t / tr.t[m] - 1)) < 1e-8
    assert np.max(np.abs(tau / tr.tau[m] - 1)) < 1e-8
    acc = ph.proper_acceleration(tr.x[m], p.a0, p.L0, p.x_stop, model)
    assert np.allclose(acc, tr.accel[m], rtol=1e-7, atol=1e-12 * p.a0)


def test_classical_far_field_quadrature_matches_closed_form():
    p = baseline()
    x = np.geomspace(p.L0, 1e7 * p.L0, 50)
    t_cf, _ = ph.time_to_distance(x, p.a0, p.L0, model=CLASSICAL)
    t_q, _ = ph.time_to_distance(x, p.a0, p.L0, model=CLASSICAL, quadrature_classical=True)
    assert np.max(np.abs(t_q / t_cf - 1)) < 1e-12


# ----------------------------------------------------------------------------- physics checks
def test_doppler_cubic_is_solved_accurately():
    q = np.geomspace(1e-24, 1e8, 400)
    u = ph.doppler_minus_one(q)
    assert np.max(np.abs(u * u * (u + 3) / q - 1)) < 1e-12


def test_low_speed_relativistic_reduces_to_classical():
    p = baseline().with_(power=1e3)        # 1 kW: beta ~ 1e-5
    s_c, s_r = ph.mission_summary(p, CLASSICAL), ph.mission_summary(p, RELATIVISTIC)
    assert s_r["beta_final"] == pytest.approx(s_c["beta_final"], rel=3e-5)
    assert s_r["t_L0"] == pytest.approx(s_c["t_L0"], rel=3e-5)


def test_speed_never_reaches_c():
    p = baseline().with_(power=1e18, payload_mass=1e-6, thickness=1e-9)
    b = ph.beta_terminal(p.a0, p.L0, model=RELATIVISTIC)
    assert 0.99 < b < 1
    assert ph.beta_terminal(p.a0, p.L0, model=CLASSICAL) > 1     # the classical model fails


def test_limiting_speed_is_sqrt2_times_speed_at_L0_classically():
    p = baseline()
    assert ph.beta_terminal(p.a0, p.L0, model=CLASSICAL) == pytest.approx(math.sqrt(2) * p.beta0_classical, rel=1e-14)


def test_work_energy_theorem():
    p = baseline().with_(power=500 * GW)
    for model in MODELS:
        tr = simulate(p, model, x_end=50 * p.L0, n_samples=20000)
        F = p.total_mass * tr.accel                          # longitudinal force (frame invariant)
        work = np.sum(0.5 * (F[1:] + F[:-1]) * np.diff(tr.x))
        ke = p.total_mass * C ** 2 * (tr.gamma[-1] - 1) if model == RELATIVISTIC \
            else 0.5 * p.total_mass * tr.v[-1] ** 2
        assert work == pytest.approx(ke, rel=1e-5)


def test_optimal_sail_maximizes_limiting_speed():
    p = baseline()
    D_opt = p.D
    Ds = np.linspace(0.3, 3, 2701)
    for model in MODELS:
        b = [float(ph.beta_terminal(q.a0, q.L0, model=model)) for q in (p.with_(sail_size=D) for D in Ds)]
        assert Ds[int(np.argmax(b))] == pytest.approx(D_opt, abs=2e-3)


def test_proper_time_shorter_than_earth_time():
    p = baseline().with_(power=1000 * GW)
    s = ph.mission_summary(p, RELATIVISTIC)
    g = 1 / math.sqrt(1 - s["beta_final"] ** 2)
    assert s["tau_target"] < s["t_target"]
    assert s["tau_target"] == pytest.approx(s["t_target"] / g, rel=1e-3)


# ----------------------------------------------------------------------------- sensitivity / uncertainty
def test_numerical_elasticities_match_analytic_classical():
    p = baseline().with_(reflectivity=0.9, sail_size=1.3)
    num = elasticities(p, "beta_final", CLASSICAL)
    ana = classical_elasticities_analytic(p)
    for k in ana:
        assert num[k] == pytest.approx(ana[k], abs=1e-7)


def test_monte_carlo_agrees_with_linear_propagation():
    p = baseline()
    mc = monte_carlo(p, n=40_000, seed=1)
    for model in MODELS:
        s = summarize(mc[model]["t_target"])
        lp = linear_propagation(p, out="t_target", model=model)
        assert s["mean"] == pytest.approx(lp["y"], rel=0.01)
        assert s["std"] == pytest.approx(lp["std"], rel=0.10)
    # reproducible
    mc2 = monte_carlo(p, n=40_000, seed=1)
    assert np.array_equal(mc[RELATIVISTIC]["t_target"], mc2[RELATIVISTIC]["t_target"])
