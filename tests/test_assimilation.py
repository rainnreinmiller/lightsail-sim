"""Live assimilation of (SIMULATED) tracking measurements: generator, universal table, estimator."""

import numpy as np
import pytest

from lightsail import PRESETS, YEAR
from lightsail import physics as ph
from lightsail.assimilation import Assimilator, UniversalTable, log_prior_moments, weighted_quantile
from lightsail.telemetry import (from_csv, sample_truth, simulated_stream, state_at_times, to_csv)
from lightsail.simulate import simulate

DESIGN = PRESETS["Kulkarni et al. 2018, Table 1 (1 g)"]


@pytest.fixture(scope="module")
def table():
    return UniversalTable.shared()


def test_state_at_times_matches_ode():
    truth = sample_truth(DESIGN, seed=1)
    tr = simulate(truth, "relativistic", x_end=1e4 * truth.L0)
    sel = tr.t > 0.5
    x, b = state_at_times(truth, tr.t[sel])
    assert np.max(np.abs(x / tr.x[sel] - 1)) < 1e-6
    assert np.max(np.abs(b / tr.beta[sel] - 1)) < 1e-6


@pytest.mark.parametrize("power,tol", [(1e10, 2e-4), (1e11, 2e-4), (1e12, 3e-4), (1e13, 3e-4)])
def test_universal_table_matches_exact_solution(table, power, tol):
    q = DESIGN.with_(power=power, illumination_stop=np.inf)
    ts = np.geomspace(0.5, 1e9, 500)
    _, b = state_at_times(q, ts, n=20000)
    assert np.max(np.abs(table.beta(ts, q.a0, q.L0) / b - 1)) < tol


def test_noiseless_stream_equals_model_and_is_ordered_by_light_time():
    truth = sample_truth(DESIGN, seed=2)
    recs = simulated_stream(truth, noise=0.0, seed=2)
    t = np.array([r.t for r in recs])
    _, b = state_at_times(truth, t)
    assert np.allclose([r.beta for r in recs], b, rtol=0, atol=1e-15)
    rec = [r.received_at for r in recs]
    assert rec == sorted(rec)
    assert all(r.received_at >= r.t for r in recs)
    assert 200 < len(recs) < 320


def test_csv_round_trip_and_speed_columns():
    truth = sample_truth(DESIGN, seed=4)
    recs = simulated_stream(truth, seed=4)[:20]
    back = from_csv(to_csv(recs))
    assert [(r.t, r.beta, r.sigma) for r in back] == [(r.t, r.beta, r.sigma) for r in recs]
    alt = from_csv("t_s,v_m_s,sigma\n10,3e6,6e5\n20,6e6,6e5\n")
    assert alt[0].beta == pytest.approx(3e6 / 299792458.0)
    assert alt[0].received_at == 10


def test_low_noise_stream_recovers_truth():
    truth = sample_truth(DESIGN, seed=11)
    A = Assimilator(DESIGN)
    A.update(simulated_stream(truth, noise=2e-4, seed=11))
    s = A.summary()
    assert s["a0"]["p50"] == pytest.approx(truth.a0, rel=0.01)
    assert s["L0"]["p50"] == pytest.approx(truth.L0, rel=0.03)
    t_true = float(ph.time_to_distance(truth.target_distance, truth.a0, truth.L0)[0])
    assert s["t_target"]["p5"] <= t_true * 1.002 and s["t_target"]["p95"] >= t_true * 0.998
    # the prediction is far sharper than the prior
    w_prior = A.prior["t_target"]["p95"] - A.prior["t_target"]["p5"]
    w_post = s["t_target"]["p95"] - s["t_target"]["p5"]
    assert w_post < 0.05 * w_prior


def test_credible_intervals_are_calibrated():
    """Truths drawn from the prior should fall inside the 90 % interval about 90 % of the time."""
    hits, n = 0, 40
    for seed in range(n):
        truth = sample_truth(DESIGN, seed=100 + seed)
        A = Assimilator(DESIGN)
        A.update(simulated_stream(truth, noise=0.002, seed=100 + seed))
        s = A.summary(travel_time=False)["beta_inf"]
        b_true = float(ph.beta_terminal(truth.a0, truth.L0))
        hits += s["p5"] <= b_true <= s["p95"]
    assert 0.75 <= hits / n <= 1.0


def test_near_field_data_fix_acceleration_but_not_L0():
    """Inside L0 the motion depends on a0 only, so early data cannot reveal L0 (the beam quality)."""
    truth = sample_truth(DESIGN, seed=5)
    recs = [r for r in simulated_stream(truth, noise=0.002, seed=5) if r.t < 0.5 * truth.t0_classical]
    A = Assimilator(DESIGN)
    sd_a_prior, sd_L_prior = A.prior["a0"]["sd"] / A.prior["a0"]["mean"], A.prior["L0"]["sd"] / A.prior["L0"]["mean"]
    A.update(recs)
    s = A.summary(travel_time=False)
    assert s["a0"]["sd"] / s["a0"]["mean"] < 0.3 * sd_a_prior
    assert s["L0"]["sd"] / s["L0"]["mean"] > 0.85 * sd_L_prior


def test_prior_moments_match_monte_carlo():
    from lightsail.uncertainty import monte_carlo
    mu, cov = log_prior_moments(DESIGN)
    mc = monte_carlo(DESIGN, n=40000, seed=9, models=())
    # monte_carlo returns inputs; rebuild ln a0, ln L0 from them
    from dataclasses import replace
    p = replace(DESIGN.resolved(), **mc["inputs"])
    th = np.vstack([np.log(p.a0), np.log(p.L0)])
    assert np.allclose(th.mean(axis=1), mu, atol=0.01)
    assert np.allclose(np.sqrt(np.diag(np.cov(th))), np.sqrt(np.diag(cov)), rtol=0.08)


def test_weighted_quantile():
    v = np.arange(10.0)
    assert weighted_quantile(v, np.ones(10), [0.5])[0] == pytest.approx(4.5)
