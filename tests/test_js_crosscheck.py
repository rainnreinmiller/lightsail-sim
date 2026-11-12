"""Cross-check: the browser's JavaScript engine (web/js/physics.js) must reproduce the Python
reference implementation. Runs the engine under Node.js."""

import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lightsail import GW, GRAM, LY, SailParams, baseline, MODELS, CLASSICAL
from lightsail import physics as ph
from lightsail.sensitivity import elasticities

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

JS_PARAM = {  # python field -> js field
    "power": "power", "array_size": "arraySize", "wavelength": "wavelength",
    "circular_array": "circularArray", "beam_quality": "beamQuality", "beam_efficiency": "beamEfficiency",
    "sail_shape": "sailShape", "reflectivity": "reflectivity", "absorption": "absorption",
    "emissivity_front": "emissivityFront", "emissivity_back": "emissivityBack",
    "payload_mass": "payloadMass", "target_distance": "targetDistance", "illumination_stop": "illuminationStop",
}

SENS_MAP = {"thickness": "arealDensity", "payload_mass": "payloadMass", "sail_size": "sailSize",
            "reflectivity": "reflectivity", "power": "power", "beam_quality": "beamQuality"}


def to_js(p: SailParams) -> dict:
    d = {js: getattr(p, py) for py, js in JS_PARAM.items()}
    d["arealDensity"] = p.areal_density
    d["optimalSail"] = p.sail_size is None
    d["sailSize"] = p.sail_size if p.sail_size is not None else 1.0
    if math.isinf(d["illuminationStop"]):
        d["illuminationStop"] = None
    return d


def cases():
    rng = np.random.default_rng(7)
    out = [baseline(), baseline().with_(illumination_stop=5.0), baseline().with_(illumination_stop=0.3),
           baseline().with_(power=1e4 * GW), baseline().with_(power=1e3),
           SailParams(power=20 * GW, array_size=3000, wavelength=800e-9, circular_array=False,
                      beam_efficiency=0.8, sail_size=2.5, thickness=0.3e-6, density=3100,
                      sail_shape="circular", reflectivity=0.9, absorption=0.5, payload_mass=0.5e-3,
                      target_distance=4.2465 * LY)]
    for _ in range(12):
        out.append(baseline().with_(
            power=10 ** rng.uniform(8, 13), array_size=10 ** rng.uniform(2.5, 5),
            thickness=10 ** rng.uniform(-8, -5), payload_mass=10 ** rng.uniform(-5, -1),
            reflectivity=rng.uniform(0, 1), absorption=rng.uniform(0, 1),
            sail_size=None if rng.uniform() < 0.5 else 10 ** rng.uniform(-1, 1.5),
            beam_quality=rng.uniform(1, 2), illumination_stop=math.inf if rng.uniform() < 0.6 else 10 ** rng.uniform(-0.5, 3)))
    return out


@pytest.fixture(scope="module")
def js_results(tmp_path_factory):
    ps = cases()
    payload = []
    for p in ps:
        pts = np.geomspace(p.L0 * 1e-6, p.target_distance, 25)
        payload.append({"params": to_js(p), "points": pts.tolist()})
    f = tmp_path_factory.mktemp("js") / "cases.json"
    f.write_text(json.dumps(payload))
    res = subprocess.run([NODE, str(ROOT / "web/js/crosscheck.mjs"), str(f)], capture_output=True, text=True, check=True)
    return ps, payload, json.loads(res.stdout)


def rel(a, b):
    return abs(a / b - 1)


def test_derived_quantities_match(js_results):
    for p, _, r in zip(*js_results):
        d = r["derived"]
        assert rel(d["D"], p.D) < 1e-13
        assert rel(d["totalMass"], p.total_mass) < 1e-13
        assert rel(d["a0"], p.a0) < 1e-13
        assert rel(d["L0"], p.L0) < 1e-13


def test_summary_and_trajectory_points_match(js_results):
    for p, case, r in zip(*js_results):
        for m in MODELS:
            js = r["models"][m]
            py = ph.mission_summary(p, m)
            for k_js, k_py in (("betaL0", "beta_L0"), ("betaFinal", "beta_final"), ("tL0", "t_L0"),
                               ("tTarget", "t_target"), ("tauTarget", "tau_target")):
                assert rel(js["summary"][k_js], py[k_py]) < 1e-10, (m, k_js)
            x = np.array(case["points"])
            beta = ph.beta_of_x(x, p.a0, p.L0, p.x_stop, m)
            t, tau = ph.time_to_distance(x, p.a0, p.L0, p.x_stop, m)
            acc = ph.proper_acceleration(x, p.a0, p.L0, p.x_stop, m)
            for i, pt in enumerate(js["points"]):
                assert rel(pt["beta"], beta[i]) < 1e-10
                assert rel(pt["t"], t[i]) < 1e-10
                assert rel(pt["tau"], tau[i]) < 1e-10
                assert abs(pt["accel"] - acc[i]) <= 1e-10 * p.a0


def test_cumulative_trajectory_grid_matches(js_results):
    """The animation's cumulative trajectory (one GL panel per grid step) vs Python."""
    for p, _, r in zip(*js_results):
        for m in MODELS:
            for pt in r["models"][m]["traj"]:
                if pt["x"] <= 0:
                    continue
                t, tau = ph.time_to_distance(pt["x"], p.a0, p.L0, p.x_stop, m)
                assert rel(pt["t"], float(t)) < 1e-9
                assert rel(pt["tau"], float(tau)) < 1e-9


def test_elasticities_match(js_results):
    for p, _, r in zip(*js_results):
        for m in MODELS:
            for out, key in (("beta_final", "elasticities"), ("t_target", "elasticitiesT")):
                py = elasticities(p, out, m, params=list(SENS_MAP))
                for k_py, k_js in SENS_MAP.items():
                    assert py[k_py] == pytest.approx(r["models"][m][key][k_js], abs=2e-6), (m, out, k_py)
