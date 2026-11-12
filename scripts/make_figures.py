"""Generate the four scientific figures (PNG 300 dpi + vector PDF) and figures/results.json.

    python scripts/make_figures.py

Fig. 1  areal density   -> acceleration and limiting speed (optimal sail)
Fig. 2  reflectivity    -> momentum transfer, limiting speed and travel time
Fig. 3  classical vs relativistic motion (ODE trajectories + universal correction curve)
Fig. 4  uncertainty     -> tornado sensitivity + Monte Carlo travel-time distributions
Fig. 5  live data       -> Bayesian assimilation of a SIMULATED tracking stream
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lightsail import C, G_N, GW, GRAM, YEAR, LY, baseline, CLASSICAL, RELATIVISTIC, MODELS  # noqa: E402
from lightsail import physics as ph  # noqa: E402
from lightsail.simulate import simulate  # noqa: E402
from lightsail.sensitivity import tornado, elasticities, PARAMETERS  # noqa: E402
from lightsail.uncertainty import (monte_carlo, summarize, linear_propagation,  # noqa: E402
                                   percentile_range, input_moments, DEFAULT_SPEC)

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MODEL_COLOR = {CLASSICAL: "#2a78d6", RELATIVISTIC: "#eb6834"}   # validated categorical pair
MODEL_LABEL = {CLASSICAL: "Classical", RELATIVISTIC: "Relativistic"}
RAMP = ["#f29b74", "#e0582a", "#9c3a14"]                         # validated ordinal ramp (light→dark)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round", "figure.dpi": 110, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def dot(ax, x, y, color, z=5, s=36):
    ax.scatter([x], [y], s=s, color=color, edgecolor="white", linewidth=1.5, zorder=z)


def end_label(ax, x, y, text, dx=4, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha=ha,
                color=INK2, fontsize=8.5)


def panel_letter(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf")


def pct_formatter(digits=0):
    return FuncFormatter(lambda v, _: f"{v:.{digits}f}%")


# ----------------------------------------------------------------------------- Fig 1
def fig_areal_density(results):
    p = baseline()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)

    # (a) peak acceleration vs areal density at fixed sail size D = 1 m
    sig = np.geomspace(0.01, 100, 300) * GRAM            # kg/m^2
    for m0, col in zip((0.1, 1.0, 10.0), RAMP):
        a = [p.with_(sail_size=1.0, payload_mass=m0 * GRAM).with_areal_density(s).a0 / G_N for s in sig]
        ax1.loglog(sig / GRAM, a, color=col, label=f"m₀ = {m0:g} g")
    a_def = p.a0 / G_N
    dot(ax1, 1.0, a_def, INK)
    ax1.annotate(f"UCSB default\n{a_def:,.0f} g", (1.0, a_def), xytext=(-8, -26),
                 textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
    ax1.text(0.03, 0.04, "1 m sail, 100 GW, perfect reflector.\nSail-dominated mass: a₀ ∝ 1/σ",
             transform=ax1.transAxes, color=INK2, fontsize=8.5)
    ax1.set(xlabel="Sail areal density σ = ρh  [g/m²]", ylabel="Peak acceleration a₀  [g]",
            title="Lighter sails accelerate harder", xlim=(0.01, 100))
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    panel_letter(ax1, "a")

    # (b) relativistic limiting speed vs sail size for several areal densities (m0 = 1 g)
    D = np.geomspace(0.1, 30, 400)
    best = []
    for s_gm2, col in zip((0.1, 1.0, 10.0), RAMP):
        q = p.with_areal_density(s_gm2 * GRAM)
        b = np.array([float(ph.beta_terminal(r.a0, r.L0)) for r in (q.with_(sail_size=d) for d in D)])
        ax2.semilogx(D, 100 * b, color=col, label=f"σ = {s_gm2:g} g/m²")
        Dopt = q.D
        bopt = float(ph.beta_terminal(q.a0, q.L0))
        best.append((s_gm2, Dopt, bopt))
        dot(ax2, Dopt, 100 * bopt, col)
    # locus of optima as sigma varies continuously
    sigs = np.geomspace(0.07, 20, 60)
    locus = [(q.D, 100 * float(ph.beta_terminal(q.a0, q.L0))) for q in (p.with_areal_density(s * GRAM) for s in sigs)]
    ax2.plot(*zip(*locus), color=MUTED, lw=1.0, zorder=1)
    ax2.annotate("optimum D* = √(m₀/σ)\n(sail mass = payload)", locus[0], xytext=(-8, 2),
                 textcoords="offset points", ha="right", va="center", color=INK2, fontsize=8.5)
    ax2.set(xlabel="Sail size D  [m]", ylabel="Limiting speed v∞  [% of c]",
            title=r"Optimal sail: $v_\infty \propto (\sigma\, m_0)^{-1/4}$", xlim=(0.1, 30), ylim=(0, 40))
    ax2.yaxis.set_major_formatter(pct_formatter())
    ax2.legend(loc="upper left")
    ax2.text(0.02, 0.70, "Relativistic model\nm₀ = 1 g, 100 GW", transform=ax2.transAxes,
             ha="left", va="top", color=INK2, fontsize=8.5)
    panel_letter(ax2, "b")
    save(fig, "fig1_areal_density")
    results["fig1"] = {"optima": [{"sigma_g_m2": s, "D_opt_m": d, "beta_inf_rel": b} for s, d, b in best],
                       "a0_default_gn": a_def}


# ----------------------------------------------------------------------------- Fig 2
def fig_reflectivity(results):
    p = baseline()
    R = np.linspace(0.0, 1.0, 201)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    rows = {}
    for m in MODELS:
        b = np.array([float(ph.beta_terminal(q.a0, q.L0, model=m)) for q in (p.with_(reflectivity=r) for r in R)])
        t = np.array([float(ph.time_to_distance(q.target_distance, q.a0, q.L0, model=m)[0])
                      for q in (p.with_(reflectivity=r) for r in R)]) / YEAR
        rows[m] = (b, t)
        ax1.plot(R, 100 * b, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax2.plot(R, t, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        dot(ax1, 1.0, 100 * b[-1], MODEL_COLOR[m])
        dot(ax2, 1.0, t[-1], MODEL_COLOR[m])
        end_label(ax1, 1.0, 100 * b[-1], f"{100 * b[-1]:.1f}%", dx=-6, dy=9, ha="right")
        end_label(ax2, 1.0, t[-1], f"{t[-1]:.1f} yr", dx=-8, dy=-2, ha="right")
        end_label(ax2, 0.0, t[0], f"{t[0]:.1f} yr", dx=6, dy=0)

    ax1.set(xlabel="Reflectivity R  (non-reflected light absorbed)", ylabel="Limiting speed v∞  [% of c]",
            title="Reflection doubles the momentum per photon", xlim=(0, 1.04), ylim=(0, None))
    ax1.yaxis.set_major_formatter(pct_formatter())
    top = ax1.secondary_xaxis("top", functions=(lambda r: 1 + r, lambda e: e - 1))
    top.set_xlabel("Momentum factor η = 2R + (1−R) = 1 + R", color=INK2)
    top.tick_params(colors=MUTED, labelcolor=INK2)
    ax1.text(0.03, 0.05, "Classical: v∞ ∝ √(1+R); R: 0 → 1 gives ×√2 (+41%)",
             transform=ax1.transAxes, color=INK2, fontsize=8.5)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12))
    panel_letter(ax1, "a")

    ax2.set(xlabel="Reflectivity R", ylabel="Travel time to α Cen (4.37 ly)  [yr]",
            title="…and shortens the trip by 26–29%", xlim=(-0.02, 1.04))
    top2 = ax2.secondary_xaxis("top", functions=(lambda r: (1 - r) * p.power / GW, lambda a: 1 - a * GW / p.power))
    top2.set_xlabel("Power absorbed by the sail (1−R)P  [GW]  (thermal load)", color=INK2)
    top2.tick_params(colors=MUTED, labelcolor=INK2)
    ax2.legend(loc="upper right")
    panel_letter(ax2, "b")
    save(fig, "fig2_reflectivity")
    results["fig2"] = {m: {"beta_inf_R0": float(rows[m][0][0]), "beta_inf_R1": float(rows[m][0][-1]),
                           "t_target_R0_yr": float(rows[m][1][0]), "t_target_R1_yr": float(rows[m][1][-1])}
                       for m in MODELS}


# ----------------------------------------------------------------------------- Fig 3
def fig_classical_vs_relativistic(results):
    p = baseline()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    out = {}
    for P, tag in ((100 * GW, "100 GW"), (10_000 * GW, "10 TW")):
        q = p.with_(power=P)
        for m in MODELS:
            tr = simulate(q, m, x_end=1e5 * q.L0, n_samples=1500)
            sel = tr.t > 0
            ax1.semilogx(tr.t[sel], tr.beta[sel], color=MODEL_COLOR[m],
                         label=MODEL_LABEL[m] if tag == "100 GW" else None)
            tL0, xL0, bL0, _ = tr.events["L0"]
            dot(ax1, tL0, bL0, MODEL_COLOR[m])
            out[f"{tag} {m}"] = {"t_L0_s": tL0, "beta_L0": float(bL0), "beta_end": float(tr.beta[-1]),
                                 "ode_nfev": tr.nfev}
        end_label(ax1, 3e5, out[f"{tag} {CLASSICAL}"]["beta_end"], f"{tag}, classical", dx=-4, dy=8, ha="right")
        end_label(ax1, 3e5, out[f"{tag} {RELATIVISTIC}"]["beta_end"], f"{tag}, relativistic", dx=-4, dy=-8, ha="right")
    ax1.axhline(1.0, color=INK, lw=0.8)
    ax1.text(1.2, 1.03, "speed of light", color=INK2, fontsize=8.5, va="bottom")
    ax1.set(xlabel="Time since launch t  [s]  (Earth frame)", ylabel="Speed β = v/c",
            title="Relativity caps the speed below c", xlim=(1, 3e5), ylim=(0, 2.6))
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax1.text(0.98, 0.42, "dots: sail reaches L0\n(beam spot = sail size)", transform=ax1.transAxes,
             ha="right", color=INK2, fontsize=8.5)
    panel_letter(ax1, "a")

    # (b) universal correction: depends only on beta0 = classical speed at L0
    b0 = np.geomspace(1e-3, 30, 400)
    ratio_L0 = ph.beta_classical_to_relativistic(b0) / b0
    # at infinity: classical v_inf = sqrt(2) b0; relativistic from the cubic with W = 2 L0
    binf_rel = ph.beta_from_u(ph.doppler_minus_one(6.0 * b0 ** 2))
    ratio_inf = binf_rel / (np.sqrt(2) * b0)
    ax2.semilogx(b0, ratio_L0, color=RAMP[0], label="at L0")
    ax2.semilogx(b0, ratio_inf, color=RAMP[2], label="limiting speed (x → ∞)")
    tab = np.array([(0.0515, 1.03), (0.107, 1.07), (0.168, 1.12), (0.234, 1.17), (0.3025, 1.21),
                    (0.384, 1.28), (0.572, 1.43), (0.815, 1.63), (1.152, 1.92), (2.0475, 2.73),
                    (3.4, 4.0), (8.6925, 9.15), (17.8654, 18.23)])
    ax2.scatter(tab[:, 0], 1 / tab[:, 1], s=22, facecolor="white", edgecolor=INK2, linewidth=1.2,
                zorder=6, label="UCSB calculator lookup table")
    bb = p.beta0_classical
    rb = float(ph.beta_classical_to_relativistic(bb) / bb)
    rinf = float(ph.beta_from_u(ph.doppler_minus_one(6 * bb ** 2)) / (np.sqrt(2) * bb))
    dot(ax2, bb, rinf, INK)
    ax2.annotate(f"UCSB default inputs\nβ₀ = {bb:.3f}: v∞ lower by {100 * (1 - rinf):.1f}%",
                 (bb, rinf), xytext=(-10, -34), textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
    ax2.axvspan(1 / np.sqrt(2), 30, color=GRID, alpha=0.5, lw=0)
    ax2.text(0.75, 0.93, "classical v∞ > c", color=INK2, fontsize=8.5)
    ax2.set(xlabel="β₀ = classical speed at L0 / c  =  √(2 a₀ L₀) / c", ylabel="v(relativistic) / v(classical)",
            title="One parameter, β₀, sets the correction", xlim=(1e-3, 30), ylim=(0, 1.05))
    ax2.legend(loc="lower left")
    panel_letter(ax2, "b")
    save(fig, "fig3_classical_vs_relativistic")
    results["fig3"] = {"trajectories": out, "baseline_beta0": bb, "ratio_L0": rb, "ratio_inf": rinf}


# ----------------------------------------------------------------------------- Fig 4
def fig_uncertainty(results):
    from dataclasses import replace
    from matplotlib.patches import Patch
    p = baseline().resolved()
    # centre the analysis on the expected (mean) inputs, the same centre the Monte Carlo has
    centre = replace(p, **{k: mu for k, (mu, sd) in input_moments(p, DEFAULT_SPEC).items()})
    rng = percentile_range(p, DEFAULT_SPEC, (5, 95))
    rows = tornado(centre, rng, "t_target", RELATIVISTIC)
    mc = monte_carlo(p, DEFAULT_SPEC, n=100_000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) tornado chart, relativistic travel time
    y0 = rows[0][6] / YEAR
    fmt = {"thickness": lambda v: f"{v * 1e6:.2f} µm", "payload_mass": lambda v: f"{v / GRAM:.2f} g",
           "sail_size": lambda v: f"{v:.2f} m", "reflectivity": lambda v: f"R = {v:.3f}",
           "power": lambda v: f"{v / GW:.0f} GW", "beam_quality": lambda v: f"×{v:.2f}"}
    n = len(rows)
    for i, (name, label, lo, hi, ylo, yhi, _) in enumerate(rows):
        yi = n - 1 - i
        for val, yv, col in ((lo, ylo / YEAR, RAMP[0]), (hi, yhi / YEAR, RAMP[2])):
            left, width = min(y0, yv), abs(yv - y0)
            ax1.barh(yi, width, left=left, height=0.55, color=col, edgecolor="white", linewidth=1)
            if width > 0.03:
                side = 1 if yv > y0 else -1
                ax1.annotate(fmt[name](val), (yv, yi), xytext=(4 * side, 0), textcoords="offset points",
                             ha="left" if side > 0 else "right", va="center", fontsize=7.5, color=INK2)
        if abs(yhi - ylo) / YEAR < 0.03:
            ax1.annotate("≈ 0 at the optimal sail size", (y0, yi), xytext=(6, 0), textcoords="offset points",
                         va="center", fontsize=7.5, color=INK2)
        group = PARAMETERS[name][1]
        ax1.text(-0.02, yi, label + ("  (beam)" if group == "beam" else ""), transform=ax1.get_yaxis_transform(),
                 ha="right", va="center", fontsize=8.5, color=INK)
    ax1.axvline(y0, color=INK, lw=0.8)
    ax1.set_yticks([])
    ax1.grid(axis="y", visible=False)
    ax1.set_ylim(-1.6, n - 0.4)
    ax1.set(xlabel=f"Travel time to α Cen  [yr]  (relativistic; centre = {y0:.1f} yr at mean inputs)",
            title="Which input uncertainty matters?", xlim=(y0 - 3.4, y0 + 3.4))
    ax1.spines["left"].set_visible(False)
    ax1.legend(handles=[Patch(color=RAMP[0], label="input at its 5th percentile"),
                        Patch(color=RAMP[2], label="input at its 95th percentile")],
               loc="lower right", ncol=2, fontsize=8, bbox_to_anchor=(1.0, -0.02))
    panel_letter(ax1, "a")

    # (b) Monte Carlo distributions
    bins = np.linspace(14, 32, 121)
    summ, hmax = {}, 0.0
    hist = {}
    for m in MODELS:
        t = mc[m]["t_target"] / YEAR
        h, _ = np.histogram(t, bins=bins, density=True)
        hist[m] = t
        hmax = max(hmax, h.max())
    for k, m in enumerate(MODELS):
        t = hist[m]
        s = summarize(t)
        lp = linear_propagation(p, DEFAULT_SPEC, "t_target", m)
        summ[m] = {"mc": s, "linear_mean_yr": lp["y"] / YEAR, "linear_std_yr": lp["std"] / YEAR,
                   "variance_share": lp["variance_share"],
                   "beta_final": summarize(mc[m]["beta_final"])}
        col = MODEL_COLOR[m]
        ax2.hist(t, bins=bins, density=True, histtype="stepfilled", color=col, alpha=0.12)
        ax2.hist(t, bins=bins, density=True, histtype="step", color=col, lw=1.4,
                 label=f"{MODEL_LABEL[m]}: median {s['p50']:.1f} yr, 90% [{s['p5']:.1f}, {s['p95']:.1f}]")
        ybar = hmax * (1.10 + 0.17 * k)
        ax2.plot([s["p5"], s["p95"]], [ybar, ybar], color=col, lw=2.2, solid_capstyle="round")
        dot(ax2, s["p50"], ybar, col)
        ax2.errorbar(lp["y"] / YEAR, ybar + hmax * 0.065, xerr=lp["std"] / YEAR, fmt="none", ecolor=INK2,
                     elinewidth=1.0, capsize=3)
    ax2.set(xlabel="Travel time to α Cen  [yr]  (Earth frame)", ylabel="Probability density  [1/yr]",
            title=f"Monte Carlo, N = {mc['n']:,}", xlim=(15, 31), ylim=(0, hmax * 1.75))
    ax2.legend(loc="upper right", fontsize=8)
    panel_letter(ax2, "b")
    save(fig, "fig4_uncertainty")
    results["fig4"] = {"tornado": [{"param": r[0], "low": r[2], "high": r[3], "t_low_yr": r[4] / YEAR,
                                    "t_high_yr": r[5] / YEAR} for r in rows],
                       "centre_t_yr": y0, "monte_carlo": summ,
                       "spec": {k: list(v) for k, v in DEFAULT_SPEC.items()}}


# ----------------------------------------------------------------------------- Fig 5
def fig_assimilation(results):
    """Live Bayesian assimilation of a SIMULATED Doppler tracking stream (no real data exist)."""
    from lightsail import PRESETS
    from lightsail.assimilation import Assimilator
    from lightsail.telemetry import sample_truth, simulated_stream, state_at_times
    MEASURED, PRIOR = "#ab4999", "#898781"
    design = PRESETS["Kulkarni et al. 2018, Table 1 (1 g)"]
    truth = sample_truth(design, seed=1)
    recs = simulated_stream(truth, noise=0.002, seed=1)
    A = Assimilator(design)
    tgrid = np.geomspace(0.5, 2e9, 160)
    prior_band = A.predictive_band(tgrid)
    hist = []
    for r in recs:
        A.update([r])
        s = A.summary()
        hist.append((r.received_at, s["t_target"]["p5"], s["t_target"]["p50"], s["t_target"]["p95"]))
    post_band = A.predictive_band(tgrid)
    hist = np.array(hist)
    t_true = float(ph.time_to_distance(truth.target_distance, truth.a0, truth.L0)[0])
    tL0 = float(ph.time_to_distance(truth.L0, truth.a0, truth.L0)[0])
    _, b_true = state_at_times(truth, tgrid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    ax1.fill_between(tgrid, prior_band[:, 0], prior_band[:, 2], color=PRIOR, alpha=0.18, lw=0, label="prior 90 %")
    ax1.fill_between(tgrid, post_band[:, 0], post_band[:, 2], color=MODEL_COLOR[RELATIVISTIC], alpha=0.25, lw=0)
    ax1.plot(tgrid, post_band[:, 1], color=MODEL_COLOR[RELATIVISTIC], label="fit to all records (median, 90 %)")
    ax1.plot(tgrid, b_true, color=INK, lw=0.9, label="hidden true sail")
    t_m = np.array([r.t for r in recs]); b_m = np.array([r.beta for r in recs])
    ax1.errorbar(t_m, b_m, yerr=0.002, fmt="o", ms=2.6, color=MEASURED, ecolor=MEASURED, elinewidth=0.6,
                 alpha=0.85, label="SIMULATED measurements ± 1σ", zorder=4)
    ax1.axvline(tL0, color=MEASURED, lw=0.8)
    ax1.text(tL0 * 1.15, 0.02, "sail passes L₀", color=INK2, fontsize=8.5)
    ax1.set_xscale("log")
    ax1.set(xlabel="Event time at the sail t  [s]", ylabel="Speed β = v/c", xlim=(0.5, 2e9), ylim=(-0.01, 0.235),
            title="Simulated Doppler tracking stream")
    ax1.legend(loc="lower right", fontsize=8)
    panel_letter(ax1, "a")

    yr = YEAR
    pr = A.prior["t_target"]
    ax2.axhspan(pr["p5"] / yr, pr["p95"] / yr, color=PRIOR, alpha=0.18, lw=0, label="prior 90 %")
    ax2.fill_between(hist[:, 0], hist[:, 1] / yr, hist[:, 3] / yr, color=MODEL_COLOR[RELATIVISTIC], alpha=0.25, lw=0, step="post")
    ax2.plot(hist[:, 0], hist[:, 2] / yr, color=MODEL_COLOR[RELATIVISTIC], drawstyle="steps-post", label="prediction (median, 90 %)")
    ax2.axhline(t_true / yr, color=INK, lw=0.9, label="hidden truth")
    rL0 = tL0 + truth.L0 / C
    ax2.axvline(rL0, color=MEASURED, lw=0.8)
    ax2.text(rL0 * 1.15, pr["p95"] / yr + 0.15, "data from beyond L₀\nstart arriving", color=INK2, fontsize=8.5, va="bottom")
    ax2.set_xscale("log")
    ax2.set(xlabel="Time the data reach Earth  [s]", ylabel="Predicted travel time to α Cen  [yr]",
            title="The prediction sharpens as data arrive", xlim=(0.8, hist[-1, 0] * 1.5),
            ylim=(pr["p5"] / yr - 1.9, pr["p95"] / yr + 1.4))
    ax2.legend(loc="lower right", fontsize=8, ncol=3)
    panel_letter(ax2, "b")
    save(fig, "fig5_live_assimilation")

    def width_at(t):
        i = np.searchsorted(hist[:, 0], t) - 1
        return float((hist[i, 3] - hist[i, 1]) / yr) if i >= 0 else float((pr["p95"] - pr["p5"]) / yr)
    s = A.summary()
    results["fig5"] = {
        "simulated": True, "n_records": len(recs), "noise_beta": 0.002,
        "truth": {"a0": truth.a0, "L0": truth.L0, "t_target_yr": t_true / yr},
        "prior_t90_width_yr": float((pr["p95"] - pr["p5"]) / yr),
        "width_at_L0_reception_yr": width_at(rL0),
        "width_after_1h_yr": width_at(3600.0), "width_after_1d_yr": width_at(86400.0),
        "final": {"t_target_yr": {k: v / yr for k, v in s["t_target"].items()},
                  "a0_rel_sd": s["a0"]["sd"] / s["a0"]["mean"], "L0_rel_sd": s["L0"]["sd"] / s["L0"]["mean"],
                  "prior_a0_rel_sd": A.prior["a0"]["sd"] / A.prior["a0"]["mean"],
                  "prior_L0_rel_sd": A.prior["L0"]["sd"] / A.prior["L0"]["mean"]},
        "reception_time_of_L0_s": rL0,
    }


def main():
    results = {"baseline": baseline().summary(),
               "mission": {m: ph.mission_summary(baseline(), m) for m in MODELS}}
    for m in MODELS:
        results["mission"][m]["t_target_yr"] = results["mission"][m]["t_target"] / YEAR
        results["mission"][m]["tau_target_yr"] = results["mission"][m]["tau_target"] / YEAR
    results["elasticities_beta_final"] = {m: elasticities(baseline(), "beta_final", m) for m in MODELS}
    fig_areal_density(results)
    fig_reflectivity(results)
    fig_classical_vs_relativistic(results)
    fig_uncertainty(results)
    fig_assimilation(results)

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "results.json").write_text(json.dumps(conv(results), indent=2, ensure_ascii=False))
    print("  wrote figures/results.json")


if __name__ == "__main__":
    main()
