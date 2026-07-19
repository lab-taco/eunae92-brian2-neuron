"""
recurrent_brian2_final.py

Presentation-oriented recurrent-network sanity check for Fig. 7/8 ideas in
Agnes & Vogels (2024) co-dependent plasticity.

This is NOT a full reproduction of the original 1000E/250I, multi-hour
simulation. It is a compressed Brian2 scaffold designed to make the key
mechanisms inspectable:

1. recurrent E/I spiking network
2. plastic E->E synapses using co-dependent excitatory plasticity
3. plastic I->E synapses using co-dependent inhibitory plasticity
4. trace-based E/I balance diagnostics
5. learned connectivity diagnostics:
   - E->E input vs E->E output strength
   - I->E input strength vs E->E input strength
6. short recall/stimulation after learning with plasticity frozen

Main changes relative to the draft:
- Use summed synaptic input/output strengths for Fig. 7-style diagnostics.
  Means are still saved, but correlations are reported from total strengths.
- Add trace-based E/I ratio diagnostics, because inhibitory plasticity uses
  E_trace and I_trace, not only instantaneous currents.
- Make recall parameters explicit and adjustable from the command line.
- Use a high-output/low-input impact score for recall stimulation, aligned with
  the input-output asymmetry logic.
- Add presentation-ready summary figures and CSV outputs.
- Add optional Brian2 import so --plot-only works on machines without Brian2.

Recommended run order:
    python recurrent_brian2_final.py --mode smoke --target numpy
    python recurrent_brian2_final.py --mode quick --target numpy
    python recurrent_brian2_final.py --mode medium --target cython

If cython is not configured on Windows, use:
    python recurrent_brian2_final.py --mode medium --target numpy

Useful parameter sweep for alpha:
    python recurrent_brian2_final.py --mode quick --alpha 1.05 --target numpy
    python recurrent_brian2_final.py --mode quick --alpha 1.10 --target numpy
    python recurrent_brian2_final.py --mode quick --alpha 1.20 --target numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from brian2 import (  # type: ignore
        Hz,
        Network,
        NeuronGroup,
        PoissonGroup,
        PopulationRateMonitor,
        SpikeMonitor,
        StateMonitor,
        Synapses,
        TimedArray,
        clip,
        defaultclock,
        exp,
        ms,
        prefs,
        second,
        seed,
        start_scope,
    )

    HAVE_BRIAN2 = True
except Exception:
    HAVE_BRIAN2 = False


EPS = 1e-12


@dataclass
class ModeConfig:
    name: str
    n_e: int
    n_i: int
    learn_s: float
    recall_s: float
    dt_ms: float
    p_ee: float
    p_ei: float
    p_ie: float
    p_ii: float
    monitor_n: int
    ext_rate_high_hz: float
    ext_rate_mid_hz: float
    ext_rate_low_hz: float
    recall_bias_e: float
    recall_bias_i: float
    recall_stim: float


MODES = {
    "smoke": ModeConfig(
        name="smoke",
        n_e=80,
        n_i=20,
        learn_s=20.0,
        recall_s=4.0,
        dt_ms=0.2,
        p_ee=0.08,
        p_ei=0.18,
        p_ie=0.25,
        p_ii=0.15,
        monitor_n=8,
        ext_rate_high_hz=90.0,
        ext_rate_mid_hz=35.0,
        ext_rate_low_hz=16.0,
        recall_bias_e=0.65,
        recall_bias_i=0.20,
        recall_stim=10.0,
    ),
    "quick": ModeConfig(
        name="quick",
        n_e=160,
        n_i=40,
        learn_s=60.0,
        recall_s=5.0,
        dt_ms=0.1,
        p_ee=0.06,
        p_ei=0.15,
        p_ie=0.20,
        p_ii=0.12,
        monitor_n=10,
        ext_rate_high_hz=65.0,
        ext_rate_mid_hz=30.0,
        ext_rate_low_hz=18.0,
        recall_bias_e=0.70,
        recall_bias_i=0.20,
        recall_stim=12.0,
    ),
    "medium": ModeConfig(
        name="medium",
        n_e=300,
        n_i=75,
        learn_s=120.0,
        recall_s=6.0,
        dt_ms=0.1,
        p_ee=0.045,
        p_ei=0.12,
        p_ie=0.16,
        p_ii=0.10,
        monitor_n=12,
        ext_rate_high_hz=80.0,
        ext_rate_mid_hz=30.0,
        ext_rate_low_hz=18.0,
        recall_bias_e=0.70,
        recall_bias_i=0.20,
        recall_stim=12.0,
    ),
}


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok]
    y = y[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + EPS)


def synaptic_stats(syn: "Synapses", n_pre: int, n_post: int) -> dict[str, np.ndarray]:
    """Return total and mean incoming/outgoing weights.

    Fig. 7-style connectivity diagnostics are more naturally expressed using
    total synaptic strengths. Mean weights are still useful for debugging, but
    degree-normalized means can hide changes in total drive.
    """
    pre = np.asarray(syn.i[:], dtype=int)
    post = np.asarray(syn.j[:], dtype=int)
    w = np.asarray(syn.w[:], dtype=float)

    in_sum = np.bincount(post, weights=w, minlength=n_post)
    in_count = np.bincount(post, minlength=n_post)
    out_sum = np.bincount(pre, weights=w, minlength=n_pre)
    out_count = np.bincount(pre, minlength=n_pre)

    return {
        "in_sum": in_sum,
        "out_sum": out_sum,
        "in_count": in_count,
        "out_count": out_count,
        "in_mean": in_sum / np.maximum(in_count, 1),
        "out_mean": out_sum / np.maximum(out_count, 1),
    }


def make_external_rate_timedarray(cfg: ModeConfig):
    """Compressed analogue of a decaying external drive during learning."""
    total_s = cfg.learn_s + cfg.recall_s
    bin_dt_s = 1.0
    n_bins = int(np.ceil(total_s / bin_dt_s)) + 3
    values = np.zeros(n_bins)

    t1 = 0.25 * cfg.learn_s
    t2 = 0.50 * cfg.learn_s

    for k in range(n_bins):
        t = k * bin_dt_s
        if t < t1:
            values[k] = cfg.ext_rate_high_hz
        elif t < t2:
            values[k] = cfg.ext_rate_mid_hz
        elif t < cfg.learn_s:
            values[k] = cfg.ext_rate_low_hz
        else:
            values[k] = 0.0

    return TimedArray(values * Hz, dt=bin_dt_s * second)


def add_regression_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return
    x_ok = x[ok]
    y_ok = y[ok]
    if np.std(x_ok) == 0:
        return
    m, b = np.polyfit(x_ok, y_ok, 1)
    xs = np.linspace(np.min(x_ok), np.max(x_ok), 100)
    ax.plot(xs, m * xs + b, linewidth=1.2)


def save_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, s=18, alpha=0.75)
    add_regression_line(ax, x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_and_run(
    mode: str,
    random_seed: int = 1,
    target: str = "numpy",
    outdir: str | Path = "results_recurrent_final",
    alpha_override: Optional[float] = None,
    recall_bias_e_override: Optional[float] = None,
    recall_bias_i_override: Optional[float] = None,
    recall_stim_override: Optional[float] = None,
) -> dict:
    if not HAVE_BRIAN2:
        raise RuntimeError("Brian2 is not installed. Install brian2 to run simulations.")

    cfg = MODES[mode]
    alpha_balance = float(alpha_override) if alpha_override is not None else 1.10
    recall_bias_e = float(recall_bias_e_override) if recall_bias_e_override is not None else cfg.recall_bias_e
    recall_bias_i = float(recall_bias_i_override) if recall_bias_i_override is not None else cfg.recall_bias_i
    recall_stim = float(recall_stim_override) if recall_stim_override is not None else cfg.recall_stim

    start_scope()
    seed(random_seed)
    np.random.seed(random_seed)

    prefs.codegen.target = target
    defaultclock.dt = cfg.dt_ms * ms

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    n_e = cfg.n_e
    n_i = cfg.n_i

    ns = {
        "tau_m": 30.0 * ms,
        "tau_ahp": 100.0 * ms,
        "tau_theta": 20.0 * ms,
        "tau_ampa": 5.0 * ms,
        "tau_nmda": 150.0 * ms,
        "tau_gaba": 10.0 * ms,
        "tau_E": 10.0 * ms,
        "tau_I": 100.0 * ms,
        "tau_pre_e": 16.8 * ms,
        "tau_post_ltd": 33.7 * ms,
        "tau_post_het": 100.0 * ms,
        "tau_i": 20.0 * ms,
        "u_rest": -65.0,
        "u_reset": -60.0,
        "theta0": -55.0,
        "theta_spike": -48.0,
        "E_ahp": -80.0,
        "E_gaba": -80.0,
        "a_nmda": 0.15,
        "b_nmda": -0.08,
        "ahp_amp": 0.35,
        # Co-dependent excitatory plasticity.
        "A_ltp_e": 1.0e-5,
        "A_ltd_e": 8.0e-5,
        "A_het_e": 2.0e-8,
        "I_control": 60.0,
        "wmin_e": 1.0e-5,
        "wmax_e": 0.08,
        # Co-dependent inhibitory plasticity.
        "A_i": 8.0e-5,
        "A_balance": 3.0e-5,
        "alpha_balance": alpha_balance,
        "wmin_i": 1.0e-5,
        "wmax_i": 3.0,
    }

    w_ee0 = 0.20 / max(1.0, n_e * cfg.p_ee)
    w_ei0 = 0.35 / max(1.0, n_e * cfg.p_ei)
    w_ie0 = 1.10 / max(1.0, n_i * cfg.p_ie)
    w_ii0 = 0.70 / max(1.0, n_i * cfg.p_ii)
    w_ext = 0.22

    eqs_e = """
    du/dt = (-(u - u_rest)
             - g_ampa * u
             - g_nmda * H_nmda * u
             - g_gaba * (u - E_gaba)
             - g_ahp * (u - E_ahp)
             + I_ext) / tau_m : 1 (unless refractory)

    dg_ampa/dt = -g_ampa / tau_ampa : 1
    dg_nmda/dt = -g_nmda / tau_nmda : 1
    dg_gaba/dt = -g_gaba / tau_gaba : 1
    dg_ahp/dt = -g_ahp / tau_ahp : 1
    dtheta/dt = -(theta - theta0) / tau_theta : 1

    dE_trace/dt = (-E_trace - g_nmda * H_nmda * u) / tau_E : 1
    dI_trace/dt = (-I_trace + g_gaba * (u - E_gaba)) / tau_I : 1

    dypost_e_ltd/dt = -ypost_e_ltd / tau_post_ltd : 1
    dypost_e_het/dt = -ypost_e_het / tau_post_het : 1
    dypost_i/dt = -ypost_i / tau_i : 1

    H_nmda = 1.0 / (1.0 + a_nmda * exp(b_nmda * u)) : 1
    E_pos = clip(E_trace, 0.0, 300.0) : 1
    I_pos = clip(I_trace, 0.0, 300.0) : 1
    gate = exp(-I_pos / I_control) : 1
    balance = A_balance * E_pos * (E_pos - alpha_balance * I_pos) : 1

    I_ext : 1
    """

    reset_e = """
    u = u_reset
    g_ahp += ahp_amp
    theta = theta_spike
    ypost_e_ltd += 1.0
    ypost_e_het += 1.0
    ypost_i += 1.0
    """

    eqs_i = """
    du/dt = (-(u - u_rest)
             - g_ampa * u
             - g_nmda * H_nmda * u
             - g_gaba * (u - E_gaba)
             - g_ahp * (u - E_ahp)
             + I_ext) / tau_m : 1 (unless refractory)

    dg_ampa/dt = -g_ampa / tau_ampa : 1
    dg_nmda/dt = -g_nmda / tau_nmda : 1
    dg_gaba/dt = -g_gaba / tau_gaba : 1
    dg_ahp/dt = -g_ahp / tau_ahp : 1
    dtheta/dt = -(theta - theta0) / tau_theta : 1

    H_nmda = 1.0 / (1.0 + a_nmda * exp(b_nmda * u)) : 1

    I_ext : 1
    """

    reset_i = """
    u = u_reset
    g_ahp += ahp_amp
    theta = theta_spike
    """

    E = NeuronGroup(n_e, eqs_e, threshold="u > theta", reset=reset_e, refractory=2.0 * ms, method="euler", namespace=ns, name="E")
    I = NeuronGroup(n_i, eqs_i, threshold="u > theta", reset=reset_i, refractory=2.0 * ms, method="euler", namespace=ns, name="I")

    for G in [E, I]:
        G.u = ns["u_rest"] + 6.0 * np.random.randn(len(G))
        G.theta = ns["theta0"] + 2.0 * np.random.randn(len(G))
        G.g_ampa = 0.0
        G.g_nmda = 0.0
        G.g_gaba = 0.0
        G.g_ahp = 0.0
        G.I_ext = 0.0

    E.E_trace = 0.0
    E.I_trace = 0.0
    E.ypost_e_ltd = 0.0
    E.ypost_e_het = 0.0
    E.ypost_i = 0.0

    model_ee = """
    w : 1
    dxpre/dt = -xpre / tau_pre_e : 1 (clock-driven)
    plasticity_on : 1 (shared)
    """

    S_EE = Synapses(
        E,
        E,
        model=model_ee,
        on_pre="""
        g_ampa_post += w
        g_nmda_post += w
        w = clip(w + plasticity_on * (-A_ltd_e * ypost_e_ltd_post * gate_post * w), wmin_e, wmax_e)
        xpre += 1.0
        """,
        on_post="""
        w = clip(w + plasticity_on * ((A_ltp_e * xpre * E_trace_post - A_het_e * ypost_e_het_post * (E_trace_post ** 2)) * gate_post), wmin_e, wmax_e)
        """,
        method="exact",
        namespace=ns,
        name="S_EE",
    )
    S_EE.connect(condition="i != j", p=cfg.p_ee)
    S_EE.w = "clip(w_ee0 * (1.0 + 0.20 * randn()), wmin_e, wmax_e)"
    S_EE.xpre = 0.0
    S_EE.plasticity_on = 1.0

    S_EI = Synapses(
        E,
        I,
        on_pre="""
        g_ampa_post += w_ei0
        g_nmda_post += w_ei0
        """,
        namespace={**ns, "w_ei0": w_ei0},
        name="S_EI",
    )
    S_EI.connect(p=cfg.p_ei)

    model_ie = """
    w : 1
    dxpre/dt = -xpre / tau_i : 1 (clock-driven)
    plasticity_on : 1 (shared)
    """

    S_IE = Synapses(
        I,
        E,
        model=model_ie,
        on_pre="""
        g_gaba_post += w
        w = clip(w + plasticity_on * (A_i * balance_post * ypost_i_post), wmin_i, wmax_i)
        xpre += 1.0
        """,
        on_post="""
        w = clip(w + plasticity_on * (A_i * balance_post * xpre), wmin_i, wmax_i)
        """,
        method="exact",
        namespace=ns,
        name="S_IE",
    )
    S_IE.connect(p=cfg.p_ie)
    S_IE.w = "clip(w_ie0 * (1.0 + 0.20 * randn()), wmin_i, wmax_i)"
    S_IE.xpre = 0.0
    S_IE.plasticity_on = 1.0

    S_II = Synapses(I, I, on_pre="g_gaba_post += w_ii0", namespace={**ns, "w_ii0": w_ii0}, name="S_II")
    S_II.connect(condition="i != j", p=cfg.p_ii)

    ext_rate = make_external_rate_timedarray(cfg)
    P_ext_e = PoissonGroup(n_e, rates="ext_rate(t)", namespace={"ext_rate": ext_rate}, name="P_ext_e")
    S_ext_e = Synapses(P_ext_e, E, on_pre="""g_ampa_post += w_ext
g_nmda_post += w_ext""", namespace={"w_ext": w_ext}, name="S_ext_e")
    S_ext_e.connect(j="i")

    P_ext_i = PoissonGroup(n_i, rates="0.5 * ext_rate(t)", namespace={"ext_rate": ext_rate}, name="P_ext_i")
    S_ext_i = Synapses(P_ext_i, I, on_pre="""g_ampa_post += w_ext
g_nmda_post += w_ext""", namespace={"w_ext": w_ext}, name="S_ext_i")
    S_ext_i.connect(j="i")

    mon_idx = np.arange(min(cfg.monitor_n, n_e))
    spike_e = SpikeMonitor(E, name="spike_e")
    spike_i = SpikeMonitor(I, name="spike_i")
    rate_e = PopulationRateMonitor(E, name="rate_e")
    rate_i = PopulationRateMonitor(I, name="rate_i")
    state_e = StateMonitor(E, ["u", "E_trace", "I_trace", "g_ampa", "g_nmda", "g_gaba"], record=mon_idx, dt=10.0 * ms, name="state_e")

    net = Network(E, I, S_EE, S_EI, S_IE, S_II, P_ext_e, S_ext_e, P_ext_i, S_ext_i, spike_e, spike_i, rate_e, rate_i, state_e)

    print("====================================")
    print("recurrent simulation")
    print("mode:", cfg.name, "seed:", random_seed, "alpha:", alpha_balance)
    print("N_E:", n_e, "N_I:", n_i)
    print("learn:", cfg.learn_s, "s", "recall:", cfg.recall_s, "s", "dt:", cfg.dt_ms, "ms")
    print("connections:", "EE", len(S_EE), "EI", len(S_EI), "IE", len(S_IE), "II", len(S_II))
    print("initial weights:", "w_ee0", w_ee0, "w_ei0", w_ei0, "w_ie0", w_ie0, "w_ii0", w_ii0)
    print("recall:", "bias_E", recall_bias_e, "bias_I", recall_bias_i, "stim", recall_stim)
    print("====================================")

    E.I_ext = 1.8
    I.I_ext = 1.3
    print("running learning period...")
    net.run(cfg.learn_s * second, report="text", namespace=ns)

    S_EE.plasticity_on = 0.0
    S_IE.plasticity_on = 0.0

    ee_stats = synaptic_stats(S_EE, n_e, n_e)
    ie_stats = synaptic_stats(S_IE, n_i, n_e)
    ee_in = ee_stats["in_sum"]
    ee_out = ee_stats["out_sum"]
    ie_in = ie_stats["in_sum"]

    spike_t = np.asarray(spike_e.t / second)
    spike_i_e = np.asarray(spike_e.i, dtype=int)
    t0_base = 0.75 * cfg.learn_s
    base_counts = np.bincount(spike_i_e[spike_t >= t0_base], minlength=n_e)
    base_rates = base_counts / max(cfg.learn_s - t0_base, EPS)

    # Fig. 7 logic: high-output and low-input E neurons are potential amplifiers.
    impact_e = 0.70 * zscore(ee_out) - 0.20 * zscore(ee_in) + 0.10 * zscore(base_rates)

    n_stim = max(5, int(0.10 * n_e))
    stim_idx = np.argsort(impact_e)[-n_stim:]

    E.I_ext = recall_bias_e
    I.I_ext = recall_bias_i
    E.I_ext[stim_idx] = recall_bias_e + recall_stim

    print("running recall stimulus...")
    net.run(1.0 * second, report="text", namespace=ns)

    E.I_ext = recall_bias_e
    I.I_ext = recall_bias_i

    print("running post-stimulus period...")
    net.run(max(cfg.recall_s - 1.0, 0.1) * second, report="text", namespace=ns)

    ee_stats = synaptic_stats(S_EE, n_e, n_e)
    ie_stats = synaptic_stats(S_IE, n_i, n_e)

    ee_in_sum = ee_stats["in_sum"]
    ee_out_sum = ee_stats["out_sum"]
    ie_in_sum = ie_stats["in_sum"]
    ee_in_mean = ee_stats["in_mean"]
    ee_out_mean = ee_stats["out_mean"]
    ie_in_mean = ie_stats["in_mean"]

    corr_in_out_sum = safe_corr(ee_in_sum, ee_out_sum)
    corr_ei_sum = safe_corr(ee_in_sum, ie_in_sum)
    corr_in_out_mean = safe_corr(ee_in_mean, ee_out_mean)
    corr_ei_mean = safe_corr(ee_in_mean, ie_in_mean)

    total_time_s = cfg.learn_s + cfg.recall_s
    rate_t = np.asarray(rate_e.t / second)
    rate_e_smooth = np.asarray(rate_e.smooth_rate(width=200.0 * ms) / Hz)
    rate_i_smooth = np.asarray(rate_i.smooth_rate(width=200.0 * ms) / Hz)

    recall_start = cfg.learn_s
    baseline_mask = (rate_t >= cfg.learn_s - min(10.0, 0.25 * cfg.learn_s)) & (rate_t < cfg.learn_s)
    recall_mask = (rate_t >= recall_start) & (rate_t <= total_time_s)
    baseline_rate = float(np.mean(rate_e_smooth[baseline_mask])) if np.any(baseline_mask) else float(np.mean(rate_e_smooth))
    response_l2 = float(np.sqrt(np.mean((rate_e_smooth[recall_mask] - baseline_rate) ** 2))) if np.any(recall_mask) else float("nan")
    peak_recall_rate = float(np.max(rate_e_smooth[recall_mask])) if np.any(recall_mask) else float("nan")

    t_state = np.asarray(state_e.t / second)
    e_trace_mean = np.mean(np.asarray(state_e.E_trace), axis=0) if len(mon_idx) > 0 else np.array([])
    i_trace_mean = np.mean(np.asarray(state_e.I_trace), axis=0) if len(mon_idx) > 0 else np.array([])
    trace_ratio = e_trace_mean / (i_trace_mean + EPS) if e_trace_mean.size else np.array([])
    trace_ratio_learning = trace_ratio[t_state < cfg.learn_s] if trace_ratio.size else np.array([])
    mean_trace_ratio = float(np.nanmean(trace_ratio_learning)) if trace_ratio_learning.size else float("nan")
    median_trace_ratio = float(np.nanmedian(trace_ratio_learning)) if trace_ratio_learning.size else float("nan")

    print("====================================")
    print("recurrent result")
    print("mode:", cfg.name)
    print("baseline E rate:", round(baseline_rate, 4), "Hz")
    print("peak recall E rate:", round(peak_recall_rate, 4), "Hz")
    print("recall l2 rate deviation:", round(response_l2, 4))
    print("corr sum(EE input, EE output):", round(corr_in_out_sum, 4))
    print("corr sum(EE input, IE input):", round(corr_ei_sum, 4))
    print("corr mean(EE input, EE output):", round(corr_in_out_mean, 4))
    print("corr mean(EE input, IE input):", round(corr_ei_mean, 4))
    print("mean trace E/I during learning:", round(mean_trace_ratio, 4), "alpha:", alpha_balance)
    print("E spikes:", int(spike_e.num_spikes), "I spikes:", int(spike_i.num_spikes))
    print("====================================")

    summary_df = pd.DataFrame(
        {
            "neuron": np.arange(n_e),
            "sum_EE_input": ee_in_sum,
            "sum_EE_output": ee_out_sum,
            "sum_IE_input": ie_in_sum,
            "mean_EE_input": ee_in_mean,
            "mean_EE_output": ee_out_mean,
            "mean_IE_input": ie_in_mean,
            "baseline_rate": base_rates,
            "impact": impact_e,
            "stimulated": np.isin(np.arange(n_e), stim_idx).astype(int),
        }
    )
    summary_df.to_csv(outdir / f"recurrent_neuron_summary_{cfg.name}_final.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            ("mode", cfg.name),
            ("seed", random_seed),
            ("n_e", n_e),
            ("n_i", n_i),
            ("learn_s", cfg.learn_s),
            ("recall_s", cfg.recall_s),
            ("alpha_balance", alpha_balance),
            ("baseline_E_rate_Hz", baseline_rate),
            ("peak_recall_E_rate_Hz", peak_recall_rate),
            ("recall_l2_rate_deviation", response_l2),
            ("corr_sum_EE_input_EE_output", corr_in_out_sum),
            ("corr_sum_EE_input_IE_input", corr_ei_sum),
            ("corr_mean_EE_input_EE_output", corr_in_out_mean),
            ("corr_mean_EE_input_IE_input", corr_ei_mean),
            ("mean_EE_input_sum", float(np.mean(ee_in_sum))),
            ("mean_EE_output_sum", float(np.mean(ee_out_sum))),
            ("mean_IE_input_sum", float(np.mean(ie_in_sum))),
            ("mean_trace_ratio_learning", mean_trace_ratio),
            ("median_trace_ratio_learning", median_trace_ratio),
            ("E_spikes", int(spike_e.num_spikes)),
            ("I_spikes", int(spike_i.num_spikes)),
        ],
        columns=["metric", "value"],
    )
    metrics_df.to_csv(outdir / f"recurrent_metrics_{cfg.name}_final.csv", index=False)

    # Population rate.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rate_t, rate_e_smooth, label="E population")
    ax.plot(rate_t, rate_i_smooth, label="I population")
    ax.axvline(cfg.learn_s, linestyle="--", linewidth=1, label="plasticity off / recall")
    ax.axvspan(cfg.learn_s, cfg.learn_s + 1.0, alpha=0.15, label="stimulus")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"Recurrent population rate ({cfg.name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"recurrent_rate_{cfg.name}_final.png", dpi=200)
    plt.close(fig)

    save_scatter(
        ee_in_sum,
        ee_out_sum,
        "total E->E input strength",
        "total E->E output strength",
        f"E->E input-output structure ({cfg.name}), corr={corr_in_out_sum:.3f}",
        outdir / f"recurrent_EE_input_output_{cfg.name}_final.png",
    )

    save_scatter(
        ee_in_sum,
        ie_in_sum,
        "total E->E input strength",
        "total I->E input strength",
        f"E/I input matching ({cfg.name}), corr={corr_ei_sum:.3f}",
        outdir / f"recurrent_EI_input_matching_{cfg.name}_final.png",
    )

    # Weight matrix image for a subset of excitatory neurons.
    subset = min(80, n_e)
    mat = np.full((subset, subset), np.nan)
    pre = np.asarray(S_EE.i[:], dtype=int)
    post = np.asarray(S_EE.j[:], dtype=int)
    w = np.asarray(S_EE.w[:], dtype=float)
    ok = (pre < subset) & (post < subset)
    mat[post[ok], pre[ok]] = w[ok]

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, label="E->E weight")
    ax.set_xlabel("presynaptic E neuron")
    ax.set_ylabel("postsynaptic E neuron")
    ax.set_title(f"E->E weight matrix subset ({cfg.name})")
    fig.tight_layout()
    fig.savefig(outdir / f"recurrent_EE_matrix_subset_{cfg.name}_final.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(min(4, len(mon_idx))):
        ax.plot(t_state, np.asarray(state_e.u[k]), linewidth=0.8, label=f"u E{k}")
    ax.axvline(cfg.learn_s, linestyle="--", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("membrane potential")
    ax.set_title(f"Example membrane traces ({cfg.name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"recurrent_voltage_examples_{cfg.name}_final.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    if e_trace_mean.size:
        ax.plot(t_state, e_trace_mean, label="mean E trace")
        ax.plot(t_state, i_trace_mean, label="mean I trace")
    ax.axvline(cfg.learn_s, linestyle="--", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trace")
    ax.set_title(f"Example co-dependent traces ({cfg.name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"recurrent_EI_traces_{cfg.name}_final.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    if trace_ratio.size:
        ax.plot(t_state, trace_ratio, linewidth=1.0, label="mean E_trace / mean I_trace")
    ax.axhline(alpha_balance, linestyle="--", linewidth=1.2, label=f"alpha = {alpha_balance}")
    ax.axvline(cfg.learn_s, linestyle="--", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trace-based E/I")
    ax.set_title(f"Trace-based E/I balance diagnostic ({cfg.name})")
    ax.set_ylim(0, np.nanpercentile(trace_ratio[np.isfinite(trace_ratio)], 95) * 1.2 if trace_ratio.size else 2.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"recurrent_trace_ratio_{cfg.name}_final.png", dpi=200)
    plt.close(fig)

    # One-page presentation summary.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(rate_t, rate_e_smooth, label="E")
    axes[0, 0].plot(rate_t, rate_i_smooth, label="I")
    axes[0, 0].axvline(cfg.learn_s, linestyle="--", linewidth=1)
    axes[0, 0].axvspan(cfg.learn_s, cfg.learn_s + 1.0, alpha=0.15)
    axes[0, 0].set_title("Population rate")
    axes[0, 0].set_ylabel("Hz")
    axes[0, 0].legend()

    if trace_ratio.size:
        axes[0, 1].plot(t_state, trace_ratio, linewidth=1.0)
    axes[0, 1].axhline(alpha_balance, linestyle="--", linewidth=1.2)
    axes[0, 1].axvline(cfg.learn_s, linestyle="--", linewidth=1)
    axes[0, 1].set_title("Trace-based E/I")

    axes[1, 0].scatter(ee_in_sum, ee_out_sum, s=16, alpha=0.75)
    add_regression_line(axes[1, 0], ee_in_sum, ee_out_sum)
    axes[1, 0].set_title(f"E->E input/output corr={corr_in_out_sum:.3f}")
    axes[1, 0].set_xlabel("total E->E input")
    axes[1, 0].set_ylabel("total E->E output")

    axes[1, 1].scatter(ee_in_sum, ie_in_sum, s=16, alpha=0.75)
    add_regression_line(axes[1, 1], ee_in_sum, ie_in_sum)
    axes[1, 1].set_title(f"E/I matching corr={corr_ei_sum:.3f}")
    axes[1, 1].set_xlabel("total E->E input")
    axes[1, 1].set_ylabel("total I->E input")

    fig.suptitle(f"Recurrent co-dependent plasticity summary ({cfg.name})", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / f"recurrent_summary_{cfg.name}_final.png", dpi=200)
    plt.close(fig)

    print("saved:", outdir.resolve())

    return {
        "baseline_rate": baseline_rate,
        "peak_recall_rate": peak_recall_rate,
        "response_l2": response_l2,
        "corr_in_out_sum": corr_in_out_sum,
        "corr_ei_sum": corr_ei_sum,
        "corr_in_out_mean": corr_in_out_mean,
        "corr_ei_mean": corr_ei_mean,
        "outdir": outdir,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES.keys()), default="smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target", choices=["cython", "numpy"], default="numpy")
    parser.add_argument("--outdir", default="results_recurrent_final")
    parser.add_argument("--alpha", type=float, default=None, help="Override co-dependent inhibitory balance alpha.")
    parser.add_argument("--recall-bias-e", type=float, default=None)
    parser.add_argument("--recall-bias-i", type=float, default=None)
    parser.add_argument("--recall-stim", type=float, default=None)
    args = parser.parse_args()

    build_and_run(
        mode=args.mode,
        random_seed=args.seed,
        target=args.target,
        outdir=args.outdir,
        alpha_override=args.alpha,
        recall_bias_e_override=args.recall_bias_e,
        recall_bias_i_override=args.recall_bias_i,
        recall_stim_override=args.recall_stim,
    )


if __name__ == "__main__":
    main()
