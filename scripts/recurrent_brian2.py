"""
recurrent_brian2.py

Compressed recurrent-network sanity check for co-dependent plasticity.

This is NOT a full reproduction of the original 1000E/250I, 10-hour simulation.
It is a runnable Brian2 scaffold that tests the core Fig. 7/8 ideas:

1. recurrent E/I spiking network
2. plastic E->E synapses
3. plastic I->E synapses
4. co-dependent E and I current traces on excitatory neurons
5. learned connectivity summary:
   - E->E input vs output weight asymmetry
   - I->E input weights vs E->E input weights
6. brief recall/stimulation period after learning

Usage:
    python recurrent_brian2.py --mode smoke
    python recurrent_brian2.py --mode quick
    python recurrent_brian2.py --mode medium

Recommended order:
    smoke -> quick -> medium
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from brian2 import (
    Hz,
    PoissonGroup,
    PopulationRateMonitor,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    NeuronGroup,
    TimedArray,
    clip,
    defaultclock,
    exp,
    ms,
    prefs,
    second,
    seed,
    start_scope,
    Network,
)


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
        ext_rate_low_hz=8.0,
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
        ext_rate_low_hz=24.0,
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
        ext_rate_mid_hz=25.0,
        ext_rate_low_hz=12.0,
    ),
}


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = x[ok]
    y = y[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def synaptic_means(syn: Synapses, n_pre: int, n_post: int) -> tuple[np.ndarray, np.ndarray]:
    """Return mean incoming weight per postsynaptic neuron and mean outgoing weight per presynaptic neuron."""
    pre = np.asarray(syn.i[:], dtype=int)
    post = np.asarray(syn.j[:], dtype=int)
    w = np.asarray(syn.w[:], dtype=float)

    in_sum = np.bincount(post, weights=w, minlength=n_post)
    in_count = np.bincount(post, minlength=n_post)
    out_sum = np.bincount(pre, weights=w, minlength=n_pre)
    out_count = np.bincount(pre, minlength=n_pre)

    in_mean = in_sum / np.maximum(in_count, 1)
    out_mean = out_sum / np.maximum(out_count, 1)
    return in_mean, out_mean


def make_external_rate_timedarray(cfg: ModeConfig):
    """Compressed analogue of the paper's decaying external drive during learning."""
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


def build_and_run(mode: str, random_seed: int = 1, target: str = "cython") -> dict:
    cfg = MODES[mode]

    start_scope()
    seed(random_seed)
    np.random.seed(random_seed)

    prefs.codegen.target = target
    defaultclock.dt = cfg.dt_ms * ms

    outdir = Path("results_recurrent")
    outdir.mkdir(exist_ok=True)

    n_e = cfg.n_e
    n_i = cfg.n_i

    # ------------------------------------------------------------------
    # Parameters.
    # All voltages are represented as dimensionless mV-like numbers.
    # Conductances are normalized by leak conductance.
    # ------------------------------------------------------------------
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
        # Slightly stronger LTP, weaker LTD, with a tighter max weight clamp.
        "A_ltp_e": 1.0e-5,
        "A_ltd_e": 8.0e-5,
        "A_het_e": 2.0e-8,
        "I_control": 60.0,
        "wmin_e": 1.0e-5,
        "wmax_e": 0.08,
        # Co-dependent inhibitory plasticity.
        "A_i": 8.0e-5,
        "A_balance": 3.0e-5,
        "alpha_balance": 1.10,
        "wmin_i": 1.0e-5,
        "wmax_i": 3.0,
    }

    # Initial weights scale approximately with expected in-degree.
    # E drive is reduced, inhibition is slightly strengthened.
    w_ee0 = 0.20 / max(1.0, n_e * cfg.p_ee)
    w_ei0 = 0.35 / max(1.0, n_e * cfg.p_ei)
    w_ie0 = 1.10 / max(1.0, n_i * cfg.p_ie)
    w_ii0 = 0.70 / max(1.0, n_i * cfg.p_ii)
    w_ext = 0.22

    # ------------------------------------------------------------------
    # Neuron equations.
    # Excitatory neurons carry E/I traces and postsynaptic traces used
    # by E->E and I->E plasticity.
    # ------------------------------------------------------------------
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
    gate = exp(-clip(I_trace, 0.0, 300.0) / I_control) : 1
    balance = A_balance * clip(E_trace, 0.0, 300.0) * (clip(E_trace, 0.0, 300.0) - alpha_balance * clip(I_trace, 0.0, 300.0)) : 1

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

    E = NeuronGroup(
        n_e,
        eqs_e,
        threshold="u > theta",
        reset=reset_e,
        refractory=2.0 * ms,
        method="euler",
        namespace=ns,
        name="E",
    )

    I = NeuronGroup(
        n_i,
        eqs_i,
        threshold="u > theta",
        reset=reset_i,
        refractory=2.0 * ms,
        method="euler",
        namespace=ns,
        name="I",
    )

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

    # ------------------------------------------------------------------
    # Recurrent synapses.
    # ------------------------------------------------------------------
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

    S_II = Synapses(
        I,
        I,
        on_pre="g_gaba_post += w_ii0",
        namespace={**ns, "w_ii0": w_ii0},
        name="S_II",
    )
    S_II.connect(condition="i != j", p=cfg.p_ii)

    # ------------------------------------------------------------------
    # External input during learning only.
    # ------------------------------------------------------------------
    ext_rate = make_external_rate_timedarray(cfg)
    P_ext_e = PoissonGroup(n_e, rates="ext_rate(t)", namespace={"ext_rate": ext_rate}, name="P_ext_e")
    S_ext_e = Synapses(
        P_ext_e,
        E,
        on_pre="""
        g_ampa_post += w_ext
        g_nmda_post += w_ext
        """,
        namespace={"w_ext": w_ext},
        name="S_ext_e",
    )
    S_ext_e.connect(j="i")

    P_ext_i = PoissonGroup(n_i, rates="0.5 * ext_rate(t)", namespace={"ext_rate": ext_rate}, name="P_ext_i")
    S_ext_i = Synapses(
        P_ext_i,
        I,
        on_pre="""
        g_ampa_post += w_ext
        g_nmda_post += w_ext
        """,
        namespace={"w_ext": w_ext},
        name="S_ext_i",
    )
    S_ext_i.connect(j="i")

    # ------------------------------------------------------------------
    # Monitors.
    # ------------------------------------------------------------------
    mon_idx = np.arange(min(cfg.monitor_n, n_e))
    spike_e = SpikeMonitor(E, name="spike_e")
    spike_i = SpikeMonitor(I, name="spike_i")
    rate_e = PopulationRateMonitor(E, name="rate_e")
    rate_i = PopulationRateMonitor(I, name="rate_i")
    state_e = StateMonitor(E, ["u", "E_trace", "I_trace", "g_ampa", "g_nmda", "g_gaba"], record=mon_idx, dt=10.0 * ms, name="state_e")

    net = Network(
        E,
        I,
        S_EE,
        S_EI,
        S_IE,
        S_II,
        P_ext_e,
        S_ext_e,
        P_ext_i,
        S_ext_i,
        spike_e,
        spike_i,
        rate_e,
        rate_i,
        state_e,
    )

    print("====================================")
    print("recurrent simulation")
    print("mode:", cfg.name)
    print("N_E:", n_e, "N_I:", n_i)
    print("learn:", cfg.learn_s, "s", "recall:", cfg.recall_s, "s", "dt:", cfg.dt_ms, "ms")
    print("connections:", "EE", len(S_EE), "EI", len(S_EI), "IE", len(S_IE), "II", len(S_II))
    print("initial weights:", "w_ee0", w_ee0, "w_ei0", w_ei0, "w_ie0", w_ie0, "w_ii0", w_ii0)
    print("====================================")

    E.I_ext = 1.8
    I.I_ext = 1.3
    print("running learning period...")
    net.run(cfg.learn_s * second, report="text", namespace=ns)

    # Freeze plasticity before recall.
    S_EE.plasticity_on = 0.0
    S_IE.plasticity_on = 0.0

    # Compute impact score after learning.
    ee_in, ee_out = synaptic_means(S_EE, n_e, n_e)
    ie_in, _ = synaptic_means(S_IE, n_i, n_e)

    # Estimate baseline rate from the last 25% of the learning period.
    spike_t = np.asarray(spike_e.t / second)
    spike_i_e = np.asarray(spike_e.i, dtype=int)
    t0_base = 0.75 * cfg.learn_s
    base_counts = np.bincount(spike_i_e[spike_t >= t0_base], minlength=n_e)
    base_rates = base_counts / max(cfg.learn_s - t0_base, 1e-9)
    ee_in_z = (ee_in - np.mean(ee_in)) / (np.std(ee_in) + 1e-12)
    ee_out_z = (ee_out - np.mean(ee_out)) / (np.std(ee_out) + 1e-12)
    rate_z = (base_rates - np.mean(base_rates)) / (np.std(base_rates) + 1e-12)

    impact_e = 0.45 * ee_in_z + 0.45 * ee_out_z + 0.10 * rate_z

    # Stimulate high-impact excitatory neurons.
    n_stim = max(5, int(0.10 * n_e))
    stim_idx = np.argsort(impact_e)[-n_stim:]
    recall_bias_e = 0.45
    recall_bias_i = 0.15
    recall_stim = 12.0

    E.I_ext = recall_bias_e
    I.I_ext = recall_bias_i
    E.I_ext[stim_idx] = recall_bias_e + recall_stim

    print("running recall stimulus...")
    net.run(1.0 * second, report="text", namespace=ns)

    E.I_ext = recall_bias_e
    I.I_ext = recall_bias_i

    print("running post-stimulus self-sustained period...")
    net.run(max(cfg.recall_s - 1.0, 0.1) * second, report="text", namespace=ns)

    # ------------------------------------------------------------------
    # Analysis after recall.
    # ------------------------------------------------------------------
    ee_in, ee_out = synaptic_means(S_EE, n_e, n_e)
    ie_in, _ = synaptic_means(S_IE, n_i, n_e)

    corr_in_out = safe_corr(ee_in, ee_out)
    corr_ei = safe_corr(ee_in, ie_in)

    total_time_s = cfg.learn_s + cfg.recall_s
    rate_t = np.asarray(rate_e.t / second)
    rate_e_smooth = np.asarray(rate_e.smooth_rate(width=200.0 * ms) / Hz)
    rate_i_smooth = np.asarray(rate_i.smooth_rate(width=200.0 * ms) / Hz)

    # L2 response norm around recall.
    recall_start = cfg.learn_s
    baseline_mask = (rate_t >= cfg.learn_s - min(10.0, 0.25 * cfg.learn_s)) & (rate_t < cfg.learn_s)
    recall_mask = (rate_t >= recall_start) & (rate_t <= total_time_s)
    baseline_rate = float(np.mean(rate_e_smooth[baseline_mask])) if np.any(baseline_mask) else float(np.mean(rate_e_smooth))
    response_l2 = float(np.sqrt(np.mean((rate_e_smooth[recall_mask] - baseline_rate) ** 2))) if np.any(recall_mask) else np.nan

    print("====================================")
    print("recurrent result")
    print("mode:", cfg.name)
    print("baseline E rate:", round(baseline_rate, 4), "Hz")
    print("recall l2 rate deviation:", round(response_l2, 4))
    print("corr(EE input, EE output):", round(corr_in_out, 4))
    print("corr(EE input, IE input):", round(corr_ei, 4))
    print("mean EE in:", round(float(np.mean(ee_in)), 5), "mean EE out:", round(float(np.mean(ee_out)), 5))
    print("mean IE in:", round(float(np.mean(ie_in)), 5))
    print("E spikes:", int(spike_e.num_spikes), "I spikes:", int(spike_i.num_spikes))
    print("====================================")

    # Save CSV summaries.
    summary = np.column_stack([
        np.arange(n_e),
        ee_in,
        ee_out,
        ie_in,
        base_rates,
        impact_e,
    ])
    header = "neuron,mean_EE_input,mean_EE_output,mean_IE_input,baseline_rate,impact"
    np.savetxt(outdir / f"recurrent_neuron_summary_{cfg.name}.csv", summary, delimiter=",", header=header, comments="")

    metrics = np.array([
        ["mode", cfg.name],
        ["n_e", str(n_e)],
        ["n_i", str(n_i)],
        ["learn_s", str(cfg.learn_s)],
        ["recall_s", str(cfg.recall_s)],
        ["baseline_E_rate_Hz", str(baseline_rate)],
        ["recall_l2_rate_deviation", str(response_l2)],
        ["corr_EE_input_EE_output", str(corr_in_out)],
        ["corr_EE_input_IE_input", str(corr_ei)],
        ["mean_EE_input", str(float(np.mean(ee_in)))],
        ["mean_EE_output", str(float(np.mean(ee_out)))],
        ["mean_IE_input", str(float(np.mean(ie_in)))],
        ["E_spikes", str(int(spike_e.num_spikes))],
        ["I_spikes", str(int(spike_i.num_spikes))],
    ], dtype=object)
    np.savetxt(outdir / f"recurrent_metrics_{cfg.name}.csv", metrics, delimiter=",", fmt="%s")

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(rate_t, rate_e_smooth, label="E population")
    plt.plot(rate_t, rate_i_smooth, label="I population")
    plt.axvline(cfg.learn_s, linestyle="--", linewidth=1, label="plasticity off / recall")
    plt.axvspan(cfg.learn_s, cfg.learn_s + 1.0, alpha=0.15, label="stimulus")
    plt.xlabel("time (s)")
    plt.ylabel("firing rate (Hz)")
    plt.title(f"Recurrent population rate ({cfg.name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"recurrent_rate_{cfg.name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.scatter(ee_in, ee_out, s=18, alpha=0.8)
    plt.xlabel("mean E->E input weight")
    plt.ylabel("mean E->E output weight")
    plt.title(f"E->E input-output asymmetry ({cfg.name}), corr={corr_in_out:.3f}")
    plt.tight_layout()
    plt.savefig(outdir / f"recurrent_EE_input_output_{cfg.name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.scatter(ee_in, ie_in, s=18, alpha=0.8)
    plt.xlabel("mean E->E input weight")
    plt.ylabel("mean I->E input weight")
    plt.title(f"E/I input matching ({cfg.name}), corr={corr_ei:.3f}")
    plt.tight_layout()
    plt.savefig(outdir / f"recurrent_EI_input_matching_{cfg.name}.png", dpi=200)
    plt.close()

    # Weight matrix image for a subset of excitatory neurons.
    subset = min(80, n_e)
    mat = np.full((subset, subset), np.nan)
    pre = np.asarray(S_EE.i[:], dtype=int)
    post = np.asarray(S_EE.j[:], dtype=int)
    w = np.asarray(S_EE.w[:], dtype=float)
    ok = (pre < subset) & (post < subset)
    mat[post[ok], pre[ok]] = w[ok]

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="E->E weight")
    plt.xlabel("presynaptic E neuron")
    plt.ylabel("postsynaptic E neuron")
    plt.title(f"E->E weight matrix subset ({cfg.name})")
    plt.tight_layout()
    plt.savefig(outdir / f"recurrent_EE_matrix_subset_{cfg.name}.png", dpi=200)
    plt.close()

    t_state = np.asarray(state_e.t / second)
    plt.figure(figsize=(10, 7))
    for k in range(min(4, len(mon_idx))):
        plt.plot(t_state, np.asarray(state_e.u[k]), linewidth=0.8, label=f"u E{k}")
    plt.axvline(cfg.learn_s, linestyle="--", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("membrane potential")
    plt.title(f"Example membrane traces ({cfg.name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"recurrent_voltage_examples_{cfg.name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 7))
    if len(mon_idx) > 0:
        e_mean = np.mean(np.asarray(state_e.E_trace), axis=0)
        i_mean = np.mean(np.asarray(state_e.I_trace), axis=0)
        plt.plot(t_state, e_mean, label="mean E trace")
        plt.plot(t_state, i_mean, label="mean I trace")
    plt.axvline(cfg.learn_s, linestyle="--", linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel("trace")
    plt.title(f"Example co-dependent traces ({cfg.name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"recurrent_EI_traces_{cfg.name}.png", dpi=200)
    plt.close()

    print("saved:", outdir.resolve())

    return {
        "baseline_rate": baseline_rate,
        "response_l2": response_l2,
        "corr_in_out": corr_in_out,
        "corr_ei": corr_ei,
        "outdir": outdir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES.keys()), default="smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target", choices=["cython", "numpy"], default="cython")
    args = parser.parse_args()

    build_and_run(mode=args.mode, random_seed=args.seed, target=args.target)


if __name__ == "__main__":
    main()
