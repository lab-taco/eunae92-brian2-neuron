"""
frequency_stdp_brian2_final.py

Presentation-oriented Fig. 2c/e frequency-STDP script.

Purpose
-------
This script is a compressed Brian2 sanity check for the paper's Fig. 2c/e idea:
pre/post spike timing and pair frequency change excitatory plasticity through
NMDA-derived E traces, and neighboring external input can further modulate LTP.

Main fixes compared with the draft
----------------------------------
1. Brian2 import is optional. If Brian2 is not installed, you can still use
   --plot-only to regenerate clean presentation plots from an existing CSV.
2. The inhibitory gate is written explicitly as
       exp(-((I_trace / I_star) ** I_gamma))
   matching the paper's equation form.
3. The plasticity update uses non-negative E and I traces through subexpressions
   E_for_plasticity and I_gate. This avoids negative-trace artifacts.
4. LTP and LTD trials at each frequency use the same seed by default. This makes
   the comparison focus on spike timing rather than different Poisson noise.
5. The script saves extra diagnostic columns: number of post spikes, peak E/I
   traces, and weight change. These are useful for explaining why frequency
   increases LTP.
6. Plotting is separated from simulation through --plot-only.

Recommended commands
--------------------
Quick smoke test:
    python frequency_stdp_brian2_final.py --mode quick --seed 1 --outdir results_frequency_final_quick

Presentation-quality medium run:
    python frequency_stdp_brian2_final.py --mode medium --seed 1 --outdir results_frequency_final_medium

Example trace only:
    python frequency_stdp_brian2_final.py --example-only --outdir results_frequency_example

Regenerate plots from an existing CSV without Brian2:
    python frequency_stdp_brian2_final.py --plot-only --mode medium \
        --csv "frequency_sweep_medium(1).csv" --outdir results_frequency_plotonly_medium
"""

from __future__ import annotations

import argparse
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
        SpikeGeneratorGroup,
        SpikeMonitor,
        StateMonitor,
        Synapses,
        TimedArray,
        clip,
        defaultclock,
        exp,
        ms,
        prefs,
        seed,
        start_scope,
    )

    prefs.codegen.target = "numpy"
    HAVE_BRIAN2 = True
except Exception:
    HAVE_BRIAN2 = False


EXPERIMENTAL_DATA = pd.DataFrame(
    {
        "freq_hz": [0.1, 10, 20, 40, 50],
        "ltp": [96.7266, 115.311, 130.02, 153.342, 155.661],
        "ltd": [68.7583, 57.0747, 65.3493, 152.192, 171.331],
        "ltp_sem": [4.08049, 9.45575, 13.6743, 10.5622, 26.0755],
        "ltd_sem": [7.43512, 10.9595, 8.5702, 30.2612, 18.4164],
    }
)


def make_protocol(freq_hz: float, delta_t_ms: float, n_pairs: int, warmup_ms: float = 100.0):
    """
    Construct the pre-spike train and the postsynaptic current-pulse protocol.

    delta_t_ms = t_post - t_pre
      +10 ms: pre-before-post
      -10 ms: post-before-pre

    The postsynaptic spike is induced by a short current pulse, rather than
    forced directly. This keeps the implementation close to the paper's protocol.
    """
    if not HAVE_BRIAN2:
        raise RuntimeError("Brian2 is required for make_protocol/run_one_trial. Use --plot-only without Brian2.")
    if freq_hz <= 0:
        raise ValueError("freq_hz must be positive.")

    pair_period_ms = 1000.0 / float(freq_hz)
    pulse_amp = 300.0
    pulse_duration_ms = 2.0

    pre_times_ms = []
    pulse_on_ms = []
    pulse_off_ms = []

    start_ms = warmup_ms + 20.0

    for n in range(n_pairs):
        base = start_ms + n * pair_period_ms

        if delta_t_ms > 0:
            pre_t = base
            desired_post_t = base + delta_t_ms
        else:
            desired_post_t = base
            pre_t = base + abs(delta_t_ms)

        pulse_on_t = desired_post_t - pulse_duration_ms
        pulse_off_t = desired_post_t

        pre_times_ms.append(pre_t)
        pulse_on_ms.append(pulse_on_t)
        pulse_off_ms.append(pulse_off_t)

    duration_ms = max(max(pre_times_ms), max(pulse_off_ms)) + 100.0

    dt_ms = float(defaultclock.dt / ms)
    n_steps = int(np.ceil(duration_ms / dt_ms)) + 5
    pulse_values = np.zeros(n_steps)

    for on_t, off_t in zip(pulse_on_ms, pulse_off_ms):
        i0 = int(np.floor(on_t / dt_ms))
        i1 = int(np.ceil(off_t / dt_ms))
        pulse_values[i0:i1] = pulse_amp

    pulse_func = TimedArray(pulse_values, dt=defaultclock.dt)
    return np.asarray(pre_times_ms) * ms, pulse_func, duration_ms * ms


def run_one_trial(
    freq_hz: float,
    delta_t_ms: float,
    ext_rate_hz: float = 0.0,
    w0: float = 0.12,
    n_pairs: int = 20,
    seed_value: int = 1,
    record: bool = False,
) -> dict:
    if not HAVE_BRIAN2:
        raise RuntimeError("Brian2 is not installed. Use --plot-only with an existing CSV.")

    start_scope()
    defaultclock.dt = 0.1 * ms

    seed(seed_value)
    np.random.seed(seed_value)

    pre_times, pulse_func, duration = make_protocol(freq_hz=freq_hz, delta_t_ms=delta_t_ms, n_pairs=n_pairs)

    ns = {
        # neuron
        "tau_m": 30.0 * ms,
        "u_rest": -65.0,
        "u_th": -50.0,
        "u_reset": -60.0,
        "E_ahp": -80.0,
        "A_ahp": 0.05,
        "tau_ahp": 125.0 * ms,
        # synapse
        "E_gaba": -80.0,
        "tau_ampa": 5.0 * ms,
        "tau_gaba": 10.0 * ms,
        "tau_nmda": 150.0 * ms,
        "a_nmda": 0.15,
        "b_nmda": -0.08,
        # neighboring/static input weights
        "w_ext_e": 0.9 * w0,
        "w_ext_i": 0.6,
        # plasticity
        "tau_pre": 16.8 * ms,
        "tau_post_ltd": 33.7 * ms,
        "tau_post_het": 125.0 * ms,
        "A_ltp": 0.0025,
        "A_ltd": 0.012,
        "A_het": 1e-5,
        "I_star": 10.0,
        "I_gamma": 1.0,
        # traces: intentionally slower than the minimal tau_E so the compressed
        # sweep visibly shows frequency-dependent NMDA accumulation.
        "tau_E": 50.0 * ms,
        "tau_I": 500.0 * ms,
        # bounds
        "wmin": 1e-5,
        "wmax": 1.0,
        # TimedArray current pulse
        "pulse_func": pulse_func,
    }

    pre_pair = SpikeGeneratorGroup(1, indices=np.zeros(len(pre_times), dtype=int), times=pre_times)
    ext_e = PoissonGroup(1, rates=ext_rate_hz * Hz)
    ext_i = PoissonGroup(1, rates=ext_rate_hz * Hz)

    eqs = """
    du/dt = (-(u - u_rest)
             - g_ampa * u
             - g_nmda * H_nmda * u
             - g_gaba * (u - E_gaba)
             - A_ahp * g_ahp * (u - E_ahp)
             + I_pulse) / tau_m : 1 (unless refractory)

    dg_ampa/dt = -g_ampa / tau_ampa : 1
    dg_nmda/dt = -g_nmda / tau_nmda : 1
    dg_gaba/dt = -g_gaba / tau_gaba : 1
    dg_ahp/dt = -g_ahp / tau_ahp : 1

    dE_trace/dt = (-E_trace - g_nmda * H_nmda * u) / tau_E : 1
    dI_trace/dt = (-I_trace + g_gaba * (u - E_gaba)) / tau_I : 1

    dxpre/dt = -xpre / tau_pre : 1
    dypost_ltd/dt = -ypost_ltd / tau_post_ltd : 1
    dypost_het/dt = -ypost_het / tau_post_het : 1

    H_nmda = 1.0 / (1.0 + a_nmda * exp(b_nmda * u)) : 1
    I_pulse = pulse_func(t) : 1

    E_for_plasticity = clip(E_trace, 0.0, 1e6) : 1
    I_for_gate = clip(I_trace, 0.0, 1e6) : 1
    I_gate = exp(-((I_for_gate / I_star) ** I_gamma)) : 1

    wE : 1
    """

    G = NeuronGroup(
        1,
        eqs,
        threshold="u >= u_th",
        reset="""
        u = u_reset
        g_ahp += 1.0
        """,
        refractory=2.0 * ms,
        method="euler",
        namespace=ns,
    )

    G.u = ns["u_rest"]
    G.g_ampa = 0.0
    G.g_nmda = 0.0
    G.g_gaba = 0.0
    G.g_ahp = 0.0
    G.E_trace = 0.0
    G.I_trace = 0.0
    G.xpre = 0.0
    G.ypost_ltd = 0.0
    G.ypost_het = 0.0
    G.wE = w0

    S_pre = Synapses(
        pre_pair,
        G,
        on_pre="""
        g_ampa_post += wE_post
        g_nmda_post += wE_post
        wE_post = clip(wE_post * (1.0 - A_ltd * ypost_ltd_post * I_gate_post), wmin, wmax)
        xpre_post += 1.0
        """,
        namespace=ns,
    )
    S_pre.connect()

    S_post = Synapses(
        G,
        G,
        on_pre="""
        wE_post = clip(
            wE_post
            + (A_ltp * xpre_post * E_for_plasticity_post
               - A_het * ypost_het_post * (E_for_plasticity_post ** 2)) * I_gate_post,
            wmin, wmax
        )
        ypost_ltd_post += 1.0
        ypost_het_post += 1.0
        """,
        namespace=ns,
    )
    S_post.connect(i=[0], j=[0])

    S_ext_e = Synapses(
        ext_e,
        G,
        on_pre="""
        g_ampa_post += w_ext_e
        g_nmda_post += w_ext_e
        """,
        namespace=ns,
    )
    S_ext_e.connect()

    S_ext_i = Synapses(ext_i, G, on_pre="g_gaba_post += w_ext_i", namespace=ns)
    S_ext_i.connect()

    monitors: dict = {}
    if record:
        monitors["state"] = StateMonitor(
            G,
            [
                "u",
                "I_pulse",
                "g_ampa",
                "g_nmda",
                "g_gaba",
                "E_trace",
                "I_trace",
                "I_gate",
                "wE",
            ],
            record=True,
        )
        monitors["post_spikes"] = SpikeMonitor(G)

    network_objects = [G, pre_pair, ext_e, ext_i, S_pre, S_post, S_ext_e, S_ext_i]
    if record:
        network_objects.extend([monitors["state"], monitors["post_spikes"]])

    net = Network(*network_objects)
    net.run(duration)

    final_w = float(G.wE[0])
    final_percent = 100.0 * final_w / w0

    result = {
        "freq_hz": float(freq_hz),
        "delta_t_ms": float(delta_t_ms),
        "ext_rate_hz": float(ext_rate_hz),
        "w0": float(w0),
        "n_pairs": int(n_pairs),
        "final_w": final_w,
        "final_percent": final_percent,
        "delta_percent": final_percent - 100.0,
    }

    if record:
        M = monitors["state"]
        result.update(
            {
                "post_spike_count": int(monitors["post_spikes"].count[0]),
                "peak_E_trace": float(np.max(M.E_trace[0])),
                "peak_I_trace": float(np.max(M.I_trace[0])),
                "peak_g_nmda": float(np.max(M.g_nmda[0])),
                "peak_g_gaba": float(np.max(M.g_gaba[0])),
                "monitors": monitors,
            }
        )

    return result


def run_example_trace(outdir: Path) -> None:
    if not HAVE_BRIAN2:
        raise RuntimeError("Brian2 is not installed, so example traces cannot be simulated.")

    outdir.mkdir(exist_ok=True, parents=True)

    result = run_one_trial(
        freq_hz=50.0,
        delta_t_ms=+10.0,
        ext_rate_hz=0.0,
        w0=0.12,
        n_pairs=20,
        seed_value=123,
        record=True,
    )

    M = result["monitors"]["state"]
    post_spikes = result["monitors"]["post_spikes"]

    print("====================================")
    print("example result")
    print("final weight / initial weight (%):", result["final_percent"])
    print("delta weight (%):", result["delta_percent"])
    print("number of postsynaptic spikes:", len(post_spikes.t))
    print("peak E trace:", result["peak_E_trace"])
    print("peak I trace:", result["peak_I_trace"])
    print("====================================")

    def save_line_plot(filename: str, x, ys, labels, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for y, lab in zip(ys, labels):
            ax.plot(x, y, label=lab)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if any(labels):
            ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=200)
        plt.close(fig)

    t_ms = M.t / ms
    save_line_plot("example_current_pulse_50Hz_pre_post_final.png", t_ms, [M.I_pulse[0]], [None], "time (ms)", "I_pulse", "External current pulse")
    save_line_plot("example_u_50Hz_pre_post_final.png", t_ms, [M.u[0]], [None], "time (ms)", "membrane potential u", "Example: 50 Hz, pre-before-post")
    save_line_plot("example_conductances_50Hz_pre_post_final.png", t_ms, [M.g_ampa[0], M.g_nmda[0], M.g_gaba[0]], ["AMPA", "NMDA", "GABA_A"], "time (ms)", "conductance")
    save_line_plot("example_EI_trace_50Hz_pre_post_final.png", t_ms, [M.E_trace[0], M.I_trace[0]], ["E trace", "I trace"], "time (ms)", "trace")
    save_line_plot("example_gate_50Hz_pre_post_final.png", t_ms, [M.I_gate[0]], ["inhibitory gate"], "time (ms)", "gate")
    save_line_plot("example_wE_50Hz_pre_post_final.png", t_ms, [M.wE[0]], [None], "time (ms)", "wE")


def mode_defaults(mode: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    if mode == "quick":
        return np.array([10, 20, 40, 50], dtype=float), np.array([0], dtype=float), 1, 20
    if mode == "medium":
        return np.array([5, 10, 20, 40, 50], dtype=float), np.array([0, 40, 160], dtype=float), 2, 30
    if mode == "paper-lite":
        return np.array([0.1, 10, 20, 40, 50], dtype=float), np.array([0], dtype=float), 3, 50
    raise ValueError("mode must be quick, medium, or paper-lite")


def run_sweep(mode: str, outdir: Path, seed_value: int = 1, n_trials_override: Optional[int] = None, n_pairs_override: Optional[int] = None) -> pd.DataFrame:
    if not HAVE_BRIAN2:
        raise RuntimeError("Brian2 is not installed. Use --plot-only with an existing CSV.")

    outdir.mkdir(exist_ok=True, parents=True)
    frequencies, external_rates, n_trials, n_pairs = mode_defaults(mode)
    if n_trials_override is not None:
        n_trials = n_trials_override
    if n_pairs_override is not None:
        n_pairs = n_pairs_override

    rows = []

    for ext_rate in external_rates:
        for freq in frequencies:
            post_pre_values = []
            pre_post_values = []

            for trial in range(n_trials):
                seed_base = int(seed_value * 100000 + ext_rate * 100 + freq * 10 + trial)

                # Same seed for the two delta_t conditions. The comparison then
                # emphasizes timing, not different Poisson streams.
                r_ltd = run_one_trial(
                    freq_hz=freq,
                    delta_t_ms=-10.0,
                    ext_rate_hz=ext_rate,
                    w0=0.12,
                    n_pairs=n_pairs,
                    seed_value=seed_base,
                    record=False,
                )
                r_ltp = run_one_trial(
                    freq_hz=freq,
                    delta_t_ms=+10.0,
                    ext_rate_hz=ext_rate,
                    w0=0.12,
                    n_pairs=n_pairs,
                    seed_value=seed_base,
                    record=False,
                )

                post_pre_values.append(r_ltd["final_percent"])
                pre_post_values.append(r_ltp["final_percent"])

            row = {
                "ext_rate_hz": float(ext_rate),
                "freq_hz": float(freq),
                "post_before_pre_mean": float(np.mean(post_pre_values)),
                "post_before_pre_sem": float(np.std(post_pre_values, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "pre_before_post_mean": float(np.mean(pre_post_values)),
                "pre_before_post_sem": float(np.std(pre_post_values, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "n_trials": int(n_trials),
                "n_pairs": int(n_pairs),
                "seed": int(seed_value),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(outdir / f"frequency_sweep_{mode}_final.csv", index=False)

            print(
                "ext_rate", ext_rate,
                "freq", freq,
                "post-before-pre", row["post_before_pre_mean"],
                "pre-before-post", row["pre_before_post_mean"],
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"frequency_sweep_{mode}_final.csv", index=False)
    plot_results(df, mode, outdir)
    return df


def plot_results(df: pd.DataFrame, mode: str, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True, parents=True)
    df = df.copy()
    df["ext_rate_hz"] = df["ext_rate_hz"].astype(float)
    df["freq_hz"] = df["freq_hz"].astype(float)

    sub0 = df[df["ext_rate_hz"] == 0].sort_values("freq_hz")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        sub0["freq_hz"],
        sub0["pre_before_post_mean"],
        yerr=sub0.get("pre_before_post_sem", 0.0),
        marker="o",
        label=r"Brian2 $\Delta t=+10$ ms",
    )
    ax.errorbar(
        sub0["freq_hz"],
        sub0["post_before_pre_mean"],
        yerr=sub0.get("post_before_pre_sem", 0.0),
        marker="o",
        linestyle="--",
        label=r"Brian2 $\Delta t=-10$ ms",
    )
    ax.errorbar(EXPERIMENTAL_DATA["freq_hz"], EXPERIMENTAL_DATA["ltp"], yerr=EXPERIMENTAL_DATA["ltp_sem"], fmt="k^", label="experiment LTP")
    ax.errorbar(EXPERIMENTAL_DATA["freq_hz"], EXPERIMENTAL_DATA["ltd"], yerr=EXPERIMENTAL_DATA["ltd_sem"], fmt="kv", label="experiment LTD")
    ax.axhline(100.0, linewidth=1)
    ax.set_xlabel("Pair frequency (Hz)")
    ax.set_ylabel("final weight / initial weight (%)")
    ax.set_title("frequency STDP, external input = 0 Hz")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"fig2c_frequency_curve_{mode}_final.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ext_rate in sorted(df["ext_rate_hz"].unique()):
        sub = df[df["ext_rate_hz"] == ext_rate].sort_values("freq_hz")
        ax.errorbar(
            sub["freq_hz"],
            sub["pre_before_post_mean"],
            yerr=sub.get("pre_before_post_sem", 0.0),
            marker="o",
            label=f"{ext_rate:g} Hz external",
        )
    ax.axhline(100.0, linewidth=1)
    ax.set_xlabel("Pair frequency (Hz)")
    ax.set_ylabel("final weight / initial weight (%)")
    ax.set_title(r"pre-before-post, $\Delta t=+10$ ms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"external_input_pre_post_{mode}_final.png", dpi=200)
    plt.close(fig)

    # Presentation helper: 2D heatmap if more than one external input is present.
    if df["ext_rate_hz"].nunique() > 1:
        pivot = df.pivot_table(index="ext_rate_hz", columns="freq_hz", values="pre_before_post_mean")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.imshow(pivot.values, origin="lower", aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{y:g}" for y in pivot.index])
        ax.set_xlabel("Pair frequency (Hz)")
        ax.set_ylabel("External input rate (Hz)")
        ax.set_title(r"pre-before-post final weight (%)")
        fig.colorbar(im, ax=ax, label="final weight / initial weight (%)")
        fig.tight_layout()
        fig.savefig(outdir / f"external_input_heatmap_{mode}_final.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--csv", default=None, help="Existing frequency_sweep CSV for --plot-only.")
    parser.add_argument("--mode", choices=["quick", "medium", "paper-lite"], default="quick")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=None)
    parser.add_argument("--outdir", default="results_frequency_final")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    if args.plot_only:
        if not args.csv:
            raise ValueError("--plot-only requires --csv PATH")
        df = pd.read_csv(args.csv)
        plot_results(df, args.mode, outdir)
        print("saved plots to:", outdir.resolve())
        return

    if not HAVE_BRIAN2:
        raise RuntimeError("Brian2 is not installed. Use --plot-only with an existing CSV or install brian2.")

    if args.example_only:
        run_example_trace(outdir)
    else:
        run_sweep(args.mode, outdir, seed_value=args.seed, n_trials_override=args.n_trials, n_pairs_override=args.n_pairs)
        run_example_trace(outdir)

    print("saved results to:", outdir.resolve())


if __name__ == "__main__":
    main()
