from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Windows-safe target
prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms


EXPERIMENTAL_DATA = pd.DataFrame(
    {
        "freq_hz": [0.1, 10, 20, 40, 50],
        "ltp": [96.7266, 115.311, 130.02, 153.342, 155.661],
        "ltd": [68.7583, 57.0747, 65.3493, 152.192, 171.331],
        "ltp_sem": [4.08049, 9.45575, 13.6743, 10.5622, 26.0755],
        "ltd_sem": [7.43512, 10.9595, 8.5702, 30.2612, 18.4164],
    }
)


def make_protocol(freq_hz, delta_t_ms, n_pairs, warmup_ms=100.0):
    """
    delta_t_ms = t_post - t_pre

    +10 ms: pre-before-post
    -10 ms: post-before-pre

    Fortran frequency_STDP:
    post spike is induced by external current x(10), not forced directly.
    """

    pair_period_ms = 1000.0 / freq_hz

    # Fortran config:
    # p_inp(11) = 300 mV
    # p_inp(12) = 2 ms
    pulse_amp = 300.0
    pulse_duration_ms = 2.0

    pre_times_ms = []
    pulse_on_ms = []
    pulse_off_ms = []

    start_ms = warmup_ms + 20.0

    for n in range(n_pairs):
        base = start_ms + n * pair_period_ms

        if delta_t_ms > 0:
            # pre-before-post
            pre_t = base
            desired_post_t = base + delta_t_ms
        else:
            # post-before-pre
            desired_post_t = base
            pre_t = base + abs(delta_t_ms)

        # Match Fortran idea:
        # current pulse happens immediately before the desired post-spike time.
        pulse_on_t = desired_post_t - pulse_duration_ms
        pulse_off_t = desired_post_t

        pre_times_ms.append(pre_t)
        pulse_on_ms.append(pulse_on_t)
        pulse_off_ms.append(pulse_off_t)

    duration_ms = max(max(pre_times_ms), max(pulse_off_ms)) + 100.0

    # TimedArray for current pulse
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
    freq_hz,
    delta_t_ms,
    ext_rate_hz=0.0,
    w0=0.12,
    n_pairs=20,
    seed_value=1,
    record=False,
):
    start_scope()
    defaultclock.dt = 0.1 * ms

    seed(seed_value)
    np.random.seed(seed_value)

    pre_times, pulse_func, duration = make_protocol(
        freq_hz=freq_hz,
        delta_t_ms=delta_t_ms,
        n_pairs=n_pairs,
    )

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

        # external input weights
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

        # traces
        "tau_E": 50.0 * ms,
        "tau_I": 500.0 * ms,

        # bounds
        "wmin": 1e-5,
        "wmax": 1.0,

        # TimedArray current pulse
        "pulse_func": pulse_func,
    }

    pre_pair = SpikeGeneratorGroup(
        1,
        indices=np.zeros(len(pre_times), dtype=int),
        times=pre_times,
    )

    ext_e = PoissonGroup(1, rates=ext_rate_hz * Hz)
    ext_i = PoissonGroup(1, rates=ext_rate_hz * Hz)

    eqs = """
    du/dt = (-(u - u_rest)
             - g_ampa * u
             - g_nmda * H_nmda * u
             - g_gaba * (u - E_gaba)
             - A_ahp * g_ahp * (u - E_ahp)
             + I_pulse) / tau_m : 1

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

    # initial_conditions.f90
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

    # Paired plastic pre input
    S_pre = Synapses(
        pre_pair,
        G,
        on_pre="""
        g_ampa_post += wE_post
        g_nmda_post += wE_post
        wE_post = clip(wE_post * (1.0 - A_ltd * ypost_ltd_post * exp(-I_trace_post / I_star)), wmin, wmax)
        xpre_post += 1.0
        """,
        namespace=ns,
    )
    S_pre.connect()

    # Postsynaptic spike triggers LTP / heterosynaptic LTD
    S_post = Synapses(
        G,
        G,
        on_pre="""
        wE_post = clip(wE_post + (A_ltp * xpre_post * E_trace_post - A_het * ypost_het_post * (E_trace_post ** 2)) * exp(-I_trace_post / I_star), wmin, wmax)
        ypost_ltd_post += 1.0
        ypost_het_post += 1.0
        """,
        namespace=ns,
    )
    S_post.connect(i=[0], j=[0])

    # External excitatory input
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

    # External inhibitory input
    S_ext_i = Synapses(
        ext_i,
        G,
        on_pre="g_gaba_post += w_ext_i",
        namespace=ns,
    )
    S_ext_i.connect()

    monitors = {}

    if record:
        monitors["state"] = StateMonitor(
            G,
            ["u", "I_pulse", "g_ampa", "g_nmda", "g_gaba", "E_trace", "I_trace", "wE"],
            record=True,
        )
        monitors["post_spikes"] = SpikeMonitor(G)

        # ------------------------------------------------------------
    # Explicit Network for Windows / VS Code stability
    # ------------------------------------------------------------
    network_objects = [
        G,
        pre_pair,
        ext_e,
        ext_i,
        S_pre,
        S_post,
        S_ext_e,
        S_ext_i,
    ]

    if record:
        network_objects.append(monitors["state"])
        network_objects.append(monitors["post_spikes"])

    net = Network(network_objects)
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
        result["monitors"] = monitors

    return result


def run_example_trace(outdir):
    outdir.mkdir(exist_ok=True)

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

    if len(M.t) == 0:
        print("ERROR: StateMonitor recorded zero time points.")
        print("Check whether the Network includes the StateMonitor.")
        return

    max_u = float(np.max(M.u[0]))
    max_pulse = float(np.max(M.I_pulse[0]))

    print("====================================")
    print("example result")
    print("final weight / initial weight (%):", result["final_percent"])
    print("delta weight (%):", result["delta_percent"])
    print("number of postsynaptic spikes:", len(post_spikes.t))
    print("max membrane potential:", max_u)
    print("max current pulse:", max_pulse)
    print("====================================")

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / ms, M.I_pulse[0])
    plt.xlabel("time (ms)")
    plt.ylabel("I_pulse")
    plt.title("External current pulse")
    plt.tight_layout()
    plt.savefig(outdir / "example_current_pulse_50Hz_pre_post.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / ms, M.u[0])
    plt.xlabel("time (ms)")
    plt.ylabel("membrane potential u")
    plt.title("Example: 50 Hz, pre-before-post")
    plt.tight_layout()
    plt.savefig(outdir / "example_u_50Hz_pre_post.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / ms, M.g_ampa[0], label="AMPA")
    plt.plot(M.t / ms, M.g_nmda[0], label="NMDA")
    plt.plot(M.t / ms, M.g_gaba[0], label="GABA_A")
    plt.xlabel("time (ms)")
    plt.ylabel("conductance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example_conductances_50Hz_pre_post.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / ms, M.E_trace[0], label="E trace")
    plt.plot(M.t / ms, M.I_trace[0], label="I trace")
    plt.xlabel("time (ms)")
    plt.ylabel("trace")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example_EI_trace_50Hz_pre_post.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / ms, M.wE[0])
    plt.xlabel("time (ms)")
    plt.ylabel("wE")
    plt.tight_layout()
    plt.savefig(outdir / "example_wE_50Hz_pre_post.png", dpi=200)
    plt.close()


def run_sweep(mode, outdir):
    outdir.mkdir(exist_ok=True)

    if mode == "quick":
        frequencies = np.array([10, 20, 40, 50], dtype=float)
        external_rates = np.array([0], dtype=float)
        n_trials = 1
        n_pairs = 20

    elif mode == "medium":
        frequencies = np.array([5, 10, 20, 40, 50], dtype=float)
        external_rates = np.array([0, 40, 160], dtype=float)
        n_trials = 2
        n_pairs = 30

    elif mode == "paper-lite":
        frequencies = np.array([0.1, 10, 20, 40, 50], dtype=float)
        external_rates = np.array([0], dtype=float)
        n_trials = 3
        n_pairs = 50

    else:
        raise ValueError("mode must be quick, medium, or paper-lite")

    rows = []

    for ext_rate in external_rates:
        for freq in frequencies:
            post_pre_values = []
            pre_post_values = []

            for trial in range(n_trials):
                seed_base = 100000 + int(ext_rate * 100) + int(freq * 10) + trial

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
                    seed_value=seed_base + 1,
                    record=False,
                )

                post_pre_values.append(r_ltd["final_percent"])
                pre_post_values.append(r_ltp["final_percent"])

            row = {
                "ext_rate_hz": ext_rate,
                "freq_hz": freq,
                "post_before_pre_mean": float(np.mean(post_pre_values)),
                "post_before_pre_sem": float(np.std(post_pre_values, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "pre_before_post_mean": float(np.mean(pre_post_values)),
                "pre_before_post_sem": float(np.std(pre_post_values, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "n_trials": n_trials,
                "n_pairs": n_pairs,
            }

            rows.append(row)

            print(
                "ext_rate", ext_rate,
                "freq", freq,
                "post-before-pre", row["post_before_pre_mean"],
                "pre-before-post", row["pre_before_post_mean"],
            )

            pd.DataFrame(rows).to_csv(outdir / f"frequency_sweep_{mode}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"frequency_sweep_{mode}.csv", index=False)

    plot_results(df, mode, outdir)


def plot_results(df, mode, outdir):
    sub0 = df[df["ext_rate_hz"] == 0].sort_values("freq_hz")

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        sub0["freq_hz"],
        sub0["pre_before_post_mean"],
        yerr=sub0["pre_before_post_sem"],
        marker="o",
        label=r"Brian2 $\Delta t=+10$ ms",
    )
    plt.errorbar(
        sub0["freq_hz"],
        sub0["post_before_pre_mean"],
        yerr=sub0["post_before_pre_sem"],
        marker="o",
        linestyle="--",
        label=r"Brian2 $\Delta t=-10$ ms",
    )

    plt.errorbar(
        EXPERIMENTAL_DATA["freq_hz"],
        EXPERIMENTAL_DATA["ltp"],
        yerr=EXPERIMENTAL_DATA["ltp_sem"],
        fmt="k^",
        label="experiment LTP",
    )
    plt.errorbar(
        EXPERIMENTAL_DATA["freq_hz"],
        EXPERIMENTAL_DATA["ltd"],
        yerr=EXPERIMENTAL_DATA["ltd_sem"],
        fmt="kv",
        label="experiment LTD",
    )

    plt.axhline(100.0, linewidth=1)
    plt.xlabel("Pair frequency (Hz)")
    plt.ylabel("final weight / initial weight (%)")
    plt.title("frequency STDP, external input = 0 Hz")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"fig2c_frequency_curve_{mode}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))

    for ext_rate in sorted(df["ext_rate_hz"].unique()):
        sub = df[df["ext_rate_hz"] == ext_rate].sort_values("freq_hz")
        plt.plot(
            sub["freq_hz"],
            sub["pre_before_post_mean"],
            marker="o",
            label=f"{ext_rate:g} Hz external",
        )

    plt.axhline(100.0, linewidth=1)
    plt.xlabel("Pair frequency (Hz)")
    plt.ylabel("final weight / initial weight (%)")
    plt.title(r"pre-before-post, $\Delta t=+10$ ms")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"external_input_pre_post_{mode}.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example-only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["quick", "medium", "paper-lite"],
        default="quick",
    )
    args = parser.parse_args()

    outdir = Path("results_frequency")
    outdir.mkdir(exist_ok=True)

    if args.example_only:
        run_example_trace(outdir)
    else:
        run_sweep(args.mode, outdir)
        run_example_trace(outdir)

    print("saved results to:", outdir.resolve())


if __name__ == "__main__":
    main()