from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

prefs.codegen.target = "numpy"


def get_mode_config(mode):
    if mode == "quick":
        return {
            "N_E": 80,
            "N_I": 20,
            "block_duration_s": 60.0,
            "dt_ms": 1.0,
            "record_dt_s": 1.0,
            "plasticity_scale": 10.0,
        }

    if mode == "medium":
        return {
            "N_E": 160,
            "N_I": 40,
            "block_duration_s": 120.0,
            "dt_ms": 1.0,
            "record_dt_s": 1.0,
            "plasticity_scale": 2.0,
        }

    if mode == "full-lite":
        return {
            "N_E": 800,
            "N_I": 200,
            "block_duration_s": 300.0,
            "dt_ms": 0.5,
            "record_dt_s": 2.0,
            "plasticity_scale": 1.0,
        }

    raise ValueError("mode must be quick, medium, or full-lite")


def target_rate_schedule(block_duration_s):
    """
    Fortran fixed point file:
        0-10 h   7 Hz
        10-20 h  11 Hz
        20-30 h  7 Hz
        30-40 h  4 Hz

    Here we compress hours into simulation blocks.
    """
    return [
        {"A_nt": 0.0094, "target_hz": 7.0, "duration_s": block_duration_s},
        {"A_nt": 0.0150, "target_hz": 11.0, "duration_s": block_duration_s},
        {"A_nt": 0.0094, "target_hz": 7.0, "duration_s": block_duration_s},
        {"A_nt": 0.0050, "target_hz": 4.0, "duration_s": block_duration_s},
    ]


def run_one_condition(rule="spike_i", mode="quick", seed_value=1):
    """
    rule:
        spike_i  : spike-based E + spike-based inhibitory plasticity
        codep_i  : spike-based E + codependent inhibitory plasticity

    This follows the structure:
        simulation1 = plasticity_orch + plasticity_istdp
        simulation2 = plasticity_orch + plasticity_i
    """

    cfg = get_mode_config(mode)

    start_scope()
    defaultclock.dt = cfg["dt_ms"] * ms

    seed(seed_value)
    np.random.seed(seed_value)

    N_E = cfg["N_E"]
    N_I = cfg["N_I"]
    record_dt = cfg["record_dt_s"] * second
    plasticity_scale = cfg["plasticity_scale"]

    # ------------------------------------------------------------
    # Fortran config parameters
    # ------------------------------------------------------------
    ns = {
        # neuron
        "tau_m": 30.0 * ms,
        "u_rest": -65.0,
        "u_th": -50.0,
        "u_reset": -60.0,
        "tau_ref": 5.0 * ms,
        "E_gaba": -80.0,
        "E_ahp": -80.0,
        "A_ahp": 5.0,
        "tau_ahp": 100.0 * ms,

        # synapse
        "tau_ampa": 5.0 * ms,
        "tau_gaba": 10.0 * ms,
        "tau_nmda": 150.0 * ms,
        "a_nmda": 0.15,
        "b_nmda": -0.08,

        # input rates
        "rate_E": 10.0 * Hz,
        "rate_I": 20.0 * Hz,

        # initial weights
        "wE0": 0.11,
        "wI0": 0.7,

        # bounds
        "wEmax": 1.15,
        "wImax": 10.0,
        "wmin": 1e-6,

        # excitatory spike-based plasticity
        "tau_e_pre": 16.8 * ms,
        "tau_e_post": 33.7 * ms,
        "tau_e_triplet": 100.0 * ms,
        "A_e_ltp": 0.001 * plasticity_scale,
        "A_e_ltd": 0.045 * plasticity_scale,

        # inhibitory spike-based plasticity
        "tau_i": 20.0 * ms,
        "A_i_spike": 0.005 * plasticity_scale,
        "beta_i": 0.28,

        # codependent inhibitory plasticity
        "A_i_codep": 0.00015 * plasticity_scale,
        "alpha_balance": 0.855,
        "balance_amp": 0.0001,

        # E/I current traces
        "tau_E_trace": 10.0 * ms,
        "tau_I_trace": 100.0 * ms,
    }

    # ------------------------------------------------------------
    # Input groups
    # ------------------------------------------------------------
    P_E = PoissonGroup(N_E, rates=ns["rate_E"])
    P_I = PoissonGroup(N_I, rates=ns["rate_I"])

    # ------------------------------------------------------------
    # Postsynaptic neuron
    # ------------------------------------------------------------
    eqs = """
    du/dt = (-(u - u_rest)
             - g_ampa * u
             - g_nmda * H_nmda * u
             - g_gaba * (u - E_gaba)
             - A_ahp * g_ahp * (u - E_ahp)) / tau_m : 1 (unless refractory)

    dg_ampa/dt = -g_ampa / tau_ampa : 1
    dg_nmda/dt = -g_nmda / tau_nmda : 1
    dg_gaba/dt = -g_gaba / tau_gaba : 1
    dg_ahp/dt = -g_ahp / tau_ahp : 1

    dE_trace/dt = (-E_trace - g_nmda * H_nmda * u) / tau_E_trace : 1
    dI_trace/dt = (-I_trace + g_gaba * (u - E_gaba)) / tau_I_trace : 1

    H_nmda = 1.0 / (1.0 + a_nmda * exp(b_nmda * u)) : 1
    balance_drive = balance_amp * E_trace * (E_trace - alpha_balance * I_trace) : 1

    I_nmda = -g_nmda * H_nmda * u : 1
    I_gaba = g_gaba * (u - E_gaba) : 1
    """

    G = NeuronGroup(
        1,
        eqs,
        threshold="u >= u_th",
        reset="""
        u = u_reset
        g_ahp += 1.0
        """,
        refractory=ns["tau_ref"],
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

    # ------------------------------------------------------------
    # Excitatory synapses: plasticity_orch()
    # ------------------------------------------------------------
    E_syn_eqs = """
    dxpre/dt = -xpre / tau_e_pre : 1 (event-driven)
    dypost/dt = -ypost / tau_e_post : 1 (event-driven)
    dytriplet/dt = -ytriplet / tau_e_triplet : 1 (event-driven)
    w : 1
    A_nt : 1 (shared)
    """

    S_E = Synapses(
        P_E,
        G,
        model=E_syn_eqs,
        on_pre="""
        g_ampa_post += w
        g_nmda_post += w
        w = clip(w - A_e_ltd * ypost + A_nt, wmin, wEmax)
        xpre += 1.0
        """,
        on_post="""
        w = clip(w + A_e_ltp * xpre * ytriplet, wmin, wEmax)
        ypost += 1.0
        ytriplet += 1.0
        """,
        namespace=ns,
    )
    S_E.connect()
    S_E.w = ns["wE0"]
    S_E.xpre = 0.0
    S_E.ypost = 0.0
    S_E.ytriplet = 0.0
    S_E.A_nt = 0.0094 * plasticity_scale

    # ------------------------------------------------------------
    # Inhibitory synapses
    # ------------------------------------------------------------
    I_syn_eqs = """
    dxpre/dt = -xpre / tau_i : 1 (event-driven)
    dypost/dt = -ypost / tau_i : 1 (event-driven)
    w : 1
    """

    if rule == "spike_i":
        # plasticity_istdp()
        S_I = Synapses(
            P_I,
            G,
            model=I_syn_eqs,
            on_pre="""
            g_gaba_post += w
            w = clip(w + A_i_spike * (ypost - beta_i), wmin, wImax)
            xpre += 1.0
            """,
            on_post="""
            w = clip(w + A_i_spike * xpre, wmin, wImax)
            ypost += 1.0
            """,
            namespace=ns,
        )

    elif rule == "codep_i":
        # plasticity_i()
        S_I = Synapses(
            P_I,
            G,
            model=I_syn_eqs,
            on_pre="""
            g_gaba_post += w
            w = clip(w + A_i_codep * balance_drive_post * ypost, wmin, wImax)
            xpre += 1.0
            """,
            on_post="""
            w = clip(w + A_i_codep * balance_drive_post * xpre, wmin, wImax)
            ypost += 1.0
            """,
            namespace=ns,
        )

    else:
        raise ValueError("rule must be spike_i or codep_i")

    S_I.connect()
    S_I.w = ns["wI0"]
    S_I.xpre = 0.0
    S_I.ypost = 0.0

    # ------------------------------------------------------------
    # Monitors
    # ------------------------------------------------------------
    M_G = StateMonitor(
        G,
        ["u", "g_ampa", "g_nmda", "g_gaba", "E_trace", "I_trace", "I_nmda", "I_gaba", "balance_drive"],
        record=True,
        dt=record_dt,
    )

    M_E_w = StateMonitor(S_E, "w", record=True, dt=record_dt)
    M_I_w = StateMonitor(S_I, "w", record=True, dt=record_dt)
    post_spikes = SpikeMonitor(G)

    net = Network(P_E, P_I, G, S_E, S_I, M_G, M_E_w, M_I_w, post_spikes)

    # ------------------------------------------------------------
    # Run compressed 4-block protocol
    # ------------------------------------------------------------
    schedule = target_rate_schedule(cfg["block_duration_s"])

    elapsed_s = 0.0
    target_segments = []

    for block in schedule:
        S_E.A_nt = block["A_nt"] * plasticity_scale

        print(
            "rule", rule,
            "block A_nt", block["A_nt"],
            "target", block["target_hz"],
            "duration", block["duration_s"], "s",
        )

        net.run(block["duration_s"] * second)

        target_segments.append(
            {
                "start_s": elapsed_s,
                "end_s": elapsed_s + block["duration_s"],
                "target_hz": block["target_hz"],
            }
        )
        elapsed_s += block["duration_s"]

    # ------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------
    t_s = np.asarray(M_G.t / second)
    dt_rec = cfg["record_dt_s"]

    spike_times_s = np.asarray(post_spikes.t / second)
    edges = np.append(t_s, t_s[-1] + dt_rec)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    firing_rate_hz = counts / dt_rec

    mean_wE = np.mean(M_E_w.w, axis=0)
    mean_wI = np.mean(M_I_w.w, axis=0)

    norm_wE = mean_wE / ns["wE0"]
    norm_wI = mean_wI / ns["wI0"]

    ei_ratio = np.asarray(M_G.I_nmda[0]) / (np.asarray(M_G.I_gaba[0]) + 1e-9)

    df = pd.DataFrame(
        {
            "time_s": t_s,
            "time_min": t_s / 60.0,
            "firing_rate_hz": firing_rate_hz,
            "mean_wE": mean_wE,
            "mean_wI": mean_wI,
            "norm_wE": norm_wE,
            "norm_wI": norm_wI,
            "E_trace": np.asarray(M_G.E_trace[0]),
            "I_trace": np.asarray(M_G.I_trace[0]),
            "I_nmda": np.asarray(M_G.I_nmda[0]),
            "I_gaba": np.asarray(M_G.I_gaba[0]),
            "ei_ratio": ei_ratio,
            "balance_drive": np.asarray(M_G.balance_drive[0]),
            "rule": rule,
        }
    )

    target_df = pd.DataFrame(target_segments)

    return df, target_df


def plot_condition(df, target_df, rule, outdir):
    outdir.mkdir(exist_ok=True)

    label = "Spike-based E + spike-based I" if rule == "spike_i" else "Spike-based E + codependent I"

    plt.figure(figsize=(8, 4))
    plt.plot(df["time_min"], df["firing_rate_hz"], label="postsynaptic firing rate")

    for _, row in target_df.iterrows():
        plt.hlines(
            row["target_hz"],
            row["start_s"] / 60.0,
            row["end_s"] / 60.0,
            linestyles="--",
        )

    plt.xlabel("time (min)")
    plt.ylabel("firing rate (Hz)")
    plt.title(label)
    plt.tight_layout()
    plt.savefig(outdir / f"{rule}_firing_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["time_min"], df["norm_wE"], label="E weight")
    plt.plot(df["time_min"], df["norm_wI"], label="I weight")
    plt.xlabel("time (min)")
    plt.ylabel("normalised weight")
    plt.title(label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{rule}_weights.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["time_min"], df["ei_ratio"])
    plt.xlabel("time (min)")
    plt.ylabel("NMDA / GABA current")
    plt.title(label)
    plt.tight_layout()
    plt.savefig(outdir / f"{rule}_ei_ratio.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["time_min"], df["E_trace"], label="E trace")
    plt.plot(df["time_min"], df["I_trace"], label="I trace")
    plt.xlabel("time (min)")
    plt.ylabel("trace")
    plt.title(label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{rule}_traces.png", dpi=200)
    plt.close()


def plot_comparison(df_spike, df_codep, target_df, mode, outdir):
    outdir.mkdir(exist_ok=True)

    plt.figure(figsize=(11, 8))

    # firing rate
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df_spike["time_min"], df_spike["firing_rate_hz"])
    for _, row in target_df.iterrows():
        ax1.hlines(row["target_hz"], row["start_s"] / 60.0, row["end_s"] / 60.0, linestyles="--")
    ax1.set_title("Spike-based E and I")
    ax1.set_ylabel("Firing rate (Hz)")

    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(df_codep["time_min"], df_codep["firing_rate_hz"])
    for _, row in target_df.iterrows():
        ax2.hlines(row["target_hz"], row["start_s"] / 60.0, row["end_s"] / 60.0, linestyles="--")
    ax2.set_title("Spike-based E + codependent I")

    # E/I ratio
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(df_spike["time_min"], df_spike["ei_ratio"])
    ax3.set_ylabel("NMDA/GABA")

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(df_codep["time_min"], df_codep["ei_ratio"])
    ax4.axhline(0.855, linestyle="--")

    # weights
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(df_spike["time_min"], df_spike["norm_wE"], label="E")
    ax5.plot(df_spike["time_min"], df_spike["norm_wI"], label="I")
    ax5.set_ylabel("Normalised weight")
    ax5.set_xlabel("time (min)")
    ax5.legend()

    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(df_codep["time_min"], df_codep["norm_wE"], label="E")
    ax6.plot(df_codep["time_min"], df_codep["norm_wI"], label="I")
    ax6.set_xlabel("time (min)")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(outdir / f"ei_balance_summary_{mode}.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["quick", "medium", "full-lite"],
        default="quick",
    )
    parser.add_argument(
        "--condition",
        choices=["both", "spike_i", "codep_i"],
        default="both",
    )

    args = parser.parse_args()

    outdir = Path("results_ei_balance")
    outdir.mkdir(exist_ok=True)

    if args.condition in ["both", "spike_i"]:
        df_spike, target_df = run_one_condition(rule="spike_i", mode=args.mode, seed_value=1)
        df_spike.to_csv(outdir / f"spike_i_{args.mode}.csv", index=False)
        plot_condition(df_spike, target_df, "spike_i", outdir)

    if args.condition in ["both", "codep_i"]:
        df_codep, target_df = run_one_condition(rule="codep_i", mode=args.mode, seed_value=2)
        df_codep.to_csv(outdir / f"codep_i_{args.mode}.csv", index=False)
        plot_condition(df_codep, target_df, "codep_i", outdir)

    if args.condition == "both":
        plot_comparison(df_spike, df_codep, target_df, args.mode, outdir)

    print("saved:", outdir.resolve())


if __name__ == "__main__":
    main()