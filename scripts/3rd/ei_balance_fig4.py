"""
ei_balance_brian2_final.py

Cleaned and presentation-oriented version of the Fig. 4 EI-balance script.

Main fixes compared with the draft:
1. The main EI-ratio diagnostic is now E_trace / I_trace, because the
   codependent inhibitory rule actually uses the low-pass traces E_trace and
   I_trace. The instantaneous current ratio I_nmda / I_gaba is still saved as
   raw_current_ratio, but it is not used as the primary balance plot.
2. The two conditions can be run with the same seed for a controlled comparison.
3. Plotting is separated from simulation. You can regenerate clean plots from
   existing CSV files with --plot-only, even on a machine without Brian2.
4. Comparison plots show firing-rate tracking, trace-based EI balance, and
   normalized weights in one presentation-ready figure.
5. The codependent balance term explicitly uses non-negative E/I traces, matching
   the interpretation E(E - alpha I).

Usage examples:
    python ei_balance_brian2_final.py --mode quick --condition both --seed 1
    python ei_balance_brian2_final.py --mode medium --condition both --seed 1

Plot existing CSVs without rerunning Brian2:
    python ei_balance_brian2_final.py --plot-only \
        --csv-spike spike_i_medium.csv \
        --csv-codep codep_i_medium.csv \
        --mode medium
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from brian2 import (  # type: ignore
        Hz,
        Network,
        NeuronGroup,
        PoissonGroup,
        SpikeMonitor,
        StateMonitor,
        Synapses,
        clip,
        defaultclock,
        exp,
        ms,
        prefs,
        second,
        seed,
        start_scope,
    )

    prefs.codegen.target = "numpy"
    HAVE_BRIAN2 = True
except Exception:
    HAVE_BRIAN2 = False


ALPHA_BALANCE = 0.855
EPS = 1e-9


def get_mode_config(mode: str) -> dict:
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


def target_rate_schedule(block_duration_s: float) -> list[dict]:
    """
    Compressed analogue of the paper's four firing-rate regimes.
    The dashed lines in the plots are only a guide for the firing-rate setpoint.
    """
    return [
        {"A_nt": 0.0094, "target_hz": 7.0, "duration_s": block_duration_s},
        {"A_nt": 0.0150, "target_hz": 11.0, "duration_s": block_duration_s},
        {"A_nt": 0.0094, "target_hz": 7.0, "duration_s": block_duration_s},
        {"A_nt": 0.0050, "target_hz": 4.0, "duration_s": block_duration_s},
    ]


def build_target_df(block_duration_s: float) -> pd.DataFrame:
    elapsed_s = 0.0
    rows = []
    for block in target_rate_schedule(block_duration_s):
        rows.append(
            {
                "start_s": elapsed_s,
                "end_s": elapsed_s + block["duration_s"],
                "target_hz": block["target_hz"],
            }
        )
        elapsed_s += block["duration_s"]
    return pd.DataFrame(rows)


def rolling_mean(values: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    return values.rolling(window=window, center=True, min_periods=1).mean()


def add_derived_columns(df: pd.DataFrame, record_dt_s: float = 1.0, smooth_s: float = 5.0) -> pd.DataFrame:
    """Add presentation-friendly diagnostics without changing the raw data."""
    out = df.copy()

    if "time_min" not in out.columns and "time_s" in out.columns:
        out["time_min"] = out["time_s"] / 60.0

    out["trace_ratio"] = out["E_trace"] / (out["I_trace"] + EPS)
    out["raw_current_ratio"] = out["I_nmda"] / (out["I_gaba"] + EPS)
    out["balance_error"] = out["trace_ratio"] - ALPHA_BALANCE

    smooth_n = max(1, int(round(smooth_s / record_dt_s)))
    for col in [
        "firing_rate_hz",
        "norm_wE",
        "norm_wI",
        "E_trace",
        "I_trace",
        "trace_ratio",
        "raw_current_ratio",
        "balance_drive",
    ]:
        if col in out.columns:
            out[f"{col}_smooth"] = rolling_mean(out[col], smooth_n)

    return out


def run_one_condition(rule: str = "spike_i", mode: str = "quick", seed_value: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    rule:
        spike_i : spike-based E + spike-based inhibitory plasticity
        codep_i : spike-based E + codependent inhibitory plasticity
    """
    if not HAVE_BRIAN2:
        raise RuntimeError(
            "Brian2 is not installed in this environment. Install brian2 or use --plot-only with existing CSV files."
        )

    cfg = get_mode_config(mode)

    start_scope()
    defaultclock.dt = cfg["dt_ms"] * ms

    seed(seed_value)
    np.random.seed(seed_value)

    N_E = cfg["N_E"]
    N_I = cfg["N_I"]
    record_dt = cfg["record_dt_s"] * second
    plasticity_scale = cfg["plasticity_scale"]

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
        "alpha_balance": ALPHA_BALANCE,
        "balance_amp": 0.0001,
        # E/I current traces
        "tau_E_trace": 10.0 * ms,
        "tau_I_trace": 100.0 * ms,
    }

    P_E = PoissonGroup(N_E, rates=ns["rate_E"])
    P_I = PoissonGroup(N_I, rates=ns["rate_I"])

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

    E_pos = clip(E_trace, 0.0, 1000.0) : 1
    I_pos = clip(I_trace, 0.0, 1000.0) : 1
    balance_drive = balance_amp * E_pos * (E_pos - alpha_balance * I_pos) : 1

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

    I_syn_eqs = """
    dxpre/dt = -xpre / tau_i : 1 (event-driven)
    dypost/dt = -ypost / tau_i : 1 (event-driven)
    w : 1
    """

    if rule == "spike_i":
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
            {"start_s": elapsed_s, "end_s": elapsed_s + block["duration_s"], "target_hz": block["target_hz"]}
        )
        elapsed_s += block["duration_s"]

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
            "balance_drive": np.asarray(M_G.balance_drive[0]),
            "rule": rule,
        }
    )

    # Compatibility with older output name, but the meaning is now trace-based.
    df = add_derived_columns(df, record_dt_s=dt_rec)
    df["ei_ratio"] = df["trace_ratio"]

    return df, pd.DataFrame(target_segments)


def add_target_lines(ax, target_df: pd.DataFrame) -> None:
    for _, row in target_df.iterrows():
        ax.hlines(
            row["target_hz"],
            row["start_s"] / 60.0,
            row["end_s"] / 60.0,
            linestyles="--",
            linewidth=1.2,
        )


def plot_condition(df: pd.DataFrame, target_df: pd.DataFrame, rule: str, outdir: Path, record_dt_s: float) -> None:
    outdir.mkdir(exist_ok=True)
    df = add_derived_columns(df, record_dt_s=record_dt_s)

    label = "Spike-based E + spike-based I" if rule == "spike_i" else "Spike-based E + codependent I"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_min"], df["firing_rate_hz"], alpha=0.45, label="raw")
    ax.plot(df["time_min"], df["firing_rate_hz_smooth"], linewidth=2.0, label="smoothed")
    add_target_lines(ax, target_df)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{rule}_firing_rate_clean.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_min"], df["norm_wE"], alpha=0.35, label="E raw")
    ax.plot(df["time_min"], df["norm_wI"], alpha=0.35, label="I raw")
    ax.plot(df["time_min"], df["norm_wE_smooth"], linewidth=2.0, label="E smoothed")
    ax.plot(df["time_min"], df["norm_wI_smooth"], linewidth=2.0, label="I smoothed")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("normalized weight")
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{rule}_weights_clean.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_min"], df["trace_ratio"], alpha=0.35, label="E_trace / I_trace raw")
    ax.plot(df["time_min"], df["trace_ratio_smooth"], linewidth=2.0, label="smoothed")
    ax.axhline(ALPHA_BALANCE, linestyle="--", linewidth=1.5, label=f"alpha = {ALPHA_BALANCE}")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("trace-based E/I")
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{rule}_trace_ratio_clean.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_min"], df["raw_current_ratio"], alpha=0.5)
    ax.axhline(ALPHA_BALANCE, linestyle="--", linewidth=1.5)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("instantaneous NMDA/GABA")
    ax.set_title(label + " (raw current ratio; diagnostic only)")
    fig.tight_layout()
    fig.savefig(outdir / f"{rule}_raw_current_ratio_diagnostic.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_min"], df["E_trace_smooth"], label="E trace")
    ax.plot(df["time_min"], df["I_trace_smooth"], label="I trace")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("trace")
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{rule}_traces_clean.png", dpi=200)
    plt.close(fig)


def plot_comparison(
    df_spike: pd.DataFrame,
    df_codep: pd.DataFrame,
    target_df: pd.DataFrame,
    mode: str,
    outdir: Path,
    record_dt_s: float,
) -> None:
    outdir.mkdir(exist_ok=True)
    df_spike = add_derived_columns(df_spike, record_dt_s=record_dt_s)
    df_codep = add_derived_columns(df_codep, record_dt_s=record_dt_s)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)

    # Row 1: firing rate
    ax = axes[0, 0]
    ax.plot(df_spike["time_min"], df_spike["firing_rate_hz_smooth"])
    add_target_lines(ax, target_df)
    ax.set_title("Spike-based E + spike-based I")
    ax.set_ylabel("Firing rate (Hz)")

    ax = axes[0, 1]
    ax.plot(df_codep["time_min"], df_codep["firing_rate_hz_smooth"])
    add_target_lines(ax, target_df)
    ax.set_title("Spike-based E + codependent I")

    # Row 2: trace-based EI ratio
    ax = axes[1, 0]
    ax.plot(df_spike["time_min"], df_spike["trace_ratio_smooth"])
    ax.axhline(ALPHA_BALANCE, linestyle="--", linewidth=1.5)
    ax.set_ylabel("E_trace / I_trace")
    ax.set_ylim(0, min(10.0, max(3.0, np.nanpercentile(df_spike["trace_ratio_smooth"], 95) * 1.1)))

    ax = axes[1, 1]
    ax.plot(df_codep["time_min"], df_codep["trace_ratio_smooth"])
    ax.axhline(ALPHA_BALANCE, linestyle="--", linewidth=1.5)
    ax.set_ylim(0, min(3.0, max(1.5, np.nanpercentile(df_codep["trace_ratio_smooth"], 99) * 1.2)))

    # Row 3: normalized weights
    ax = axes[2, 0]
    ax.plot(df_spike["time_min"], df_spike["norm_wE_smooth"], label="E")
    ax.plot(df_spike["time_min"], df_spike["norm_wI_smooth"], label="I")
    ax.set_ylabel("Normalized weight")
    ax.set_xlabel("time (min)")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(df_codep["time_min"], df_codep["norm_wE_smooth"], label="E")
    ax.plot(df_codep["time_min"], df_codep["norm_wI_smooth"], label="I")
    ax.set_xlabel("time (min)")
    ax.legend()

    fig.suptitle(f"EI balance comparison ({mode})", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / f"ei_balance_summary_{mode}_clean.png", dpi=200)
    plt.close(fig)

    # Focused plot for the presentation: codependent condition only.
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(df_codep["time_min"], df_codep["firing_rate_hz_smooth"])
    add_target_lines(axes[0], target_df)
    axes[0].set_ylabel("Firing rate (Hz)")
    axes[0].set_title("Codependent inhibitory plasticity: main diagnostics")

    axes[1].plot(df_codep["time_min"], df_codep["trace_ratio_smooth"], label="E_trace / I_trace")
    axes[1].axhline(ALPHA_BALANCE, linestyle="--", linewidth=1.5, label=f"alpha = {ALPHA_BALANCE}")
    axes[1].set_ylabel("Trace-based E/I")
    axes[1].legend()

    axes[2].plot(df_codep["time_min"], df_codep["norm_wE_smooth"], label="E weight")
    axes[2].plot(df_codep["time_min"], df_codep["norm_wI_smooth"], label="I weight")
    axes[2].set_ylabel("Normalized weight")
    axes[2].set_xlabel("time (min)")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(outdir / f"codep_i_main_diagnostics_{mode}_clean.png", dpi=200)
    plt.close(fig)


def read_csv_for_plot(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Backward compatibility: older CSVs may have ei_ratio as instantaneous ratio.
    if "raw_current_ratio" not in df.columns and "ei_ratio" in df.columns:
        df["raw_current_ratio"] = df["ei_ratio"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "medium", "full-lite"], default="quick")
    parser.add_argument("--condition", choices=["both", "spike_i", "codep_i"], default="both")
    parser.add_argument("--seed", type=int, default=1, help="Seed used for both conditions by default.")
    parser.add_argument("--outdir", default="results_ei_balance_final")
    parser.add_argument("--plot-only", action="store_true", help="Do not run Brian2; regenerate plots from existing CSV files.")
    parser.add_argument("--csv-spike", default=None, help="CSV from the spike_i condition.")
    parser.add_argument("--csv-codep", default=None, help="CSV from the codep_i condition.")
    args = parser.parse_args()

    cfg = get_mode_config(args.mode)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    target_df = build_target_df(cfg["block_duration_s"])

    if args.plot_only:
        if not args.csv_spike and not args.csv_codep:
            raise ValueError("For --plot-only, provide --csv-spike and/or --csv-codep.")

        df_spike = read_csv_for_plot(args.csv_spike) if args.csv_spike else None
        df_codep = read_csv_for_plot(args.csv_codep) if args.csv_codep else None

        if df_spike is not None:
            plot_condition(df_spike, target_df, "spike_i", outdir, cfg["record_dt_s"])
        if df_codep is not None:
            plot_condition(df_codep, target_df, "codep_i", outdir, cfg["record_dt_s"])
        if df_spike is not None and df_codep is not None:
            plot_comparison(df_spike, df_codep, target_df, args.mode, outdir, cfg["record_dt_s"])

        print("saved:", outdir.resolve())
        return

    if not HAVE_BRIAN2:
        raise RuntimeError("Brian2 is not installed. Use --plot-only or install brian2.")

    df_spike: Optional[pd.DataFrame] = None
    df_codep: Optional[pd.DataFrame] = None

    if args.condition in ["both", "spike_i"]:
        df_spike, target_df = run_one_condition(rule="spike_i", mode=args.mode, seed_value=args.seed)
        df_spike.to_csv(outdir / f"spike_i_{args.mode}_final.csv", index=False)
        plot_condition(df_spike, target_df, "spike_i", outdir, cfg["record_dt_s"])

    if args.condition in ["both", "codep_i"]:
        # Same seed by default: the comparison should isolate the inhibitory rule, not random input differences.
        df_codep, target_df = run_one_condition(rule="codep_i", mode=args.mode, seed_value=args.seed)
        df_codep.to_csv(outdir / f"codep_i_{args.mode}_final.csv", index=False)
        plot_condition(df_codep, target_df, "codep_i", outdir, cfg["record_dt_s"])

    if args.condition == "both" and df_spike is not None and df_codep is not None:
        plot_comparison(df_spike, df_codep, target_df, args.mode, outdir, cfg["record_dt_s"])

    print("saved:", outdir.resolve())


if __name__ == "__main__":
    main()
