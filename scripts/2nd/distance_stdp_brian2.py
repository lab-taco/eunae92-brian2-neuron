"""
distance_interaction_brian2_final.py

Cleaned, presentation-oriented version of the Fig. 2h-j two-synapse
interaction script.

Purpose
-------
This is a compressed Brian2 sanity check for three ideas:
1. single-synapse pre-before-post LTP decreases as pre-post delay increases,
2. a strong LTP event can temporarily boost a later weak LTP event at a nearby synapse,
3. the neighboring boost decays with synaptic distance through a Gaussian coupling.

Important fixes compared with the draft
---------------------------------------
1. Brian2 import is optional. If Brian2 is not installed, --plot-only still works
   from existing CSV files.
2. Network construction uses Network(*objects), which is the safer Brian2 pattern.
3. The weak baseline used in temporal/distance plots is now measured from an
   isolated weak synapse (synapse 2, initial w2), not from the strong synapse.
4. n_pre is now centralized per mode, so the single-synapse baseline and the
   temporal/distance protocols use the same number of pairings.
5. Recorded examples use a configurable monitor dt, avoiding very large StateMonitor
   arrays for long protocols.
6. E-effective terms are rectified in plasticity updates, matching the interpretation
   of E as a non-negative NMDA-current trace.
7. Plotting is separated from simulation via --plot-only.

Usage
-----
Run simulation:
    python distance_interaction_brian2_final.py --mode quick --outdir results_distance_final_quick
    python distance_interaction_brian2_final.py --mode medium --outdir results_distance_final_medium

Regenerate plots from existing CSV files without Brian2:
    python distance_interaction_brian2_final.py --plot-only --mode medium \
        --stdp-csv "stdp_curve_medium(3).csv" \
        --temporal-csv "temporal_interaction_medium(3).csv" \
        --distance-csv "distance_interaction_medium(3).csv" \
        --outdir results_distance_plotonly_medium

Run one recorded example:
    python distance_interaction_brian2_final.py --example-only --outdir results_distance_example
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from brian2 import (  # type: ignore
        SpikeGeneratorGroup,
        StateMonitor,
        SpikeMonitor,
        Synapses,
        TimedArray,
        NeuronGroup,
        Network,
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
    defaultclock.dt = 0.1 * ms
    HAVE_BRIAN2 = True
except Exception:
    # These placeholders allow --plot-only to run on machines without Brian2.
    HAVE_BRIAN2 = False
    ms = 1.0
    second = 1000.0

DEFAULT_DT_MS = 0.1


# ============================================================
# Parameters
# ============================================================

def get_params() -> Dict[str, object]:
    """Return parameters. Brian2 quantities are used only when Brian2 is available."""
    return {
        # neuron
        "tau_m": 30.0 * ms,
        "u_rest": -65.0,
        "u_th": -50.0,
        "u_reset": -60.0,
        "tau_ref": 5.0 * ms,
        "E_ahp": -80.0,
        "A_ahp": 5.0,
        "tau_ahp": 100.0 * ms,
        # synapses
        "tau_ampa": 5.0 * ms,
        "tau_nmda": 150.0 * ms,
        "a_nmda": 0.15,
        "b_nmda": -0.08,
        # external post-burst current pulse
        "pulse_amp": 600.0,
        # initial weights
        "w1_init": 0.12,
        "w2_init": 0.15 * 0.12,
        # plasticity
        "wmin": 0.0001,
        "wmax": 1.0,
        "tau_pre": 16.8 * ms,
        "tau_post_ltd": 33.7 * ms,
        "tau_post_het": 300.0 * ms,
        # These large factors compensate for the intentionally very slow E trace.
        # This is a compressed mechanism check, not a full reproduction.
        "A_ltp": 0.5,
        "A_ltd": 0.02,
        "A_het": 2.02 * 0.5,
        # very slow E trace for temporal interaction over minutes
        "tau_E": 250000.0 * ms,
        # distance Gaussian: exp(-0.5*d^2/sigma_sq_um2)
        "sigma_sq_um2": 10.0,
    }


def mode_settings(mode: str) -> Dict[str, object]:
    """Centralized sweep settings so all panels use consistent n_pre per mode."""
    if mode == "quick":
        return {
            "stdp_intervals_ms": np.array([1, 5, 10, 15, 25, 35, 45, 50], dtype=float),
            "temporal_gaps_sec": np.array([0, 120, 360, 720, 1080], dtype=float),
            "distances_um": np.array([0, 1, 2, 3, 5, 8, 12, 16, 20], dtype=float),
            "n_pre": 20,
        }
    if mode == "medium":
        return {
            "stdp_intervals_ms": np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float),
            "temporal_gaps_sec": np.array([0, 60, 180, 360, 540, 720, 900, 1080], dtype=float),
            "distances_um": np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 16, 20], dtype=float),
            "n_pre": 60,
        }
    if mode == "full":
        return {
            "stdp_intervals_ms": np.arange(1, 51, dtype=float),
            "temporal_gaps_sec": np.arange(0, 1200, 60, dtype=float),
            "distances_um": np.arange(0, 20.0, 0.2, dtype=float),
            "n_pre": 60,
        }
    raise ValueError("mode must be quick, medium, or full")


def alpha_from_distance(distance_um: float, sigma_sq_um2: float = 10.0) -> float:
    """Gaussian cross-synapse coupling: exp(-0.5*d^2/sigma^2)."""
    return float(np.exp(-0.5 * (distance_um ** 2) / sigma_sq_um2))


def require_brian2() -> None:
    if not HAVE_BRIAN2:
        raise RuntimeError(
            "Brian2 is not installed. Install brian2 to run simulations, "
            "or use --plot-only with existing CSV files."
        )


# ============================================================
# Protocol construction
# ============================================================

def make_pre_burst_protocol(
    start_ms: float,
    syn_id: int,
    interval_dt_ms: float,
    n_pre: int = 60,
    pre_interval_ms: float = 500.0,
) -> tuple[List[float], List[float], List[float]]:
    """
    One pre spike is followed by a 3-spike postsynaptic burst:
        interval_dt_ms, interval_dt_ms + 20 ms, interval_dt_ms + 40 ms.

    Returns pre1_times_ms, pre2_times_ms, post_pulse_times_ms.
    """
    pre_times: List[float] = []
    post_pulse_times: List[float] = []

    for k in range(n_pre):
        pre_t = start_ms + k * pre_interval_ms
        pre_times.append(pre_t)
        for b in range(3):
            post_pulse_times.append(pre_t + interval_dt_ms + b * 20.0)

    if syn_id == 1:
        return pre_times, [], post_pulse_times
    if syn_id == 2:
        return [], pre_times, post_pulse_times
    raise ValueError("syn_id must be 1 or 2")


def build_timed_pulse(post_pulse_times_ms: List[float], duration_ms: float, params: Dict[str, object]):
    """Build a one-time-step current pulse array to force postsynaptic spikes."""
    dt_ms = float(defaultclock.dt / ms)
    n_steps = int(np.ceil(duration_ms / dt_ms)) + 10
    values = np.zeros(n_steps)
    for t_ms in post_pulse_times_ms:
        idx = int(round(t_ms / dt_ms))
        if 0 <= idx < n_steps:
            values[idx] = float(params["pulse_amp"])
    return TimedArray(values, dt=defaultclock.dt)


# ============================================================
# Simulation
# ============================================================

def run_sequence(
    protocols: List[Dict[str, float]],
    distance_um: float = 3.0,
    n_pre: int = 60,
    forced_duration_ms: Optional[float] = None,
    seed_value: int = 1,
    record: bool = False,
    record_dt_ms: float = 1.0,
) -> Dict[str, object]:
    """Run one sequence with one or two pre-burst protocols."""
    require_brian2()

    params = get_params()

    start_scope()
    defaultclock.dt = DEFAULT_DT_MS * ms
    seed(seed_value)
    np.random.seed(seed_value)

    alpha = alpha_from_distance(distance_um, float(params["sigma_sq_um2"]))

    pre1_times_ms: List[float] = []
    pre2_times_ms: List[float] = []
    post_pulse_times_ms: List[float] = []

    for p in protocols:
        pre1, pre2, post = make_pre_burst_protocol(
            start_ms=float(p["start_ms"]),
            syn_id=int(p["syn_id"]),
            interval_dt_ms=float(p["interval_dt_ms"]),
            n_pre=n_pre,
        )
        pre1_times_ms.extend(pre1)
        pre2_times_ms.extend(pre2)
        post_pulse_times_ms.extend(post)

    max_event = 0.0
    for seq in [pre1_times_ms, pre2_times_ms, post_pulse_times_ms]:
        if seq:
            max_event = max(max_event, max(seq))

    duration_ms = max_event + 1000.0
    if forced_duration_ms is not None:
        duration_ms = max(duration_ms, forced_duration_ms)

    pulse_func = build_timed_pulse(post_pulse_times_ms, duration_ms, params)

    pre1 = SpikeGeneratorGroup(1, indices=np.zeros(len(pre1_times_ms), dtype=int), times=np.asarray(pre1_times_ms) * ms)
    pre2 = SpikeGeneratorGroup(1, indices=np.zeros(len(pre2_times_ms), dtype=int), times=np.asarray(pre2_times_ms) * ms)

    ns = dict(params)
    ns["alpha"] = alpha
    ns["pulse_func"] = pulse_func
    ns["pulse_dt"] = defaultclock.dt

    eqs = """
    du/dt = (-(u - u_rest)
             - g_ampa * u
             - g_nmda * H_nmda * u
             - A_ahp * g_ahp * (u - E_ahp)) / tau_m
             + I_pulse / pulse_dt : 1 (unless refractory)

    dg_ampa/dt = -g_ampa / tau_ampa : 1
    dg_nmda/dt = -g_nmda / tau_nmda : 1
    dg_ahp/dt = -g_ahp / tau_ahp : 1

    decond1/dt = -econd1 / tau_nmda : 1
    decond2/dt = -econd2 / tau_nmda : 1

    dE1/dt = (-E1 + econd1 * H_nmda * (-u)) / tau_E : 1
    dE2/dt = (-E2 + econd2 * H_nmda * (-u)) / tau_E : 1

    dxpre1/dt = -xpre1 / tau_pre : 1
    dxpre2/dt = -xpre2 / tau_pre : 1

    dypost_ltd1/dt = -ypost_ltd1 / tau_post_ltd : 1
    dypost_ltd2/dt = -ypost_ltd2 / tau_post_ltd : 1
    dypost_het/dt = -ypost_het / tau_post_het : 1

    H_nmda = 1.0 / (1.0 + a_nmda * exp(b_nmda * u)) : 1
    I_pulse = pulse_func(t) : 1

    w1 : 1
    w2 : 1
    """

    G = NeuronGroup(
        1,
        eqs,
        threshold="u >= u_th",
        reset="""
        u = u_reset
        g_ahp += 1.0
        """,
        refractory=params["tau_ref"],
        method="euler",
        namespace=ns,
    )

    G.u = params["u_rest"]
    G.g_ampa = 0.0
    G.g_nmda = 0.0
    G.g_ahp = 0.0
    G.econd1 = 0.0
    G.econd2 = 0.0
    G.E1 = 0.0
    G.E2 = 0.0
    G.xpre1 = 0.0
    G.xpre2 = 0.0
    G.ypost_ltd1 = 0.0
    G.ypost_ltd2 = 0.0
    G.ypost_het = 0.0
    G.w1 = params["w1_init"]
    G.w2 = params["w2_init"]

    S1 = Synapses(
        pre1,
        G,
        on_pre="""
        g_ampa_post += w1_post
        g_nmda_post += w1_post
        econd1_post += w1_post
        w1_post = clip(w1_post * (1.0 - A_ltd * ypost_ltd1_post), wmin, wmax)
        xpre1_post += 1.0
        """,
        namespace=ns,
    )
    S1.connect()

    S2 = Synapses(
        pre2,
        G,
        on_pre="""
        g_ampa_post += w2_post
        g_nmda_post += w2_post
        econd2_post += w2_post
        w2_post = clip(w2_post * (1.0 - A_ltd * ypost_ltd2_post), wmin, wmax)
        xpre2_post += 1.0
        """,
        namespace=ns,
    )
    S2.connect()

    S_post = Synapses(
        G,
        G,
        on_pre="""
        w1_post = clip(
            w1_post
            + A_ltp * xpre1_post * clip(E1_post + alpha * E2_post, 0.0, 1e9)
            - A_het * ypost_het_post * (clip(E1_post + alpha * E2_post, 0.0, 1e9) ** 2),
            wmin,
            wmax,
        )
        w2_post = clip(
            w2_post
            + A_ltp * xpre2_post * clip(E2_post + alpha * E1_post, 0.0, 1e9)
            - A_het * ypost_het_post * (clip(E2_post + alpha * E1_post, 0.0, 1e9) ** 2),
            wmin,
            wmax,
        )
        ypost_ltd1_post += 1.0
        ypost_ltd2_post += 1.0
        ypost_het_post += 1.0
        """,
        namespace=ns,
    )
    S_post.connect(i=[0], j=[0])

    monitors: Dict[str, object] = {}
    objects = [G, pre1, pre2, S1, S2, S_post]

    if record:
        monitors["state"] = StateMonitor(
            G,
            ["u", "I_pulse", "g_ampa", "g_nmda", "E1", "E2", "w1", "w2", "xpre1", "xpre2", "ypost_het"],
            record=True,
            dt=record_dt_ms * ms,
        )
        monitors["spikes"] = SpikeMonitor(G)
        objects.extend([monitors["state"], monitors["spikes"]])

    net = Network(*objects)
    net.run(duration_ms * ms)

    result: Dict[str, object] = {
        "distance_um": float(distance_um),
        "alpha": float(alpha),
        "w1_final": float(G.w1[0]),
        "w2_final": float(G.w2[0]),
        "w1_percent": 100.0 * float(G.w1[0]) / float(params["w1_init"]),
        "w2_percent": 100.0 * float(G.w2[0]) / float(params["w2_init"]),
    }

    if record:
        result["monitors"] = monitors
    return result


# ============================================================
# Sweeps
# ============================================================

def run_stdp_curve(mode: str, outdir: Path) -> pd.DataFrame:
    settings = mode_settings(mode)
    intervals = settings["stdp_intervals_ms"]
    n_pre = int(settings["n_pre"])

    rows = []
    for interval_dt in intervals:
        print("STDP curve interval_dt:", interval_dt)
        result = run_sequence(
            protocols=[{"start_ms": 1000.0, "syn_id": 1, "interval_dt_ms": float(interval_dt)}],
            distance_um=3.0,
            n_pre=n_pre,
            forced_duration_ms=31000.0,
            seed_value=int(interval_dt * 1000) + 10,
            record=False,
        )
        rows.append({"interval_dt_ms": float(interval_dt), "w1_percent": result["w1_percent"], "n_pre": n_pre})
        pd.DataFrame(rows).to_csv(outdir / f"stdp_curve_{mode}.csv", index=False)
    return pd.DataFrame(rows)


def run_weak_baseline(interval_dt_weak: float, mode: str) -> float:
    """Isolated weak-synapse baseline: same initial w2 and same n_pre as interaction sweeps."""
    n_pre = int(mode_settings(mode)["n_pre"])
    result = run_sequence(
        protocols=[{"start_ms": 1000.0, "syn_id": 2, "interval_dt_ms": float(interval_dt_weak)}],
        distance_um=3.0,
        n_pre=n_pre,
        forced_duration_ms=31000.0,
        seed_value=int(interval_dt_weak * 1000) + 99,
        record=False,
    )
    return float(result["w2_percent"])


def run_temporal_interaction(mode: str, outdir: Path, weak_baseline_percent: float) -> pd.DataFrame:
    interval_dt_strong = 5.0
    interval_dt_weak = 35.0
    settings = mode_settings(mode)
    intervals_sec = settings["temporal_gaps_sec"]
    n_pre = int(settings["n_pre"])

    rows = []
    for interval_ltp_sec in intervals_sec:
        print("Temporal interval_ltp_sec:", interval_ltp_sec)
        start_strong = 1000.0
        strong_duration = 30000.0
        start_weak = start_strong + strong_duration + float(interval_ltp_sec) * 1000.0
        forced_duration = start_weak + 50000.0

        result = run_sequence(
            protocols=[
                {"start_ms": start_strong, "syn_id": 1, "interval_dt_ms": interval_dt_strong},
                {"start_ms": start_weak, "syn_id": 2, "interval_dt_ms": interval_dt_weak},
            ],
            distance_um=3.0,
            n_pre=n_pre,
            forced_duration_ms=forced_duration,
            seed_value=int(interval_ltp_sec) + 1000,
            record=False,
        )
        rows.append(
            {
                "interval_ltp_sec": float(interval_ltp_sec),
                "interval_ltp_min": float(interval_ltp_sec) / 60.0,
                "w2_percent": result["w2_percent"],
                "weak_baseline_percent": weak_baseline_percent,
                "n_pre": n_pre,
            }
        )
        pd.DataFrame(rows).to_csv(outdir / f"temporal_interaction_{mode}.csv", index=False)
    return pd.DataFrame(rows)


def run_distance_interaction(mode: str, outdir: Path, weak_baseline_percent: float) -> pd.DataFrame:
    interval_dt_strong = 5.0
    interval_dt_weak = 35.0
    settings = mode_settings(mode)
    distances = settings["distances_um"]
    n_pre = int(settings["n_pre"])

    rows = []
    for distance_um in distances:
        print("Distance:", distance_um)
        start_strong = 5000.0
        strong_duration = 30000.0
        gap_duration = 90000.0
        start_weak = start_strong + strong_duration + gap_duration
        forced_duration = start_weak + 50000.0

        result = run_sequence(
            protocols=[
                {"start_ms": start_strong, "syn_id": 1, "interval_dt_ms": interval_dt_strong},
                {"start_ms": start_weak, "syn_id": 2, "interval_dt_ms": interval_dt_weak},
            ],
            distance_um=float(distance_um),
            n_pre=n_pre,
            forced_duration_ms=forced_duration,
            seed_value=int(distance_um * 1000) + 2000,
            record=False,
        )
        rows.append(
            {
                "distance_um": float(distance_um),
                "alpha": result["alpha"],
                "w2_percent": result["w2_percent"],
                "weak_baseline_percent": weak_baseline_percent,
                "n_pre": n_pre,
            }
        )
        pd.DataFrame(rows).to_csv(outdir / f"distance_interaction_{mode}.csv", index=False)
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_summary(stdp_df: pd.DataFrame, temporal_df: pd.DataFrame, distance_df: pd.DataFrame, mode: str, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    ax = axes[0]
    ax.plot(stdp_df["interval_dt_ms"], stdp_df["w1_percent"], marker="o")
    ax.axhline(100.0, linewidth=1)
    ax.axvline(5.0, linestyle=":", linewidth=1)
    ax.axvline(35.0, linestyle=":", linewidth=1)
    ax.set_xlabel(r"$\Delta t$ (ms)")
    ax.set_ylabel("final weight / initial weight (%)")
    ax.set_title("Single synapse")
    ax.set_ylim(90, max(160.0, float(stdp_df["w1_percent"].max()) * 1.08))

    ax = axes[1]
    ax.plot(temporal_df["interval_ltp_min"], temporal_df["w2_percent"], marker="o", label="neighbor")
    ax.plot(temporal_df["interval_ltp_min"], temporal_df["weak_baseline_percent"], linestyle="--", label="isolated weak")
    ax.set_xlabel("Time gap (min)")
    ax.set_title("Temporal interaction")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(distance_df["distance_um"], distance_df["w2_percent"], marker="o", label="neighbor")
    ax.plot(distance_df["distance_um"], distance_df["weak_baseline_percent"], linestyle="--", label="isolated weak")
    ax.set_xlabel("Distance (um)")
    ax.set_title("Distance interaction")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / f"distance_summary_{mode}_final.png", dpi=200)
    plt.close(fig)


def run_example(outdir: Path, record_dt_ms: float = 1.0) -> None:
    outdir.mkdir(exist_ok=True, parents=True)

    start_strong = 1000.0
    strong_duration = 30000.0
    gap_sec = 120.0
    start_weak = start_strong + strong_duration + gap_sec * 1000.0
    forced_duration = start_weak + 50000.0

    result = run_sequence(
        protocols=[
            {"start_ms": start_strong, "syn_id": 1, "interval_dt_ms": 5.0},
            {"start_ms": start_weak, "syn_id": 2, "interval_dt_ms": 35.0},
        ],
        distance_um=3.0,
        n_pre=20,
        forced_duration_ms=forced_duration,
        seed_value=123,
        record=True,
        record_dt_ms=record_dt_ms,
    )

    M = result["monitors"]["state"]
    spikes = result["monitors"]["spikes"]

    print("====================================")
    print("Example result")
    print("alpha:", result["alpha"])
    print("w1 percent:", result["w1_percent"])
    print("w2 percent:", result["w2_percent"])
    print("postsynaptic spikes:", len(spikes.t))
    print("====================================")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(M.t / second, M.u[0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("membrane potential u")
    fig.tight_layout()
    fig.savefig(outdir / "example_u_final.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(M.t / second, M.E1[0], label="E1 strong synapse")
    ax.plot(M.t / second, M.E2[0], label="E2 weak synapse")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("E trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "example_E_traces_final.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(M.t / second, M.w1[0], label="w1 strong")
    ax.plot(M.t / second, M.w2[0], label="w2 weak")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "example_weights_final.png", dpi=200)
    plt.close(fig)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example-only", action="store_true", help="Run one recorded two-synapse example only.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate summary plot from existing CSV files.")
    parser.add_argument("--mode", choices=["quick", "medium", "full"], default="quick")
    parser.add_argument("--outdir", default="results_distance_final")
    parser.add_argument("--stdp-csv", default=None)
    parser.add_argument("--temporal-csv", default=None)
    parser.add_argument("--distance-csv", default=None)
    parser.add_argument("--record-dt-ms", type=float, default=1.0)
    parser.add_argument("--no-example", action="store_true", help="Skip the recorded example after sweeps.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    if args.plot_only:
        if not (args.stdp_csv and args.temporal_csv and args.distance_csv):
            raise ValueError("--plot-only requires --stdp-csv, --temporal-csv, and --distance-csv.")
        stdp_df = load_csv(args.stdp_csv)
        temporal_df = load_csv(args.temporal_csv)
        distance_df = load_csv(args.distance_csv)
        plot_summary(stdp_df, temporal_df, distance_df, args.mode, outdir)
        print("saved:", outdir.resolve())
        return

    if args.example_only:
        run_example(outdir, record_dt_ms=args.record_dt_ms)
        print("saved:", outdir.resolve())
        return

    require_brian2()

    stdp_df = run_stdp_curve(args.mode, outdir)
    weak_baseline = run_weak_baseline(interval_dt_weak=35.0, mode=args.mode)
    print("isolated weak baseline at dt=35 ms:", weak_baseline)

    temporal_df = run_temporal_interaction(args.mode, outdir, weak_baseline)
    distance_df = run_distance_interaction(args.mode, outdir, weak_baseline)
    plot_summary(stdp_df, temporal_df, distance_df, args.mode, outdir)

    if not args.no_example:
        run_example(outdir, record_dt_ms=args.record_dt_ms)

    print("saved:", outdir.resolve())


if __name__ == "__main__":
    main()
