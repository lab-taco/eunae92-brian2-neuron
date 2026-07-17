from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Windows-safe target
prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms


# ============================================================
# Fortran distance/receptive-field plasticity parameters
# ============================================================

PARAMS = {
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

    # external post-burst current
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
    "A_ltp": 0.5,
    "A_ltd": 0.02,
    "A_het": 2.02 * 0.5,

    # very slow E trace
    "tau_E": 250000.0 * ms,

    # distance Gaussian
    "sigma2": 10.0,
}


def alpha_from_distance(distance_um, sigma2=10.0):
    """
    Fortran:
        p_plast(42) = exp(-0.5 * distance^2 / sigma^2)
    """
    return float(np.exp(-0.5 * (distance_um ** 2) / sigma2))


def make_pre_burst_protocol(
    start_ms,
    syn_id,
    interval_dt_ms,
    n_pre=60,
    pre_interval_ms=500.0,
):
    """
    Fortran input_LTP_protocol:

    pre spike:
        t = 1 + (pre_spk - 1) * 5000 steps
        5000 steps * 0.1 ms = 500 ms

    post burst current:
        t0 + 10*time_dt + (post_spk - 1)*200 steps
        10*time_dt steps * 0.1 ms = time_dt ms
        200 steps * 0.1 ms = 20 ms

    So each pre spike is followed by a 3-spike postsynaptic burst:
        interval_dt_ms, interval_dt_ms + 20 ms, interval_dt_ms + 40 ms
    """

    pre_times = []
    post_pulse_times = []

    for k in range(n_pre):
        pre_t = start_ms + k * pre_interval_ms
        pre_times.append(pre_t)

        for b in range(3):
            post_t = pre_t + interval_dt_ms + b * 20.0
            post_pulse_times.append(post_t)

    if syn_id == 1:
        return pre_times, [], post_pulse_times
    elif syn_id == 2:
        return [], pre_times, post_pulse_times
    else:
        raise ValueError("syn_id must be 1 or 2")


def build_timed_pulse(post_pulse_times_ms, duration_ms):
    """
    Fortran LIF adds x(14) directly to membrane potential for one dt.
    Here we mimic that by adding pulse_amp / dt to du/dt for one dt.
    """

    dt_ms = float(defaultclock.dt / ms)
    n_steps = int(np.ceil(duration_ms / dt_ms)) + 10

    values = np.zeros(n_steps)
    for t_ms in post_pulse_times_ms:
        idx = int(round(t_ms / dt_ms))
        if 0 <= idx < n_steps:
            values[idx] = PARAMS["pulse_amp"]

    return TimedArray(values, dt=defaultclock.dt)


def run_sequence(
    protocols,
    distance_um=3.0,
    n_pre=60,
    forced_duration_ms=None,
    seed_value=1,
    record=False,
):
    """
    Run one complete sequence with one or two protocols.

    protocols example:
        [
          {"start_ms": 1000, "syn_id": 1, "interval_dt_ms": 5},
          {"start_ms": 121000, "syn_id": 2, "interval_dt_ms": 35},
        ]
    """

    start_scope()
    defaultclock.dt = 0.1 * ms

    seed(seed_value)
    np.random.seed(seed_value)

    alpha = alpha_from_distance(distance_um, PARAMS["sigma2"])

    pre1_times_ms = []
    pre2_times_ms = []
    post_pulse_times_ms = []

    for p in protocols:
        a, b, post = make_pre_burst_protocol(
            start_ms=p["start_ms"],
            syn_id=p["syn_id"],
            interval_dt_ms=p["interval_dt_ms"],
            n_pre=n_pre,
        )
        pre1_times_ms.extend(a)
        pre2_times_ms.extend(b)
        post_pulse_times_ms.extend(post)

    max_event = 0.0
    if pre1_times_ms:
        max_event = max(max_event, max(pre1_times_ms))
    if pre2_times_ms:
        max_event = max(max_event, max(pre2_times_ms))
    if post_pulse_times_ms:
        max_event = max(max_event, max(post_pulse_times_ms))

    duration_ms = max_event + 1000.0
    if forced_duration_ms is not None:
        duration_ms = max(duration_ms, forced_duration_ms)

    pulse_func = build_timed_pulse(post_pulse_times_ms, duration_ms)

    pre1 = SpikeGeneratorGroup(
        1,
        indices=np.zeros(len(pre1_times_ms), dtype=int),
        times=np.asarray(pre1_times_ms) * ms,
    )

    pre2 = SpikeGeneratorGroup(
        1,
        indices=np.zeros(len(pre2_times_ms), dtype=int),
        times=np.asarray(pre2_times_ms) * ms,
    )

    ns = dict(PARAMS)
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
        refractory=PARAMS["tau_ref"],
        method="euler",
        namespace=ns,
    )

    # initial_conditions.f90
    G.u = PARAMS["u_rest"]
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
    G.w1 = PARAMS["w1_init"]
    G.w2 = PARAMS["w2_init"]

    # Synapse 1 pre event
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

    # Synapse 2 pre event
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

    # Postsynaptic spike event
    # Fortran:
    # IF(j.EQ.1) exc_eff = exc_pre(1) + p_plast(42)*exc_pre(2)
    # IF(j.EQ.2) exc_eff = exc_pre(2) + p_plast(42)*exc_pre(1)
    S_post = Synapses(
        G,
        G,
        on_pre="""
        w1_post = clip(w1_post + A_ltp * xpre1_post * (E1_post + alpha * E2_post) - A_het * ypost_het_post * ((E1_post + alpha * E2_post) ** 2), wmin, wmax)
        w2_post = clip(w2_post + A_ltp * xpre2_post * (E2_post + alpha * E1_post) - A_het * ypost_het_post * ((E2_post + alpha * E1_post) ** 2), wmin, wmax)
        ypost_ltd1_post += 1.0
        ypost_ltd2_post += 1.0
        ypost_het_post += 1.0
        """,
        namespace=ns,
    )
    S_post.connect(i=[0], j=[0])

    monitors = {}
    if record:
        monitors["state"] = StateMonitor(
            G,
            [
                "u",
                "I_pulse",
                "g_ampa",
                "g_nmda",
                "E1",
                "E2",
                "w1",
                "w2",
                "xpre1",
                "xpre2",
                "ypost_het",
            ],
            record=True,
        )
        monitors["spikes"] = SpikeMonitor(G)

    objects = [G, pre1, pre2, S1, S2, S_post]
    if record:
        objects.extend([monitors["state"], monitors["spikes"]])

    net = Network(objects)
    net.run(duration_ms * ms)

    result = {
        "distance_um": float(distance_um),
        "alpha": float(alpha),
        "w1_final": float(G.w1[0]),
        "w2_final": float(G.w2[0]),
        "w1_percent": 100.0 * float(G.w1[0]) / PARAMS["w1_init"],
        "w2_percent": 100.0 * float(G.w2[0]) / PARAMS["w2_init"],
    }

    if record:
        result["monitors"] = monitors

    return result


def run_stdp_curve(mode, outdir):
    """
    Fortran data01:
        interval_dt, 100*msynw(1)/p_con(5)
    """

    if mode == "quick":
        intervals = np.array([1, 5, 10, 15, 25, 35, 45, 50], dtype=int)
        n_pre = 20
    elif mode == "medium":
        intervals = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=int)
        n_pre = 60
    elif mode == "full":
        intervals = np.arange(1, 51, dtype=int)
        n_pre = 60
    else:
        raise ValueError("mode must be quick, medium, or full")

    rows = []

    for interval_dt in intervals:
        print("STDP curve interval_dt:", interval_dt)

        result = run_sequence(
            protocols=[
                {
                    "start_ms": 1000.0,
                    "syn_id": 1,
                    "interval_dt_ms": float(interval_dt),
                }
            ],
            distance_um=3.0,
            n_pre=n_pre,
            forced_duration_ms=31000.0,
            seed_value=interval_dt,
            record=False,
        )

        rows.append(
            {
                "interval_dt_ms": interval_dt,
                "w1_percent": result["w1_percent"],
            }
        )

        pd.DataFrame(rows).to_csv(outdir / f"stdp_curve_{mode}.csv", index=False)

    return pd.DataFrame(rows)


def run_temporal_interaction(mode, outdir, weak_baseline_percent):
    """
    Fortran data02:
        interval_ltp, 100*msynw(2)/p_con(17), w_stdp(interval_dt_weak)
    """

    interval_dt_strong = 5
    interval_dt_weak = 35

    if mode == "quick":
        intervals_sec = np.array([0, 120, 360, 720, 1080], dtype=float)
        n_pre = 20
    elif mode == "medium":
        intervals_sec = np.array([0, 60, 180, 360, 540, 720, 900, 1080], dtype=float)
        n_pre = 40
    elif mode == "full":
        intervals_sec = np.arange(0, 1200, 60, dtype=float)
        n_pre = 60
    else:
        raise ValueError("mode must be quick, medium, or full")

    rows = []

    for interval_ltp_sec in intervals_sec:
        print("Temporal interval_ltp_sec:", interval_ltp_sec)

        start_strong = 1000.0
        strong_duration = 30000.0
        start_weak = start_strong + strong_duration + interval_ltp_sec * 1000.0
        forced_duration = start_weak + 50000.0

        result = run_sequence(
            protocols=[
                {
                    "start_ms": start_strong,
                    "syn_id": 1,
                    "interval_dt_ms": float(interval_dt_strong),
                },
                {
                    "start_ms": start_weak,
                    "syn_id": 2,
                    "interval_dt_ms": float(interval_dt_weak),
                },
            ],
            distance_um=3.0,
            n_pre=n_pre,
            forced_duration_ms=forced_duration,
            seed_value=int(interval_ltp_sec) + 1000,
            record=False,
        )

        rows.append(
            {
                "interval_ltp_sec": interval_ltp_sec,
                "interval_ltp_min": interval_ltp_sec / 60.0,
                "w2_percent": result["w2_percent"],
                "weak_baseline_percent": weak_baseline_percent,
            }
        )

        pd.DataFrame(rows).to_csv(outdir / f"temporal_interaction_{mode}.csv", index=False)

    return pd.DataFrame(rows)


def run_distance_interaction(mode, outdir, weak_baseline_percent):
    """
    Fortran data03:
        distance_ltp, 100*msynw(2)/p_con(17), w_stdp(interval_dt_weak)
    """

    interval_dt_strong = 5
    interval_dt_weak = 35

    if mode == "quick":
        distances = np.array([0, 1, 2, 3, 5, 8, 12, 16, 20], dtype=float)
        n_pre = 20
    elif mode == "medium":
        distances = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 16, 20], dtype=float)
        n_pre = 40
    elif mode == "full":
        distances = np.arange(0, 20.0, 0.2, dtype=float)
        n_pre = 60
    else:
        raise ValueError("mode must be quick, medium, or full")

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
                {
                    "start_ms": start_strong,
                    "syn_id": 1,
                    "interval_dt_ms": float(interval_dt_strong),
                },
                {
                    "start_ms": start_weak,
                    "syn_id": 2,
                    "interval_dt_ms": float(interval_dt_weak),
                },
            ],
            distance_um=float(distance_um),
            n_pre=n_pre,
            forced_duration_ms=forced_duration,
            seed_value=int(distance_um * 1000) + 2000,
            record=False,
        )

        rows.append(
            {
                "distance_um": distance_um,
                "alpha": result["alpha"],
                "w2_percent": result["w2_percent"],
                "weak_baseline_percent": weak_baseline_percent,
            }
        )

        pd.DataFrame(rows).to_csv(outdir / f"distance_interaction_{mode}.csv", index=False)

    return pd.DataFrame(rows)


def run_example(outdir):
    outdir.mkdir(exist_ok=True)

    start_strong = 1000.0
    strong_duration = 30000.0
    gap_sec = 120.0
    start_weak = start_strong + strong_duration + gap_sec * 1000.0
    forced_duration = start_weak + 50000.0

    result = run_sequence(
        protocols=[
            {
                "start_ms": start_strong,
                "syn_id": 1,
                "interval_dt_ms": 5.0,
            },
            {
                "start_ms": start_weak,
                "syn_id": 2,
                "interval_dt_ms": 35.0,
            },
        ],
        distance_um=3.0,
        n_pre=20,
        forced_duration_ms=forced_duration,
        seed_value=123,
        record=True,
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

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / second, M.u[0])
    plt.xlabel("time (s)")
    plt.ylabel("membrane potential u")
    plt.tight_layout()
    plt.savefig(outdir / "example_u.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / second, M.E1[0], label="E1")
    plt.plot(M.t / second, M.E2[0], label="E2")
    plt.xlabel("time (s)")
    plt.ylabel("E trace")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example_E_traces.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(M.t / second, M.w1[0], label="w1 strong")
    plt.plot(M.t / second, M.w2[0], label="w2 weak")
    plt.xlabel("time (s)")
    plt.ylabel("weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example_weights.png", dpi=200)
    plt.close()


def plot_summary(stdp_df, temporal_df, distance_df, mode, outdir):
    plt.figure(figsize=(12, 3.5))

    plt.subplot(1, 3, 1)
    plt.plot(stdp_df["interval_dt_ms"], stdp_df["w1_percent"], marker="o")
    plt.xlabel(r"$\Delta t$ (ms)")
    plt.ylabel("final weight / initial weight (%)")
    plt.title("Single synapse")
    plt.ylim(90, max(200, stdp_df["w1_percent"].max() * 1.1))

    plt.subplot(1, 3, 2)
    plt.plot(temporal_df["interval_ltp_min"], temporal_df["w2_percent"], marker="o", label="neighbor")
    plt.plot(
        temporal_df["interval_ltp_min"],
        temporal_df["weak_baseline_percent"],
        linestyle="--",
        label="weak baseline",
    )
    plt.xlabel("Time (min)")
    plt.title("Temporal interaction")
    plt.legend(fontsize=8)

    plt.subplot(1, 3, 3)
    plt.plot(distance_df["distance_um"], distance_df["w2_percent"], marker="o", label="neighbor")
    plt.plot(
        distance_df["distance_um"],
        distance_df["weak_baseline_percent"],
        linestyle="--",
        label="weak baseline",
    )
    plt.xlabel("Distance (um)")
    plt.title("Distance interaction")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outdir / f"distance_summary_{mode}.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example-only",
        action="store_true",
        help="Run one two-synapse example only.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "medium", "full"],
        default="quick",
        help="Start with quick. full is slow.",
    )

    args = parser.parse_args()

    outdir = Path("results_distance")
    outdir.mkdir(exist_ok=True)

    if args.example_only:
        run_example(outdir)
        print("saved:", outdir.resolve())
        return

    stdp_df = run_stdp_curve(args.mode, outdir)

    # Ensure weak baseline at interval_dt = 35 ms
    weak_rows = stdp_df[stdp_df["interval_dt_ms"] == 35]
    if len(weak_rows) == 0:
        weak_result = run_sequence(
            protocols=[
                {
                    "start_ms": 1000.0,
                    "syn_id": 1,
                    "interval_dt_ms": 35.0,
                }
            ],
            distance_um=3.0,
            n_pre=20 if args.mode == "quick" else 60,
            forced_duration_ms=31000.0,
            seed_value=35,
            record=False,
        )
        weak_baseline = weak_result["w1_percent"]
    else:
        weak_baseline = float(weak_rows.iloc[0]["w1_percent"])

    temporal_df = run_temporal_interaction(args.mode, outdir, weak_baseline)
    distance_df = run_distance_interaction(args.mode, outdir, weak_baseline)

    plot_summary(stdp_df, temporal_df, distance_df, args.mode, outdir)
    run_example(outdir)

    print("saved:", outdir.resolve())


if __name__ == "__main__":
    main()