from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

prefs.codegen.target = "cython"
defaultclock.dt = 0.1 * ms


def make_protocol():
    pre_times = []
    post_times = []

    quiet_1 = 5000.0      # ms
    burst_duration = 100.0
    quiet_2 = 5000.0
    rep_duration = quiet_1 + burst_duration + quiet_2

    for k in range(15):
        base = k * rep_duration + quiet_1

        for n in range(5):
            pre_times.append((base + 10.0 + 20.0 * n) * ms)
            post_times.append((base + 20.0 + 20.0 * n) * ms)

    duration = 15 * rep_duration * ms
    return pre_times, post_times, duration, quiet_1, burst_duration, rep_duration


def run_one_condition(w0=0.005, inp_clamp=0.0, ca_amp=0.0, make_plots=True):
    start_scope()
    defaultclock.dt = 0.1 * ms

    # -----------------------------
    # Parameters from config.f90
    # -----------------------------
    ns = {
        "tau_m": 30.0 * ms,
        "u_rest": -65.0,
        "E_bap": 50.0,

        "tau_ampa": 5.0 * ms,
        "tau_nmda": 150.0 * ms,
        "tau_bap": 1.0 * ms,
        "tau_E": 50.0 * ms,

        "a_nmda": 0.15,
        "b_nmda": -0.08,

        "tau_pre": 16.8 * ms,
        "tau_post_ltd": 33.7 * ms,
        "tau_post_het": 100.0 * ms,

        "A_ltp": 0.00025,
        "A_ltd": 0.000398,
        "A_het": 1e-9,

        "wmin": 1e-5,
        "wmax": 1.0,

        "inp_clamp": inp_clamp,
        "ca_amp": ca_amp,
    }

    pre_times, post_times, duration, quiet_1, burst_duration, rep_duration = make_protocol()

    pre_group = SpikeGeneratorGroup(
        1,
        indices=np.zeros(len(pre_times), dtype=int),
        times=pre_times,
    )

    post_group = SpikeGeneratorGroup(
        1,
        indices=np.zeros(len(post_times), dtype=int),
        times=post_times,
    )

    # ----------------------------------------------------
    # Fortran LIF equivalent, simplified for voltage_STDP
    #
    # Fortran:
    # g_tot = 1 + x(3) + x(6)*x(8) + x(11)
    # u_inf numerator = u_rest + inp_clamp + x(11)*E_bap
    #
    # Equivalent:
    # tau_m du/dt =
    # -(u-u_rest)
    # - g_ampa*u
    # - g_nmda*H_nmda*u
    # - g_bap*(u-E_bap)
    # + inp_clamp
    # ----------------------------------------------------
    eqs = """
    du/dt = (-(u - u_rest)
             - g_ampa * u
             - g_nmda * H_nmda * u
             - g_bap * (u - E_bap)
             + inp_clamp) / tau_m : 1

    dg_ampa/dt = -g_ampa / tau_ampa : 1
    dg_nmda/dt = -g_nmda / tau_nmda : 1
    dg_bap/dt = -g_bap / tau_bap : 1

    dE_trace/dt = (-E_trace - g_nmda * H_nmda * u) / tau_E : 1

    dxpre/dt = -xpre / tau_pre : 1
    dypost_ltd/dt = -ypost_ltd / tau_post_ltd : 1
    dypost_het/dt = -ypost_het / tau_post_het : 1

    H_nmda = 1.0 / (1.0 + a_nmda * exp(b_nmda * u)) : 1

    wE : 1
    """

    G = NeuronGroup(
        1,
        eqs,
        method="exponential_euler",
        namespace=ns,
    )

    # initial_conditions.f90 대응
    G.u = ns["u_rest"]
    G.g_ampa = 0.0
    G.g_nmda = 0.0
    G.g_bap = 0.0
    G.E_trace = 0.0
    G.xpre = 0.0
    G.ypost_ltd = 0.0
    G.ypost_het = 0.0
    G.wE = w0

    # ----------------------------------------------------
    # input_pre_post + plasticity_e, pre spike part
    # ----------------------------------------------------
    Spre = Synapses(
        pre_group,
        G,
        on_pre="""
        g_ampa_post += wE_post
        g_nmda_post += wE_post
        wE_post = clip(wE_post * (1.0 - A_ltd * ypost_ltd_post), wmin, wmax)
        xpre_post += 1.0
        """,
        namespace=ns,
    )
    Spre.connect()

    # ----------------------------------------------------
    # post spike part
    # Fortran: spk_post triggers bAP and post-side plasticity
    # ----------------------------------------------------
    Spost = Synapses(
        post_group,
        G,
        on_pre="""
        g_bap_post += ca_amp
        wE_post = clip(wE_post + A_ltp * xpre_post * E_trace_post - A_het * ypost_het_post * (E_trace_post ** 2), wmin, wmax)
        ypost_ltd_post += 1.0
        ypost_het_post += 1.0
        """,
        namespace=ns,
    )
    Spost.connect()

    M = StateMonitor(
        G,
        ["u", "g_ampa", "g_nmda", "g_bap", "H_nmda", "E_trace", "wE"],
        record=True,
    )

    run(duration, namespace=ns)

    t_ms = M.t / ms
    u = M.u[0]

    burst_mask = np.zeros_like(t_ms, dtype=bool)
    for k in range(15):
        start = k * rep_duration + quiet_1
        stop = start + burst_duration
        burst_mask |= (t_ms >= start) & (t_ms < stop)

    avg_u_burst = float(np.mean(u[burst_mask]))
    depol = avg_u_burst - ns["u_rest"]

    final_w = float(G.wE[0])
    final_percent = 100.0 * final_w / w0
    delta_percent = final_percent - 100.0

    print("====================================")
    print("single condition result")
    print("w0:", w0)
    print("inp_clamp:", inp_clamp)
    print("ca_amp:", ca_amp)
    print("avg burst u:", avg_u_burst)
    print("depolarization from rest:", depol)
    print("final wE:", final_w)
    print("final weight / initial weight (%):", final_percent)
    print("delta weight (%):", delta_percent)
    print("====================================")

    if make_plots:
        outdir = Path("results_single")
        outdir.mkdir(exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(t_ms / 1000.0, M.u[0])
        plt.xlabel("time (s)")
        plt.ylabel("membrane potential u")
        plt.tight_layout()
        plt.savefig(outdir / "u_trace.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(t_ms / 1000.0, M.E_trace[0])
        plt.xlabel("time (s)")
        plt.ylabel("E trace")
        plt.tight_layout()
        plt.savefig(outdir / "E_trace.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(t_ms / 1000.0, M.wE[0])
        plt.xlabel("time (s)")
        plt.ylabel("wE")
        plt.tight_layout()
        plt.savefig(outdir / "wE_trace.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(t_ms / 1000.0, M.g_nmda[0], label="g_nmda")
        plt.plot(t_ms / 1000.0, M.g_ampa[0], label="g_ampa")
        plt.xlabel("time (s)")
        plt.ylabel("conductance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "conductances.png", dpi=200)
        plt.close()

        print(f"plots saved to: {outdir.resolve()}")

    return {
        "w0": w0,
        "inp_clamp": inp_clamp,
        "ca_amp": ca_amp,
        "avg_u_burst": avg_u_burst,
        "depol": depol,
        "final_w": final_w,
        "final_percent": final_percent,
        "delta_percent": delta_percent,
    }


if __name__ == "__main__":
    run_one_condition(
        w0=0.005,
        inp_clamp=0.0,
        ca_amp=0.0,
        make_plots=True,
    )