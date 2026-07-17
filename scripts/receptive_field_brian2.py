from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import csv

prefs.codegen.target = "numpy"


class ReceptiveFieldModel:
    def __init__(
        self,
        n_pw=8,
        ne_pw=25,
        ni_pw=8,
        dt_ms=0.2,
        seed=1,
        plasticity_scale=1.0,
    ):
        self.rng = np.random.default_rng(seed)

        self.n_pw = n_pw
        self.ne_pw = ne_pw
        self.ni_pw = ni_pw
        self.ne_input = n_pw * ne_pw
        self.ni_input = n_pw * ni_pw

        self.dt_ms = dt_ms
        self.t_ms = 0.0
        self.step_index = 0

        # -------------------------
        # neuron parameters
        # -------------------------
        self.tau_m = 30.0
        self.u_rest = -65.0
        self.refrac_e = 5.0
        self.threshold = -50.0
        self.E_ahp = -80.0
        self.g_ahp_amp = 5.0
        self.u_reset = -60.0

        # -------------------------
        # synapse parameters
        # -------------------------
        self.E_gaba = -80.0
        self.tau_ampa = 5.0
        self.tau_gaba = 10.0
        self.tau_nmda = 150.0
        self.tau_ahp = 100.0

        self.a_nmda = 0.15
        self.b_nmda = -0.08

        self.decay_ampa = np.exp(-dt_ms / self.tau_ampa)
        self.decay_gaba = np.exp(-dt_ms / self.tau_gaba)
        self.decay_nmda = np.exp(-dt_ms / self.tau_nmda)
        self.decay_ahp = np.exp(-dt_ms / self.tau_ahp)

        # -------------------------
        # input probabilities
        # Fortran: probability = dt * Hz / 1000
        # -------------------------
        self.p_bg_e = dt_ms * 2.0 / 1000.0
        self.p_bg_i = dt_ms * 2.0 / 1000.0
        self.p_burst_e = dt_ms * 50.0 / 1000.0
        self.p_burst_i = dt_ms * 2.0 / 1000.0

        # OU pathway envelope
        # Fortran: pr(80)=50, then pr(80)=0.1*pr(80)
        self.tau_ou = 5.0
        self.ou_decay = np.exp(-dt_ms / self.tau_ou)
        self.ou_amp_hz = 5.0
        self.ou_update_steps = max(1, int(round(1.0 / dt_ms)))

        # inhibitory population gain
        self.inh_rate_gain = 2.0

        # -------------------------
        # initial weights
        # -------------------------
        self.wE0 = 0.12
        self.wI0 = 0.9

        # -------------------------
        # plasticity parameters
        # -------------------------
        self.wE_max = 1.0
        self.wE_min = 0.0001
        self.wI_max = 10.0
        self.wI_min = 0.001

        self.tau_pre_e = 16.8
        self.tau_post_e = 33.7

        self.tau_pre_i = 20.0
        self.tau_post_i = 20.0

        # compressed simulation에서 변화가 너무 작으면 --scale을 올린다.
        self.A_ltp = (0.0005 / 3.0) * plasticity_scale
        self.A_ltd = 1000.0 * self.A_ltp
        self.A_het = 0.00002 * self.A_ltp

        self.I_block_scale = 150.0
        self.I_block_power = 3.0
        self.I_rectify_threshold = 170.0

        # inhibitory plasticity가 fullish 압축 시뮬레이션 안에서 너무 느려서 I profile이 flat하게 남음
        self.A_i = 0.0015 * plasticity_scale
        self.alpha_balance = 0.93
        self.A_balance = 0.00001

        self.tau_E_trace = 10.0
        self.tau_I_trace = 100.0
        self.decay_E_trace = np.exp(-dt_ms / self.tau_E_trace)
        self.decay_I_trace = np.exp(-dt_ms / self.tau_I_trace)
        self.inc_E_trace = 1.0 - self.decay_E_trace
        self.inc_I_trace = 1.0 - self.decay_I_trace

        self.decay_pre_e = np.exp(-dt_ms / self.tau_pre_e)
        self.decay_post_e = np.exp(-dt_ms / self.tau_post_e)
        self.decay_pre_i = np.exp(-dt_ms / self.tau_pre_i)
        self.decay_post_i = np.exp(-dt_ms / self.tau_post_i)

        self.reset_state()

        self.records = []
        self.snapshots = []

    def reset_state(self):
        self.u = self.u_rest
        self.g_ahp = 0.0
        self.g_ampa = 0.0
        self.g_gaba = 0.0
        self.g_nmda = 0.0
        self.H_nmda = 0.0

        # Fortran initial condition:
        # x(19)=600, x(24)=100000 to suppress early E plasticity
        self.E_trace = 0.0
        self.I_trace = 600.0
        self.balance_signal = 0.0

        self.t_ms = 0.0
        self.step_index = 0
        self.last_post_spike = -10.0
        self.post_spike = False

        self.wE = np.ones(self.ne_input) * self.wE0
        self.wI = np.ones(self.ni_input) * self.wI0

        self.last_pre_e = np.ones(self.ne_input) * -10.0
        self.last_pre_i = np.ones(self.ni_input) * -10.0

        self.ypre_e = np.zeros(self.ne_input)
        self.ypost_e = np.zeros(self.ne_input)

        self.ypre_i = np.zeros(self.ni_input)
        self.ypost_i = np.zeros(self.ni_input)

        self.patt_time = np.zeros(self.n_pw)
        self.inp_patt = np.zeros(self.n_pw)

        self.window_spikes = 0

    def pathway_slice_e(self, pw_zero_based):
        start = pw_zero_based * self.ne_pw
        stop = (pw_zero_based + 1) * self.ne_pw
        return slice(start, stop)

    def pathway_slice_i(self, pw_zero_based):
        start = pw_zero_based * self.ni_pw
        stop = (pw_zero_based + 1) * self.ni_pw
        return slice(start, stop)

    def update_ou_input(self):
        if self.step_index % self.ou_update_steps != 0:
            return

        randg = self.rng.normal(0.0, 1.0, size=self.n_pw)
        self.patt_time = self.patt_time * self.ou_decay + randg

        positive = np.maximum(self.patt_time, 0.0)
        self.inp_patt = self.ou_amp_hz * self.dt_ms * positive / 1000.0

    def get_pathway_probabilities(self, mode, active_pw=None):
        pE = np.zeros(self.n_pw)
        pI = np.zeros(self.n_pw)

        if mode == "g":
            # inhomogeneous OU input
            self.update_ou_input()
            pE[:] = self.p_bg_e + self.inp_patt
            pI[:] = self.inh_rate_gain * (self.p_bg_i + self.inp_patt)

        elif mode == "h":
            # homogeneous input used for plotting before/after RF induction
            pE[:] = 5.0 * self.p_bg_e
            pI[:] = self.inh_rate_gain * self.p_bg_i

        elif mode == "bn":
            # receptive-field burst input
            # active_pw is 1-based, matching Fortran call simulationbn(..., 6)
            if active_pw is None:
                raise ValueError("active_pw must be provided for mode='bn'")

            for pw0 in range(self.n_pw):
                pw1 = pw0 + 1
                dist = abs(pw1 - active_pw)

                if dist == 0:
                    prop = 0.8
                elif dist == 1:
                    prop = 0.6
                elif dist == 2:
                    prop = 0.4
                elif dist == 3:
                    prop = 0.3
                elif dist == 4:
                    prop = 0.2
                elif dist == 5:
                    prop = 0.15
                else:
                    prop = 0.0

                pE[pw0] = self.p_bg_e + prop * self.p_burst_e

            pI[:] = self.inh_rate_gain * self.p_bg_i

        else:
            raise ValueError(f"unknown input mode: {mode}")

        return pE, pI

    def generate_presynaptic_spikes(self, mode, active_pw=None):
        pE, pI = self.get_pathway_probabilities(mode, active_pw)

        e_spikes = np.zeros(self.ne_input, dtype=bool)
        i_spikes = np.zeros(self.ni_input, dtype=bool)

        for pw0 in range(self.n_pw):
            sl = self.pathway_slice_e(pw0)
            ready = (self.t_ms - self.last_pre_e[sl]) > self.refrac_e
            random_values = self.rng.random(self.ne_pw)
            mask = (random_values <= pE[pw0]) & ready

            e_spikes[sl] = mask
            tmp = self.last_pre_e[sl].copy()
            tmp[mask] = self.t_ms
            self.last_pre_e[sl] = tmp

        for pw0 in range(self.n_pw):
            sl = self.pathway_slice_i(pw0)
            ready = (self.t_ms - self.last_pre_i[sl]) > (0.5 * self.refrac_e)
            random_values = self.rng.random(self.ni_pw)
            mask = (random_values <= pI[pw0]) & ready

            i_spikes[sl] = mask
            tmp = self.last_pre_i[sl].copy()
            tmp[mask] = self.t_ms
            self.last_pre_i[sl] = tmp

        # spikes -> conductances
        e_input = np.sum(e_spikes * self.wE)
        i_input = np.sum(i_spikes * self.wI)

        self.g_ampa += e_input
        self.g_nmda += e_input
        self.g_gaba += i_input

        return e_spikes, i_spikes

    def lif_step(self):
        self.post_spike = False

        if (self.t_ms - self.last_post_spike) > self.refrac_e:
            self.H_nmda = 1.0 / (1.0 + self.a_nmda * np.exp(self.b_nmda * self.u))

            g_tot = (
                1.0
                + self.g_ampa
                + self.g_nmda * self.H_nmda
                + self.g_gaba
                + self.g_ahp * self.g_ahp_amp
            )

            u_inf = (
                self.u_rest
                + self.g_gaba * self.E_gaba
                + self.g_ahp * self.g_ahp_amp * self.E_ahp
            ) / g_tot

            tau_eff = self.tau_m / g_tot
            decay = np.exp(-self.dt_ms / tau_eff)

            self.u = self.u * decay + u_inf * (1.0 - decay)

            if self.u >= self.threshold:
                self.last_post_spike = self.t_ms
                self.post_spike = True
                self.u = self.u_reset
                self.g_ahp += 1.0
                self.window_spikes += 1

        # decay conductances
        self.g_ahp *= self.decay_ahp
        self.g_ampa *= self.decay_ampa
        self.g_gaba *= self.decay_gaba
        self.g_nmda *= self.decay_nmda

        # codependent current traces
        self.E_trace = (
            self.E_trace * self.decay_E_trace
            - self.g_nmda * self.H_nmda * self.u * self.inc_E_trace
        )

        self.I_trace = (
            self.I_trace * self.decay_I_trace
            + self.g_gaba * (self.u - self.E_gaba) * self.inc_I_trace
        )

        self.balance_signal = self.A_balance * self.E_trace * (
            self.E_trace - self.alpha_balance * self.I_trace
        )

    def decay_plasticity_traces(self):
        self.ypre_e *= self.decay_pre_e
        self.ypost_e *= self.decay_post_e
        self.ypre_i *= self.decay_pre_i
        self.ypost_i *= self.decay_post_i

    def plasticity_e(self, e_spikes):
        # block excitatory plasticity if inhibition is too high
        if self.I_trace < self.I_rectify_threshold:
            I_control = self.I_trace
        else:
            I_control = 1e8

        gate = np.exp(-((I_control / self.I_block_scale) ** self.I_block_power))

        # pre event: LTD
        if np.any(e_spikes):
            self.wE[e_spikes] = self.wE[e_spikes] * (
                1.0 - self.A_ltd * self.ypost_e[e_spikes] * gate
            )

        # post event: LTP + heterosynaptic LTD
        if self.post_spike:
            delta = (
                self.A_ltp * self.ypre_e * self.E_trace
                - self.A_het * self.g_ahp * (self.E_trace ** 2)
            ) * gate
            self.wE += delta
            self.ypost_e += 1.0

        # final pre trace update
        if np.any(e_spikes):
            self.ypre_e[e_spikes] += 1.0

        self.wE = np.clip(self.wE, self.wE_min, self.wE_max)

    def plasticity_i(self, i_spikes):
        # inhibitory pre event
        if np.any(i_spikes):
            self.wI[i_spikes] += self.A_i * self.balance_signal * self.ypost_i[i_spikes]

        # inhibitory post event
        if self.post_spike:
            self.wI += self.A_i * self.balance_signal * self.ypre_i
            self.ypost_i += 1.0

        # final inhibitory pre trace update
        if np.any(i_spikes):
            self.ypre_i[i_spikes] += 1.0

        self.wI = np.clip(self.wI, self.wI_min, self.wI_max)

    def step(self, mode, active_pw=None, plasticity=True):
        self.t_ms += self.dt_ms
        self.step_index += 1

        e_spikes, i_spikes = self.generate_presynaptic_spikes(mode, active_pw)
        self.lif_step()

        if plasticity:
            self.decay_plasticity_traces()
            self.plasticity_e(e_spikes)
            self.plasticity_i(i_spikes)

    def mean_weights_by_pathway(self):
        mean_e = np.zeros(self.n_pw)
        mean_i = np.zeros(self.n_pw)

        for pw0 in range(self.n_pw):
            mean_e[pw0] = np.mean(self.wE[self.pathway_slice_e(pw0)])
            mean_i[pw0] = np.mean(self.wI[self.pathway_slice_i(pw0)])

        return mean_e, mean_i

    def record(self, label, record_interval_ms):
        mean_e, mean_i = self.mean_weights_by_pathway()

        firing_rate = self.window_spikes / (record_interval_ms / 1000.0)
        self.window_spikes = 0

        ei_ratio = np.nan
        if self.I_trace > 1e-12:
            ei_ratio = self.E_trace / self.I_trace

        row = {
            "time_s": self.t_ms / 1000.0,
            "label": label,
            "firing_rate_hz": firing_rate,
            "E_trace": self.E_trace,
            "I_trace": self.I_trace,
            "EI_ratio": ei_ratio,
        }

        for k in range(self.n_pw):
            row[f"E_path_{k + 1}"] = mean_e[k]
            row[f"I_path_{k + 1}"] = mean_i[k]

        self.records.append(row)

    def snapshot(self, name):
        mean_e, mean_i = self.mean_weights_by_pathway()

        norm_e = mean_e / np.max(mean_e)
        norm_i = mean_i / np.max(mean_i)

        self.snapshots.append(
            {
                "name": name,
                "time_s": self.t_ms / 1000.0,
                "E_norm": norm_e.copy(),
                "I_norm": norm_i.copy(),
                "E_raw": mean_e.copy(),
                "I_raw": mean_i.copy(),
            }
        )

    def run_segment(
        self,
        seconds,
        mode,
        active_pw=None,
        plasticity=True,
        label="segment",
        record_interval_ms=200.0,
    ):
        n_steps = int(round(seconds * 1000.0 / self.dt_ms))
        rec_every = max(1, int(round(record_interval_ms / self.dt_ms)))

        print(
            f"running {label}: mode={mode}, active_pw={active_pw}, "
            f"duration={seconds:.2f}s, steps={n_steps}"
        )

        for step in range(n_steps):
            self.step(mode=mode, active_pw=active_pw, plasticity=plasticity)

            if step % rec_every == 0:
                self.record(label=label, record_interval_ms=record_interval_ms)


def mode_settings(mode):
    if mode == "quick":
        return {
            "ne_pw": 25,
            "ni_pw": 8,
            "dt_ms": 0.2,
            "plasticity_scale": 2.0,
            "durations": {
                "warmup": 2.0,
                "baseline": 1.0,
                "rf_burst": 1.0,
                "after_rf": 5.0,
                "settle_1": 8.0,
                "settle_2": 8.0,
            },
        }

    if mode == "medium":
        return {
            "ne_pw": 50,
            "ni_pw": 12,
            "dt_ms": 0.2,
            "plasticity_scale": 1.0,
            "durations": {
                "warmup": 5.0,
                "baseline": 2.0,
                "rf_burst": 2.0,
                "after_rf": 10.0,
                "settle_1": 20.0,
                "settle_2": 20.0,
            },
        }

    if mode == "fullish":
        return {
            "ne_pw": 100,
            "ni_pw": 25,
            "dt_ms": 0.1,
            "plasticity_scale": 1.0,
            "durations": {
                "warmup": 50.0,
                "baseline": 0.5,
                "rf_burst": 0.2,
                "after_rf": 5.0,
                "settle_1": 60.0,
                "settle_2": 120.0,
            },
        }

    raise ValueError("mode must be quick, medium, or fullish")


def run_receptive_field_protocol(mode, seed, outdir):
    settings = mode_settings(mode)

    model = ReceptiveFieldModel(
        n_pw=8,
        ne_pw=settings["ne_pw"],
        ni_pw=settings["ni_pw"],
        dt_ms=settings["dt_ms"],
        seed=seed,
        plasticity_scale=settings["plasticity_scale"],
    )

    d = settings["durations"]

    # -------------------------
    # initial steady state
    # -------------------------
    model.run_segment(
        d["warmup"],
        mode="g",
        plasticity=False,
        label="warmup_no_plasticity",
    )
    model.snapshot("after_warmup")

    # -------------------------
    # first RF: pathway 6
    # -------------------------
    model.run_segment(
        d["baseline"],
        mode="h",
        plasticity=True,
        label="baseline_before_rf1",
    )
    model.snapshot("before_rf1")

    model.run_segment(
        d["rf_burst"],
        mode="bn",
        active_pw=6,
        plasticity=True,
        label="rf1_burst_pathway_6",
    )
    model.snapshot("after_rf1_burst")

    model.run_segment(
        d["after_rf"],
        mode="g",
        plasticity=True,
        label="after_rf1_ou",
    )
    model.snapshot("after_rf1_ou")

    model.run_segment(
        d["settle_1"],
        mode="g",
        plasticity=True,
        label="rf1_settle_1",
    )
    model.snapshot("after_rf1_settle_1")

    model.run_segment(
        d["settle_2"],
        mode="g",
        plasticity=True,
        label="rf1_settle_2",
    )
    model.snapshot("rf1_final")

    # -------------------------
    # second RF: pathway 4
    # -------------------------
    model.run_segment(
        d["baseline"],
        mode="h",
        plasticity=True,
        label="baseline_before_rf2",
    )
    model.snapshot("before_rf2")

    model.run_segment(
        d["rf_burst"],
        mode="bn",
        active_pw=4,
        plasticity=True,
        label="rf2_burst_pathway_4",
    )
    model.snapshot("after_rf2_burst")

    model.run_segment(
        d["after_rf"],
        mode="g",
        plasticity=True,
        label="after_rf2_ou",
    )
    model.snapshot("after_rf2_ou")

    model.run_segment(
        d["settle_1"],
        mode="g",
        plasticity=True,
        label="rf2_settle_1",
    )
    model.snapshot("after_rf2_settle_1")

    model.run_segment(
        d["settle_2"],
        mode="g",
        plasticity=True,
        label="rf2_settle_2",
    )
    model.snapshot("rf2_final")

    save_results(model, outdir, mode)
    plot_results(model, outdir, mode)

    return model


def save_results(model, outdir, mode):
    outdir.mkdir(exist_ok=True, parents=True)

    # time course CSV
    time_csv = outdir / f"rf_timecourse_{mode}.csv"
    if model.records:
        fieldnames = list(model.records[0].keys())
        with open(time_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(model.records)

    # snapshot CSV
    snap_csv = outdir / f"rf_snapshots_{mode}.csv"
    with open(snap_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["snapshot", "time_s", "pathway", "E_norm", "I_norm", "E_raw", "I_raw"]
        writer.writerow(header)

        for snap in model.snapshots:
            for pw in range(model.n_pw):
                writer.writerow(
                    [
                        snap["name"],
                        snap["time_s"],
                        pw + 1,
                        snap["E_norm"][pw],
                        snap["I_norm"][pw],
                        snap["E_raw"][pw],
                        snap["I_raw"][pw],
                    ]
                )

    print(f"saved CSV files to: {outdir.resolve()}")


def get_snapshot(model, name):
    for snap in model.snapshots:
        if snap["name"] == name:
            return snap
    raise ValueError(f"snapshot not found: {name}")


def records_to_arrays(model):
    times = np.array([r["time_s"] for r in model.records])
    firing = np.array([r["firing_rate_hz"] for r in model.records])
    E_trace = np.array([r["E_trace"] for r in model.records])
    I_trace = np.array([r["I_trace"] for r in model.records])
    EI_ratio = np.array([r["EI_ratio"] for r in model.records])

    E_paths = np.zeros((len(model.records), model.n_pw))
    I_paths = np.zeros((len(model.records), model.n_pw))

    for i, r in enumerate(model.records):
        for pw in range(model.n_pw):
            E_paths[i, pw] = r[f"E_path_{pw + 1}"]
            I_paths[i, pw] = r[f"I_path_{pw + 1}"]

    return times, firing, E_trace, I_trace, EI_ratio, E_paths, I_paths


def plot_results(model, outdir, mode):
    outdir.mkdir(exist_ok=True, parents=True)

    pathways = np.arange(1, model.n_pw + 1)

    rf1 = get_snapshot(model, "rf1_final")
    rf2 = get_snapshot(model, "rf2_final")

    times, firing, E_trace, I_trace, EI_ratio, E_paths, I_paths = records_to_arrays(model)
    times_min = times / 60.0

    # ------------------------------------------------------------
    # Summary figure: RF profiles + long time course
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(pathways, rf1["E_norm"], marker="o", label="1st RF")
    ax.plot(pathways, rf2["E_norm"], marker="s", linestyle="--", label="2nd RF")
    ax.set_title("Excitatory receptive-field profile")
    ax.set_xlabel("pathway")
    ax.set_ylabel("normalised weight")
    ax.set_xticks(pathways)
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(pathways, rf1["I_norm"], marker="o", label="1st RF")
    ax.plot(pathways, rf2["I_norm"], marker="s", linestyle="--", label="2nd RF")
    ax.set_title("Inhibitory receptive-field profile")
    ax.set_xlabel("pathway")
    ax.set_ylabel("normalised weight")
    ax.set_xticks(pathways)
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax = axes[1, 0]
    for pw in range(model.n_pw):
        ax.plot(times_min, 10.0 * E_paths[:, pw], label=f"path {pw + 1}")
    ax.set_title("Excitatory weights over time")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("weight (nS, scaled)")
    ax.legend(ncol=4, fontsize=7)

    ax = axes[1, 1]
    for pw in range(model.n_pw):
        ax.plot(times_min, 10.0 * I_paths[:, pw], label=f"path {pw + 1}")
    ax.set_title("Inhibitory weights over time")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("weight (nS, scaled)")
    ax.legend(ncol=4, fontsize=7)

    fig.tight_layout()
    fig.savefig(outdir / f"rf_summary_{mode}.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------
    # Trace figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(times_min, firing)
    axes[0].set_ylabel("firing rate (Hz)")
    axes[0].set_title("Postsynaptic firing rate")

    axes[1].plot(times_min, E_trace, label="E trace")
    axes[1].plot(times_min, I_trace, label="I trace")
    axes[1].set_ylabel("trace")
    axes[1].legend()

    axes[2].plot(times_min, EI_ratio)
    axes[2].set_ylabel("E/I")
    axes[2].set_xlabel("time (min)")

    fig.tight_layout()
    fig.savefig(outdir / f"rf_traces_{mode}.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------
    # Snapshot-only figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(pathways, rf1["E_norm"], marker="o", label="1st RF")
    axes[0].plot(pathways, rf2["E_norm"], marker="s", linestyle="--", label="2nd RF")
    axes[0].set_title("Excitatory")
    axes[0].set_xlabel("pathway")
    axes[0].set_ylabel("normalised weight")
    axes[0].set_xticks(pathways)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(pathways, rf1["I_norm"], marker="o", label="1st RF")
    axes[1].plot(pathways, rf2["I_norm"], marker="s", linestyle="--", label="2nd RF")
    axes[1].set_title("Inhibitory")
    axes[1].set_xlabel("pathway")
    axes[1].set_ylabel("normalised weight")
    axes[1].set_xticks(pathways)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / f"rf_profiles_{mode}.png", dpi=200)
    plt.close(fig)

    print(f"saved figures to: {outdir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["quick", "medium", "fullish"],
        default="quick",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    outdir = Path("results_receptive_field")
    model = run_receptive_field_protocol(
        mode=args.mode,
        seed=args.seed,
        outdir=outdir,
    )

    rf1 = get_snapshot(model, "rf1_final")
    rf2 = get_snapshot(model, "rf2_final")

    print("====================================")
    print("receptive field result")
    print("mode:", args.mode)
    print("RF1 final E profile:", np.round(rf1["E_norm"], 3))
    print("RF2 final E profile:", np.round(rf2["E_norm"], 3))
    print("RF1 final I profile:", np.round(rf1["I_norm"], 3))
    print("RF2 final I profile:", np.round(rf2["I_norm"], 3))
    print("====================================")


if __name__ == "__main__":
    main()