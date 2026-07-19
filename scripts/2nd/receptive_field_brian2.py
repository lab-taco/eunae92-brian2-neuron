"""
receptive_field_final.py

Presentation-oriented receptive-field sanity check for Fig. 5.

This is not a full reproduction of the Nature Neuroscience model.
It is a compact NumPy simulation that keeps the important mechanisms visible:

1. excitatory plasticity is gated by inhibition,
2. active pathways are potentiated during disinhibited learning windows,
3. inactive/weakly active pathways are weakened by a heterosynaptic term,
4. inhibitory plasticity uses a co-dependent balance term E(E - alpha I),
5. the first receptive field can be reshaped by a second disinhibited stimulus.

Recommended usage:
    python receptive_field_final.py --mode quick --seed 1 --outdir results_rf_final_quick
    python receptive_field_final.py --mode medium --seed 1 --outdir results_rf_final_medium

For the presentation, use the medium figures if time allows.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


EPS = 1e-12


@dataclass
class ModeSettings:
    ne_pw: int
    ni_pw: int
    dt_ms: float
    plasticity_scale: float
    warmup_s: float
    baseline_s: float
    rf_burst_s: float
    after_rf_s: float
    settle_1_s: float
    settle_2_s: float
    record_interval_ms: float


def mode_settings(mode: str) -> ModeSettings:
    if mode == "quick":
        return ModeSettings(
            ne_pw=25,
            ni_pw=8,
            dt_ms=0.2,
            plasticity_scale=2.0,
            warmup_s=3.0,
            baseline_s=1.0,
            rf_burst_s=1.0,
            after_rf_s=5.0,
            settle_1_s=8.0,
            settle_2_s=8.0,
            record_interval_ms=200.0,
        )
    if mode == "medium":
        return ModeSettings(
            ne_pw=50,
            ni_pw=12,
            dt_ms=0.2,
            plasticity_scale=1.0,
            warmup_s=5.0,
            baseline_s=2.0,
            rf_burst_s=2.0,
            after_rf_s=10.0,
            settle_1_s=20.0,
            settle_2_s=20.0,
            record_interval_ms=200.0,
        )
    if mode == "fullish":
        return ModeSettings(
            ne_pw=100,
            ni_pw=25,
            dt_ms=0.1,
            plasticity_scale=1.0,
            warmup_s=20.0,
            baseline_s=3.0,
            rf_burst_s=2.0,
            after_rf_s=15.0,
            settle_1_s=40.0,
            settle_2_s=60.0,
            record_interval_ms=250.0,
        )
    raise ValueError("mode must be quick, medium, or fullish")


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) == 0:
        return y
    window = min(window, len(y))
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    ypad = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


class ReceptiveFieldModel:
    def __init__(
        self,
        n_pw: int = 8,
        ne_pw: int = 25,
        ni_pw: int = 8,
        dt_ms: float = 0.2,
        seed: int = 1,
        plasticity_scale: float = 1.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_pw = n_pw
        self.ne_pw = ne_pw
        self.ni_pw = ni_pw
        self.ne_input = n_pw * ne_pw
        self.ni_input = n_pw * ni_pw
        self.dt_ms = dt_ms
        self.t_ms = 0.0
        self.step_index = 0

        # Neuron parameters. Voltages are mV-like dimensionless numbers.
        self.tau_m = 30.0
        self.u_rest = -65.0
        self.refrac_e = 5.0
        self.threshold = -50.0
        self.u_reset = -60.0
        self.E_gaba = -80.0
        self.E_ahp = -80.0
        self.g_ahp_amp = 5.0

        # Synaptic conductance time constants.
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

        # Input probabilities per time step. probability = dt_ms * Hz / 1000.
        self.p_bg_e = dt_ms * 2.0 / 1000.0
        self.p_bg_i = dt_ms * 2.0 / 1000.0
        self.p_burst_e = dt_ms * 55.0 / 1000.0
        # Inhibitory pathway activity is weaker during the disinhibited RF window,
        # but it is not zero; otherwise the inhibitory RF cannot become pathway-tuned.
        self.p_burst_i = dt_ms * 14.0 / 1000.0

        # OU-like spontaneous pathway envelope.
        self.tau_ou = 5.0
        self.ou_decay = np.exp(-dt_ms / self.tau_ou)
        self.ou_amp_hz = 5.0
        self.ou_update_steps = max(1, int(round(1.0 / dt_ms)))
        self.inh_rate_gain = 1.4

        # Initial weights.
        self.wE0 = 0.12
        self.wI0 = 0.90
        self.wE_max = 1.0
        self.wE_min = 1e-4
        self.wI_max = 10.0
        self.wI_min = 1e-3

        # Plasticity parameters. These are compressed-time values.
        self.tau_pre_e = 16.8
        self.tau_post_e = 33.7
        self.tau_pre_i = 20.0
        self.tau_post_i = 20.0

        self.A_ltp = (0.0005 / 3.0) * plasticity_scale
        # LTD is kept moderate here; otherwise the active pathway can be punished
        # by post-before-pre coincidences during the compressed burst protocol.
        self.A_ltd = 120.0 * self.A_ltp
        self.A_het = 0.000025 * self.A_ltp

        # Smooth inhibitory gate rather than a hard threshold.
        self.I_block_scale = 140.0
        self.I_block_power = 3.0
        # Outside RF induction windows, excitatory plasticity is effectively off.
        # During RF bursts the smooth inhibition gate is used.
        self.gate_floor = 0.0

        # Co-dependent inhibitory plasticity.
        self.alpha_balance = 0.93
        self.A_balance = 1.0e-5
        self.balance_clip = 0.08
        self.A_i = 0.006 * plasticity_scale

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

        self.records: List[dict] = []
        self.snapshots: List[dict] = []
        self.segment_spans: List[dict] = []
        self.reset_state()

    def reset_state(self) -> None:
        self.u = self.u_rest
        self.g_ahp = 0.0
        self.g_ampa = 0.0
        self.g_gaba = 0.0
        self.g_nmda = 0.0
        self.H_nmda = 0.0

        # Start with moderately elevated inhibition to suppress early plasticity,
        # but avoid the enormous first-point artifact in the previous plots.
        self.E_trace = 0.0
        self.I_trace = 180.0
        self.balance_signal = 0.0
        self.last_gate = 0.0

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

    def pathway_slice_e(self, pw_zero_based: int) -> slice:
        return slice(pw_zero_based * self.ne_pw, (pw_zero_based + 1) * self.ne_pw)

    def pathway_slice_i(self, pw_zero_based: int) -> slice:
        return slice(pw_zero_based * self.ni_pw, (pw_zero_based + 1) * self.ni_pw)

    def update_ou_input(self) -> None:
        if self.step_index % self.ou_update_steps != 0:
            return
        self.patt_time = self.patt_time * self.ou_decay + self.rng.normal(0.0, 1.0, size=self.n_pw)
        self.inp_patt = self.ou_amp_hz * self.dt_ms * np.maximum(self.patt_time, 0.0) / 1000.0

    def stimulus_profile(self, active_pw: int) -> np.ndarray:
        profile = np.zeros(self.n_pw)
        for pw0 in range(self.n_pw):
            pw1 = pw0 + 1
            dist = abs(pw1 - active_pw)
            if dist == 0:
                prop = 1.00
            elif dist == 1:
                prop = 0.45
            elif dist == 2:
                prop = 0.18
            elif dist == 3:
                prop = 0.08
            elif dist == 4:
                prop = 0.04
            else:
                prop = 0.02
            profile[pw0] = prop
        return profile

    def get_pathway_probabilities(self, mode: str, active_pw: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        pE = np.zeros(self.n_pw)
        pI = np.zeros(self.n_pw)

        if mode == "g":
            self.update_ou_input()
            pE[:] = self.p_bg_e + self.inp_patt
            pI[:] = self.inh_rate_gain * (self.p_bg_i + 0.6 * self.inp_patt)
        elif mode == "h":
            pE[:] = 5.0 * self.p_bg_e
            pI[:] = self.inh_rate_gain * self.p_bg_i
        elif mode == "bn":
            if active_pw is None:
                raise ValueError("active_pw must be provided for mode='bn'")
            prof = self.stimulus_profile(active_pw)
            pE[:] = self.p_bg_e + prof * self.p_burst_e
            # Key fix: pathway-specific inhibitory activity is present, but weaker.
            # This lets ISP form a co-tuned inhibitory RF instead of a flat profile.
            pI[:] = self.inh_rate_gain * (self.p_bg_i + 0.35 * prof * self.p_burst_i)
        else:
            raise ValueError(f"unknown input mode: {mode}")
        return pE, pI

    def generate_presynaptic_spikes(self, mode: str, active_pw: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        pE, pI = self.get_pathway_probabilities(mode, active_pw)
        e_spikes = np.zeros(self.ne_input, dtype=bool)
        i_spikes = np.zeros(self.ni_input, dtype=bool)

        for pw0 in range(self.n_pw):
            sl = self.pathway_slice_e(pw0)
            ready = (self.t_ms - self.last_pre_e[sl]) > self.refrac_e
            mask = (self.rng.random(self.ne_pw) <= pE[pw0]) & ready
            e_spikes[sl] = mask
            self.last_pre_e[sl][mask] = self.t_ms

        for pw0 in range(self.n_pw):
            sl = self.pathway_slice_i(pw0)
            ready = (self.t_ms - self.last_pre_i[sl]) > (0.5 * self.refrac_e)
            mask = (self.rng.random(self.ni_pw) <= pI[pw0]) & ready
            i_spikes[sl] = mask
            self.last_pre_i[sl][mask] = self.t_ms

        e_input = float(np.sum(e_spikes * self.wE))
        i_input = float(np.sum(i_spikes * self.wI))
        self.g_ampa += e_input
        self.g_nmda += e_input
        self.g_gaba += i_input
        return e_spikes, i_spikes

    def lif_step(self) -> None:
        self.post_spike = False
        self.H_nmda = 1.0 / (1.0 + self.a_nmda * np.exp(self.b_nmda * self.u))

        if (self.t_ms - self.last_post_spike) > self.refrac_e:
            g_tot = 1.0 + self.g_ampa + self.g_nmda * self.H_nmda + self.g_gaba + self.g_ahp * self.g_ahp_amp
            u_inf = (self.u_rest + self.g_gaba * self.E_gaba + self.g_ahp * self.g_ahp_amp * self.E_ahp) / g_tot
            tau_eff = self.tau_m / g_tot
            decay = np.exp(-self.dt_ms / max(tau_eff, 1e-6))
            self.u = self.u * decay + u_inf * (1.0 - decay)

            if self.u >= self.threshold:
                self.last_post_spike = self.t_ms
                self.post_spike = True
                self.u = self.u_reset
                self.g_ahp += 1.0
                self.window_spikes += 1

        self.g_ahp *= self.decay_ahp
        self.g_ampa *= self.decay_ampa
        self.g_gaba *= self.decay_gaba
        self.g_nmda *= self.decay_nmda

        E_drive = max(0.0, -self.g_nmda * self.H_nmda * self.u)
        I_drive = max(0.0, self.g_gaba * (self.u - self.E_gaba))
        self.E_trace = self.E_trace * self.decay_E_trace + E_drive * self.inc_E_trace
        self.I_trace = self.I_trace * self.decay_I_trace + I_drive * self.inc_I_trace

        balance_raw = self.A_balance * self.E_trace * (self.E_trace - self.alpha_balance * self.I_trace)
        self.balance_signal = float(np.clip(balance_raw, -self.balance_clip, self.balance_clip))

    def decay_plasticity_traces(self) -> None:
        self.ypre_e *= self.decay_pre_e
        self.ypost_e *= self.decay_post_e
        self.ypre_i *= self.decay_pre_i
        self.ypost_i *= self.decay_post_i

    def inhibitory_gate(self) -> float:
        x = max(0.0, self.I_trace) / self.I_block_scale
        gate = np.exp(-min(50.0, x ** self.I_block_power))
        gate = max(self.gate_floor, gate)
        self.last_gate = gate
        return gate

    def plasticity_e(self, e_spikes: np.ndarray, force_gate: Optional[float] = None) -> None:
        gate = self.inhibitory_gate() if force_gate is None else force_gate
        if np.any(e_spikes):
            self.wE[e_spikes] *= 1.0 - self.A_ltd * self.ypost_e[e_spikes] * gate

        if self.post_spike:
            # Heterosynaptic term is non-specific and uses the global E trace.
            delta = (self.A_ltp * self.ypre_e * self.E_trace - self.A_het * (self.E_trace ** 2)) * gate
            self.wE += delta
            self.ypost_e += 1.0

        if np.any(e_spikes):
            self.ypre_e[e_spikes] += 1.0
        self.wE = np.clip(self.wE, self.wE_min, self.wE_max)

    def plasticity_i(self, i_spikes: np.ndarray) -> None:
        if np.any(i_spikes):
            self.wI[i_spikes] += self.A_i * self.balance_signal * self.ypost_i[i_spikes]
        if self.post_spike:
            self.wI += self.A_i * self.balance_signal * self.ypre_i
            self.ypost_i += 1.0
        if np.any(i_spikes):
            self.ypre_i[i_spikes] += 1.0
        self.wI = np.clip(self.wI, self.wI_min, self.wI_max)

    def step(
        self,
        mode: str,
        active_pw: Optional[int] = None,
        plasticity: bool = True,
        excitatory_plasticity: bool = True,
    ) -> None:
        self.t_ms += self.dt_ms
        self.step_index += 1
        e_spikes, i_spikes = self.generate_presynaptic_spikes(mode, active_pw)
        self.lif_step()
        if plasticity:
            self.decay_plasticity_traces()
            if excitatory_plasticity:
                # During RF induction we explicitly model disinhibition: excitatory
                # plasticity is allowed even if the slow I trace has not fully decayed.
                gate_override = 1.0 if mode == "bn" else None
                self.plasticity_e(e_spikes, force_gate=gate_override)
            else:
                # Keep the current gate value for plotting, but do not update E weights.
                self.inhibitory_gate()
            self.plasticity_i(i_spikes)

    def mean_weights_by_pathway(self) -> Tuple[np.ndarray, np.ndarray]:
        mean_e = np.zeros(self.n_pw)
        mean_i = np.zeros(self.n_pw)
        for pw0 in range(self.n_pw):
            mean_e[pw0] = np.mean(self.wE[self.pathway_slice_e(pw0)])
            mean_i[pw0] = np.mean(self.wI[self.pathway_slice_i(pw0)])
        return mean_e, mean_i

    @staticmethod
    def profile_max_norm(x: np.ndarray) -> np.ndarray:
        m = float(np.max(x))
        return x / (m + EPS)

    @staticmethod
    def profile_minmax_norm(x: np.ndarray) -> np.ndarray:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-12:
            return np.ones_like(x)
        return (x - lo) / (hi - lo)

    def record(self, label: str, record_interval_ms: float) -> None:
        mean_e, mean_i = self.mean_weights_by_pathway()
        firing_rate = self.window_spikes / (record_interval_ms / 1000.0)
        self.window_spikes = 0
        row = {
            "time_s": self.t_ms / 1000.0,
            "label": label,
            "firing_rate_hz": firing_rate,
            "E_trace": self.E_trace,
            "I_trace": self.I_trace,
            "EI_ratio": self.E_trace / (self.I_trace + EPS),
            "plasticity_gate": self.last_gate,
            "balance_signal": self.balance_signal,
        }
        for k in range(self.n_pw):
            row[f"E_path_{k + 1}"] = mean_e[k]
            row[f"I_path_{k + 1}"] = mean_i[k]
        self.records.append(row)

    def snapshot(self, name: str) -> None:
        mean_e, mean_i = self.mean_weights_by_pathway()
        self.snapshots.append(
            {
                "name": name,
                "time_s": self.t_ms / 1000.0,
                "E_norm": self.profile_max_norm(mean_e),
                "I_norm": self.profile_max_norm(mean_i),
                "E_shape": self.profile_minmax_norm(mean_e),
                "I_shape": self.profile_minmax_norm(mean_i),
                "E_raw": mean_e.copy(),
                "I_raw": mean_i.copy(),
            }
        )

    def run_segment(
        self,
        seconds: float,
        mode: str,
        active_pw: Optional[int] = None,
        plasticity: bool = True,
        excitatory_plasticity: Optional[bool] = None,
        label: str = "segment",
        record_interval_ms: float = 200.0,
    ) -> None:
        if excitatory_plasticity is None:
            # Fig. 5 mechanism: excitatory RF changes occur during disinhibited RF bursts;
            # outside those windows, ongoing E plasticity is gated off and the RF remains stable.
            excitatory_plasticity = (mode == "bn")

        start_s = self.t_ms / 1000.0
        n_steps = int(round(seconds * 1000.0 / self.dt_ms))
        rec_every = max(1, int(round(record_interval_ms / self.dt_ms)))
        print(
            f"running {label}: mode={mode}, active_pw={active_pw}, "
            f"plasticity={plasticity}, E_plasticity={excitatory_plasticity}, "
            f"duration={seconds:.2f}s, steps={n_steps}"
        )
        for step in range(n_steps):
            self.step(
                mode=mode,
                active_pw=active_pw,
                plasticity=plasticity,
                excitatory_plasticity=excitatory_plasticity,
            )
            if step % rec_every == 0:
                self.record(label=label, record_interval_ms=record_interval_ms)
        end_s = self.t_ms / 1000.0
        self.segment_spans.append({"label": label, "start_s": start_s, "end_s": end_s, "active_pw": active_pw})


def run_receptive_field_protocol(mode: str, seed: int, outdir: Path) -> ReceptiveFieldModel:
    s = mode_settings(mode)
    model = ReceptiveFieldModel(
        n_pw=8,
        ne_pw=s.ne_pw,
        ni_pw=s.ni_pw,
        dt_ms=s.dt_ms,
        seed=seed,
        plasticity_scale=s.plasticity_scale,
    )

    model.run_segment(s.warmup_s, mode="g", plasticity=False, label="warmup_no_plasticity", record_interval_ms=s.record_interval_ms)
    model.snapshot("after_warmup")

    model.run_segment(s.baseline_s, mode="h", plasticity=True, label="baseline_before_rf1", record_interval_ms=s.record_interval_ms)
    model.snapshot("before_rf1")
    model.run_segment(s.rf_burst_s, mode="bn", active_pw=6, plasticity=True, label="rf1_burst_pathway_6", record_interval_ms=s.record_interval_ms)
    model.snapshot("after_rf1_burst")
    model.run_segment(s.after_rf_s, mode="g", plasticity=True, label="after_rf1_ou", record_interval_ms=s.record_interval_ms)
    model.snapshot("after_rf1_ou")
    model.run_segment(s.settle_1_s, mode="g", plasticity=True, label="rf1_settle_1", record_interval_ms=s.record_interval_ms)
    model.snapshot("after_rf1_settle_1")
    model.run_segment(s.settle_2_s, mode="g", plasticity=True, label="rf1_settle_2", record_interval_ms=s.record_interval_ms)
    model.snapshot("rf1_final")

    model.run_segment(s.baseline_s, mode="h", plasticity=True, label="baseline_before_rf2", record_interval_ms=s.record_interval_ms)
    model.snapshot("before_rf2")
    model.run_segment(s.rf_burst_s, mode="bn", active_pw=4, plasticity=True, label="rf2_burst_pathway_4", record_interval_ms=s.record_interval_ms)
    model.snapshot("after_rf2_burst")
    model.run_segment(s.after_rf_s, mode="g", plasticity=True, label="after_rf2_ou", record_interval_ms=s.record_interval_ms)
    model.snapshot("after_rf2_ou")
    model.run_segment(s.settle_1_s, mode="g", plasticity=True, label="rf2_settle_1", record_interval_ms=s.record_interval_ms)
    model.snapshot("after_rf2_settle_1")
    model.run_segment(s.settle_2_s, mode="g", plasticity=True, label="rf2_settle_2", record_interval_ms=s.record_interval_ms)
    model.snapshot("rf2_final")

    save_results(model, outdir, mode)
    plot_results(model, outdir, mode)
    return model


def save_results(model: ReceptiveFieldModel, outdir: Path, mode: str) -> None:
    outdir.mkdir(exist_ok=True, parents=True)
    time_csv = outdir / f"rf_timecourse_{mode}_final.csv"
    if model.records:
        with open(time_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(model.records[0].keys()))
            writer.writeheader()
            writer.writerows(model.records)

    snap_csv = outdir / f"rf_snapshots_{mode}_final.csv"
    with open(snap_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot", "time_s", "pathway", "E_norm", "I_norm", "E_shape", "I_shape", "E_raw", "I_raw"])
        for snap in model.snapshots:
            for pw in range(model.n_pw):
                writer.writerow([
                    snap["name"],
                    snap["time_s"],
                    pw + 1,
                    snap["E_norm"][pw],
                    snap["I_norm"][pw],
                    snap["E_shape"][pw],
                    snap["I_shape"][pw],
                    snap["E_raw"][pw],
                    snap["I_raw"][pw],
                ])
    print(f"saved CSV files to: {outdir.resolve()}")


def get_snapshot(model: ReceptiveFieldModel, name: str) -> dict:
    for snap in model.snapshots:
        if snap["name"] == name:
            return snap
    raise ValueError(f"snapshot not found: {name}")


def records_to_arrays(model: ReceptiveFieldModel) -> Tuple[np.ndarray, ...]:
    times = np.array([r["time_s"] for r in model.records])
    firing = np.array([r["firing_rate_hz"] for r in model.records])
    E_trace = np.array([r["E_trace"] for r in model.records])
    I_trace = np.array([r["I_trace"] for r in model.records])
    EI_ratio = np.array([r["EI_ratio"] for r in model.records])
    gate = np.array([r["plasticity_gate"] for r in model.records])
    balance = np.array([r["balance_signal"] for r in model.records])
    E_paths = np.zeros((len(model.records), model.n_pw))
    I_paths = np.zeros((len(model.records), model.n_pw))
    for i, r in enumerate(model.records):
        for pw in range(model.n_pw):
            E_paths[i, pw] = r[f"E_path_{pw + 1}"]
            I_paths[i, pw] = r[f"I_path_{pw + 1}"]
    return times, firing, E_trace, I_trace, EI_ratio, gate, balance, E_paths, I_paths


def add_segment_shading(ax, model: ReceptiveFieldModel) -> None:
    for span in model.segment_spans:
        label = span["label"]
        if "burst" in label:
            color = "0.85"
            ax.axvspan(span["start_s"] / 60.0, span["end_s"] / 60.0, alpha=0.35, color=color)


def plot_results(model: ReceptiveFieldModel, outdir: Path, mode: str) -> None:
    outdir.mkdir(exist_ok=True, parents=True)
    pathways = np.arange(1, model.n_pw + 1)
    rf1 = get_snapshot(model, "rf1_final")
    rf2 = get_snapshot(model, "rf2_final")
    times, firing, E_trace, I_trace, EI_ratio, gate, balance, E_paths, I_paths = records_to_arrays(model)
    times_min = times / 60.0
    smooth_n = 5

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(pathways, rf1["E_norm"], marker="o", label="1st RF")
    axes[0, 0].plot(pathways, rf2["E_norm"], marker="s", linestyle="--", label="2nd RF")
    axes[0, 0].set_title("Excitatory receptive-field profile")
    axes[0, 0].set_xlabel("pathway")
    axes[0, 0].set_ylabel("max-normalized weight")
    axes[0, 0].set_xticks(pathways)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend()

    # Show inhibitory shape using min-max normalization; max-normalization hides small but real tuning.
    axes[0, 1].plot(pathways, rf1["I_shape"], marker="o", label="1st RF")
    axes[0, 1].plot(pathways, rf2["I_shape"], marker="s", linestyle="--", label="2nd RF")
    axes[0, 1].set_title("Inhibitory receptive-field shape")
    axes[0, 1].set_xlabel("pathway")
    axes[0, 1].set_ylabel("min-max normalized weight")
    axes[0, 1].set_xticks(pathways)
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend()

    ax = axes[1, 0]
    for pw in range(model.n_pw):
        ax.plot(times_min, E_paths[:, pw], label=f"path {pw + 1}")
    add_segment_shading(ax, model)
    ax.set_title("Excitatory weights over time")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("mean weight")
    ax.legend(ncol=4, fontsize=7)

    ax = axes[1, 1]
    for pw in range(model.n_pw):
        ax.plot(times_min, I_paths[:, pw], label=f"path {pw + 1}")
    add_segment_shading(ax, model)
    ax.set_title("Inhibitory weights over time")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("mean weight")
    ax.legend(ncol=4, fontsize=7)

    fig.suptitle(f"Receptive-field plasticity sanity check ({mode})", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / f"rf_summary_{mode}_final.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(times_min, firing, alpha=0.45, label="raw")
    axes[0].plot(times_min, smooth(firing, smooth_n), linewidth=2, label="smoothed")
    axes[0].set_ylabel("firing rate (Hz)")
    axes[0].set_title("Postsynaptic firing and co-dependent traces")
    axes[0].legend()

    axes[1].plot(times_min, smooth(E_trace, smooth_n), label="E trace")
    axes[1].plot(times_min, smooth(I_trace, smooth_n), label="I trace")
    axes[1].set_ylabel("trace")
    axes[1].legend()

    axes[2].plot(times_min, smooth(EI_ratio, smooth_n), label="E/I")
    axes[2].axhline(model.alpha_balance, linestyle="--", linewidth=1.2, label=f"alpha = {model.alpha_balance}")
    axes[2].set_ylabel("E/I")
    axes[2].legend()

    axes[3].plot(times_min, smooth(gate, smooth_n), label="excitatory plasticity gate")
    axes[3].plot(times_min, smooth(balance, smooth_n), label="ISP balance signal")
    axes[3].set_ylabel("gate / balance")
    axes[3].set_xlabel("time (min)")
    axes[3].legend()

    for ax in axes:
        add_segment_shading(ax, model)

    fig.tight_layout()
    fig.savefig(outdir / f"rf_traces_{mode}_final.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(pathways, rf1["E_norm"], marker="o", label="1st RF")
    axes[0].plot(pathways, rf2["E_norm"], marker="s", linestyle="--", label="2nd RF")
    axes[0].set_title("Excitatory")
    axes[0].set_xlabel("pathway")
    axes[0].set_ylabel("max-normalized weight")
    axes[0].set_xticks(pathways)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(pathways, rf1["I_shape"], marker="o", label="1st RF")
    axes[1].plot(pathways, rf2["I_shape"], marker="s", linestyle="--", label="2nd RF")
    axes[1].set_title("Inhibitory shape")
    axes[1].set_xlabel("pathway")
    axes[1].set_ylabel("min-max normalized weight")
    axes[1].set_xticks(pathways)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / f"rf_profiles_{mode}_final.png", dpi=200)
    plt.close(fig)

    print(f"saved figures to: {outdir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "medium", "fullish"], default="quick")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="results_receptive_field_final")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    model = run_receptive_field_protocol(args.mode, args.seed, outdir)
    rf1 = get_snapshot(model, "rf1_final")
    rf2 = get_snapshot(model, "rf2_final")

    print("====================================")
    print("receptive field result")
    print("mode:", args.mode)
    print("RF1 final E profile:", np.round(rf1["E_norm"], 3))
    print("RF2 final E profile:", np.round(rf2["E_norm"], 3))
    print("RF1 final I shape:", np.round(rf1["I_shape"], 3))
    print("RF2 final I shape:", np.round(rf2["I_shape"], 3))
    print("====================================")


if __name__ == "__main__":
    main()
