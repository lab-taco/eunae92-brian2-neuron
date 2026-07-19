#!/usr/bin/env python3
"""
Compressed dendritic clustering simulation for Fig. 6-style explanation.

This script is designed for presentation/sanity-check use rather than exact
full-scale reproduction. It keeps the model fully in NumPy so it can run without
Brian2, while preserving the key quantities used in the paper-style mechanism:

  - correlated vs uncorrelated excitatory groups on one dendrite
  - local E and I traces
  - excitatory plasticity gated by inhibition
  - codependent inhibitory plasticity driven by E * (E - alpha I)
  - clustering index = (w_corr - w_uncorr) / (w_corr + w_uncorr)

Recommended usage:
  python dendritic_clustering_final.py --mode smoke --outdir results_dendrites_final_smoke
  python dendritic_clustering_final.py --mode quick --n-repeats 3 --outdir results_dendrites_final_quick
  python dendritic_clustering_final.py --mode medium --n-repeats 3 --outdir results_dendrites_final_medium

Plot existing CSV without rerunning:
  python dendritic_clustering_final.py --plot-only --csv dendrites_clustering_medium.csv \
      --trace-csv dendrites_example_trace_medium.csv --mode medium --outdir results_dendrites_plotonly
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def np_clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def sem(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


@dataclass(frozen=True)
class ModeConfig:
    group_sizes: List[int]
    warmup_s: float
    plastic_min: float
    dt_ms: float


def get_mode_config(mode: str) -> ModeConfig:
    if mode == "smoke":
        return ModeConfig(group_sizes=[2, 8, 16, 24, 30], warmup_s=1.0, plastic_min=0.15, dt_ms=0.2)
    if mode == "quick":
        return ModeConfig(group_sizes=list(range(2, 31)), warmup_s=2.0, plastic_min=0.5, dt_ms=0.1)
    if mode == "medium":
        return ModeConfig(group_sizes=list(range(2, 31)), warmup_s=5.0, plastic_min=2.0, dt_ms=0.1)
    if mode == "long":
        return ModeConfig(group_sizes=list(range(2, 31)), warmup_s=5.0, plastic_min=10.0, dt_ms=0.1)
    raise ValueError("mode must be smoke, quick, medium, or long")


class DendriteSimulation:
    """Pure NumPy compressed two-compartment dendritic clustering simulation."""

    def __init__(self, ne_corr: int, dt_ms: float = 0.1, seed: int = 1):
        self.rng = np.random.default_rng(seed)

        # Network size. We avoid ne_corr=0 or 32 in scans because one of the
        # two means in the clustering index would otherwise be undefined.
        self.ne_input = 32
        self.ni_input = self.ne_input // 2
        self.ne_corr = int(ne_corr)
        if not (1 <= self.ne_corr <= self.ne_input - 1):
            raise ValueError("ne_corr must be between 1 and ne_input-1")

        self.ni_corr = 8
        self.n_total = self.ne_input + self.ni_input
        self.n_pw = self.n_total * 2

        # Time and membrane parameters.
        self.dt = float(dt_ms)
        self.tau_m = 30.0
        self.u_rest = -65.0
        self.refrac = 5.0
        self.u_th = -50.0
        self.E_ahp = -80.0
        self.g_ahp_amp = 0.5
        self.u_reset = -60.0
        self.decay_ahp = np.exp(-self.dt / 100.0)

        # Synaptic conductances.
        self.E_gaba = -80.0
        self.decay_ampa = np.exp(-self.dt / 5.0)
        self.decay_gaba = np.exp(-self.dt / 10.0)
        self.decay_nmda = np.exp(-self.dt / 150.0)
        self.a_nmda = 0.15
        self.b_nmda = -0.08

        # Input probabilities per step. These are intentionally compressed,
        # not exact paper parameters.
        self.p_e_base = self.dt * 2.0 / 1000.0
        self.p_i_base = self.dt * 4.0 / 1000.0
        self.p_i_factor = 2.0

        # OU-like envelope for shared vs independent input modulation.
        self.tau_ou = 5.0
        self.decay_ou = np.exp(-self.dt / self.tau_ou)
        self.one_minus_decay_ou = 1.0 - self.decay_ou
        self.ou_amp = 250.0

        # Initial weights and limits.
        self.w_e0 = 0.3
        self.w_i0 = 0.5
        self.w_e_max = 1.0
        self.w_i_max = 10.0
        self.w_e_min = 1e-5
        self.w_i_min = 1e-4

        # Excitatory plasticity: Eq. style
        # dw_E = [A_ltp x_pre E - A_het y_post^E E^2 - A_ltd y_post^- w] * G_I(I)
        self.tau_pre_e = 16.8
        self.tau_post_e = 33.7
        self.tau_post_het_e = 125.0
        self.A_ltp_e = 0.00003
        self.A_ltd_e = 50.0 * self.A_ltp_e
        self.A_het_e = 0.0002 * self.A_ltp_e
        self.I_star = 50.0
        self.I_gamma = 1.0

        # Inhibitory plasticity: dw_I proportional to E * (E - alpha I).
        self.tau_i = 20.0
        self.A_i = 0.001
        self.balance_alpha = 1.75
        self.balance_amp = 0.0001

        # E/I traces. Use a faster E trace and slower I trace to expose balance.
        self.decay_E = np.exp(-self.dt / 10.0)
        self.one_minus_decay_E = 1.0 - self.decay_E
        self.decay_I = np.exp(-self.dt / 100.0)
        self.one_minus_decay_I = 1.0 - self.decay_I

        # Soma-dendrite coupling.
        self.g_c = 8.0

        self.reset_state()

    def reset_state(self):
        self.u = self.u_rest
        self.g_ahp = 0.0

        self.ud = self.u_rest
        self.gd_ampa = 0.0
        self.gd_gaba = 0.0
        self.gd_nmda = 0.0

        self.ub = self.u_rest
        self.gb_ampa = 0.0
        self.gb_gaba = 0.0
        self.gb_nmda = 0.0

        self.E_trace = 0.0
        self.I_trace = 100.0
        self.balance_signal = 0.0
        self.y_post_het = 0.0

        self.t_ms = 0.0
        self.last_post = -10.0

        self.w = np.zeros(self.n_total)
        self.w[: self.ne_input] = self.w_e0
        self.w[self.ne_input :] = self.w_i0

        self.spk = np.zeros(self.n_total, dtype=bool)
        self.last_pre = np.full(self.n_total, -10.0)
        self.ypre = np.zeros(self.n_total)
        self.ypost = np.zeros(self.n_total)
        self.last_ypre_update = np.full(self.n_total, -10.0)
        self.last_ypost_update = np.full(self.n_total, -10.0)

        self.patt_time = np.zeros(self.n_pw)
        self.inp_patt = np.zeros(self.n_pw)

    def update_ou_input(self, step: int):
        if step % 10 != 0:
            return
        noise = self.rng.normal(0.0, 1.0, size=self.n_pw)
        self.patt_time = self.patt_time * self.decay_ou + self.ou_amp * noise * self.one_minus_decay_ou
        rate_hz = np.maximum(self.patt_time, 0.0)
        self.inp_patt = np_clip(self.dt * rate_hz / 1000.0, 0.0, 0.95)

    def generate_input(self, step: int):
        self.spk[:] = False
        self.update_ou_input(step)

        rand = self.rng.random(self.n_total)

        # Dendrite 1: plastic correlated excitatory inputs share inp_patt[0].
        if self.ne_corr > 0:
            idx = np.arange(0, self.ne_corr)
            prob = np_clip(self.p_e_base + self.inp_patt[0], 0.0, 0.95)
            ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > self.refrac)
            self.spk[idx[ok]] = True
            self.last_pre[idx[ok]] = self.t_ms

        # Dendrite 1: remaining excitatory inputs have independent envelopes.
        if self.ne_corr < self.ne_input:
            idx = np.arange(self.ne_corr, self.ne_input)
            prob = np_clip(self.p_e_base + self.inp_patt[idx], 0.0, 0.95)
            ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > self.refrac)
            self.spk[idx[ok]] = True
            self.last_pre[idx[ok]] = self.t_ms

        # Dendrite 1: correlated inhibitory group.
        idx = np.arange(self.ne_input, self.ne_input + self.ni_corr)
        prob = np_clip(self.p_i_base + self.p_i_factor * self.inp_patt[0], 0.0, 0.95)
        ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > 0.5 * self.refrac)
        self.spk[idx[ok]] = True
        self.last_pre[idx[ok]] = self.t_ms

        # Dendrite 1: uncorrelated inhibitory group.
        idx = np.arange(self.ne_input + self.ni_corr, self.n_total)
        prob = np_clip(self.p_i_base + self.p_i_factor * self.inp_patt[idx], 0.0, 0.95)
        ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > 0.5 * self.refrac)
        self.spk[idx[ok]] = True
        self.last_pre[idx[ok]] = self.t_ms

        e_spikes = self.spk[: self.ne_input]
        i_spikes = self.spk[self.ne_input :]
        e_input = float(np.sum(e_spikes * self.w[: self.ne_input]))
        i_input = float(np.sum(i_spikes * self.w[self.ne_input :]))
        self.gd_ampa += e_input
        self.gd_nmda += e_input
        self.gd_gaba += i_input

        # Dendrite 2: background-only compartment.
        rand2 = self.rng.random(self.n_total)
        e_prob = np_clip(self.p_e_base + self.inp_patt[self.n_total : self.n_total + self.ne_input], 0.0, 0.95)
        e_bg = rand2[: self.ne_input] <= e_prob
        self.gb_ampa += float(np.sum(e_bg)) * self.w_e0
        self.gb_nmda += float(np.sum(e_bg)) * self.w_e0

        i_prob = np_clip(
            self.p_i_base + self.p_i_factor * self.inp_patt[self.n_total + self.ne_input : self.n_pw],
            0.0,
            0.95,
        )
        i_bg = rand2[self.ne_input :] <= i_prob
        self.gb_gaba += float(np.sum(i_bg)) * self.w_i0 * 2.2

    def lif_step(self) -> bool:
        spk_post = False

        # Soma.
        if (self.t_ms - self.last_post) > self.refrac:
            total_g = 1.0 + self.g_ahp * self.g_ahp_amp + 2.0 * self.g_c
            inv_g = 1.0 / total_g
            u_inf = (self.u_rest + self.g_ahp * self.g_ahp_amp * self.E_ahp + self.g_c * (self.ud + self.ub)) * inv_g
            decay = np.exp(-self.dt * total_g / self.tau_m)
            self.u = self.u * decay + u_inf * (1.0 - decay)
            self.u = float(np.clip(self.u, -90.0, 30.0))
            if self.u >= self.u_th:
                self.last_post = self.t_ms
                spk_post = True
                self.u = self.u_reset
                self.g_ahp += 1.0

        self.g_ahp *= self.decay_ahp

        # Dendrite 1.
        H_d = 1.0 / (1.0 + self.a_nmda * np.exp(np.clip(self.b_nmda * self.ud, -50.0, 50.0)))
        total_g_d = 1.0 + self.gd_ampa + self.gd_nmda * H_d + self.gd_gaba + self.g_c
        inv_g_d = 1.0 / total_g_d
        u_inf_d = (self.u_rest + self.gd_gaba * self.E_gaba + self.g_c * self.u) * inv_g_d
        decay_d = np.exp(-self.dt * total_g_d / self.tau_m)
        self.ud = float(np.clip(self.ud * decay_d + u_inf_d * (1.0 - decay_d), -90.0, 30.0))

        self.gd_ampa *= self.decay_ampa
        self.gd_gaba *= self.decay_gaba
        self.gd_nmda *= self.decay_nmda

        # Dendrite 2.
        H_b = 1.0 / (1.0 + self.a_nmda * np.exp(np.clip(self.b_nmda * self.ub, -50.0, 50.0)))
        total_g_b = 1.0 + self.gb_ampa + self.gb_nmda * H_b + self.gb_gaba + self.g_c
        inv_g_b = 1.0 / total_g_b
        u_inf_b = (self.u_rest + self.gb_gaba * self.E_gaba + self.g_c * self.u) * inv_g_b
        decay_b = np.exp(-self.dt * total_g_b / self.tau_m)
        self.ub = float(np.clip(self.ub * decay_b + u_inf_b * (1.0 - decay_b), -90.0, 30.0))

        self.gb_ampa *= self.decay_ampa
        self.gb_gaba *= self.decay_gaba
        self.gb_nmda *= self.decay_nmda

        # Local codependent traces on plastic dendrite.
        self.E_trace = self.E_trace * self.decay_E - self.gd_nmda * H_d * self.ud * self.one_minus_decay_E
        self.I_trace = self.I_trace * self.decay_I + self.gd_gaba * (self.ud - self.E_gaba) * self.one_minus_decay_I
        self.E_trace = float(np.clip(self.E_trace, -1e4, 1e4))
        self.I_trace = float(np.clip(self.I_trace, -1e4, 1e4))

        # Decay heterosynaptic post trace. It is incremented after post update.
        self.y_post_het *= np.exp(-self.dt / self.tau_post_het_e)

        E_pos = max(self.E_trace, 0.0)
        I_pos = max(self.I_trace, 0.0)
        self.balance_signal = self.balance_amp * E_pos * (E_pos - self.balance_alpha * I_pos)
        self.balance_signal = float(np.clip(self.balance_signal, -1e2, 1e2))
        return spk_post

    def inhibitory_gate(self) -> float:
        I_pos = max(self.I_trace, 0.0)
        return float(np.exp(-((I_pos / self.I_star) ** self.I_gamma)))

    def plasticity_e(self, spk_post: bool):
        e_idx = np.arange(0, self.ne_input)
        pre_spiked = e_idx[self.spk[e_idx]]
        gate = self.inhibitory_gate()
        E_pos = max(self.E_trace, 0.0)

        # LTD on pre-spike when post trace is present.
        if pre_spiked.size > 0:
            self.ypost[pre_spiked] *= np.exp((self.last_ypost_update[pre_spiked] - self.t_ms) / self.tau_post_e)
            self.last_ypost_update[pre_spiked] = self.t_ms
            self.w[pre_spiked] *= 1.0 - self.A_ltd_e * self.ypost[pre_spiked] * gate

        # LTP and heterosynaptic weakening on post spike.
        if spk_post:
            self.ypre[e_idx] *= np.exp((self.last_ypre_update[e_idx] - self.t_ms) / self.tau_pre_e)
            self.last_ypre_update[e_idx] = self.t_ms
            dw = (self.A_ltp_e * self.ypre[e_idx] * E_pos - self.A_het_e * self.y_post_het * (E_pos ** 2)) * gate
            self.w[e_idx] += dw

            self.ypost[e_idx] *= np.exp((self.last_ypost_update[e_idx] - self.t_ms) / self.tau_post_e)
            self.ypost[e_idx] += 1.0
            self.last_ypost_update[e_idx] = self.t_ms
            self.y_post_het += 1.0

        # Update pre traces after applying LTD.
        if pre_spiked.size > 0:
            self.ypre[pre_spiked] *= np.exp((self.last_ypre_update[pre_spiked] - self.t_ms) / self.tau_pre_e)
            self.ypre[pre_spiked] += 1.0
            self.last_ypre_update[pre_spiked] = self.t_ms

        self.w[e_idx] = np_clip(self.w[e_idx], self.w_e_min, self.w_e_max)

    def plasticity_i(self, spk_post: bool):
        i_idx = np.arange(self.ne_input, self.n_total)
        pre_spiked = i_idx[self.spk[i_idx]]

        if pre_spiked.size > 0:
            self.ypost[pre_spiked] *= np.exp((self.last_ypost_update[pre_spiked] - self.t_ms) / self.tau_i)
            self.last_ypost_update[pre_spiked] = self.t_ms
            self.w[pre_spiked] += self.A_i * self.balance_signal * self.ypost[pre_spiked]

        if spk_post:
            self.ypre[i_idx] *= np.exp((self.last_ypre_update[i_idx] - self.t_ms) / self.tau_i)
            self.last_ypre_update[i_idx] = self.t_ms
            self.w[i_idx] += self.A_i * self.balance_signal * self.ypre[i_idx]

            self.ypost[i_idx] *= np.exp((self.last_ypost_update[i_idx] - self.t_ms) / self.tau_i)
            self.ypost[i_idx] += 1.0
            self.last_ypost_update[i_idx] = self.t_ms

        if pre_spiked.size > 0:
            self.ypre[pre_spiked] *= np.exp((self.last_ypre_update[pre_spiked] - self.t_ms) / self.tau_i)
            self.ypre[pre_spiked] += 1.0
            self.last_ypre_update[pre_spiked] = self.t_ms

        self.w[i_idx] = np_clip(self.w[i_idx], self.w_i_min, self.w_i_max)

    def run(self, duration_s: float, plasticity: bool = False, record: bool = False, record_interval_ms: float = 100.0):
        n_steps = int(duration_s * 1000.0 / self.dt)
        rec: Dict[str, List[float]] = {
            "time_s": [],
            "u_soma": [],
            "u_dend1": [],
            "u_dend2": [],
            "E_trace": [],
            "I_trace": [],
            "EI_ratio": [],
            "balance_signal": [],
            "inhibitory_gate": [],
            "w_corr": [],
            "w_uncorr": [],
            "w_i_mean": [],
        }
        record_interval = max(1, int(record_interval_ms / self.dt))

        for step in range(n_steps):
            self.t_ms += self.dt
            self.generate_input(step)
            spk_post = self.lif_step()
            if plasticity:
                self.plasticity_e(spk_post)
                self.plasticity_i(spk_post)
            if record and step % record_interval == 0:
                w_corr = float(np.mean(self.w[: self.ne_corr]))
                w_uncorr = float(np.mean(self.w[self.ne_corr : self.ne_input]))
                w_i = float(np.mean(self.w[self.ne_input :]))
                E_pos = max(self.E_trace, 0.0)
                I_pos = max(self.I_trace, 0.0)
                rec["time_s"].append(self.t_ms / 1000.0)
                rec["u_soma"].append(self.u)
                rec["u_dend1"].append(self.ud)
                rec["u_dend2"].append(self.ub)
                rec["E_trace"].append(E_pos)
                rec["I_trace"].append(I_pos)
                rec["EI_ratio"].append(E_pos / (I_pos + 1e-9))
                rec["balance_signal"].append(self.balance_signal)
                rec["inhibitory_gate"].append(self.inhibitory_gate())
                rec["w_corr"].append(w_corr)
                rec["w_uncorr"].append(w_uncorr)
                rec["w_i_mean"].append(w_i)
        return pd.DataFrame(rec) if record else None

    def clustering_index(self) -> Tuple[float, float, float]:
        w_corr = float(np.mean(self.w[: self.ne_corr]))
        w_uncorr = float(np.mean(self.w[self.ne_corr : self.ne_input]))
        return float((w_corr - w_uncorr) / (w_corr + w_uncorr + 1e-12)), w_corr, w_uncorr


def run_one_group(ne_corr: int, cfg: ModeConfig, seed: int, record: bool = False, record_interval_ms: float = 100.0):
    sim = DendriteSimulation(ne_corr=ne_corr, dt_ms=cfg.dt_ms, seed=seed)
    sim.run(cfg.warmup_s, plasticity=False, record=False)
    trace = sim.run(cfg.plastic_min * 60.0, plasticity=True, record=record, record_interval_ms=record_interval_ms)
    ci, w_corr, w_uncorr = sim.clustering_index()
    row = {
        "ne_corr": ne_corr,
        "coactive_fraction": ne_corr / 32.0,
        "clustering_index": ci,
        "w_corr": w_corr,
        "w_uncorr": w_uncorr,
        "w_i_mean": float(np.mean(sim.w[sim.ne_input :])),
        "E_trace_final": max(float(sim.E_trace), 0.0),
        "I_trace_final": max(float(sim.I_trace), 0.0),
    }
    return row, trace


def aggregate_rows(rows: List[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    if "repeat" not in raw.columns:
        return raw
    grouped = []
    for ne_corr, sub in raw.groupby("ne_corr", sort=True):
        item = {
            "ne_corr": ne_corr,
            "coactive_fraction": float(sub["coactive_fraction"].iloc[0]),
            "n_repeats": int(len(sub)),
        }
        for col in ["clustering_index", "w_corr", "w_uncorr", "w_i_mean", "E_trace_final", "I_trace_final"]:
            item[f"{col}_mean"] = float(sub[col].mean())
            item[f"{col}_sem"] = sem(sub[col])
        grouped.append(item)
    return pd.DataFrame(grouped)


def plot_results(df: pd.DataFrame, outdir: Path, mode: str):
    outdir.mkdir(parents=True, exist_ok=True)
    # Handle both raw single-run CSV and aggregated CSV.
    if "clustering_index_mean" in df.columns:
        x = df["coactive_fraction"]
        ci = df["clustering_index_mean"]
        ci_sem = df.get("clustering_index_sem", pd.Series(np.zeros(len(df))))
        wc = df["w_corr_mean"]
        wu = df["w_uncorr_mean"]
        wc_sem = df.get("w_corr_sem", pd.Series(np.zeros(len(df))))
        wu_sem = df.get("w_uncorr_sem", pd.Series(np.zeros(len(df))))
    else:
        x = df["coactive_fraction"]
        ci = df["clustering_index"]
        ci_sem = np.zeros(len(df))
        wc = df["w_corr"]
        wu = df["w_uncorr"]
        wc_sem = np.zeros(len(df))
        wu_sem = np.zeros(len(df))

    plt.figure(figsize=(7, 4))
    plt.errorbar(x, ci, yerr=ci_sem, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Co-active group fraction")
    plt.ylabel("Clustering index")
    plt.title(f"Dendritic clustering ({mode})")
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_clustering_{mode}_final.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.errorbar(x, wc, yerr=wc_sem, marker="o", label="correlated")
    plt.errorbar(x, wu, yerr=wu_sem, marker="o", label="uncorrelated")
    plt.xlabel("Co-active group fraction")
    plt.ylabel("mean excitatory weight")
    plt.title(f"Correlated vs uncorrelated weights ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_weights_{mode}_final.png", dpi=200)
    plt.close()


def plot_trace(trace: Optional[pd.DataFrame], outdir: Path, mode: str):
    if trace is None or len(trace) == 0:
        return
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["u_soma"], label="soma")
    plt.plot(trace["time_s"], trace["u_dend1"], label="dendrite 1")
    plt.plot(trace["time_s"], trace["u_dend2"], label="dendrite 2")
    plt.xlabel("time (s)")
    plt.ylabel("membrane potential")
    plt.title("Example voltage traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_voltage_trace_{mode}_final.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["E_trace"], label="E trace")
    plt.plot(trace["time_s"], trace["I_trace"], label="I trace")
    plt.xlabel("time (s)")
    plt.ylabel("trace")
    plt.title("Example codependent traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_EI_trace_{mode}_final.png", dpi=200)
    plt.close()

    if "EI_ratio" in trace.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(trace["time_s"], trace["EI_ratio"], label="E/I")
        plt.axhline(1.75, linestyle="--", label=r"$\alpha=1.75$")
        plt.xlabel("time (s)")
        plt.ylabel("trace ratio")
        plt.title("Trace-based E/I ratio")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"dendrites_EI_ratio_{mode}_final.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["w_corr"], label="correlated E")
    plt.plot(trace["time_s"], trace["w_uncorr"], label="uncorrelated E")
    plt.xlabel("time (s)")
    plt.ylabel("mean excitatory weight")
    plt.title("Example excitatory weight evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_E_weight_trace_{mode}_final.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["w_i_mean"], label="mean I")
    plt.xlabel("time (s)")
    plt.ylabel("mean inhibitory weight")
    plt.title("Example inhibitory weight evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_I_weight_trace_{mode}_final.png", dpi=200)
    plt.close()


def run_scan(args):
    cfg = get_mode_config(args.mode)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    example_trace = None

    for repeat in range(args.n_repeats):
        for i, ne_corr in enumerate(cfg.group_sizes):
            seed = args.seed + 10000 * repeat + i
            record = (ne_corr == args.record_ne_corr and repeat == 0)
            print(
                f"running ne_corr={ne_corr}, repeat={repeat+1}/{args.n_repeats}, "
                f"warmup={cfg.warmup_s}s, plastic={cfg.plastic_min}min, dt={cfg.dt_ms}ms"
            )
            row, trace = run_one_group(
                ne_corr=ne_corr,
                cfg=cfg,
                seed=seed,
                record=record,
                record_interval_ms=args.record_interval_ms,
            )
            row["repeat"] = repeat
            row["seed"] = seed
            rows.append(row)
            pd.DataFrame(rows).to_csv(outdir / f"dendrites_clustering_{args.mode}_raw_partial.csv", index=False)
            if record and trace is not None:
                example_trace = trace
                trace.to_csv(outdir / f"dendrites_example_trace_{args.mode}_final.csv", index=False)
            print(
                "  clustering_index:", round(row["clustering_index"], 4),
                "w_corr:", round(row["w_corr"], 4),
                "w_uncorr:", round(row["w_uncorr"], 4),
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / f"dendrites_clustering_{args.mode}_raw_final.csv", index=False)
    summary = aggregate_rows(rows)
    summary.to_csv(outdir / f"dendrites_clustering_{args.mode}_final.csv", index=False)
    plot_results(summary, outdir, args.mode)
    plot_trace(example_trace, outdir, args.mode)
    print("saved:", outdir.resolve())


def plot_only(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    plot_results(df, outdir, args.mode)
    if args.trace_csv:
        trace = pd.read_csv(args.trace_csv)
        plot_trace(trace, outdir, args.mode)
    print("saved plots to:", outdir.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="smoke", choices=["smoke", "quick", "medium", "long"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--outdir", default="results_dendrites_final")
    parser.add_argument("--record-ne-corr", type=int, default=16)
    parser.add_argument("--record-interval-ms", type=float, default=100.0)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--csv", default=None, help="CSV for plot-only mode")
    parser.add_argument("--trace-csv", default=None, help="Optional trace CSV for plot-only mode")
    args = parser.parse_args()

    if args.plot_only:
        if not args.csv:
            raise ValueError("--csv is required with --plot-only")
        plot_only(args)
    else:
        run_scan(args)


if __name__ == "__main__":
    main()
