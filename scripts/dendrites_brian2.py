from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

prefs.codegen.target = "numpy"

# ============================================================
# Dendritic clustering simulation
# Fortran folder: dendrites
#
# Main output:
#   clustering index = (w_corr - w_uncorr) / (w_corr + w_uncorr)
#
# This is a compressed Brian2/Python translation of the Fortran
# two-layer neuron dendrite simulation.
# ============================================================


def get_mode_config(mode):
    if mode == "smoke":
        return {
            "group_sizes": [1, 4, 8, 16, 24, 31],
            "warmup_s": 1.0,
            "plastic_min": 0.2,
            "dt_ms": 0.1,
        }

    if mode == "quick":
        return {
            "group_sizes": list(range(1, 32)),
            "warmup_s": 2.0,
            "plastic_min": 0.5,
            "dt_ms": 0.1,
        }

    if mode == "medium":
        return {
            "group_sizes": list(range(1, 32)),
            "warmup_s": 5.0,
            "plastic_min": 2.0,
            "dt_ms": 0.1,
        }

    if mode == "long":
        return {
            "group_sizes": list(range(1, 32)),
            "warmup_s": 5.0,
            "plastic_min": 10.0,
            "dt_ms": 0.1,
        }

    raise ValueError("mode must be smoke, quick, medium, or long")


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


class DendriteSimulation:
    def __init__(self, ne_corr, dt_ms=0.1, seed=1):
        self.rng = np.random.default_rng(seed)

        # -----------------------------
        # Network size
        # -----------------------------
        self.ne_input = 32
        self.ni_input = self.ne_input // 2
        self.ne_corr = int(ne_corr)
        self.ni_corr = 8
        self.n_total = self.ne_input + self.ni_input
        self.n_pw = self.n_total * 2

        # -----------------------------
        # Parameters from config.f90
        # -----------------------------
        self.dt = float(dt_ms)

        self.tau_m = 30.0
        self.u_rest = -65.0
        self.refrac = 5.0
        self.u_th = -50.0
        self.E_ahp = -80.0
        self.g_ahp_amp = 0.5
        self.u_reset = -60.0

        self.decay_ahp = np.exp(-self.dt / 100.0)

        self.E_gaba = -80.0
        self.decay_ampa = np.exp(-self.dt / 5.0)
        self.decay_gaba = np.exp(-self.dt / 10.0)
        self.decay_nmda = np.exp(-self.dt / 150.0)

        self.a_nmda = 0.15
        self.b_nmda = -0.08

        # Input probabilities per time step
        self.p_e_base = self.dt * 2.0 / 1000.0
        self.p_i_base = self.dt * 4.0 / 1000.0
        self.p_i_factor = 2.0

        # OU-like input envelope
        self.tau_ou = 5.0
        self.decay_ou = np.exp(-self.dt / self.tau_ou)
        self.one_minus_decay_ou = 1.0 - self.decay_ou
        self.ou_amp = 250.0

        # Weights
        self.w_e0 = 0.3
        self.w_i0 = 0.5

        # Plasticity limits
        self.w_e_max = 1.0
        self.w_i_max = 10.0
        self.w_e_min = 1e-5
        self.w_i_min = 1e-4

        # Excitatory plasticity
        self.tau_pre_e = 16.8
        self.tau_post_e = 33.7
        self.A_ltp_e = 0.00003
        self.A_ltd_e = 50.0 * self.A_ltp_e
        self.A_het_e = 0.0002 * self.A_ltp_e
        self.inhib_control_tau = 50.0
        self.A_pre_alone = 0.1 * self.A_ltp_e

        # Inhibitory plasticity
        self.tau_i = 20.0
        self.A_i = 0.001
        self.balance_alpha = 1.75
        self.balance_amp = 0.0001

        # E/I traces
        self.decay_E = np.exp(-self.dt / 10.0)
        self.one_minus_decay_E = 1.0 - self.decay_E
        self.decay_I = np.exp(-self.dt / 100.0)
        self.one_minus_decay_I = 1.0 - self.decay_I

        # Coupling
        self.g_c = 8.0

        self.reset_state()

    def reset_state(self):
        # Soma variables
        self.u = self.u_rest
        self.g_ahp = 0.0

        # Dendrite 1 variables
        self.ud = self.u_rest
        self.gd_ampa = 0.0
        self.gd_gaba = 0.0
        self.gd_nmda = 0.0

        # Dendrite 2 variables
        self.ub = self.u_rest
        self.gb_ampa = 0.0
        self.gb_gaba = 0.0
        self.gb_nmda = 0.0

        # Plasticity traces
        self.E_trace = 0.0
        self.I_trace = 100.0
        self.balance_signal = 0.0

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

    def update_ou_input(self, step):
        if step % 10 != 0:
            return

        noise = self.rng.normal(0.0, 1.0, size=self.n_pw)
        self.patt_time = (
            self.patt_time * self.decay_ou
            + self.ou_amp * noise * self.one_minus_decay_ou
        )
        rate_hz = np.maximum(self.patt_time, 0.0)
        self.inp_patt = self.dt * rate_hz / 1000.0

    def generate_input(self, step):
        self.spk[:] = False
        self.update_ou_input(step)

        rand = self.rng.random(self.n_total)

        # -----------------------------
        # Dendrite 1 plastic inputs
        # -----------------------------
        # Correlated excitatory inputs
        if self.ne_corr > 0:
            idx = np.arange(0, self.ne_corr)
            prob = self.p_e_base + self.inp_patt[0]
            ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > self.refrac)
            self.spk[idx[ok]] = True
            self.last_pre[idx[ok]] = self.t_ms

        # Uncorrelated excitatory inputs
        if self.ne_corr < self.ne_input:
            idx = np.arange(self.ne_corr, self.ne_input)
            prob = self.p_e_base + self.inp_patt[idx]
            ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > self.refrac)
            self.spk[idx[ok]] = True
            self.last_pre[idx[ok]] = self.t_ms

        # Correlated inhibitory inputs
        idx = np.arange(self.ne_input, self.ne_input + self.ni_corr)
        prob = self.p_i_base + self.p_i_factor * self.inp_patt[0]
        ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > 0.5 * self.refrac)
        self.spk[idx[ok]] = True
        self.last_pre[idx[ok]] = self.t_ms

        # Uncorrelated inhibitory inputs
        idx = np.arange(self.ne_input + self.ni_corr, self.n_total)
        prob = self.p_i_base + self.p_i_factor * self.inp_patt[idx]
        ok = (rand[idx] <= prob) & ((self.t_ms - self.last_pre[idx]) > 0.5 * self.refrac)
        self.spk[idx[ok]] = True
        self.last_pre[idx[ok]] = self.t_ms

        # Conductances onto dendrite 1
        e_spikes = self.spk[: self.ne_input]
        i_spikes = self.spk[self.ne_input :]

        e_input = np.sum(e_spikes * self.w[: self.ne_input])
        i_input = np.sum(i_spikes * self.w[self.ne_input :])

        # Fortran dendrites code uses negative sign here
        self.gd_ampa += e_input
        self.gd_nmda += e_input
        self.gd_gaba += i_input

        # -----------------------------
        # Dendrite 2 background inputs
        # -----------------------------
        rand2 = self.rng.random(self.n_total)

        e_prob = self.p_e_base + self.inp_patt[self.n_total : self.n_total + self.ne_input]
        e_bg = rand2[: self.ne_input] <= e_prob
        self.gb_ampa += np.sum(e_bg) * self.w_e0
        self.gb_nmda += np.sum(e_bg) * self.w_e0

        i_prob = self.p_i_base + self.p_i_factor * self.inp_patt[
            self.n_total + self.ne_input : self.n_pw
        ]
        i_bg = rand2[self.ne_input :] <= i_prob
        self.gb_gaba += np.sum(i_bg) * self.w_i0 * 2.2

    def lif_step(self):
        spk_post = False

        # -----------------------------
        # Soma
        # -----------------------------
        if (self.t_ms - self.last_post) > self.refrac:
            inv_g = 1.0 / (1.0 + self.g_ahp * self.g_ahp_amp + 2.0 * self.g_c)
            u_inf = (
                self.u_rest
                + self.g_ahp * self.g_ahp_amp * self.E_ahp
                + self.g_c * (self.ud + self.ub)
            ) * inv_g
            decay = np.exp((-self.dt / self.tau_m) / inv_g)
            self.u = self.u * decay + u_inf * (1.0 - decay)
            self.u = float(np.clip(self.u, -90.0, 30.0))

            if self.u >= self.u_th:
                self.last_post = self.t_ms
                spk_post = True
                self.u = self.u_reset
                self.g_ahp += 1.0

        self.g_ahp *= self.decay_ahp

        # -----------------------------
        # Dendrite 1
        # -----------------------------
        H_d = 1.0 / (1.0 + self.a_nmda * np.exp(np.clip(self.b_nmda * self.ud, -50.0, 50.0)))
        inv_g_d = 1.0 / (1.0 + self.gd_ampa + self.gd_nmda * H_d + self.gd_gaba + self.g_c)
        u_inf_d = (self.u_rest + self.gd_gaba * self.E_gaba + self.g_c * self.u) * inv_g_d
        decay_d = np.exp((-self.dt / self.tau_m) / inv_g_d)
        self.ud = self.ud * decay_d + u_inf_d * (1.0 - decay_d)
        self.ud = float(np.clip(self.ud, -90.0, 30.0))

        self.gd_ampa *= self.decay_ampa
        self.gd_gaba *= self.decay_gaba
        self.gd_nmda *= self.decay_nmda

        # -----------------------------
        # Dendrite 2
        # -----------------------------
        H_b = 1.0 / (1.0 + self.a_nmda * np.exp(np.clip(self.b_nmda * self.ub, -50.0, 50.0)))
        inv_g_b = 1.0 / (1.0 + self.gb_ampa + self.gb_nmda * H_b + self.gb_gaba + self.g_c)
        u_inf_b = (self.u_rest + self.gb_gaba * self.E_gaba + self.g_c * self.u) * inv_g_b
        decay_b = np.exp((-self.dt / self.tau_m) / inv_g_b)
        self.ub = self.ub * decay_b + u_inf_b * (1.0 - decay_b)
        self.ub = float(np.clip(self.ub, -90.0, 30.0))

        self.gb_ampa *= self.decay_ampa
        self.gb_gaba *= self.decay_gaba
        self.gb_nmda *= self.decay_nmda

        # -----------------------------
        # Codependent traces on dendrite 1
        # -----------------------------
        self.E_trace = (
            self.E_trace * self.decay_E
            - self.gd_nmda * H_d * self.ud * self.one_minus_decay_E
        )
        self.I_trace = (
            self.I_trace * self.decay_I
            + self.gd_gaba * (self.ud - self.E_gaba) * self.one_minus_decay_I
        )
        self.E_trace = float(np.clip(self.E_trace, -1e4, 1e4))
        self.I_trace = float(np.clip(self.I_trace, -1e4, 1e4))

        self.balance_signal = self.balance_amp * (
            self.E_trace * (self.E_trace - self.I_trace * self.balance_alpha)
        )
        self.balance_signal = float(np.clip(self.balance_signal, -1e2, 1e2))
        return spk_post

    def plasticity_e(self, spk_post):
        e_idx = np.arange(0, self.ne_input)
        pre_spiked = e_idx[self.spk[e_idx]]

        if pre_spiked.size > 0:
            self.ypost[pre_spiked] *= np.exp(
                (self.last_ypost_update[pre_spiked] - self.t_ms) / self.tau_post_e
            )
            self.last_ypost_update[pre_spiked] = self.t_ms

            inhib_gate = np.exp(np.clip(-self.I_trace / self.inhib_control_tau, -50.0, 0.0))
            self.w[pre_spiked] *= 1.0 - self.A_ltd_e * self.ypost[pre_spiked] * inhib_gate

        if spk_post:
            self.ypre[e_idx] *= np.exp(
                (self.last_ypre_update[e_idx] - self.t_ms) / self.tau_pre_e
            )
            self.last_ypre_update[e_idx] = self.t_ms

            inhib_gate = np.exp(np.clip(-self.I_trace / self.inhib_control_tau, -50.0, 0.0))
            dw = (
                self.A_ltp_e * self.ypre[e_idx] * self.E_trace
                - self.A_het_e * self.g_ahp * (self.E_trace ** 2)
            ) * inhib_gate
            self.w[e_idx] += dw

            self.ypost[e_idx] *= np.exp(
                (self.last_ypost_update[e_idx] - self.t_ms) / self.tau_post_e
            )
            self.ypost[e_idx] += 1.0
            self.last_ypost_update[e_idx] = self.t_ms

        if pre_spiked.size > 0:
            self.ypre[pre_spiked] *= np.exp(
                (self.last_ypre_update[pre_spiked] - self.t_ms) / self.tau_pre_e
            )
            self.ypre[pre_spiked] += 1.0
            self.last_ypre_update[pre_spiked] = self.t_ms

        self.w[e_idx] = clip(self.w[e_idx], self.w_e_min, self.w_e_max)

    def plasticity_i(self, spk_post):
        i_idx = np.arange(self.ne_input, self.n_total)
        pre_spiked = i_idx[self.spk[i_idx]]

        if pre_spiked.size > 0:
            self.ypost[pre_spiked] *= np.exp(
                (self.last_ypost_update[pre_spiked] - self.t_ms) / self.tau_i
            )
            self.last_ypost_update[pre_spiked] = self.t_ms
            self.w[pre_spiked] += self.A_i * self.balance_signal * self.ypost[pre_spiked]

        if spk_post:
            self.ypre[i_idx] *= np.exp(
                (self.last_ypre_update[i_idx] - self.t_ms) / self.tau_i
            )
            self.last_ypre_update[i_idx] = self.t_ms
            self.w[i_idx] += self.A_i * self.balance_signal * self.ypre[i_idx]

            self.ypost[i_idx] *= np.exp(
                (self.last_ypost_update[i_idx] - self.t_ms) / self.tau_i
            )
            self.ypost[i_idx] += 1.0
            self.last_ypost_update[i_idx] = self.t_ms

        if pre_spiked.size > 0:
            self.ypre[pre_spiked] *= np.exp(
                (self.last_ypre_update[pre_spiked] - self.t_ms) / self.tau_i
            )
            self.ypre[pre_spiked] += 1.0
            self.last_ypre_update[pre_spiked] = self.t_ms

        self.w[i_idx] = clip(self.w[i_idx], self.w_i_min, self.w_i_max)

    def run(self, duration_s, plasticity=False, record=False):
        n_steps = int(duration_s * 1000.0 / self.dt)

        rec = {
            "time_s": [],
            "u_soma": [],
            "u_dend1": [],
            "u_dend2": [],
            "E_trace": [],
            "I_trace": [],
            "w_corr": [],
            "w_uncorr": [],
            "w_i_mean": [],
        }

        record_interval = max(1, int(100.0 / self.dt))

        for step in range(n_steps):
            self.t_ms += self.dt
            self.generate_input(step)
            spk_post = self.lif_step()

            if plasticity:
                self.plasticity_e(spk_post)
                self.plasticity_i(spk_post)

            if record and step % record_interval == 0:
                w_corr = np.mean(self.w[: self.ne_corr])
                w_uncorr = np.mean(self.w[self.ne_corr : self.ne_input])
                w_i = np.mean(self.w[self.ne_input :])

                rec["time_s"].append(self.t_ms / 1000.0)
                rec["u_soma"].append(self.u)
                rec["u_dend1"].append(self.ud)
                rec["u_dend2"].append(self.ub)
                rec["E_trace"].append(self.E_trace)
                rec["I_trace"].append(self.I_trace)
                rec["w_corr"].append(w_corr)
                rec["w_uncorr"].append(w_uncorr)
                rec["w_i_mean"].append(w_i)

        if record:
            return pd.DataFrame(rec)

        return None

    def clustering_index(self):
        w_corr = np.mean(self.w[: self.ne_corr])
        w_uncorr = np.mean(self.w[self.ne_corr : self.ne_input])
        return (w_corr - w_uncorr) / (w_corr + w_uncorr), w_corr, w_uncorr


def run_one_group(ne_corr, warmup_s, plastic_min, dt_ms, seed, record=False):
    sim = DendriteSimulation(ne_corr=ne_corr, dt_ms=dt_ms, seed=seed)

    sim.run(warmup_s, plasticity=False, record=False)

    trace = sim.run(plastic_min * 60.0, plasticity=True, record=record)

    ci, w_corr, w_uncorr = sim.clustering_index()

    row = {
        "ne_corr": ne_corr,
        "coactive_fraction": ne_corr / 32.0,
        "clustering_index": ci,
        "w_corr": w_corr,
        "w_uncorr": w_uncorr,
    }

    return row, trace


def plot_results(df, outdir, mode):
    plt.figure(figsize=(7, 4))
    plt.plot(df["coactive_fraction"], df["clustering_index"], marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Co-active group fraction")
    plt.ylabel("Clustering index")
    plt.title(f"Dendritic clustering ({mode})")
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_clustering_{mode}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(df["coactive_fraction"], df["w_corr"], marker="o", label="correlated")
    plt.plot(df["coactive_fraction"], df["w_uncorr"], marker="o", label="uncorrelated")
    plt.xlabel("Co-active group fraction")
    plt.ylabel("mean excitatory weight")
    plt.title(f"Correlated vs uncorrelated weights ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_weights_{mode}.png", dpi=200)
    plt.close()


def plot_trace(trace, outdir, mode):
    if trace is None or len(trace) == 0:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["u_soma"], label="soma")
    plt.plot(trace["time_s"], trace["u_dend1"], label="dendrite 1")
    plt.plot(trace["time_s"], trace["u_dend2"], label="dendrite 2")
    plt.xlabel("time (s)")
    plt.ylabel("membrane potential")
    plt.title("Example voltage traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_voltage_trace_{mode}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["E_trace"], label="E trace")
    plt.plot(trace["time_s"], trace["I_trace"], label="I trace")
    plt.xlabel("time (s)")
    plt.ylabel("trace")
    plt.title("Example codependent traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_EI_trace_{mode}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["w_corr"], label="correlated E")
    plt.plot(trace["time_s"], trace["w_uncorr"], label="uncorrelated E")
    plt.plot(trace["time_s"], trace["w_i_mean"], label="mean I")
    plt.xlabel("time (s)")
    plt.ylabel("weight")
    plt.title("Example weight evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"dendrites_weight_trace_{mode}.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="smoke", choices=["smoke", "quick", "medium", "long"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = get_mode_config(args.mode)

    outdir = Path("results_dendrites")
    outdir.mkdir(exist_ok=True)

    rows = []
    example_trace = None

    for i, ne_corr in enumerate(cfg["group_sizes"]):
        print(
            f"running ne_corr={ne_corr}, "
            f"warmup={cfg['warmup_s']} s, "
            f"plastic={cfg['plastic_min']} min, "
            f"dt={cfg['dt_ms']} ms"
        )

        record = ne_corr == 16
        row, trace = run_one_group(
            ne_corr=ne_corr,
            warmup_s=cfg["warmup_s"],
            plastic_min=cfg["plastic_min"],
            dt_ms=cfg["dt_ms"],
            seed=args.seed + i,
            record=record,
        )

        rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / f"dendrites_clustering_{args.mode}_partial.csv", index=False)

        if record:
            example_trace = trace
            trace.to_csv(outdir / f"dendrites_example_trace_{args.mode}.csv", index=False)

        print(
            "  clustering_index:",
            round(row["clustering_index"], 4),
            "w_corr:",
            round(row["w_corr"], 4),
            "w_uncorr:",
            round(row["w_uncorr"], 4),
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"dendrites_clustering_{args.mode}.csv", index=False)

    plot_results(df, outdir, args.mode)
    plot_trace(example_trace, outdir, args.mode)

    print("====================================")
    print("dendrites result")
    print("mode:", args.mode)
    print(df)
    print("saved:", outdir.resolve())
    print("====================================")


if __name__ == "__main__":
    main()