"""
fig3_simulation.py
==================

Presentation-oriented reduced simulation for Fig. 3 of:
Agnes & Vogels, Nature Neuroscience (2024),
"Co-dependent excitatory and inhibitory plasticity accounts for quick,
stable and long-lasting memories in biological networks".

IMPORTANT
---------
This is NOT the authors' original Fig. 3 Fortran program and should not be
called an exact reproduction. The public repository currently provides code
for Figs. 2, 4, 5, and 6, but does not expose a Fig. 3 folder. This script is a
transparent, rate-based reduction of Methods equations (16), (21), and (24),
written to demonstrate the mechanisms needed for a presentation:

1. Gaussian distance-dependent neighbouring interaction.
2. Approximately sigma-independent mean excitatory current setpoint.
3. Sigma-dependent competition and current dispersion.
4. Lower setpoint for larger heterosynaptic/LTP learning-rate ratio.
5. Loss of dependence on the initial excitatory weight after convergence.
6. A weak dependence on static inhibitory strength through a reduced
   postsynaptic-gain proxy.

Reduced model
-------------
For synapse i,

    q_i = r_i * w_i
    E_i = sum_j K_ij(sigma) q_j

and the averaged weight dynamics are

    dw_i/dt = eta * [
        A_ltp * G_I(w_I) * r_i * E_i
        - A_het * E_i**2
        - A_ltd * r_i * w_i
    ]

where K is a normalized Gaussian kernel and G_I is a mild inhibitory gain
factor used only to reproduce the qualitative trend of Fig. 3g. Spike traces,
membrane voltage, NMDA channel kinetics, and postsynaptic spikes have been
averaged out. Therefore use the phrase "reduced mechanism simulation" in the
presentation.

Dependencies
------------
    numpy, pandas, matplotlib

Examples
--------
Quick test:
    python fig3_simulation.py --mode quick --seed 1

Presentation run:
    python fig3_simulation.py --mode medium --seed 1

Results are written to results_fig3_reduced/ by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class ModeConfig:
    n_synapses: int
    max_steps: int
    dt: float
    replicates: int
    sigma_points: int
    check_every: int
    convergence_tol: float
    convergence_checks: int


def get_mode_config(mode: str) -> ModeConfig:
    if mode == "quick":
        return ModeConfig(
            n_synapses=128,
            max_steps=4500,
            dt=0.05,
            replicates=2,
            sigma_points=13,
            check_every=100,
            convergence_tol=2e-6,
            convergence_checks=4,
        )
    if mode == "medium":
        return ModeConfig(
            n_synapses=256,
            max_steps=6500,
            dt=0.04,
            replicates=3,
            sigma_points=17,
            check_every=125,
            convergence_tol=1e-6,
            convergence_checks=5,
        )
    if mode == "full-lite":
        return ModeConfig(
            n_synapses=800,
            max_steps=8000,
            dt=0.03,
            replicates=3,
            sigma_points=21,
            check_every=150,
            convergence_tol=8e-7,
            convergence_checks=5,
        )
    raise ValueError("mode must be quick, medium, or full-lite")


@dataclass
class ModelParameters:
    a_ltp: float = 1.0
    a_het: float = 2.0
    a_ltd: float = 0.03
    eta: float = 1.0
    tau_e: float = 1.0
    initial_weight: float = 0.30
    inhibitory_weight: float = 1.0
    inhibitory_gain_slope: float = 0.08
    w_min: float = 0.0
    w_max: float = 0.70
    rate_floor: float = 0.02
    equal_rate: float = 0.50


def circular_gaussian_kernel(n: int, sigma: float) -> np.ndarray:
    """Return a periodic 1D Gaussian kernel whose sum is one.

    The paper uses a long 1D line. A ring is used here to remove edge effects;
    for n much larger than sigma, the centre behaviour is effectively the same.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    index = np.arange(n)
    distance = np.minimum(index, n - index).astype(float)
    kernel = np.exp(-0.5 * (distance / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def convolve_periodic(values: np.ndarray, kernel_fft: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(values) * kernel_fft, n=values.size)


def make_rates(n: int, pattern: str, rng: np.random.Generator, params: ModelParameters) -> np.ndarray:
    if pattern == "equal":
        return np.full(n, params.equal_rate, dtype=float)
    if pattern == "uniform":
        return rng.uniform(params.rate_floor, 1.0, size=n)
    raise ValueError("pattern must be equal or uniform")


def simulate_equilibrium(
    *,
    cfg: ModeConfig,
    sigma: float,
    rates: np.ndarray,
    params: ModelParameters,
    initial_weight: float | np.ndarray | None = None,
) -> dict:
    """Integrate the reduced weight/current dynamics to equilibrium."""
    n = rates.size
    if n != cfg.n_synapses:
        raise ValueError("rates length must match cfg.n_synapses")

    if initial_weight is None:
        initial_weight = params.initial_weight

    if np.isscalar(initial_weight):
        weight = np.full(n, float(initial_weight), dtype=float)
    else:
        weight = np.asarray(initial_weight, dtype=float).copy()
        if weight.shape != (n,):
            raise ValueError("initial_weight array has the wrong shape")

    e_trace = np.zeros(n, dtype=float)
    kernel = circular_gaussian_kernel(n, sigma)
    kernel_fft = np.fft.rfft(kernel)

    # This proxy produces the weak inhibitory dependence described for Fig. 3g.
    post_gain = 1.0 / (1.0 + params.inhibitory_gain_slope * params.inhibitory_weight)

    stable_count = 0
    previous_checkpoint = weight.copy()
    steps_used = cfg.max_steps

    # Explicit Euler is sufficient because this is a smooth reduced system.
    e_fraction = min(cfg.dt / max(params.tau_e, cfg.dt), 1.0)

    for step in range(cfg.max_steps):
        synapse_current = rates * weight
        neighbouring_current = convolve_periodic(synapse_current, kernel_fft)
        e_trace += e_fraction * (neighbouring_current - e_trace)

        ltp = params.a_ltp * post_gain * rates * e_trace
        heterosynaptic = params.a_het * e_trace**2
        ltd = params.a_ltd * rates * weight
        drive = params.eta * (ltp - heterosynaptic - ltd)

        weight = np.clip(weight + cfg.dt * drive, params.w_min, params.w_max)

        if (step + 1) % cfg.check_every == 0:
            max_change = float(np.max(np.abs(weight - previous_checkpoint)))
            previous_checkpoint = weight.copy()
            if max_change < cfg.convergence_tol:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= cfg.convergence_checks:
                steps_used = step + 1
                break

    synapse_current = rates * weight
    neighbouring_current = convolve_periodic(synapse_current, kernel_fft)

    return {
        "weights": weight,
        "rates": rates,
        "synapse_current": synapse_current,
        "e_trace": e_trace,
        "neighbouring_current": neighbouring_current,
        "steps": steps_used,
        "mean_current": float(np.mean(synapse_current)),
        "std_current": float(np.std(synapse_current)),
        "total_current": float(np.sum(synapse_current)),
        "surviving_fraction": float(np.mean(weight > 1e-5)),
    }


def mean_sem(values: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size))


def sigma_values(cfg: ModeConfig) -> np.ndarray:
    # Includes the transition region near sigma about 0.6 and the fitted scale 4.4.
    return np.geomspace(0.15, 32.0, cfg.sigma_points)


def run_distance_sweep(cfg: ModeConfig, base: ModelParameters, seed_value: int) -> pd.DataFrame:
    rows: list[dict] = []
    taus = [0.25, 1.0, 4.0]

    for tau_e in taus:
        for pattern in ["equal", "uniform"]:
            for sigma in sigma_values(cfg):
                mean_values = []
                std_values = []
                survival_values = []
                step_values = []

                for rep in range(cfg.replicates):
                    rep_seed = seed_value + 100000 * rep + int(round(1000 * sigma)) + int(100 * tau_e)
                    rng = np.random.default_rng(rep_seed)
                    rates = make_rates(cfg.n_synapses, pattern, rng, base)
                    p = ModelParameters(**{**asdict(base), "tau_e": tau_e})
                    result = simulate_equilibrium(cfg=cfg, sigma=float(sigma), rates=rates, params=p)
                    mean_values.append(result["mean_current"])
                    std_values.append(result["std_current"])
                    survival_values.append(result["surviving_fraction"])
                    step_values.append(result["steps"])

                mean_current, mean_sem_value = mean_sem(mean_values)
                std_current, std_sem_value = mean_sem(std_values)
                survival, survival_sem = mean_sem(survival_values)

                rows.append(
                    {
                        "sigma": float(sigma),
                        "tau_e_reduced": tau_e,
                        "input_pattern": pattern,
                        "mean_current": mean_current,
                        "mean_current_sem": mean_sem_value,
                        "std_current": std_current,
                        "std_current_sem": std_sem_value,
                        "surviving_fraction": survival,
                        "surviving_fraction_sem": survival_sem,
                        "mean_steps": float(np.mean(step_values)),
                    }
                )

    return pd.DataFrame(rows)


def run_parameter_sweep(
    *,
    cfg: ModeConfig,
    base: ModelParameters,
    seed_value: int,
    parameter_name: str,
    parameter_values: Iterable[float],
    sigma: float = 4.4,
) -> pd.DataFrame:
    rows: list[dict] = []

    for value_index, value in enumerate(parameter_values):
        totals = []
        means = []
        stds = []
        survivals = []
        steps = []

        for rep in range(cfg.replicates):
            rng = np.random.default_rng(seed_value + rep)
            rates = make_rates(cfg.n_synapses, "uniform", rng, base)
            p_dict = asdict(base)

            initial_weight = base.initial_weight
            if parameter_name == "het_ltp_ratio":
                p_dict["a_het"] = float(value) * base.a_ltp
            elif parameter_name == "initial_weight":
                initial_weight = float(value)
            elif parameter_name == "inhibitory_weight":
                p_dict["inhibitory_weight"] = float(value)
            else:
                raise ValueError("unknown parameter sweep")

            p = ModelParameters(**p_dict)
            result = simulate_equilibrium(
                cfg=cfg,
                sigma=sigma,
                rates=rates,
                params=p,
                initial_weight=initial_weight,
            )
            totals.append(result["total_current"])
            means.append(result["mean_current"])
            stds.append(result["std_current"])
            survivals.append(result["surviving_fraction"])
            steps.append(result["steps"])

        total_mean, total_sem = mean_sem(totals)
        current_mean, current_sem = mean_sem(means)
        std_mean, std_sem = mean_sem(stds)
        survival_mean, survival_sem = mean_sem(survivals)

        rows.append(
            {
                parameter_name: float(value),
                "total_current": total_mean,
                "total_current_sem": total_sem,
                "mean_current": current_mean,
                "mean_current_sem": current_sem,
                "std_current": std_mean,
                "std_current_sem": std_sem,
                "surviving_fraction": survival_mean,
                "surviving_fraction_sem": survival_sem,
                "mean_steps": float(np.mean(steps)),
            }
        )

    return pd.DataFrame(rows)


def plot_kernel_examples(cfg: ModeConfig, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    centre = cfg.n_synapses // 2
    offsets = np.arange(-20, 21)
    for sigma in [0.3, 1.0, 3.0, 4.4]:
        kernel = circular_gaussian_kernel(cfg.n_synapses, sigma)
        shifted = np.roll(kernel, centre)
        ax.plot(offsets, shifted[centre - 20 : centre + 21], marker="o", markersize=3, label=fr"$\sigma={sigma}$")
    ax.set_xlabel(r"synaptic distance $\Delta x$")
    ax.set_ylabel("normalized neighbouring influence")
    ax.set_title("Fig. 3b mechanism: Gaussian distance kernel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig3_b_kernel_examples.png", dpi=220)
    plt.close(fig)


def plot_distance_results(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)

    styles = {
        "equal": {"linestyle": "--", "marker": "o", "label": "equal input rates"},
        "uniform": {"linestyle": "-", "marker": "s", "label": "uniform input rates"},
    }

    for tau_e in sorted(df["tau_e_reduced"].unique()):
        for pattern in ["equal", "uniform"]:
            sub = df[(df["tau_e_reduced"] == tau_e) & (df["input_pattern"] == pattern)].sort_values("sigma")
            style = styles[pattern]
            label = f"{style['label']}, tau={tau_e:g}"
            axes[0].plot(sub["sigma"], sub["mean_current"], linestyle=style["linestyle"], marker=style["marker"], markersize=3.5, label=label)
            axes[1].plot(sub["sigma"], sub["std_current"], linestyle=style["linestyle"], marker=style["marker"], markersize=3.5, label=label)

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(0.6, linestyle=":", linewidth=1.4, label=r"paper $\sigma_{th}\approx0.6$")
        ax.axvline(4.4, linestyle="-.", linewidth=1.2, label=r"paper $\sigma_{fit}\approx4.4$")
        ax.set_xlabel(r"interaction length $\sigma$")

    axes[0].set_ylabel("mean synapse-specific current (reduced units)")
    axes[0].set_title("Fig. 3c analogue: mean current setpoint")
    axes[1].set_ylabel("SD of synapse-specific current")
    axes[1].set_title("Fig. 3d analogue: competition increases dispersion")

    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(unique.values(), unique.keys(), fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "fig3_cd_distance_stability.png", dpi=220)
    plt.close(fig)


def plot_parameter_result(
    df: pd.DataFrame,
    *,
    x: str,
    xlabel: str,
    title: str,
    filename: str,
    outdir: Path,
    xscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(
        df[x],
        df["total_current"],
        yerr=df["total_current_sem"],
        marker="o",
        capsize=3,
        linewidth=1.8,
    )
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("total excitatory current (reduced units)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=220)
    plt.close(fig)


def plot_weight_distributions(cfg: ModeConfig, base: ModelParameters, seed_value: int, outdir: Path) -> None:
    rng = np.random.default_rng(seed_value)
    rates = make_rates(cfg.n_synapses, "uniform", rng, base)
    sigmas = [0.25, 0.6, 4.4, 16.0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, sigma in zip(axes.flat, sigmas):
        result = simulate_equilibrium(cfg=cfg, sigma=sigma, rates=rates, params=base)
        ax.scatter(rates, result["synapse_current"], s=12, alpha=0.7)
        ax.set_title(fr"$\sigma={sigma:g}$; survive={result['surviving_fraction']:.2f}")
        ax.set_xlabel("normalized presynaptic rate")
        ax.set_ylabel("synapse-specific current")
    fig.suptitle("Mechanism check: spatial coupling creates synaptic competition", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_current_distributions.png", dpi=220)
    plt.close(fig)


def make_summary(
    distance_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    inhibitory_df: pd.DataFrame,
    cfg: ModeConfig,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # Kernel panel
    centre = cfg.n_synapses // 2
    offsets = np.arange(-15, 16)
    for sigma in [0.3, 1.0, 3.0, 4.4]:
        kernel = np.roll(circular_gaussian_kernel(cfg.n_synapses, sigma), centre)
        axes[0, 0].plot(offsets, kernel[centre - 15 : centre + 16], label=fr"$\sigma={sigma}$")
    axes[0, 0].set_title("B. Distance kernel")
    axes[0, 0].set_xlabel(r"$\Delta x$")
    axes[0, 0].set_ylabel("influence")
    axes[0, 0].legend(fontsize=8)

    # Distance mean and SD: use tau=1 for uncluttered summary.
    summary_distance = distance_df[distance_df["tau_e_reduced"] == 1.0]
    for pattern, marker in [("equal", "o"), ("uniform", "s")]:
        sub = summary_distance[summary_distance["input_pattern"] == pattern].sort_values("sigma")
        axes[0, 1].plot(sub["sigma"], sub["mean_current"], marker=marker, label=pattern)
        axes[0, 2].plot(sub["sigma"], sub["std_current"], marker=marker, label=pattern)
    for ax in [axes[0, 1], axes[0, 2]]:
        ax.set_xscale("log")
        ax.axvline(0.6, linestyle=":", linewidth=1.2)
        ax.axvline(4.4, linestyle="-.", linewidth=1.2)
        ax.set_xlabel(r"$\sigma$")
        ax.legend(fontsize=8)
    axes[0, 1].set_title("C. Mean current")
    axes[0, 1].set_ylabel("mean current")
    axes[0, 2].set_title("D. Current dispersion")
    axes[0, 2].set_ylabel("current SD")

    axes[1, 0].errorbar(ratio_df["het_ltp_ratio"], ratio_df["total_current"], yerr=ratio_df["total_current_sem"], marker="o", capsize=3)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("E. Heterosynaptic/LTP ratio")
    axes[1, 0].set_xlabel(r"$A_{het}/A_{LTP}$")
    axes[1, 0].set_ylabel("total current")

    axes[1, 1].errorbar(initial_df["initial_weight"], initial_df["total_current"], yerr=initial_df["total_current_sem"], marker="o", capsize=3)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("F. Initial E weight")
    axes[1, 1].set_xlabel(r"$w_E(0)$")
    axes[1, 1].set_ylabel("total current")

    axes[1, 2].errorbar(inhibitory_df["inhibitory_weight"], inhibitory_df["total_current"], yerr=inhibitory_df["total_current_sem"], marker="o", capsize=3)
    axes[1, 2].set_title("G. Static inhibitory weight")
    axes[1, 2].set_xlabel(r"$w_I$")
    axes[1, 2].set_ylabel("total current")

    fig.suptitle("Fig. 3 reduced mechanism simulation (not exact paper reproduction)", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_reduced_summary.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduced mechanism simulation for paper Fig. 3")
    parser.add_argument("--mode", choices=["quick", "medium", "full-lite"], default="quick")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="results_fig3_reduced")
    args = parser.parse_args()

    cfg = get_mode_config(args.mode)
    base = ModelParameters()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running Fig. 3 reduced mechanism simulation")
    print("mode:", args.mode)
    print("n_synapses:", cfg.n_synapses)
    print("seed:", args.seed)

    plot_kernel_examples(cfg, outdir)

    print("1/5 distance and tau_E sweep")
    distance_df = run_distance_sweep(cfg, base, args.seed)
    distance_df.to_csv(outdir / "fig3_cd_distance_sweep.csv", index=False)
    plot_distance_results(distance_df, outdir)

    print("2/5 heterosynaptic/LTP ratio sweep")
    ratio_values = np.geomspace(0.45, 7.0, 12)
    ratio_df = run_parameter_sweep(
        cfg=cfg,
        base=base,
        seed_value=args.seed + 20000,
        parameter_name="het_ltp_ratio",
        parameter_values=ratio_values,
    )
    ratio_df.to_csv(outdir / "fig3_e_hetero_ltp_ratio.csv", index=False)
    plot_parameter_result(
        ratio_df,
        x="het_ltp_ratio",
        xlabel=r"heterosynaptic/LTP ratio $A_{het}/A_{LTP}$",
        title="Fig. 3e analogue: stronger heterosynaptic plasticity lowers the setpoint",
        filename="fig3_e_hetero_ltp_ratio.png",
        outdir=outdir,
        xscale="log",
    )

    print("3/5 initial excitatory weight sweep")
    initial_values = np.geomspace(0.04, 0.65, 10)
    initial_df = run_parameter_sweep(
        cfg=cfg,
        base=base,
        seed_value=args.seed + 30000,
        parameter_name="initial_weight",
        parameter_values=initial_values,
    )
    initial_df.to_csv(outdir / "fig3_f_initial_weight.csv", index=False)
    plot_parameter_result(
        initial_df,
        x="initial_weight",
        xlabel=r"initial excitatory weight $w_E(0)$",
        title="Fig. 3f analogue: the converged current loses memory of initial weight",
        filename="fig3_f_initial_weight.png",
        outdir=outdir,
        xscale="log",
    )

    print("4/5 static inhibitory weight sweep")
    inhibitory_values = np.linspace(0.0, 8.0, 11)
    inhibitory_df = run_parameter_sweep(
        cfg=cfg,
        base=base,
        seed_value=args.seed + 40000,
        parameter_name="inhibitory_weight",
        parameter_values=inhibitory_values,
    )
    inhibitory_df.to_csv(outdir / "fig3_g_inhibitory_weight.csv", index=False)
    plot_parameter_result(
        inhibitory_df,
        x="inhibitory_weight",
        xlabel=r"static inhibitory weight $w_I$",
        title="Fig. 3g analogue: inhibition weakly shifts the excitatory-current setpoint",
        filename="fig3_g_inhibitory_weight.png",
        outdir=outdir,
    )

    print("5/5 current-distribution and summary plots")
    plot_weight_distributions(cfg, base, args.seed + 50000, outdir)
    make_summary(distance_df, ratio_df, initial_df, inhibitory_df, cfg, outdir)

    metadata = {
        "script": "fig3_simulation.py",
        "status": "reduced mechanism simulation; not exact paper reproduction",
        "mode": args.mode,
        "seed": args.seed,
        "mode_config": asdict(cfg),
        "model_parameters": asdict(base),
        "notes": [
            "Uses a rate-based average of Methods equations 16, 21, and 24.",
            "Uses a periodic 1D ring to avoid edge effects.",
            "The inhibitory-weight effect is implemented as a mild postsynaptic-gain proxy.",
            "Do not label the outputs as an exact reproduction of the paper.",
        ],
    }
    with (outdir / "fig3_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("saved:", outdir.resolve())
    print("main presentation figure:", (outdir / "fig3_reduced_summary.png").resolve())


if __name__ == "__main__":
    main()
