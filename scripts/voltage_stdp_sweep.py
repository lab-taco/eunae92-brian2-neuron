import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from voltage_stdp_single import run_one_condition


outdir = Path("results_sweep")
outdir.mkdir(exist_ok=True)

# Fortran config.f90 대응
p_sim_3 = 0.04
p_sim_4 = 0.005 - p_sim_3

n_weight = 5
n_clamp = 25

rows = []

for k0 in range(1, n_weight + 1):
    w0 = p_sim_4 + p_sim_3 * k0

    inp_clamp = 0.0

    for k1 in range(1, n_clamp + 1):
        if k1 == 1:
            inp_clamp = 0.0
        else:
            inp_clamp += 10.0 ** (0.12 * float(k1 - 2) - 2.0)

        k = 0
        keep_going = True

        while keep_going:
            k += 1

            v_inf = 10.0 ** (0.04 * float(k - 1) - 1.0)
            ca_amp = v_inf / (50.0 - (-65.0) - v_inf)
            ca_amp = ca_amp / (50.0 * 0.05)

            if k == 1:
                ca_amp = 0.0

            print("running:", "k0", k0, "k1", k1, "k", k,
                  "w0", w0, "inp_clamp", inp_clamp, "ca_amp", ca_amp)

            result = run_one_condition(
                w0=w0,
                inp_clamp=inp_clamp,
                ca_amp=ca_amp,
                make_plots=False,
            )

            rows.append(result)

            if result["depol"] > 7.0:
                keep_going = False

df = pd.DataFrame(rows)
df.to_csv(outdir / "voltage_stdp_sweep_raw.csv", index=False)

plt.figure(figsize=(6, 4))
plt.scatter(df["depol"], df["final_percent"], s=10)
plt.xscale("log")
plt.xlabel("Depolarization from rest")
plt.ylabel("final weight / initial weight (%)")
plt.tight_layout()
plt.savefig(outdir / "fig2b_scatter.png", dpi=200)
plt.close()

print(df.head())
print("saved:", outdir.resolve())