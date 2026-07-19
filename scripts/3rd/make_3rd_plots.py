import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out = Path(__file__).resolve().parent / "assets"
out.mkdir(exist_ok=True)

# 1. Spike traces: exponential decay
τ_pre = 16.8
τ_post = 33.7
t = np.linspace(0, 120, 600)
xpre = np.exp(-t / τ_pre)
ypost = np.exp(-t / τ_post)
plt.figure(figsize=(6.4, 3.8))
plt.plot(t, xpre, label=r"$x_{pre}(t)=e^{-t/\tau_+}$")
plt.plot(t, ypost, label=r"$y_{post}(t)=e^{-t/\tau_-}$")
plt.xlabel("time after spike (ms)")
plt.ylabel("trace amplitude")
plt.title("Spike traces are short-term memory variables")
plt.legend()
plt.tight_layout()
plt.savefig(out / "trace_decay.png", dpi=220)
plt.close()

# 2. NMDA voltage gate
u = np.linspace(-90, -20, 600)
a_nmda = 0.15
b_nmda = -0.08
H = 1.0 / (1.0 + a_nmda * np.exp(b_nmda * u))
plt.figure(figsize=(6.4, 3.8))
plt.plot(u, H)
plt.xlabel("membrane potential u (mV-like units)")
plt.ylabel(r"$H_{NMDA}(u)$")
plt.title("Depolarization increases NMDA contribution")
plt.tight_layout()
plt.savefig(out / "nmda_gate.png", dpi=220)
plt.close()

# 3. LTP vs heterosynaptic weakening setpoint
E = np.linspace(0, 2.0, 600)
A_ltp = 1.0
A_het = 0.8
ltp = A_ltp * E
het = -A_het * E**2
total = ltp + het
E_star = A_ltp / A_het
plt.figure(figsize=(6.4, 3.8))
plt.axhline(0, linewidth=0.8)
plt.plot(E, ltp, label=r"LTP: $A_{LTP}E$")
plt.plot(E, het, label=r"Het.: $-A_{het}E^2$")
plt.plot(E, total, linewidth=2.2, label=r"Total")
plt.axvline(E_star, linestyle="--", linewidth=1.2, label=r"setpoint $E^*$")
plt.xlabel("effective excitatory trace E")
plt.ylabel("weight-change drive")
plt.title("Linear LTP + quadratic weakening creates a setpoint")
plt.legend()
plt.tight_layout()
plt.savefig(out / "ltp_het_setpoint.png", dpi=220)
plt.close()

# 4. inhibitory gate
I = np.linspace(0, 4, 600)
I_star = 1.0
gamma = 3.0
G = np.exp(-(I / I_star)**gamma)
plt.figure(figsize=(6.4, 3.8))
plt.plot(I, G)
plt.xlabel(r"inhibitory trace $I/I^*$")
plt.ylabel(r"$G_I(I)$")
plt.title("Inhibition gates excitatory plasticity")
plt.tight_layout()
plt.savefig(out / "inhibitory_gate.png", dpi=220)
plt.close()

# 5. ISP balance heatmap
alpha = 1.0
E_grid = np.linspace(0, 2.0, 251)
I_grid = np.linspace(0, 2.0, 251)
EE, II = np.meshgrid(E_grid, I_grid)
dwI = EE * (EE - alpha * II)
plt.figure(figsize=(6.2, 4.8))
lim = np.max(np.abs(dwI))
plt.imshow(dwI, origin="lower", extent=[0, 2, 0, 2], aspect="auto", vmin=-lim, vmax=lim, cmap="coolwarm")
plt.colorbar(label=r"drive $E(E-\alpha I)$")
plt.plot(I_grid * alpha, I_grid, "k--", linewidth=1.4, label=r"$E=\alpha I$")
plt.xlabel("E")
plt.ylabel("I")
plt.title("Co-dependent ISP pushes the system to E/I = alpha")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(out / "isp_balance_heatmap.png", dpi=220)
plt.close()

# 6. distance kernel
sigma = np.sqrt(10.0)
d = np.linspace(0, 18, 600)
alpha_d = np.exp(-0.5 * (d**2) / (sigma**2))
plt.figure(figsize=(6.4, 3.8))
plt.plot(d, alpha_d)
plt.xlabel(r"synaptic distance $d$ ($\mu$m)")
plt.ylabel(r"neighbor coupling $\alpha(d)$")
plt.title("Neighboring influence decays with distance")
plt.tight_layout()
plt.savefig(out / "distance_kernel.png", dpi=220)
plt.close()

# 7. recurrent metrics concept
n = np.arange(1, 11)
w_in = np.linspace(1.0, 0.2, 10)
w_out = np.linspace(0.15, 0.95, 10)
w_i = np.linspace(0.9, 0.25, 10)
plt.figure(figsize=(6.4, 3.8))
plt.plot(n, w_in, marker="o", label=r"mean $E\to E$ input")
plt.plot(n, w_out, marker="o", label=r"mean $E\to E$ output")
plt.plot(n, w_i, marker="o", label=r"mean $I\to E$ input")
plt.xlabel("neurons ordered by learned E->E input")
plt.ylabel("normalized mean weight")
plt.title("Fig. 7 metrics to compute in recurrent network")
plt.legend()
plt.tight_layout()
plt.savefig(out / "fig7_metric_concept.png", dpi=220)
plt.close()

print(f"Saved plots to {out}")
