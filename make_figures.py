"""
Regenerates the figures of manuscript/main.tex from the repository's own
example code (run from the repository root):

    python manuscript/make_figures.py

Needs the [all] extras (Qiskit, Aer, qiskit-ibm-runtime, matplotlib).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from qiskit.circuit.library import quantum_volume
from qiskit_aer.noise import amplitude_damping_error, phase_damping_error

from nisq_hardware_validation import get_backend, run_relaxation_experiment
from quantum_volume_qg_s_realistic_noise import layered_noisy_density_matrix, t1_aware_profile

OUT = os.path.dirname(os.path.abspath(__file__))


def channel_sweep():
    n = 4
    qc = quantum_volume(n, depth=n, seed=0)
    grid = np.linspace(0.0, 1.0, 21)
    ad = [t1_aware_profile(layered_noisy_density_matrix(qc, n, amplitude_damping_error(g)), n) for g in grid]
    pd = [t1_aware_profile(layered_noisy_density_matrix(qc, n, phase_damping_error(g)), n) for g in grid]
    return grid, np.array(ad), np.array(pd)


def main():
    grid, ad, pd = channel_sweep()
    rows = run_relaxation_experiment(get_backend("fake", "fake_brisbane"), shots=4000, seed=42)
    ideal, measured = rows[0], rows[1:]
    delays = [r["delay_us"] for r in measured]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    a.plot(grid, ad[:, 0], "-", color="#1f5fa8", lw=1.8, label=r"$qg_S$, amplitude damping")
    a.plot(grid, ad[:, 1], "--", color="#1f5fa8", lw=1.8, label=r"mean $qg_Z$, amplitude damping")
    a.plot(grid, pd[:, 0], "-", color="#c2571a", lw=1.4, label=r"$qg_S$, dephasing")
    a.plot(grid, pd[:, 1], "--", color="#c2571a", lw=1.4, label=r"mean $qg_Z$, dephasing")
    a.set_xlabel(r"channel strength $\gamma$ or $\lambda$ (after every 2-qubit block)")
    a.set_ylabel("value")
    a.set_title("(a) idealized channels, exact", loc="left", fontsize=9)
    a.set_ylim(-0.1, 1.38)
    a.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)

    b.plot(delays, [r["qg_s"] for r in measured], "o-", color="#1f5fa8", lw=1.8, label=r"$qg_S$ (Miller–Madow)")
    b.plot(delays, [r["mean_qg_z"] for r in measured], "s--", color="#1f5fa8", lw=1.8, label=r"mean $qg_Z$")
    b.plot(delays, [r["hop"] for r in measured], "^-", color="#6b6b6b", lw=1.2, label="heavy output prob.")
    b.axhline(ideal["qg_s"], color="#1f5fa8", lw=0.8, ls=":", label=r"noiseless $qg_S$")
    b.set_xlabel(r"idle delay before readout ($\mu$s)")
    b.set_title("(b) fake_brisbane calibration noise, 4000 shots", loc="left", fontsize=9)
    b.set_ylim(-0.1, 1.38)
    b.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    for ax in (a, b):
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.savefig(os.path.join(OUT, "fig_t1.pdf"))
    print("wrote", os.path.join(OUT, "fig_t1.pdf"))


if __name__ == "__main__":
    main()
