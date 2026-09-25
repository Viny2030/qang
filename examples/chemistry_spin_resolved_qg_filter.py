"""
Chemistry with the spin sectors filtered separately: one qg filter
(total electron number) vs two (spin-up and spin-down numbers).

Same H2 problem as examples/chemistry_qg_symmetry_witness.py (§21): 4
qubits in the Jordan-Wigner encoding with interleaved spin-orbitals,
qubits 0, 2 = spin up and 1, 3 = spin down, 2 electrons (one of each
spin). The Hamiltonian conserves N_up and N_down separately, so each
spin register has its own known qg:

    mean qg_Z over the up qubits   = 1 - 2 N_up / 2   = 0
    mean qg_Z over the down qubits = 1 - 2 N_down / 2 = 0

Filters on the Z-basis shots (all on top of readout mitigation; the
four XXYY-type terms are measured in rotated bases and cannot be
filtered this way):

  total   keep Hamming weight 2 (§21): 6 of 16 bit strings
  spin    keep one up AND one down electron: 4 of 16 bit strings.
          It also removes |0101> and |1010> (two electrons of the same
          spin), which the total filter lets through.

Witnesses from the same shots: mean qg_Z of each spin register, and the
fraction of weight-2 shots that sit in the wrong spin sector
("spin leak"). Noise: the fake_brisbane calibration model with an idle
delay (as §21), and controlled noise on every CX (as §24).

Findings (20,000 shots per circuit; energy error vs FCI in mHa,
mean ± std over 10 seeds; HF 20.3 mHa, chemical accuracy 1.6 mHa):

  noise              qg_up   qg_down  spin leak | readout | total filter | spin filters
  fake_brisbane 0us  +0.012  +0.010   0.002     |  20.2   |  6.5 ± 1.5   |  5.9 ± 1.5
  fake_brisbane 20us +0.060  +0.056   0.005     |  92.4   | 20.2 ± 1.4   | 20.0 ± 1.4
  fake_brisbane 50us +0.124  +0.121   0.008     | 189.0   | 31.0 ± 1.3   | 31.4 ± 1.5
  T1 p=0.10          +0.101  +0.098   0.001     | 128.9   |  7.4 ± 1.5   |  6.7 ± 1.4
  depolariz. p=0.03  -0.013  +0.028   0.015     |  60.8   | 23.2 ± 2.3   | 14.7 ± 2.2
  depolariz. p=0.10  -0.040  +0.082   0.052     | 193.5   | 81.3 ± 3.8   | 53.9 ± 3.6
  dephasing p=0.10    0.000   0.000   0.000     |  10.2   | 10.2 ± 1.7   | 10.2 ± 1.7

  (spin leak = fraction of weight-2 shots with both electrons of the
  same spin, before any mitigation)

  * The second filter helps when the spin-leak witness is clearly
    nonzero. Depolarizing noise on the CXs flips pairs of qubits and
    moves weight-2 states into the wrong spin sector: the spin filters
    remove a further 37% of the error at p = 0.03 (23.2 -> 14.7 mHa)
    and 34% at p = 0.10 (81.3 -> 53.9).
  * On the realistic fake_brisbane model the spin leak is below 1% and
    the gain is within the seed spread (6.5 -> 5.9 mHa at zero delay,
    none with idle delay). T1 changes N, which the total filter already
    catches; dephasing changes nothing in Z. The expectation that the
    second filter would cut the §21 residual (~5.5 mHa) does not hold:
    that residual is in the XXYY terms and in N- and spin-preserving
    errors.
  * The two spin witnesses separate under depolarizing noise (up -0.040,
    down +0.082 at p = 0.10) while their average, the §21 witness,
    reads +0.021. The asymmetry follows the circuit: the CXs fan out of
    an up qubit (qubit 2). Under T1 and on fake_brisbane both registers
    drift together.

Honest summary: a second, free filter from a second conserved quantity
(same shots, no extra circuits). It pays off only when the spin-leak
witness says so, which on this realistic noise model it does not. No
classical-vs-quantum advantage: FCI is exact at this size.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemistry_qg_symmetry_witness import (  # noqa: E402
    FCI_ENERGY,
    HF_ENERGY,
    N_QUBITS,
    XY_TERMS,
    _TERMS,
    _WEIGHT,
    _calibration_circuits,
    _energy,
    _measure_circuit,
    _readout_inverse,
    optimal_angle,
)
from error_mitigation_qg_vs_zne import _run, controlled_noise_model  # noqa: E402

UP, DOWN = (0, 2), (1, 3)
_IDX = np.arange(2**N_QUBITS)
_N_UP = sum((_IDX >> q) & 1 for q in UP)
_N_DOWN = sum((_IDX >> q) & 1 for q in DOWN)
TOTAL_MASK = _WEIGHT == 2
SPIN_MASK = (_N_UP == 1) & (_N_DOWN == 1)


def filtered(p, mask):
    kept = p * mask
    return kept / kept.sum()


def register_qg(p, qubits):
    """Mean qg_Z over the given qubits: 1 - 2 <N_sector> / len(qubits)."""
    n = sum((_IDX >> q) & 1 for q in qubits)
    return float(np.sum(p * (1.0 - 2.0 * n / len(qubits))))


def run_case(noise, delay_us=0.0, shots=20000, seed=11):
    """Energies (Hartree) with the total and the spin filter, plus witnesses."""
    if noise == "brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane

        from nisq_hardware_validation import choose_layout

        backend = FakeBrisbane()
        nm, layout = None, choose_layout(backend, N_QUBITS)
    else:
        nm, backend, layout = controlled_noise_model(*noise), None, None
    t = optimal_angle()
    circuits = ([_measure_circuit(t, "ZZZZ", delay_us)] + [_measure_circuit(t, l, delay_us) for l in XY_TERMS]
                + _calibration_circuits())
    P = _run(circuits, shots, seed, nm, backend, layout)
    inv = _readout_inverse(P[-2], P[-1])
    p_z_raw = P[0]
    p_z = inv @ P[0]
    p_xy = {l: inv @ p for l, p in zip(XY_TERMS, P[1:-2])}
    weight2 = float(np.sum(p_z_raw * TOTAL_MASK))
    return {
        "qg_up": register_qg(p_z_raw, UP),
        "qg_down": register_qg(p_z_raw, DOWN),
        "spin_leak": float(np.sum(p_z_raw * (TOTAL_MASK & ~SPIN_MASK))) / weight2,
        "kept_total": weight2,
        "kept_spin": float(np.sum(p_z_raw * SPIN_MASK)),
        "readout": _energy(p_z, p_xy, _TERMS),
        "total": _energy(filtered(p_z, TOTAL_MASK), p_xy, _TERMS),
        "spin": _energy(filtered(p_z, SPIN_MASK), p_xy, _TERMS),
    }


CASES = [
    ("brisbane", 0.0), ("brisbane", 20.0), ("brisbane", 50.0),
    (("T1", 0.10), 0.0),
    (("depolarizing", 0.03), 0.0), (("depolarizing", 0.10), 0.0),
    (("dephasing", 0.10), 0.0),
]
METHODS = ("readout", "total", "spin")


def case_table(noise, delay_us=0.0, seeds=range(11, 21), shots=20000):
    runs = [run_case(noise, delay_us, shots, s) for s in seeds]
    out = {m: (1e3 * (float(np.mean([r[m] for r in runs])) - FCI_ENERGY), 1e3 * float(np.std([r[m] for r in runs])))
           for m in METHODS}
    for k in ("qg_up", "qg_down", "spin_leak", "kept_total", "kept_spin"):
        out[k] = float(np.mean([r[k] for r in runs]))
    return out


def _label(noise, delay):
    return f"fake_brisbane {delay:.0f}us" if noise == "brisbane" else f"{noise[0]} p={noise[1]}"


def make_figure(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"readout": "#8c8c8c", "total": "#1f6fb2", "spin": "#2e8b57"}
    names = {"readout": "readout-mitigated", "total": "+ total-N qg filter", "spin": "+ spin-resolved qg filters"}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(rows))
    for j, m in enumerate(METHODS):
        ax.bar(x + (j - 1) * 0.26, [max(abs(r[m][0]), 0.05) for _, r in rows], 0.24,
               yerr=[r[m][1] for _, r in rows], color=colors[m], label=names[m], capsize=2)
    ax.axhline(1.6, color="k", ls=":", lw=1)
    ax.axhline(1e3 * (HF_ENERGY - FCI_ENERGY), color="#c0392b", ls="--", lw=1)
    ax.text(-0.45, 23, "Hartree-Fock", fontsize=8, color="#c0392b")
    ax.text(-0.45, 1.3, "chemical accuracy", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 400)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nspin leak {r['spin_leak']:.3f}" for lab, r in rows], fontsize=7.5)
    ax.set_ylabel("|E - E_FCI| (mHa)")
    ax.set_title("H2: one qg filter (N) vs two (N_up, N_down)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    print(f"{'noise':22s} {'qg_up':>7} {'qg_dn':>7} {'leak':>6} {'kept N':>6} {'kept s':>6} | "
          + " | ".join(f"{m:>12s}" for m in METHODS))
    rows = []
    for noise, delay in CASES:
        r = case_table(noise, delay)
        rows.append((_label(noise, delay).replace(" ", "\n", 1), r))
        print(f"{_label(noise, delay):22s} {r['qg_up']:+7.3f} {r['qg_down']:+7.3f} {r['spin_leak']:6.3f} "
              f"{r['kept_total']:6.2f} {r['kept_spin']:6.2f} | "
              + " | ".join(f"{r[m][0]:6.1f} ±{r[m][1]:4.1f}" for m in METHODS))
    if "--figure" in sys.argv:
        make_figure(rows, __file__.replace(".py", ".png"))
