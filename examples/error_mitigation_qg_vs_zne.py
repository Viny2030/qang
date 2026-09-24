"""
Decoherence and error mitigation: which noise does the qg filter fix,
which does zero-noise extrapolation (ZNE) fix, and can the qg witness
tell us in advance which one to use?

Same problem as examples/chemistry_qg_symmetry_witness.py: the exact H2
VQE state (4 qubits, 2 electrons, Jordan-Wigner) whose ideal register
mean qg_Z is known, 1 - 2N/n = 0. Energy error vs FCI in mHa.

Mitigation methods (all on top of standard readout-error mitigation):

  raw        readout-mitigated energy, nothing else
  qg filter  keep Z-basis shots whose register qg_Z = 1 - 2N/n
  ZNE        fold every CX into CX^(2k+1) (noise scale 1, 3, 5) and
             Richardson-extrapolate to zero noise
  ZNE + qg   filter at every scale, then extrapolate

Witnesses read from the same Z-basis shots, before any mitigation:
mean qg_Z (ideal 0) and the kept fraction (ideal 1).

Findings (20,000 shots per circuit; mean error ± std over 10 simulator
seeds, mHa; Hartree-Fock is 20.3 mHa, chemical accuracy 1.6 mHa). The
controlled models put the noise on every CX (so folding scales it
exactly) or on readout only.

    noise              qg_Z   kept |  raw   | qg filter | ZNE        | ZNE + qg
    T1 p=0.03         +0.029  0.94 |  36.6  |  2.0±1.6  | -4.0±3.1   | -2.0±2.9
    T1 p=0.10         +0.099  0.81 | 128.9  |  7.4±1.5  | -5.7±5.3   | -0.6±3.5
    dephasing p=0.03  +0.000  1.00 |   2.9  |  2.9±1.7  | -0.8±3.2   | -0.8±3.2
    dephasing p=0.10  +0.000  1.00 |  10.2  | 10.2±1.7  |  2.1±3.3   |  2.1±3.3
    depolariz. p=0.03 +0.007  0.96 |  60.8  | 23.2±2.3  |  0.3±4.5   | -2.3±3.8
    depolariz. p=0.10 +0.021  0.86 | 193.5  | 81.3±3.8  | 17.5±8.2   | -3.3±8.8
    readout p=0.03    -0.000  0.89 |  -0.6  | -0.6±1.2  | -1.9±2.9   | -1.9±2.1
    brisbane, 0 us    +0.011  0.90 |  20.6  |  6.5±1.6  |  6.8±5.9   |  2.0±3.9
    brisbane, 50 us   +0.122  0.72 | 189.3  | 31.1±1.3  | 176.6±4.1  | 27.2±4.1

  * The witness says in advance whether the filter will help. Under
    dephasing mean qg_Z stays at its ideal 0 and every shot is kept, and
    the filter changes nothing; dephasing preserves the electron number,
    so only ZNE helps (10.2 -> 2.1). Under T1, mean qg_Z rises by about
    p and the filter removes 94-95% of the error for free (no extra
    circuits, lower spread than ZNE).
  * The two methods are complementary. ZNE over-shoots on T1 (-4 to -6
    mHa: amplitude damping is not linear in the fold count) and doubles
    or triples the spread; the filter cannot touch number-preserving
    errors. Under strong depolarizing noise ZNE + qg is the only method
    within a few mHa (193 -> 3.3, vs 17.5 for ZNE alone).
  * Idle decoherence (a 50 us delay before measurement on fake_brisbane)
    is invisible to ZNE, because gate folding does not scale it (189 ->
    177), while mean qg_Z flags it (+0.122) and the filter removes 84% of
    the error (-> 31). No method reaches chemical accuracy there.
  * On the realistic fake_brisbane model without delay, ZNE + qg gives
    2.0 ± 3.9 mHa, 10x better than raw and better than Hartree-Fock, at 3x
    the circuits of the filter alone (6.5 ± 1.6).
  * Symmetric readout errors lower the kept fraction (0.89) without
    moving mean qg_Z; standard readout mitigation already removes them.

Decision rule from the witness: if mean qg_Z moves away from 1 - 2N/n
or the kept fraction drops, apply the qg filter (free); if the residual
is still large and the kept fraction is ~1, the remaining noise
preserves the symmetry and needs ZNE. Classically, FCI is exact at this
size: this ranks mitigation strategies for a quantum computation, it is
not a quantum advantage.
"""

import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemistry_qg_symmetry_witness import (  # noqa: E402
    FCI_ENERGY,
    HF_ENERGY,
    N_ELECTRONS,
    N_QUBITS,
    XY_TERMS,
    _WEIGHT,
    _calibration_circuits,
    _energy,
    _measure_circuit,
    _probs,
    _readout_inverse,
    _TERMS,
    optimal_angle,
    qg_filter,
)

SCALES = (1, 3, 5)
RICHARDSON = np.array([15.0, -10.0, 3.0]) / 8.0  # zero-noise weights for scales 1, 3, 5


def fold_cx(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Replace every CX by CX^scale (scale odd): same unitary, `scale`
    times the CX noise."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be a positive odd integer.")
    out = qc.copy_empty_like()
    for inst in qc.data:
        reps = scale if inst.operation.name == "cx" else 1
        for _ in range(reps):
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def controlled_noise_model(kind: str, p: float):
    """Noise on CX only (so folding scales it exactly), or on readout only."""
    from qiskit_aer.noise import (NoiseModel, ReadoutError, amplitude_damping_error,
                                  depolarizing_error, phase_damping_error)

    nm = NoiseModel()
    if kind == "T1":
        nm.add_all_qubit_quantum_error(amplitude_damping_error(p).tensor(amplitude_damping_error(p)), ["cx"])
    elif kind == "dephasing":
        nm.add_all_qubit_quantum_error(phase_damping_error(p).tensor(phase_damping_error(p)), ["cx"])
    elif kind == "depolarizing":
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    elif kind == "readout":
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    else:
        raise ValueError(kind)
    return nm


def _circuits(t, delay_us):
    return [_measure_circuit(t, "ZZZZ", delay_us)] + [_measure_circuit(t, l, delay_us) for l in XY_TERMS]


def _run(circuits, shots, seed, noise_model=None, backend=None, layout=None):
    from qiskit_aer import AerSimulator

    if backend is None:
        sim = AerSimulator(noise_model=noise_model)
        tc = circuits
    else:
        sim = AerSimulator.from_backend(backend)
        tc = transpile(circuits, backend=backend, initial_layout=layout, optimization_level=0,
                       seed_transpiler=1, scheduling_method="alap")
    res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    return [_probs(res.get_counts(i), shots) for i in range(len(tc))]


def mitigated_energies(noise=("T1", 0.05), delay_us=0.0, shots=20000, seed=11, backend=None, layout=None):
    """Energies (Hartree) for every method, plus the witnesses.

    noise: (kind, p) for a controlled model, or "brisbane" for the
    calibration-based fake_brisbane model (then delay_us adds idle T1)."""
    if noise == "brisbane":
        if backend is None:
            from qiskit_ibm_runtime.fake_provider import FakeBrisbane

            backend = FakeBrisbane()
        if layout is None:
            from nisq_hardware_validation import choose_layout

            layout = choose_layout(backend, N_QUBITS)
        nm = None
    else:
        nm, backend = controlled_noise_model(*noise), None
    t = optimal_angle()
    base = _circuits(t, delay_us)
    batch = [fold_cx(c, s) for s in SCALES for c in base] + _calibration_circuits()
    P = _run(batch, shots, seed, nm, backend, layout)
    inv = _readout_inverse(P[-2], P[-1])
    k = len(base)
    e_raw, e_qg = [], []
    for i in range(len(SCALES)):
        block = [inv @ p for p in P[i * k:(i + 1) * k]]
        p_z, p_xy = block[0], dict(zip(XY_TERMS, block[1:]))
        e_raw.append(_energy(p_z, p_xy, _TERMS))
        e_qg.append(_energy(qg_filter(p_z), p_xy, _TERMS))
    p_z_unmitigated = P[0]
    return {
        "mean_qg_z": float(np.sum(p_z_unmitigated * (1 - 2 * _WEIGHT / N_QUBITS))),
        "kept_fraction": float(np.sum(p_z_unmitigated * (_WEIGHT == N_ELECTRONS))),
        "raw": e_raw[0],
        "qg_filter": e_qg[0],
        "zne": float(RICHARDSON @ np.array(e_raw)),
        "zne_qg": float(RICHARDSON @ np.array(e_qg)),
    }


METHODS = ("raw", "qg_filter", "zne", "zne_qg")


def error_table(noise, delay_us=0.0, seeds=range(11, 21), shots=20000):
    """Mean |error| and std over seeds (mHa) per method, and witnesses."""
    runs = [mitigated_energies(noise, delay_us, shots, s) for s in seeds]
    out = {m: (1e3 * (float(np.mean([r[m] for r in runs])) - FCI_ENERGY),
               1e3 * float(np.std([r[m] for r in runs]))) for m in METHODS}
    out["mean_qg_z"] = float(np.mean([r["mean_qg_z"] for r in runs]))
    out["kept_fraction"] = float(np.mean([r["kept_fraction"] for r in runs]))
    return out


CASES = [
    (("T1", 0.03), 0.0), (("T1", 0.10), 0.0),
    (("dephasing", 0.03), 0.0), (("dephasing", 0.10), 0.0),
    (("depolarizing", 0.03), 0.0), (("depolarizing", 0.10), 0.0),
    (("readout", 0.03), 0.0),
    ("brisbane", 0.0), ("brisbane", 50.0),
]


def _label(noise, delay):
    return f"fake_brisbane, delay {delay:.0f} us" if noise == "brisbane" else f"{noise[0]} p={noise[1]}"


def make_figure(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"raw": "#8c8c8c", "qg_filter": "#1f6fb2", "zne": "#e0a030", "zne_qg": "#2e8b57"}
    names = {"raw": "readout-mitigated", "qg_filter": "+ qg filter", "zne": "+ ZNE", "zne_qg": "+ ZNE + qg filter"}
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(rows))
    for j, m in enumerate(METHODS):
        vals = [max(abs(r[m][0]), 0.05) for _, r in rows]
        errs = [r[m][1] for _, r in rows]
        ax.bar(x + (j - 1.5) * 0.2, vals, 0.18, yerr=errs, color=colors[m], label=names[m], capsize=2)
    ax.axhline(1.6, color="k", ls=":", lw=1)
    ax.text(-0.45, 1.3, "chemical accuracy", ha="left", fontsize=8)
    ax.axhline(1e3 * (HF_ENERGY - FCI_ENERGY), color="#c0392b", ls="--", lw=1)
    ax.text(-0.45, 23, "Hartree-Fock (classical)", ha="left", fontsize=8, color="#c0392b")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 1500)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nqg_Z {r['mean_qg_z']:+.3f}\nkept {r['kept_fraction']:.2f}" for lab, r in rows],
                       fontsize=7.5)
    ax.set_ylabel("|E - E_FCI| (mHa)")
    ax.set_title("H2: which mitigation fixes which noise (qg witness under each group)")
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    print(f"Hartree-Fock error {1e3 * (HF_ENERGY - FCI_ENERGY):.1f} mHa; chemical accuracy 1.6 mHa")
    print(f"{'noise':30s} {'qg_Z':>6} {'kept':>5} | " + " | ".join(f"{m:>13s}" for m in METHODS))
    rows = []
    for noise, delay in CASES:
        r = error_table(noise, delay)
        short = f"brisbane\ndelay {delay:.0f} us" if noise == "brisbane" else f"{noise[0]}\np = {noise[1]}"
        rows.append((short, r))
        print(f"{_label(noise, delay):30s} {r['mean_qg_z']:+6.3f} {r['kept_fraction']:5.2f} | "
              + " | ".join(f"{r[m][0]:6.1f} ±{r[m][1]:4.1f}" for m in METHODS))
    if "--figure" in sys.argv:
        make_figure(rows, __file__.replace(".py", ".png"))
