"""P3 follow-up: finite-shot gradient magnitude, global Z^n vs qg-local cost.

Exploratory (not pre-registered): for shallow HEA (n=6, L=2, 300 random inits,
100 shots per parameter-shift circuit, first parameter), compare median absolute
shot-estimated gradient in theta-space and after clipped/raw qg conversion.

Scope: ideal statevector probabilities sampled as multinomial shots; no hardware noise.
Measurement-level toy, not a hardware claim.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qang.ansatze import hardware_efficient_ansatz
from qang.gradients import inverse_jacobian_clipped, inverse_jacobian_raw
from qang.multiqubit import mean_qg_z


def probs(n_qubits, reps, params):
    qc = hardware_efficient_ansatz(n_qubits, reps, params)
    return np.abs(Statevector.from_instruction(qc).data) ** 2


def cost_from_probs(p, n_qubits, cost):
    if cost == "global":
        return float(p[0])
    if cost == "qg_local":
        # mean_qg_z expects statevector; compute from probs
        idx = np.arange(2**n_qubits, dtype=np.uint64)
        zsum = 0.0
        for i in range(n_qubits):
            signs = 1.0 - 2.0 * ((idx >> i) & 1).astype(float)
            zsum += float(p @ signs)
        return (1.0 - zsum / n_qubits) / 2.0
    raise ValueError(cost)


def shot_grad(n_qubits, reps, params, param_index, cost, shots, rng):
    shift = np.zeros_like(params)
    shift[param_index] = np.pi / 2.0
    p_plus = probs(n_qubits, reps, params + shift)
    p_minus = probs(n_qubits, reps, params - shift)
    c_plus = rng.multinomial(shots, p_plus) / shots
    c_minus = rng.multinomial(shots, p_minus) / shots
    return 0.5 * (cost_from_probs(c_plus, n_qubits, cost) - cost_from_probs(c_minus, n_qubits, cost))


def run(n_qubits=6, reps=2, n_samples=300, shots=100, seed=7, param_index=0, eps=0.05):
    rng = np.random.default_rng(seed)
    n_params = n_qubits * (reps + 1)
    rows = []
    for _ in range(n_samples):
        params = rng.uniform(0.0, 2.0 * np.pi, size=n_params)
        theta0 = float(params[param_index])
        rec = {"theta0": theta0}
        for cost in ("global", "qg_local"):
            g = shot_grad(n_qubits, reps, params, param_index, cost, shots, rng)
            rec[f"{cost}_theta"] = g
            rec[f"{cost}_qg_clipped"] = g * inverse_jacobian_clipped(theta0, eps=eps)
            rec[f"{cost}_qg_raw"] = g * inverse_jacobian_raw(theta0)
        rows.append(rec)
    return rows


def summarize(rows):
    def med_abs(key):
        vals = np.array([abs(r[key]) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals))

    def max_abs(key):
        vals = np.array([abs(r[key]) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.max(vals))

    out = {}
    for space in ("theta", "qg_clipped", "qg_raw"):
        g = med_abs(f"global_{space}")
        l = med_abs(f"qg_local_{space}")
        out[space] = {
            "median_abs_global": g,
            "median_abs_qg_local": l,
            "ratio_local_over_global": l / max(g, 1e-30),
            "max_abs_global": max_abs(f"global_{space}"),
            "max_abs_qg_local": max_abs(f"qg_local_{space}"),
        }
    return out


if __name__ == "__main__":
    rows = run()
    s = summarize(rows)
    for space, v in s.items():
        print(space)
        for k, val in v.items():
            print(f"  {k}: {val:.6g}")
