"""
Barren plateaus, qg-space vs theta-space: connecting two previously
separate limitations of the qang framework and the wider VQE
literature.

Barren plateaus (McClean, Boixo, Smelyanskiy, Babbush, & Neven, 2018,
"Barren plateaus in quantum neural network training landscapes", Nature
Communications 9, 4812; cost-function-dependence result: Cerezo,
Sone, Volkoff, Cincio, & Coles, 2021, "Cost function dependent barren
plateaus in shallow parametrized quantum circuits", Nature
Communications 12, 1791) is the well-known phenomenon that, for
sufficiently random parametrized circuits evaluated against a *global*
cost function (one built from an observable acting on every qubit, such
as Z^{\\otimes n} here), the variance of the cost gradient with respect
to any single parameter vanishes exponentially in the number of qubits
-- even for shallow, constant-depth circuits, which is what makes it a
genuine obstacle for near-term hardware-efficient ansatze rather than
only a deep-circuit effect.

This file reproduces that effect for qang.ansatze.hardware_efficient_ansatz
in theta-space (the standard setting), then asks a question specific to
this framework: what happens to that gradient if the ansatz is instead
parameterized in qg-space (qang.ansatze.hardware_efficient_ansatz_qg),
using the same regularized inverse-Jacobian conversions from
qang.gradients that Section 4.1 of the paper already introduces for a
single-qubit toy problem?

The answer connects the two previously separate limitations directly:
qg-space gradients pick up an extra multiplicative factor
d(theta)/d(qg) = inverse_jacobian_*(theta) relative to theta-space
gradients. Away from the poles this factor is a generic O(1) number and
the exponential-in-n decay survives unchanged (checked below for the
regularized variants). But exactly AT a pole (theta = 0 or pi) the raw
inverse Jacobian diverges, so raw qg-space gradients are not just
"small everywhere" the way a barren plateau is usually described --
they are unboundedly LARGE at isolated points even in the same shallow,
highly-entangling regime that produces the plateau in theta-space. The
regularized variants (clipped, Tikhonov) cap this blow-up, at the cost
of no longer being an exact reparameterization of the theta-space
gradient near the poles -- exactly the tradeoff Section 4.1 already
describes for the single-qubit case, now shown to persist unchanged
when the ansatz is embedded in a many-qubit barren-plateau setting.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from qiskit.quantum_info import Statevector, SparsePauliOp

from qang.ansatze import hardware_efficient_ansatz
from qang.gradients import (
    inverse_jacobian_clipped,
    inverse_jacobian_raw,
    inverse_jacobian_tikhonov,
)


def global_z_observable(n_qubits: int) -> SparsePauliOp:
    """Z^{\\otimes n}, the standard 'global' cost-function observable
    that triggers barren plateaus even for shallow circuits (Cerezo et
    al. 2021), unlike a single-qubit local Z observable."""
    return SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])


def global_z_cost(n_qubits: int, reps: int, params: Sequence[float]) -> float:
    qc = hardware_efficient_ansatz(n_qubits, reps, params)
    sv = Statevector.from_instruction(qc)
    return sv.expectation_value(global_z_observable(n_qubits)).real


def parameter_shift_grad(
    n_qubits: int, reps: int, params: Sequence[float], param_index: int
) -> float:
    """Exact parameter-shift-rule gradient of global_z_cost with respect
    to params[param_index] (every rotation here is a single-Pauli-
    generator RY, so the rule is exact)."""
    params = np.asarray(params, dtype=float)
    shift = np.zeros_like(params)
    shift[param_index] = np.pi / 2.0
    return 0.5 * (
        global_z_cost(n_qubits, reps, params + shift)
        - global_z_cost(n_qubits, reps, params - shift)
    )


def sample_theta_space_gradients(
    n_qubits: int, reps: int, n_samples: int, seed: int, param_index: int = 0
):
    """Sample ``n_samples`` random parameter vectors (each entry uniform
    in [0, 2*pi)) and return (gradients, values of params[param_index])
    -- the latter is needed to convert each gradient into qg-space at
    the same point it was computed."""
    rng = np.random.default_rng(seed)
    n_params = n_qubits * (reps + 1)
    grads = np.empty(n_samples, dtype=float)
    theta_values = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        params = rng.uniform(0.0, 2.0 * np.pi, size=n_params)
        grads[i] = parameter_shift_grad(n_qubits, reps, params, param_index)
        theta_values[i] = params[param_index]
    return grads, theta_values


def convert_to_qg_space_gradient(
    grads_theta: np.ndarray, theta_values: np.ndarray, mode: str = "raw", eps: float = 0.05
) -> np.ndarray:
    """dE/d(qg) = dE/d(theta) * d(theta)/d(qg), using the same
    qang.gradients inverse-Jacobian conversions (raw / clipped /
    Tikhonov) already defined for the single-qubit case."""
    if mode == "raw":
        factors = np.array([inverse_jacobian_raw(t) for t in theta_values])
    elif mode == "clipped":
        factors = np.array([inverse_jacobian_clipped(t, eps=eps) for t in theta_values])
    elif mode == "tikhonov":
        factors = np.array([inverse_jacobian_tikhonov(t, eps=eps) for t in theta_values])
    else:
        raise ValueError("mode must be 'raw', 'clipped', or 'tikhonov'.")
    return grads_theta * factors


def gradient_variance_by_qubit_count(
    n_qubits_list: Sequence[int],
    reps: int,
    n_samples: int,
    seed: int,
    param_index: int = 0,
    space: str = "theta",
    eps: float = 0.05,
) -> Dict[int, float]:
    """Variance of the (theta- or qg-space) gradient across random
    samples, for each n_qubits in n_qubits_list."""
    result: Dict[int, float] = {}
    for n in n_qubits_list:
        grads, thetas = sample_theta_space_gradients(n, reps, n_samples, seed + n, param_index)
        values = grads if space == "theta" else convert_to_qg_space_gradient(grads, thetas, mode=space, eps=eps)
        result[n] = float(np.var(values))
    return result


def near_pole_gradient_blowup(
    n_qubits: int, reps: int, seed: int, theta_pole_value: float, eps: float = 0.05
):
    """
    Deterministic (non-random) demonstration of the pole singularity:
    fix every parameter except params[0], set params[0] to a value very
    close to a pole (0 or pi), and compare the theta-space gradient to
    its three qg-space conversions at that exact point.

    Returns a dict with keys 'theta', 'qg_raw', 'qg_clipped', 'qg_tikhonov'.
    """
    rng = np.random.default_rng(seed)
    n_params = n_qubits * (reps + 1)
    params = rng.uniform(0.0, 2.0 * np.pi, size=n_params)
    params[0] = theta_pole_value

    grad_theta = parameter_shift_grad(n_qubits, reps, params, 0)
    return {
        "theta": grad_theta,
        "qg_raw": grad_theta * inverse_jacobian_raw(theta_pole_value),
        "qg_clipped": grad_theta * inverse_jacobian_clipped(theta_pole_value, eps=eps),
        "qg_tikhonov": grad_theta * inverse_jacobian_tikhonov(theta_pole_value, eps=eps),
    }


if __name__ == "__main__":
    n_qubits_list = [4, 6, 8, 10]
    reps = 1
    n_samples = 150
    seed = 0

    print("Gradient variance vs number of qubits, global Z^n cost, theta-space:")
    theta_var = gradient_variance_by_qubit_count(n_qubits_list, reps, n_samples, seed, space="theta")
    for n, v in theta_var.items():
        print(f"  n_qubits={n:2d}  Var(dE/dtheta) = {v:.6e}")

    log_var = np.log([theta_var[n] for n in n_qubits_list])
    slope, _ = np.polyfit(n_qubits_list, log_var, 1)
    print(f"  log-linear fit slope (per qubit): {slope:.4f}  (negative => exponential decay)")
    print()

    print("Same scan, converted to qg-space (clipped, eps=0.05) -- regularization")
    print("preserves the exponential-in-n decay away from the poles:")
    qg_clipped_var = gradient_variance_by_qubit_count(
        n_qubits_list, reps, n_samples, seed, space="clipped"
    )
    for n, v in qg_clipped_var.items():
        print(f"  n_qubits={n:2d}  Var(dE/dqg_clipped) = {v:.6e}")
    print()

    print("Near-pole blow-up (deterministic, theta[0] = 1e-4, n_qubits=6):")
    blowup = near_pole_gradient_blowup(6, reps=1, seed=0, theta_pole_value=1e-4)
    for space, val in blowup.items():
        print(f"  {space:12s}: {val:+.4e}")
    print()
    print("Guideline: away from the poles, qg-space regularization just rescales")
    print("the barren-plateau variance by an O(1) factor; AT a pole, raw qg-space")
    print("gradients diverge regardless of qubit count -- the Section 4.1")
    print("coordinate singularity and the barren-plateau phenomenon are")
    print("independent effects that compound rather than cancel.")
