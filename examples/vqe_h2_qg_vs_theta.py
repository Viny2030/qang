"""
VQE for the H2 molecule: theta-space vs qg-space optimization.

This extends Future Research Direction #3 (qang.gradients' toy benchmark
in the paper) to a real molecule, using the standard minimal (2-qubit,
parity-mapped, 2-qubit-reduced) H2 qubit Hamiltonian at the equilibrium
bond length (0.735 Angstrom, STO-3G basis). The electronic Hamiltonian
coefficients and nuclear repulsion energy below were independently
derived via PySCF + qiskit-nature (ParityMapper with two_qubit_reduction),
not taken from memory or a single literature source.

Headline finding #1: starting from the physically natural Hartree-Fock
point (theta ~= 0), theta-space optimization reaches the exact FCI
ground-state energy in a handful of steps, while every qg-space variant
(raw, clipped, Tikhonov -- qang.gradients) is trapped exactly at the
Hartree-Fock energy and recovers zero correlation energy. This is not
slow convergence -- it's a hard structural limit: arccos()'s range
[0, pi] cannot reach the true minimum, which lies in the other half of
the circle. This is a second, distinct limitation from the Jacobian
singularity already documented in Section 4.1 of the paper, now
demonstrated on a real chemistry problem instead of a toy case.

Headline finding #2: a fifth space, ``theta_pole_damped`` (qang.gradients),
sidesteps finding #1 entirely -- it never leaves theta-space, so there is
no arccos round trip to get trapped by -- and adds a measurable benefit of
its own. At a safe, well-tuned learning rate (lr <= 2.0 here) it matches
plain theta-space optimization. At a badly-tuned, too-aggressive learning
rate (lr = 3.0, 5.0), plain theta-space gradient descent overshoots and
diverges -- it never reaches chemical accuracy -- while theta_pole_damped,
started from the exact same Hartree-Fock point, still reaches the exact
FCI energy, in a handful of steps. See qang.gradients' module docstring
for the honest framing (this is Levenberg-Marquardt-style trust-region
damping, not a new optimizer class) and
examples/pole_damped_gradient_descent_robustness.py for the same effect
established statistically over 300 random one-qubit landscapes rather
than this one real molecule.
"""

import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from qang.gradients import (
    inverse_jacobian_clipped,
    inverse_jacobian_raw,
    inverse_jacobian_tikhonov,
    pole_damping_factor,
)

# Minimal (2-qubit) H2 electronic Hamiltonian, STO-3G, R=0.735 Angstrom,
# ParityMapper with two_qubit_reduction=True (PySCF + qiskit-nature).
H2_ELECTRONIC = SparsePauliOp.from_list([
    ("II", -1.05237325),
    ("IZ", 0.39793742),
    ("ZI", -0.39793742),
    ("ZZ", -0.01128010),
    ("XX", 0.18093120),
])
NUCLEAR_REPULSION = 0.7199689944489797
EXACT_FCI_ENERGY = -1.1373060357534004
CHEMICAL_ACCURACY = 1.6e-3


def h2_ansatz(theta: float) -> QuantumCircuit:
    """Minimal single-parameter UCC-like ansatz: X on qubit 0 prepares the
    Hartree-Fock reference, Ry(theta) on qubit 1 plus a CX injects the
    single active excitation."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.ry(theta, 1)
    qc.cx(1, 0)
    return qc


def h2_energy(theta: float) -> float:
    sv = Statevector.from_instruction(h2_ansatz(theta))
    return sv.expectation_value(H2_ELECTRONIC).real + NUCLEAR_REPULSION


def h2_energy_grad(theta: float) -> float:
    """Exact parameter-shift-rule gradient (single Pauli generator)."""
    return 0.5 * (h2_energy(theta + math.pi / 2) - h2_energy(theta - math.pi / 2))


def run_vqe(space: str, theta0: float, lr: float = 0.3, steps: int = 300, eps: float = 0.05):
    """Gradient descent in one of five spaces:
      - "theta":             directly on theta
      - "qg_raw":            qg_Z = cos(theta), raw inverse-Jacobian step
      - "qg_clipped":        qg_Z = cos(theta), clipped inverse-Jacobian step
      - "qg_tikhonov":       qg_Z = cos(theta), Tikhonov-regularized inverse-Jacobian step
      - "theta_pole_damped": directly on theta, with a pole-proximity
                              damping factor applied to the step (no qg
                              round trip at all -- see this module's
                              docstring, headline finding #2)
    Returns the list of energies visited (including the starting point).
    """
    theta = theta0
    energies = [h2_energy(theta)]

    for _ in range(steps):
        grad_theta = h2_energy_grad(theta)

        if space == "theta":
            theta = theta - lr * grad_theta
        elif space == "theta_pole_damped":
            theta = theta - lr * pole_damping_factor(theta, eps=eps) * grad_theta
        else:
            qg = math.cos(theta)
            # dE/dqg = dE/dtheta * dtheta/dqg ; dtheta/dqg = -1/sin(theta) (raw)
            if space == "qg_raw":
                dtheta_dqg = inverse_jacobian_raw(theta)
            elif space == "qg_clipped":
                dtheta_dqg = inverse_jacobian_clipped(theta, eps=eps)
            elif space == "qg_tikhonov":
                dtheta_dqg = inverse_jacobian_tikhonov(theta, eps=eps)
            else:
                raise ValueError(f"unknown space: {space}")

            grad_qg = grad_theta * dtheta_dqg
            qg_new = qg - lr * grad_qg
            qg_new = max(-1.0, min(1.0, qg_new))
            theta = math.acos(qg_new)

        energies.append(h2_energy(theta))

    return energies


def steps_to_chemical_accuracy(energies):
    """First index at which |E - E_exact| < chemical accuracy, or None."""
    for i, e in enumerate(energies):
        if abs(e - EXACT_FCI_ENERGY) < CHEMICAL_ACCURACY:
            return i
    return None


if __name__ == "__main__":
    print(f"Exact FCI ground-state energy: {EXACT_FCI_ENERGY:.10f} Ha")
    print(f"Hartree-Fock energy (theta=0): {h2_energy(0.0):.10f} Ha")
    print()

    print("Headline finding #1 (lr=0.3, safe): qg-space variants are trapped")
    print("at the Hartree-Fock energy; theta and theta_pole_damped both reach FCI.")
    for space in ["theta", "qg_raw", "qg_clipped", "qg_tikhonov", "theta_pole_damped"]:
        energies = run_vqe(space, theta0=1e-6, lr=0.3, steps=300)
        final = energies[-1]
        steps = steps_to_chemical_accuracy(energies)
        steps_str = str(steps) if steps is not None else "never"
        print(f"{space:18s}  final E = {final:.10f} Ha  "
              f"error = {abs(final - EXACT_FCI_ENERGY):.2e}  "
              f"steps to chem. accuracy = {steps_str}")
    print()

    print("Headline finding #2: badly-tuned (too-aggressive) learning rate --")
    print("plain theta-space diverges, theta_pole_damped still reaches FCI energy.")
    for lr in [0.3, 1.0, 2.0, 3.0, 5.0]:
        e_theta = run_vqe("theta", theta0=1e-6, lr=lr, steps=300)
        e_damped = run_vqe("theta_pole_damped", theta0=1e-6, lr=lr, steps=300)
        s_theta = steps_to_chemical_accuracy(e_theta)
        s_damped = steps_to_chemical_accuracy(e_damped)
        print(
            f"  lr={lr:4.1f}  theta: final E={e_theta[-1]:+.6f} Ha "
            f"(steps={s_theta if s_theta is not None else 'never'})   "
            f"theta_pole_damped: final E={e_damped[-1]:+.6f} Ha "
            f"(steps={s_damped if s_damped is not None else 'never'})"
        )
