"""
H2 VQE with a genuinely two-parameter ansatz: extending
examples/vqe_h2_qg_vs_theta.py (a single parameter) and
examples/multi_parameter_pole_damped_vqe.py (a separable toy landscape)
to a real molecule where the two parameters are *entangled* by an actual
two-qubit circuit, not merely independent terms in a sum.

The ansatz generalizes the original single-parameter one
(examples/vqe_h2_qg_vs_theta.py's ``h2_ansatz``, which hard-codes an X gate
to prepare the Hartree-Fock reference) by replacing that X gate with its
own Ry rotation:

    qc.ry(theta0, 0)   # generalizes the hard-coded X = Ry(pi) on qubit 0
    qc.ry(theta1, 1)   # the original single excitation parameter
    qc.cx(1, 0)

Since Ry(pi)|0> = X|0> exactly, theta0 = pi reproduces the original
ansatz's action bit for bit (verified in
tests/test_h2_vqe_multi_parameter_ansatz.py), and the joint global minimum
sits at theta0 = pi, theta1 ~= -0.2235 -- both parameters have their exact
minimum, and theta0's happens to sit exactly AT a pole of this framework's
own angle chart (Ry(pi) is one of the two Bloch-sphere poles). That is
what makes this ansatz worth studying beyond the toy separable case: it
lets us ask, on a real coupled molecular Hamiltonian, both directions of
the pole-proximity question already raised in qang.gradients --
*fleeing* a pole (theta1, started near theta=0, same stress test as the
single-parameter study) and *converging onto* one (theta0, whose own
target is a pole).

Two honestly-measured findings, both starting from (theta0, theta1) =
(1.0, 1e-6) -- theta1 starts near the theta=0 pole exactly as in the
single-parameter study, while theta0 starts away from either pole (a
starting point close to the theta0=0 pole was tried and rejected: at
theta0 ~= 0 this particular ansatz has a spurious stationary point at
(theta0=0, theta1=-pi/2) where BOTH partial gradients vanish together --
a bad-local-minimum trap that pole damping is not designed to fix and
that has nothing to do with pole proximity, so it is avoided here rather
than misrepresented as a pole-damping result):

  Finding A (an honest cost, sharper here than in the single-parameter
  case): at a safe, well-tuned learning rate, theta_pole_damped needs
  MORE iterations than plain theta-space gradient descent -- roughly 3x
  more at lr=0.05 (603 steps vs 196 to reach chemical accuracy) -- because
  theta0's own true minimum sits exactly at the pole theta=pi, and the
  damping factor keeps shrinking theta0's step as it approaches that
  target. This is the same "modest extra iterations" cost already noted
  in qang.gradients' module docstring, now shown to be more pronounced
  when a parameter's optimum itself coincides with a pole rather than
  merely starting near one.

  Finding B (the established benefit, now confirmed on a real, non
  -separable two-parameter molecular Hamiltonian): at a badly-tuned,
  too-aggressive learning rate (lr >= 2.5), plain theta-space gradient
  descent overshoots and never reaches chemical accuracy, while
  theta_pole_damped -- started from the exact same point -- reaches the
  exact FCI energy in a handful of steps at every aggressive learning
  rate tested, up to lr=5.0.
"""

import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from qang.gradients import pole_damping_factor

# Same minimal (2-qubit) H2 electronic Hamiltonian as
# examples/vqe_h2_qg_vs_theta.py (STO-3G, R=0.735 Angstrom, ParityMapper
# with two_qubit_reduction=True, via PySCF + qiskit-nature).
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


def h2_ansatz_multi(theta0: float, theta1: float) -> QuantumCircuit:
    """Two-parameter ansatz: Ry(theta0) generalizes the hard-coded X gate
    that prepared the Hartree-Fock reference in the single-parameter
    version; Ry(theta1) plus the CX is the same single active excitation
    as before."""
    qc = QuantumCircuit(2)
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.cx(1, 0)
    return qc


def h2_energy_multi(theta0: float, theta1: float) -> float:
    sv = Statevector.from_instruction(h2_ansatz_multi(theta0, theta1))
    return sv.expectation_value(H2_ELECTRONIC).real + NUCLEAR_REPULSION


def h2_energy_grad_multi(theta0: float, theta1: float):
    """Exact parameter-shift-rule gradient for each parameter
    independently (valid regardless of the entanglement between them,
    since each is the angle of its own single-qubit Pauli-rotation
    generator)."""
    g0 = 0.5 * (h2_energy_multi(theta0 + math.pi / 2, theta1) - h2_energy_multi(theta0 - math.pi / 2, theta1))
    g1 = 0.5 * (h2_energy_multi(theta0, theta1 + math.pi / 2) - h2_energy_multi(theta0, theta1 - math.pi / 2))
    return g0, g1


def run_vqe_multi(space: str, theta0_0: float, theta1_0: float, lr: float = 0.1, steps: int = 300, eps: float = 0.05):
    """Gradient descent on both parameters at once, in one of two spaces:
      - "theta":             each parameter updated directly, unclamped
                              (matching examples/vqe_h2_qg_vs_theta.py's
                              own "theta" space -- see this module's
                              docstring on why clamping to [0, pi] would
                              reintroduce the arccos-range trapping issue
                              even for a space that never touches qg).
      - "theta_pole_damped": each parameter's own pole_damping_factor
                              applied to its own partial gradient,
                              independently -- no cross terms.
    Returns the list of (theta0, theta1) energies visited (including the
    starting point).
    """
    t0, t1 = theta0_0, theta1_0
    energies = [h2_energy_multi(t0, t1)]

    for _ in range(steps):
        g0, g1 = h2_energy_grad_multi(t0, t1)
        if space == "theta":
            t0 = t0 - lr * g0
            t1 = t1 - lr * g1
        elif space == "theta_pole_damped":
            t0 = t0 - lr * pole_damping_factor(t0, eps=eps) * g0
            t1 = t1 - lr * pole_damping_factor(t1, eps=eps) * g1
        else:
            raise ValueError(f"unknown space: {space}")
        energies.append(h2_energy_multi(t0, t1))

    return energies


def steps_to_chemical_accuracy(energies):
    """First index at which |E - E_exact| < chemical accuracy, or None."""
    for i, e in enumerate(energies):
        if abs(e - EXACT_FCI_ENERGY) < CHEMICAL_ACCURACY:
            return i
    return None


if __name__ == "__main__":
    theta0_0, theta1_0 = 1.0, 1e-6
    print(f"Exact FCI ground-state energy: {EXACT_FCI_ENERGY:.10f} Ha")
    print(f"Starting point: theta0={theta0_0}, theta1={theta1_0}")
    print()

    print("Finding A: at a safe, well-tuned learning rate, theta_pole_damped needs")
    print("more iterations than plain theta, because theta0's true optimum sits")
    print("exactly at the pole theta=pi.")
    for lr in [0.05, 0.1, 0.3, 1.0, 2.0]:
        for space in ("theta", "theta_pole_damped"):
            energies = run_vqe_multi(space, theta0_0, theta1_0, lr=lr, steps=700)
            steps = steps_to_chemical_accuracy(energies)
            steps_str = str(steps) if steps is not None else "never (within 700)"
            print(f"  lr={lr:4.2f}  {space:18s}  steps to chem. accuracy = {steps_str}")
        print()

    print("Finding B: at a badly-tuned (aggressive) learning rate, plain theta")
    print("overshoots and never reaches chemical accuracy, while theta_pole_damped")
    print("still reaches the exact FCI energy in a handful of steps.")
    for lr in [2.5, 3.0, 4.0, 5.0]:
        for space in ("theta", "theta_pole_damped"):
            energies = run_vqe_multi(space, theta0_0, theta1_0, lr=lr, steps=500)
            final = energies[-1]
            steps = steps_to_chemical_accuracy(energies)
            steps_str = str(steps) if steps is not None else "never"
            print(
                f"  lr={lr:4.1f}  {space:18s}  final E = {final:.6f} Ha  "
                f"error = {abs(final - EXACT_FCI_ENERGY):.2e}  steps = {steps_str}"
            )
        print()
