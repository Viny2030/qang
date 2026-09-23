
"""
LiH VQE with a genuinely mixed-axis ansatz: three Ry rotations plus one Rx
rotation, extending examples/h2_vqe_multi_parameter_ansatz.py (which uses
Ry exclusively) to (a) a real molecule bigger than H2 -- 4 qubits, 52 Pauli
terms, instead of H2's 2 qubits, 5 terms -- and (b) a rotation axis other
than Ry, closing the "generalize qang's gradient machinery beyond Ry" gap
left open by qang.gradients (whose pole_damping_factor was previously only
ever exercised on Ry-parametrized qubits).

The molecule and Hamiltonian
-----------------------------
LiH at bond length 1.5459 Angstrom, STO-3G basis, via PySCF + qiskit-nature
(ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=3),
ParityMapper). This active-space reduction keeps 2 electrons in the 3
spatial orbitals closest to the frontier (roughly, Li's 2s/2p-derived
orbitals and H's 1s orbital) and freezes the rest: 3 spatial orbitals are
6 spin-orbitals, which the parity mapping's two-qubit reduction brings
down to a 4-qubit, 52-term qubit Hamiltonian -- the same scale as the 4-qubit LiH problems studied in the
early VQE hardware literature (Kandala et al. 2017; O'Malley et al. 2016).
LIH_ELECTRONIC and NUCLEAR_REPULSION below are exactly this derivation's
output, hard-coded the same way examples/h2_vqe_multi_parameter_ansatz.py
hard-codes H2_ELECTRONIC -- PySCF and qiskit-nature are needed only to
*derive* these numbers once, never to run this module or its tests.

The ansatz
----------
    qc.ry(theta0, 0)   # generalizes the HF-preparing X gate on qubit 0
    qc.ry(theta1, 1)   # generalizes the HF-preparing X gate on qubit 1
    qc.ry(theta2, 2)   #  |0>-initialized "virtual" qubit, entangled in
    qc.rx(theta3, 3)   #  |0>-initialized "virtual" qubit, entangled in --
                       #  deliberately Rx rather than Ry, see below
    qc.cx(2, 0)
    qc.cx(3, 1)

Qubits 0 and 1 start from the two occupied spin-orbitals of this active
space's Hartree-Fock reference (Ry(pi)|0> = X|0>, exactly as in the H2
ansatz); qubits 2 and 3 start from the two unoccupied ("virtual")
spin-orbitals, and are wired as CX *controls* onto 0 and 1 so that at
theta2 = theta3 = 0 (qubits 2, 3 exactly |0>) the CX gates are no-ops and
the whole circuit reduces, bit for bit, to the bare Hartree-Fock
preparation. Concretely, at (theta0, theta1, theta2, theta3) =
(pi, pi, 0, 0):

  * this ansatz's statevector has overlap EXACTLY 1.0 (verified to
    floating-point precision) with qiskit-nature's own official
    ``HartreeFock`` reference circuit for this active space, and
  * lih_energy(pi, pi, 0, 0) == HF_ENERGY == -0.04378130564745453,
    matching that official circuit's own energy to the same precision.

That HF-point energy differs from EXACT_GROUND_STATE (the active space's
FCI ground state, -0.044830902021270935) by ~0.00105 Hartree. This is not
a bug in the Hamiltonian or the ansatz -- it is the genuine electron
correlation energy for this active space, which VQE recovers (partially,
for a 4-parameter ansatz -- see below) by moving theta2, theta3 away from
their Hartree-Fock values of 0. Note this correlation energy happens to
be smaller than the conventional "chemical accuracy" bar of 1.6e-3
Hartree used throughout this codebase's H2 examples -- so, unlike H2,
the *bare, untrained* Hartree-Fock point already sits within chemical
accuracy of the exact answer here, a property specific to this active
space's small correlation energy, not a general feature of LiH. For that
reason, Findings A and B below measure convergence to this ansatz's own
attainable optimum (ANSATZ_OPTIMUM, found by direct classical
minimization, and only ~0.0008 Hartree above the true FCI ground state --
itself inside chemical accuracy) rather than to chemical accuracy, since
the latter is met almost everywhere in this particular landscape and
would not distinguish the optimizers.

Generalizing pole_damping_factor beyond Ry
--------------------------------------------
qang.gradients' pole_damping_factor(theta, eps) = max(|sin(theta)|, eps)
was derived from d(qg_Z)/d(theta) = -sin(theta) for a Ry-parametrized
qubit. This module confirms, both by direct computation and in
tests/test_lih_vqe_ry_rx_ansatz.py, that the same relation holds
identically for an Rx-parametrized qubit: for a qubit starting in |0>,
Rx(theta)|0> = cos(theta/2)|0> - i*sin(theta/2)|1> has EXACTLY the same
measurement-basis populations, |cos(theta/2)|^2 and |sin(theta/2)|^2, as
Ry(theta)|0> -- the two circuits differ only in the relative phase (-i
versus real) between |0> and |1>, which qg_Z (a population-only quantity)
cannot see at all. So qg_Z(theta) = cos(theta), and therefore
pole_damping_factor(theta), are identical functions of theta regardless
of which axis prepared that population -- pole_damping_factor needs no
generalization at all to be applied to theta3 below; it already applies,
unchanged, because it was never really Ry-specific, only
population-specific.

The relative phase Rx leaves behind is not merely cosmetic, though: once
qubit 3 is entangled into the rest of the register via CX(3 -> 1), that
phase changes the resulting joint state (and hence the energy) relative
to what an Ry rotation on qubit 3 would have produced at the same theta3
(tests/test_lih_vqe_ry_rx_ansatz.py confirms both halves of this
directly: identical single-qubit populations before the CX, different
whole-register energies after it). Rx is therefore a genuine, physically
distinct choice of generator here -- not a relabeling of Ry -- while still
inheriting the exact same pole geometry that makes pole damping meaningful.

Two findings, both starting from the Hartree-Fock point nudged just off
the ansatz's own pole point, (theta0, theta1, theta2, theta3) =
(pi, pi, 1e-6, 1e-6) -- deliberately the same "near-pole start" stress
test used throughout qang.gradients and the H2 example, now on all four
parameters simultaneously, since (see below) all four of this ansatz's
own true optimum values happen to sit at or extremely close to a pole:

  Finding A (an honest cost, substantially larger here than for H2):
  at every safe, well-tuned learning rate tested, theta_pole_damped needs
  MANY more iterations than plain theta -- roughly 20x more at lr=1.0 (106
  steps vs 5) and 20x more at lr=0.3 (355 steps vs 17) -- because ALL FOUR
  of this ansatz's true optimum parameter values sit at or within ~0.04
  radians of a pole (theta1's optimum is theta=pi exactly; theta3's
  optimum is theta=0 exactly), not just one parameter as in the H2 case.
  The damping factor keeps shrinking every parameter's step as it
  approaches its own target, so the extra-iterations cost compounds
  across all four coordinates at once.

  Finding B (the established benefit, now confirmed for a mixed-axis,
  larger, non-separable four-parameter molecular ansatz): at every
  aggressive learning rate tested (lr = 2.0 through 10.0), plain
  theta-space gradient descent overshoots and never converges to this
  ansatz's optimum, while theta_pole_damped -- started from the exact
  same point -- converges reliably every time, and, notably, converges in
  FEWER steps as the learning rate grows more aggressive (53 steps at
  lr=2.0, down to 10 steps at lr=10.0): the very early, most-damped steps
  do most of the work of escaping the near-pole starting region, and a
  larger nominal learning rate simply means a larger effective step once
  the damping factor grows away from the poles.
"""

import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from qang.gradients import pole_damping_factor

# The 4-qubit, 52-term LiH electronic Hamiltonian for this active space
# (STO-3G, R=1.5459 Angstrom, ActiveSpaceTransformer(num_electrons=2,
# num_spatial_orbitals=3), ParityMapper), derived once via PySCF +
# qiskit-nature and hard-coded here exactly as
# examples/h2_vqe_multi_parameter_ansatz.py hard-codes H2_ELECTRONIC.
LIH_ELECTRONIC = SparsePauliOp.from_list([
    ("IIII", -0.47118832730937993),
    ("IIIZ", -0.03248738302506973),
    ("IIZZ", -0.20749200075650553),
    ("IIZI", 0.21357886680908028),
    ("IZII", 0.03248738302506968),
    ("IZIZ", -0.12276860077166406),
    ("ZXII", 0.014909887026983443),
    ("ZXIZ", 0.011919376142463375),
    ("IXII", 0.014909887026983443),
    ("IXIZ", 0.011919376142463375),
    ("ZZII", -0.20749200075650542),
    ("ZZIZ", 0.05630012515323693),
    ("ZIII", -0.21357886680908028),
    ("ZIIZ", 0.06821276873666593),
    ("IIZX", -0.014909887026983443),
    ("IIIX", 0.014909887026983443),
    ("IZZX", 0.011919376142463375),
    ("IZIX", -0.011919376142463375),
    ("ZXZX", -0.0031360347376337195),
    ("IXZX", -0.0031360347376337195),
    ("ZXIX", 0.0031360347376337195),
    ("IXIX", 0.0031360347376337195),
    ("ZZZX", 0.0016925109480380209),
    ("ZZIX", -0.0016925109480380209),
    ("ZIZX", 0.0012979961000870702),
    ("ZIIX", -0.0012979961000870702),
    ("XXXX", 0.0059310384473942445),
    ("YYXX", -0.0059310384473942445),
    ("XXYY", -0.0059310384473942445),
    ("YYYY", 0.0059310384473942445),
    ("XZXX", -0.004808961060959648),
    ("XIXX", 0.004808961060959648),
    ("XZYY", 0.004808961060959648),
    ("XIYY", -0.004808961060959648),
    ("IZZZ", -0.05630012515323693),
    ("ZXZZ", -0.0016925109480380209),
    ("IXZZ", -0.0016925109480380209),
    ("ZZZZ", 0.08460478367313734),
    ("ZIZZ", 0.07052983162552193),
    ("XXXZ", 0.004808961060959648),
    ("YYXZ", -0.004808961060959648),
    ("XXXI", 0.004808961060959648),
    ("YYXI", -0.004808961060959648),
    ("XZXZ", -0.010323248524099883),
    ("XIXZ", 0.010323248524099883),
    ("XZXI", -0.010323248524099883),
    ("XIXI", 0.010323248524099883),
    ("IZZI", 0.06821276873666593),
    ("ZXZI", 0.0012979961000870702),
    ("IXZI", 0.0012979961000870702),
    ("ZZZI", -0.07052983162552193),
    ("ZIZI", -0.07823637778985229),
])
NUCLEAR_REPULSION = 1.0269303530370657

# Exact (FCI) ground-state energy of this active space, and this specific
# ansatz's own best-attainable energy (found by direct classical
# minimization from many random starting points -- see this module's
# docstring for why chemical accuracy is not a useful convergence bar
# here).
EXACT_GROUND_STATE = -0.044830902021270935
HF_ENERGY = -0.04378130564745453
ANSATZ_OPTIMUM = -0.044015242088863316
CHEMICAL_ACCURACY = 1.6e-3
CONVERGENCE_TOL = 1e-5


def lih_ansatz(thetas, axis3: str = "rx") -> QuantumCircuit:
    """The mixed Ry/Rx ansatz described in this module's docstring.
    axis3 selects the rotation gate used on qubit 3 ("rx", the delivered
    choice, or "ry", used only by
    tests/test_lih_vqe_ry_rx_ansatz.py to isolate the effect of that
    single axis choice)."""
    t0, t1, t2, t3 = thetas
    qc = QuantumCircuit(4)
    qc.ry(t0, 0)
    qc.ry(t1, 1)
    qc.ry(t2, 2)
    if axis3 == "rx":
        qc.rx(t3, 3)
    elif axis3 == "ry":
        qc.ry(t3, 3)
    else:
        raise ValueError(f"axis3 must be 'rx' or 'ry', got {axis3!r}.")
    qc.cx(2, 0)
    qc.cx(3, 1)
    return qc


def lih_energy(thetas, axis3: str = "rx") -> float:
    sv = Statevector.from_instruction(lih_ansatz(thetas, axis3=axis3))
    return float(sv.expectation_value(LIH_ELECTRONIC).real) + NUCLEAR_REPULSION


def lih_energy_grad(thetas, axis3: str = "rx"):
    """Exact parameter-shift-rule gradient for each of the four
    parameters independently. The rule dE/d(theta_i) =
    [E(theta_i + pi/2) - E(theta_i - pi/2)] / 2 holds for any parameter
    that is the angle of a single-qubit Pauli rotation -- Rx included --
    since Rx(theta) = exp(-i*theta*X/2) has the same +-1/2 generator
    eigenvalue spectrum as Ry(theta) = exp(-i*theta*Y/2). This is
    verified against finite differences for all four parameters,
    including theta3 (the Rx one), in
    tests/test_lih_vqe_ry_rx_ansatz.py."""
    grads = []
    for i in range(4):
        plus = list(thetas)
        plus[i] += math.pi / 2
        minus = list(thetas)
        minus[i] -= math.pi / 2
        grads.append(0.5 * (lih_energy(plus, axis3=axis3) - lih_energy(minus, axis3=axis3)))
    return grads


def run_vqe_lih(space: str, thetas0, lr: float = 0.1, steps: int = 300, eps: float = 0.05):
    """Gradient descent on all four parameters at once, in one of two
    spaces:
      - "theta":             each parameter updated directly, unclamped.
      - "theta_pole_damped": each parameter's own pole_damping_factor
                              applied to its own partial gradient,
                              independently -- no cross terms, and, per
                              this module's docstring, applied identically
                              to theta3 (Rx) as to theta0-theta2 (Ry).
    Returns the list of energies visited (including the starting point).
    Stops early (leaving the returned list shorter than steps + 1) if the
    energy or any parameter diverges, exactly like
    examples/quantum_volume_qg_s_realistic_noise.py's neighbors avoid
    doing unbounded floating-point work on a run that has already failed.
    """
    thetas = list(thetas0)
    energies = [lih_energy(thetas)]

    for _ in range(steps):
        grads = lih_energy_grad(thetas)
        if space == "theta":
            thetas = [t - lr * g for t, g in zip(thetas, grads)]
        elif space == "theta_pole_damped":
            thetas = [t - lr * pole_damping_factor(t, eps=eps) * g for t, g in zip(thetas, grads)]
        else:
            raise ValueError("space must be 'theta' or 'theta_pole_damped'.")

        e = lih_energy(thetas)
        energies.append(e)
        if not math.isfinite(e) or any(abs(t) > 1e6 for t in thetas):
            break

    return energies


def steps_to_sustained_convergence(energies, tol: float = CONVERGENCE_TOL):
    """The earliest index i such that energies[i:] are ALL within tol of
    ANSATZ_OPTIMUM, or None if no such index exists. Unlike a plain
    first-crossing check, this is not fooled by a divergent trajectory
    that transiently passes near the optimum before overshooting away
    from it again (which plain theta does at several of the aggressive
    learning rates studied here)."""
    n = len(energies)
    ok = [abs(e - ANSATZ_OPTIMUM) < tol for e in energies]
    i = n
    for k in range(n - 1, -1, -1):
        if ok[k]:
            i = k
        else:
            break
    return i if i < n else None


if __name__ == "__main__":
    print(f"Hartree-Fock energy (this active space):  {HF_ENERGY:.10f} Ha")
    print(f"Exact (FCI) ground-state energy:           {EXACT_GROUND_STATE:.10f} Ha")
    print(f"This ansatz's own best-attainable energy:  {ANSATZ_OPTIMUM:.10f} Ha")
    print()

    theta0 = [math.pi, math.pi, 1e-6, 1e-6]

    print("Finding A: at every safe learning rate, theta_pole_damped needs many more")
    print("iterations than plain theta, because ALL FOUR parameters' true optima sit")
    print("at or within ~0.04 rad of a pole (not just one, as in the H2 example).")
    for lr in [0.3, 1.0]:
        for space in ("theta", "theta_pole_damped"):
            energies = run_vqe_lih(space, theta0, lr=lr, steps=400)
            steps = steps_to_sustained_convergence(energies)
            print(f"  lr={lr:4.2f}  {space:18s}  steps to convergence = {steps}")
        print()

    print("Finding B: at every aggressive learning rate, plain theta fails to converge")
    print("while theta_pole_damped succeeds every time -- and converges in FEWER steps")
    print("as the learning rate grows, since the early, most-damped steps do most of")
    print("the work of escaping the near-pole starting region.")
    for lr in [2.0, 2.5, 4.0, 7.0, 10.0]:
        for space in ("theta", "theta_pole_damped"):
            energies = run_vqe_lih(space, theta0, lr=lr, steps=300)
            steps = steps_to_sustained_convergence(energies)
            final = energies[-1]
            steps_str = str(steps) if steps is not None else "never"
            print(
                f"  lr={lr:5.2f}  {space:18s}  final E = {final:.6f} Ha  "
                f"error = {abs(final - ANSATZ_OPTIMUM):.2e}  steps = {steps_str}"
            )
        print()

