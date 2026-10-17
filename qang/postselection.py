"""
Teleportation with postselection: closed timelike curves (P-CTC) and black-hole decoding in qg terms
(RESEARCH_NOTES §164).

Both are information-theoretic models, not statements about real spacetime.

P-CTC (Lloyd et al. 2011). A qubit C "sent to the past" is half of a Bell pair (C, A); after C interacts with
the system S, (C, A) is projected on the Bell state (postselection). The effective map is S -> Tr_C(U) S
Tr_C(U)^dagger, normalized. If C alone is rotated by RX(phi), the postselection succeeds with probability
|Tr RX(phi) / 2|^2 = cos^2(phi/2): its qang angle arccos(2P - 1) is phi itself, and the grandfather paradox
(phi = pi, C flips its own past) has probability 0. If S controls the rotation, Tr_C(U) = diag(1, cos(phi/2)): the
paradoxical branch is suppressed, and from |+> the postselected qg_Z of S is (1 - c^2) / (1 + c^2),
c = cos(phi/2).

Black holes (Hayden-Preskill; Yoshida-Kitaev decoder). A message qubit A, entangled with a reference R, falls
into a black hole B (entangled with early radiation B'); a scrambler U acts on (A, B); one qubit D of the output
is radiated. Applying U* on (A', B') and postselecting (D, D') on a Bell state teleports A's entanglement to R'.
For a scrambler (every single-qubit Pauli spread over all qubits) the postselection succeeds with probability
1/d_A^2 = 1/4 and the fidelity of (R, R') with a Bell pair is 1; with no scrambling it is 1/4. The postselection
is a filter on the Bell sector of (D, D').
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# ---------------------------------------------------------------- P-CTC

def pctc_success(phi: float) -> float:
    """Postselection success of the uncontrolled P-CTC, cos^2(phi / 2)."""
    return math.cos(phi / 2) ** 2


def pctc_controlled_qg_z(phi: float) -> float:
    """Postselected qg_Z of S (input |+>) when S controls RX(phi) on the CTC qubit: (1 - c^2) / (1 + c^2)."""
    c2 = math.cos(phi / 2) ** 2
    return (1 - c2) / (1 + c2)


def pctc_circuit(phi: float, controlled: bool, measure: bool = True):
    """Qubits 0 = S (prepared in |+>), 1 = C, 2 = A. Bell pair (C, A), RX(phi) on C (controlled by S if asked),
    Bell measurement of (C, A): success = both read 0."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.cx(1, 2)
    if controlled:
        qc.crx(phi, 0, 1)
    else:
        qc.rx(phi, 1)
    qc.cx(1, 2)
    qc.h(1)
    if measure:
        qc.measure_all()
    return qc


def pctc_from_counts(counts: Dict[str, int]) -> dict:
    """{"success", "qg_z_S"}: P(C = A = 0) and the qg_Z of S among those shots."""
    tot = n = z = 0
    for k, c in counts.items():
        b = k.replace(" ", "")
        tot += c
        if b[-2] == "0" and b[-3] == "0":
            n += c
            z += c * (1 - 2 * int(b[-1]))
    return {"success": n / tot, "qg_z_S": z / n if n else float("nan")}


# ---------------------------------------------------------------- Hayden-Preskill / Yoshida-Kitaev

def is_scrambler(clifford) -> bool:
    """True when every single-qubit Pauli is mapped to a Pauli acting on all qubits."""
    from qiskit.quantum_info import Pauli

    n = clifford.num_qubits
    for q in range(n):
        for P in "XYZ":
            lab = ["I"] * n
            lab[n - 1 - q] = P
            w = Pauli("".join(lab)).evolve(clifford, frame="s").to_label().lstrip("-i")
            if sum(ch != "I" for ch in w) != n:
                return False
    return True


def find_scrambler(n: int = 3, seed: int = 164, tries: int = 5000):
    """First random n-qubit Clifford (from ``seed``) that is a scrambler, as a circuit."""
    from qiskit.quantum_info import random_clifford

    rng = np.random.default_rng(seed)
    for _ in range(tries):
        C = random_clifford(n, seed=int(rng.integers(1e9)))
        if is_scrambler(C):
            return C.to_circuit()
    raise RuntimeError("no scrambler found")


YK_D = 3  # the radiated qubit D is the output wire of B2; D' the wire of B2'


def yk_decoder_circuit(u_circuit, measure: bool = True):
    """8 qubits: 0 R, 1 A, 2 B1, 3 B2, 4 B1', 5 B2', 6 A', 7 R'. Bell pairs (R, A), (B1, B1'), (B2, B2'), (A', R');
    U on (A, B1, B2), U* on (A', B1', B2'); Bell measurements of (D, D') = (B2, B2') and of (R, R')."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(8)
    for a, b in ((0, 1), (2, 4), (3, 5), (6, 7)):
        qc.h(a)
        qc.cx(a, b)
    U = Operator(u_circuit).data
    qc.append(UnitaryGate(U, label="U"), [1, 2, 3])
    qc.append(UnitaryGate(U.conj(), label="U*"), [6, 4, 5])
    for a, b in ((3, 5), (0, 7)):
        qc.cx(a, b)
        qc.h(a)
    if measure:
        qc.measure_all()
    return qc


def yk_from_counts(counts: Dict[str, int]) -> dict:
    """{"success", "fidelity"}: P(Bell outcome on (D, D')) and the Bell fidelity of (R, R') given it."""
    tot = n = f = 0
    for k, c in counts.items():
        b = k.replace(" ", "")
        bit = lambda q: b[-1 - q]  # noqa: E731
        tot += c
        if bit(3) == "0" and bit(5) == "0":
            n += c
            if bit(0) == "0" and bit(7) == "0":
                f += c
    return {"success": n / tot, "fidelity": f / n if n else float("nan")}
