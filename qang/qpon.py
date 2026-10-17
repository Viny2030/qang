"""
Teleportation through a quantum passive optical network, discrete variables, in qg terms (RESEARCH_NOTES §157).

A downlink QPON: the central office (Alice) shares the resource cos(alpha)|00> + sin(alpha)|11> and sends the
second qubit (a photon present or absent, single rail) over a fibre of transmission T to a passive 1:N splitter;
Bob is one of the N outputs. For Bob's mode both are beam splitters, i.e. amplitude damping, and the photon
reaches him with probability eta = T / N.

Non-assisted protocol (only Alice and Bob act): the resource after the channel has singlet fraction

    f = (cos alpha + sqrt(eta) sin alpha)^2 / 2,

and standard teleportation has average fidelity F = (2 f + 1) / 3 (classical limit 2/3). The maximum is not at
maximal entanglement: tan alpha* = sqrt(eta), F* = (2 + eta) / 3. This is the discrete-variable version of the
optimum finite squeezing of continuous-variable teleportation over lossy networks (Yaghoobianzadeh & Salehi,
Academia Quantum 2026, 10.20935/AcadQuant8460). In qg terms the optimal resource has qg_Z = cos 2 alpha* =
(1 - eta) / (1 + eta) on each qubit.

Assisted protocol: the other N - 1 users report whether the photon arrived at their port (it then went to them,
not to Bob), and those shots are discarded. This is a sector filter on the register of the other users. The loss
that remains unheralded is the fibre loss 1 - T, and with kappa = eta + (1 - T) (the probability that no other
user clicks, given the photon was sent)

    f = (cos alpha + sqrt(eta) sin alpha)^2 / (2 (cos^2 alpha + kappa sin^2 alpha)),
    tan alpha* = sqrt(eta) / kappa,   F* = (2 + eta / kappa) / 3.

Non-assisted is kappa = 1. Without fibre loss (T = 1) kappa = eta = 1/N, so F* = 1 at tan alpha* = sqrt(N): the
best resource is more than maximally entangled before the splitter, so that the splitter's filtering brings it
back to a Bell state; the price is the kept fraction.

Circuit (5 qubits): q0 input, q1 Alice, q2 Bob, q3 fibre environment, q4 the other users. The fibre and the
splitter are RBS gates (XX+YY) between q2 and q3, q4: they conserve the number of photons. Teleportation in
deferred-measurement form (CX, H, CX, CZ); the fidelity of input k is P(q2 = 0) after undoing the preparation of
input k on q2. The six cardinal states form a spherical 3-design, so their mean fidelity is the average over
the Bloch sphere.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

CLASSICAL_FIDELITY = 2 / 3
INPUTS = ("0", "1", "+", "-", "+i", "-i")


# ---------------------------------------------------------------- channel

def fibre_transmission(length_km: float, attenuation_db_per_km: float = 0.2) -> float:
    """T = 10^(-a L / 10)."""
    return 10 ** (-attenuation_db_per_km * length_km / 10)


def link(length_km: float, n_users: int, assisted: bool = False,
         attenuation_db_per_km: float = 0.2) -> Tuple[float, float]:
    """(eta, kappa) of a downlink QPON: eta = T / N, kappa = 1 (non-assisted) or eta + 1 - T (assisted)."""
    T = fibre_transmission(length_km, attenuation_db_per_km)
    eta = T / n_users
    return eta, (eta + 1 - T) if assisted else 1.0


# ---------------------------------------------------------------- closed forms

def heralding_probability(alpha: float, kappa: float = 1.0) -> float:
    """Probability that no other user clicks (the kept fraction): cos^2 alpha + kappa sin^2 alpha."""
    return math.cos(alpha) ** 2 + kappa * math.sin(alpha) ** 2


def singlet_fraction(alpha: float, eta: float, kappa: float = 1.0) -> float:
    """<Phi+| rho |Phi+> of the (kept) resource after the channel."""
    return (math.cos(alpha) + math.sqrt(eta) * math.sin(alpha)) ** 2 / (2 * heralding_probability(alpha, kappa))


def average_fidelity(alpha: float, eta: float, kappa: float = 1.0) -> float:
    """Average teleportation fidelity (2 f + 1) / 3 over the Bloch sphere."""
    return (2 * singlet_fraction(alpha, eta, kappa) + 1) / 3


def optimal_angle(eta: float, kappa: float = 1.0) -> float:
    """alpha* = arctan(sqrt(eta) / kappa)."""
    return math.atan(math.sqrt(eta) / kappa)


def optimal_fidelity(eta: float, kappa: float = 1.0) -> float:
    """F* = (2 + eta / kappa) / 3."""
    return (2 + eta / kappa) / 3


def optimal_qg_z(eta: float, kappa: float = 1.0) -> float:
    """qg_Z of each resource qubit at alpha*: cos 2 alpha* = (kappa^2 - eta) / (kappa^2 + eta)."""
    return (kappa ** 2 - eta) / (kappa ** 2 + eta)


def resource_angle_from_qg_z(qg_z: float) -> float:
    """alpha = arccos(qg_Z) / 2: the resource is set by the qang angle of one of its qubits."""
    return math.acos(max(-1.0, min(1.0, qg_z))) / 2


# ---------------------------------------------------------------- circuit

def _prepare(qc, k: int, q: int, inverse: bool = False):
    name = INPUTS[k]
    ops = {"0": [], "1": [("x",)], "+": [("h",)], "-": [("x",), ("h",)],
           "+i": [("h",), ("s",)], "-i": [("h",), ("sdg",)]}[name]
    inv = {"x": "x", "h": "h", "s": "sdg", "sdg": "s"}
    seq = [inv[o[0]] for o in reversed(ops)] if inverse else [o[0] for o in ops]
    for g in seq:
        getattr(qc, g)(q)


def _beam_splitter(qc, transmission: float, a: int, b: int):
    from qiskit.circuit.library import XXPlusYYGate

    # on (e_a, e_b): e_a -> sqrt(T) e_a + sqrt(1 - T) e_b
    t = math.acos(math.sqrt(max(0.0, min(1.0, transmission))))
    qc.append(XXPlusYYGate(2 * t, math.pi / 2), [a, b])


def teleport_circuit(alpha: float, transmission: float, n_users: int, input_index: int, measure: bool = True):
    """5-qubit downlink QPON teleportation of cardinal input ``input_index`` (see INPUTS).

    Qubits: 0 input, 1 Alice, 2 Bob, 3 fibre loss, 4 other users. Fidelity = P(q2 = 0); assisted: conditioned on
    q4 = 0.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5)
    _prepare(qc, input_index, 0)
    qc.ry(2 * alpha, 1)
    qc.cx(1, 2)
    _beam_splitter(qc, transmission, 2, 3)
    if n_users > 1:
        _beam_splitter(qc, 1 / n_users, 2, 4)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.cz(0, 2)
    _prepare(qc, input_index, 2, inverse=True)
    if measure:
        qc.measure_all()
    return qc


def fidelity_from_counts(counts_per_input: Sequence[Dict[str, int]], assisted: bool = False) -> dict:
    """Average fidelity and kept fraction from the counts of the six inputs (Qiskit bit order: q0 rightmost).

    Returns {"fidelity", "per_input", "kept", "se"}.
    """
    per, kept, var = [], [], []
    for c in counts_per_input:
        total = sum(c.values())
        sel = {k: v for k, v in c.items() if not assisted or k.replace(" ", "")[-5] == "0"}
        n = sum(sel.values())
        good = sum(v for k, v in sel.items() if k.replace(" ", "")[-3] == "0")
        p = good / n if n else float("nan")
        per.append(p)
        kept.append(n / total)
        var.append(p * (1 - p) / n if n else float("nan"))
    m = len(per)
    return {"fidelity": sum(per) / m, "per_input": per, "kept": sum(kept) / m, "se": math.sqrt(sum(var)) / m}
