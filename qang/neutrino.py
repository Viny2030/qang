"""
Neutrino flavour oscillations in qg terms (RESEARCH_NOTES §151).

Two flavours on one qubit: |0> = nu_e, |1> = nu_mu. Going to the mass basis
(RY(-2 theta)), letting the mass states pick up their relative phase
Delta = dm2 L / (2E) (RZ), and coming back (RY(2 theta)) gives

    P(e -> mu) = sin^2(2 theta) sin^2(Delta / 2),
    qg_Z      = 1 - 2 P = cos^2(2 theta) + sin^2(2 theta) cos(Delta).

The second line is the spherical law of cosines: the Bloch vector precesses on a
cone of half-angle 2 theta around the mass axis, and the qang angle
vartheta = arccos(qg_Z) is the great-circle distance travelled. At the
oscillation maximum (Delta = pi) vartheta = 4 theta exactly (for theta <= pi/4),
so the mixing angle is a quarter of a qang angle and inherits its
theta-independent shot-noise variance (§125).

Three flavours on three qubits in the weight-1 sector: |e_k> = only qubit k
excited, flavours (e, mu, tau) = qubits (0, 1, 2). The PMNS matrix
U = R23 U_delta R13 U_delta^dagger R12 is three real RBS rotations and two
single-qubit phase gates; propagation is one phase gate per qubit. Every gate
conserves the Hamming weight, so the qg filter applies (§20).

Units: dm2 in eV^2, L in km, E in GeV; Delta = 2 x 1.267 dm2 L / E.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

K_OSC = 1.26693  # dm2 [eV^2] L [km] / (4 E [GeV]) in natural units: Delta / 2 = K_OSC dm2 L / E

# Approximate global-fit values (normal ordering, NuFIT 5.2 era); used as defaults only.
NUFIT_NO = {"theta12": math.radians(33.4), "theta13": math.radians(8.6), "theta23": math.radians(49.0),
            "dm21": 7.42e-5, "dm31": 2.51e-3}

FLAVOURS = ("e", "mu", "tau")


def phase(dm2: float, L: float, E: float) -> float:
    """Delta = dm2 L / (2E) in natural units, i.e. 2 x 1.267 dm2[eV^2] L[km] / E[GeV]."""
    return 2 * K_OSC * dm2 * L / E


# ---------------------------------------------------------------- two flavours

def two_flavour_probability(theta: float, delta: float) -> float:
    """P(nu_e -> nu_mu) in vacuum for mixing angle theta and phase Delta."""
    return math.sin(2 * theta) ** 2 * math.sin(delta / 2) ** 2


def two_flavour_qg_z(theta: float, delta: float) -> float:
    """qg_Z of the flavour qubit: cos^2(2 theta) + sin^2(2 theta) cos(Delta)."""
    return math.cos(2 * theta) ** 2 + math.sin(2 * theta) ** 2 * math.cos(delta)


def two_flavour_qang_angle(theta: float, delta: float) -> float:
    """qang angle arccos(qg_Z): the great-circle distance travelled on the precession cone."""
    return math.acos(max(-1.0, min(1.0, two_flavour_qg_z(theta, delta))))


def mixing_angle_from_maximum(qg_z: float) -> float:
    """theta = arccos(qg_Z) / 4, exact at the oscillation maximum Delta = pi for theta <= pi/4."""
    return math.acos(max(-1.0, min(1.0, qg_z))) / 4


def mixing_angle_std(n_shots: int) -> float:
    """Shot-noise standard deviation of theta = vartheta / 4 at the maximum: 1 / (4 sqrt(N + 1/2)) (§125)."""
    return 1 / (4 * math.sqrt(n_shots + 0.5))


def two_flavour_circuit(theta: float, delta: float, measure: bool = True):
    """One qubit starting as nu_e; P(1) = P(nu_e -> nu_mu)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.ry(-2 * theta, 0)
    qc.rz(delta, 0)
    qc.ry(2 * theta, 0)
    if measure:
        qc.measure_all()
    return qc


# ---------------------------------------------------------------- three flavours

def _r(i: int, j: int, t: float) -> np.ndarray:
    R = np.eye(3)
    R[i, i] = R[j, j] = math.cos(t)
    R[i, j] = math.sin(t)
    R[j, i] = -math.sin(t)
    return R


def pmns(theta12: float, theta13: float, theta23: float, delta_cp: float) -> np.ndarray:
    """Standard parametrization U = R23 U13(delta) R12 (rows: e, mu, tau; columns: masses 1, 2, 3)."""
    Ud = np.diag([1, 1, np.exp(1j * delta_cp)])
    return _r(1, 2, theta23) @ Ud @ _r(0, 2, theta13) @ Ud.conj().T @ _r(0, 1, theta12)


def three_flavour_probabilities(L: float, E: float, delta_cp: float, antineutrino: bool = False,
                                params: dict = None) -> np.ndarray:
    """P[alpha, beta] = P(nu_alpha -> nu_beta) in vacuum; antineutrinos use U*."""
    p = dict(NUFIT_NO, **(params or {}))
    U = pmns(p["theta12"], p["theta13"], p["theta23"], delta_cp)
    if antineutrino:
        U = U.conj()
    D = np.diag(np.exp(-1j * np.array([0.0, phase(p["dm21"], L, E), phase(p["dm31"], L, E)])))
    A = U @ D @ U.conj().T  # A[beta, alpha] = <nu_beta| evolution |nu_alpha>
    return (np.abs(A) ** 2).T


def _rbs(qc, t, a, b):
    from qiskit.circuit.library import XXPlusYYGate

    # on (e_a, e_b): [[cos t, -sin t], [sin t, cos t]]
    qc.append(XXPlusYYGate(2 * t, math.pi / 2), [a, b])


def _apply_rotation(qc, i, j, t):
    # R_ij(t) = [[c, s], [-s, c]] on (e_i, e_j) is RBS(-t)
    _rbs(qc, -t, i, j)


def three_flavour_circuit(initial: str, L: float, E: float, delta_cp: float, antineutrino: bool = False,
                          params: dict = None, measure: bool = True):
    """Weight-1 circuit on 3 qubits: start in flavour ``initial``, apply U D U^dagger; qubit k excited = flavour k.

    P(qubit beta excited) = P(nu_initial -> nu_beta). Gates: X, RBS (XX+YY) and phase gates, all
    weight-preserving.
    """
    from qiskit import QuantumCircuit

    p = dict(NUFIT_NO, **(params or {}))
    d = -delta_cp if antineutrino else delta_cp
    qc = QuantumCircuit(3)
    qc.x(FLAVOURS.index(initial))
    _vacuum_step(qc, L, E, d, p)
    if measure:
        qc.measure_all()
    return qc


def cp_asymmetry(L: float, E: float, delta_cp: float, params: dict = None, rho: float = 0.0,
                 ye: float = 0.5) -> float:
    """A_CP = P(nu_mu -> nu_e) - P(anti-nu_mu -> anti-nu_e); in vacuum unless a density ``rho`` is given."""
    if rho:
        a = matter_probabilities(L, E, delta_cp, rho, ye, False, params)[1, 0]
        b = matter_probabilities(L, E, delta_cp, rho, ye, True, params)[1, 0]
    else:
        a = three_flavour_probabilities(L, E, delta_cp, False, params)[1, 0]
        b = three_flavour_probabilities(L, E, delta_cp, True, params)[1, 0]
    return float(a - b)


# ---------------------------------------------------------------- constant-density matter (MSW, §152)

def matter_potential(rho: float, E: float, ye: float = 0.5) -> float:
    """A = 2 sqrt(2) G_F N_e E in eV^2: 1.52e-4 Y_e rho[g/cm^3] E[GeV]."""
    return 1.52e-4 * ye * rho * E


def matter_probabilities(L: float, E: float, delta_cp: float, rho: float, ye: float = 0.5,
                         antineutrino: bool = False, params: dict = None) -> np.ndarray:
    """Exact P[alpha, beta] in constant-density matter (expm of the flavour Hamiltonian)."""
    from scipy.linalg import expm

    p = dict(NUFIT_NO, **(params or {}))
    U = pmns(p["theta12"], p["theta13"], p["theta23"], delta_cp)
    A = matter_potential(rho, E, ye)
    if antineutrino:
        U, A = U.conj(), -A
    M = U @ np.diag([0.0, p["dm21"], p["dm31"]]) @ U.conj().T + np.diag([A, 0.0, 0.0])
    S = expm(-1j * 2 * K_OSC * L / E * M)  # S[beta, alpha]
    return (np.abs(S) ** 2).T


def _vacuum_step(qc, L, E, d, p):
    # U D(L) U^dagger on the weight-1 sector, appended to qc (d = delta_CP, sign already set):
    # U^dagger = R12^T U_d R13^T U_d^dagger R23^T, then e_k -> exp(-i phi_k) e_k, then U = R23 U_d R13 U_d^dagger R12
    _apply_rotation(qc, 1, 2, -p["theta23"])
    qc.p(-d, 2)
    _apply_rotation(qc, 0, 2, -p["theta13"])
    qc.p(d, 2)
    _apply_rotation(qc, 0, 1, -p["theta12"])
    qc.p(-phase(p["dm21"], L, E), 1)
    qc.p(-phase(p["dm31"], L, E), 2)
    _apply_rotation(qc, 0, 1, p["theta12"])
    qc.p(-d, 2)
    _apply_rotation(qc, 0, 2, p["theta13"])
    qc.p(d, 2)
    _apply_rotation(qc, 1, 2, p["theta23"])


def matter_trotter_circuit(initial: str, L: float, E: float, delta_cp: float, rho: float, steps: int,
                           ye: float = 0.5, antineutrino: bool = False, params: dict = None,
                           measure: bool = True):
    """Second-order (Strang) Trotter circuit for constant-density matter on the weight-1 sector.

    Each step: matter phase on the nu_e qubit for L/2r, exact vacuum propagation for L/r, matter phase
    for L/2r. Every gate conserves the weight, so the qg filter applies.
    """
    from qiskit import QuantumCircuit

    p = dict(NUFIT_NO, **(params or {}))
    d = -delta_cp if antineutrino else delta_cp
    A = matter_potential(rho, E, ye) * (-1 if antineutrino else 1)
    dl = L / steps
    half = 2 * K_OSC * A * dl / (2 * E)
    qc = QuantumCircuit(3)
    qc.x(FLAVOURS.index(initial))
    for _ in range(steps):
        qc.p(-half, 0)
        _vacuum_step(qc, dl, E, d, p)
        qc.p(-half, 0)
    if measure:
        qc.measure_all()
    return qc


def first_maximum_energy(L: float, dm2: float = NUFIT_NO["dm31"]) -> float:
    """Energy (GeV) of the first oscillation maximum, Delta = pi."""
    return 2 * K_OSC * dm2 * L / math.pi


def occupation_profile(probs: Sequence[float]) -> np.ndarray:
    """qg_Z of each flavour qubit from P(flavour): 1 - 2 P."""
    return 1 - 2 * np.asarray(probs, float)
