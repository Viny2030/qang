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


# ---------------------------------------------------------------- collective oscillations (§154)
#
# N neutrinos, two flavours, one qubit each, single-angle approximation:
#
#     H = sum_i (omega_i / 2) B . sigma_i + (mu / 2N) sum_{i<j} sigma_i . sigma_j,
#
# omega_i = dm2 / 2E_i the vacuum frequency of neutrino i, B the mass axis (at angle 2 theta from the
# flavour z axis), mu the neutrino-neutrino coupling. sigma_i . sigma_j = XX + YY + ZZ is the same in every
# basis, so it conserves the number of excitations in the flavour basis AND in the mass basis; the vacuum
# term is diagonal only in the mass basis. In the mass basis (|0> = nu_1, |1> = nu_2)
#
#     H_m = - sum_i omega_i Z_i / 2 + (mu / 2N) sum_{i<j} (X_i X_j + Y_i Y_j + Z_i Z_j),
#
# every term conserves the weight: the conserved quantity is the total number of nu_2, not the total flavour
# number (which the vacuum term changes). A flavour state is a superposition of mass sectors:
# nu_e = cos theta nu_1 + sin theta nu_2, nu_x = -sin theta nu_1 + cos theta nu_2, i.e. RY(2 theta) on |0> or |1>.


def collective_mass_hamiltonian(omegas: Sequence[float], mu: float) -> np.ndarray:
    """Dense H_m (mass basis, qubit i = bit i of the index)."""
    n = len(omegas)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.diag([1.0, -1.0]).astype(complex)

    def op(single):
        out = np.ones((1, 1), complex)
        for q in reversed(range(n)):
            out = np.kron(out, single.get(q, np.eye(2)))
        return out

    H = sum(-0.5 * w * op({i: Z}) for i, w in enumerate(omegas))
    J = mu / (2 * n)
    for i in range(n):
        for j in range(i + 1, n):
            for P in (X, Y, Z):
                H = H + J * op({i: P, j: P})
    return H


def _initial_state(initial: str, theta: float, basis: str) -> np.ndarray:
    """Product state in the mass basis. basis 'flavour': characters e / x; basis 'mass': 0 / 1."""
    one = []
    for ch in initial:
        if basis == "mass":
            one.append(np.array([1, 0], complex) if ch == "0" else np.array([0, 1], complex))
        elif ch == "e":
            one.append(np.array([math.cos(theta), math.sin(theta)], complex))
        else:
            one.append(np.array([-math.sin(theta), math.cos(theta)], complex))
    psi = np.ones(1, complex)
    for v in reversed(one):
        psi = np.kron(psi, v)
    return psi


def collective_exact(omegas: Sequence[float], mu: float, theta: float, initial: str, t: float,
                     initial_basis: str = "flavour", measure_basis: str = "mass") -> np.ndarray:
    """Exact P(qubit i = 1) at time t: P(nu_2) (measure_basis 'mass') or P(nu_x) ('flavour')."""
    from scipy.linalg import expm

    n = len(omegas)
    psi = expm(-1j * t * collective_mass_hamiltonian(omegas, mu)) @ _initial_state(initial, theta, initial_basis)
    if measure_basis == "flavour":
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, s], [-s, c]], complex)  # mass -> flavour amplitudes
        full = np.ones((1, 1), complex)
        for _ in range(n):
            full = np.kron(full, R)
        psi = full @ psi
    p = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    return np.array([p[(idx >> i) & 1 == 1].sum() for i in range(n)])


def mass_sector_probabilities(initial: str, theta: float, initial_basis: str = "flavour") -> np.ndarray:
    """P(total number of nu_2 = w), w = 0..N: conserved by H, so known from the initial state alone."""
    p = np.abs(_initial_state(initial, theta, initial_basis)) ** 2
    n = len(initial)
    w = np.array([bin(k).count("1") for k in range(2 ** n)])
    return np.bincount(w, weights=p, minlength=n + 1)


def _exchange_unitary(a: float) -> np.ndarray:
    from scipy.linalg import expm

    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    return expm(-1j * a * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)))


def collective_trotter_circuit(omegas: Sequence[float], mu: float, theta: float, initial: str, t: float,
                               steps: int, initial_basis: str = "flavour", measure_basis: str = "mass",
                               measure: bool = True):
    """Trotter circuit of H_m in the mass basis. Each step: fields for dt/2, the exchange gate
    exp(-i (mu dt / 2N) sigma.sigma) on every pair (order reversed on every other step), fields for dt/2.
    The initial flavour state is prepared with X and RY(2 theta); measure_basis 'flavour' rotates back with
    RY(-2 theta) before measuring. Everything between preparation and read-out conserves the weight."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    n = len(omegas)
    dt = t / steps
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    U = UnitaryGate(_exchange_unitary(mu * dt / (2 * n)), label="exch")
    qc = QuantumCircuit(n)
    for q, ch in enumerate(initial):
        if ch in "x1":
            qc.x(q)
        if initial_basis == "flavour":
            qc.ry(2 * theta, q)
    for s in range(steps):
        for q, w in enumerate(omegas):
            qc.rz(-w * dt / 2, q)
        for i, j in (pairs if s % 2 == 0 else pairs[::-1]):
            qc.append(U, [i, j])
        for q, w in enumerate(omegas):
            qc.rz(-w * dt / 2, q)
    if measure_basis == "flavour":
        for q in range(n):
            qc.ry(-2 * theta, q)
    if measure:
        qc.measure_all()
    return qc
