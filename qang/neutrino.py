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


# ---------------------------------------------------------------- solar neutrinos: adiabatic MSW (§156)
#
# Two effective flavours (nu_3 decoupled): P_ee = c13^4 P_2nu + s13^4, with dm21, theta12 and the potential
# A c13^2. The electron density falls exponentially from the core, A(x) = A_core exp(-x / h), with
# A_core = 1.52e-4 (rho Y_e)_core E. Solar neutrinos reach the Earth as incoherent mass states, so the
# read-out is in the vacuum mass basis: p_2 = P(nu_2) and P_2nu = cos^2 theta12 (1 - p_2) + sin^2 theta12 p_2,
# i.e. P_2nu = 1/2 + cos(2 theta12) qg_Z(nu_2) / 2, linear in qg_Z. Two encodings: one qubit (|0> = nu_e,
# |1> = nu_x), or two qubits in the weight-1 sector (qubit 0 = nu_e, qubit 1 = nu_x; after the read-out
# rotation qubit 0 = nu_1, qubit 1 = nu_2), where the qg filter applies.

SOLAR_RHO_YE = 100.0  # (rho Y_e) at the solar core, g/cm^3


def _solar_params(params):
    p = dict(NUFIT_NO, **(params or {}))
    return p["theta12"], p["theta13"], p["dm21"]


def solar_core_potential(E: float, rho_ye: float = SOLAR_RHO_YE, params: dict = None) -> float:
    """A_core c13^2 in eV^2 (E in GeV): the effective two-flavour potential at the centre of the Sun."""
    _, t13, _ = _solar_params(params)
    return 1.52e-4 * rho_ye * E * math.cos(t13) ** 2


def solar_matter_angle(a: float, params: dict = None) -> float:
    """Two-flavour mixing angle in matter, theta_m in [0, pi/2], for the effective potential a (eV^2)."""
    t12, _, dm21 = _solar_params(params)
    return 0.5 * math.atan2(math.sin(2 * t12) * dm21, math.cos(2 * t12) * dm21 - a)


def solar_pee_from_p2(p2: float, params: dict = None) -> float:
    """P_ee = c13^4 (cos^2 theta12 (1 - p2) + sin^2 theta12 p2) + s13^4 from the nu_2 population p2."""
    t12, t13, _ = _solar_params(params)
    return math.cos(t13) ** 4 * (math.cos(t12) ** 2 * (1 - p2) + math.sin(t12) ** 2 * p2) + math.sin(t13) ** 4


def solar_pee_adiabatic(E: float, rho_ye: float = SOLAR_RHO_YE, params: dict = None) -> float:
    """Adiabatic P_ee: the nu_e born at the core leaves as nu_2 with probability sin^2 theta_m(core)."""
    return solar_pee_from_p2(math.sin(solar_matter_angle(solar_core_potential(E, rho_ye, params), params)) ** 2,
                             params)


def solar_slab_edges(E: float, h: float, X: float, r: int, grid: str = "theta", rho_ye: float = SOLAR_RHO_YE,
                     params: dict = None) -> np.ndarray:
    """Boundaries (km) of r constant-density slabs from the core (0) to X: uniform in x (grid="x") or
    uniform in the matter angle theta_m (grid="theta", finer where theta_m changes fastest)."""
    if grid == "x":
        return np.linspace(0.0, X, r + 1)
    t12, _, dm21 = _solar_params(params)
    a0 = solar_core_potential(E, rho_ye, params)
    a1 = a0 * math.exp(-X / h)
    xs = []
    for th in np.linspace(solar_matter_angle(a0, params), solar_matter_angle(a1, params), r + 1):
        tn = math.tan(2 * th)
        a = dm21 * (math.cos(2 * t12) - math.sin(2 * t12) / tn) if abs(tn) > 1e-12 else 0.0
        a = min(max(a, a1), a0)
        xs.append(-h * math.log(a / a0))
    xs[0], xs[-1] = 0.0, X
    return np.array(xs)


def solar_slab_unitaries(E: float, h: float, X: float, r: int, grid: str = "theta",
                         rho_ye: float = SOLAR_RHO_YE, params: dict = None):
    """2x2 flavour-basis propagators of the r slabs (density at each slab's midpoint), in order."""
    from scipy.linalg import expm

    t12, _, dm21 = _solar_params(params)
    U2 = np.array([[math.cos(t12), math.sin(t12)], [-math.sin(t12), math.cos(t12)]])
    M0 = U2 @ np.diag([0.0, dm21]) @ U2.T
    a0 = solar_core_potential(E, rho_ye, params)
    edges = solar_slab_edges(E, h, X, r, grid, rho_ye, params)
    out = []
    for x0, x1 in zip(edges[:-1], edges[1:]):
        a = a0 * math.exp(-(x0 + x1) / (2 * h))
        out.append(expm(-1j * 2 * K_OSC / E * (x1 - x0) * (M0 + np.diag([a, 0.0]))))
    return out


def _solar_p2(mats, t12):
    psi = np.array([1.0, 0.0], complex)
    for S in mats:
        psi = S @ psi
    return float(abs(math.sin(t12) * psi[0] + math.cos(t12) * psi[1]) ** 2)  # nu_2 = s nu_e + c nu_x


def solar_p2(E: float, h: float, X: float, r: int, grid: str = "theta", rho_ye: float = SOLAR_RHO_YE,
             params: dict = None) -> float:
    """nu_2 population at the exit after r slabs (what the noiseless slab circuit gives)."""
    return _solar_p2(solar_slab_unitaries(E, h, X, r, grid, rho_ye, params), _solar_params(params)[0])


def solar_p2_exact(E: float, h: float, X: float, n: int = 20000, rho_ye: float = SOLAR_RHO_YE,
                   params: dict = None) -> float:
    """nu_2 population at the exit from a fine evolution (n uniform slabs): the reference."""
    return solar_p2(E, h, X, n, "x", rho_ye, params)


def _zyz(S):
    # S = exp(i a) RZ(b) RY(c) RZ(d); returns (b, c, d)
    V = S / np.sqrt(np.linalg.det(S))
    c = 2 * math.atan2(abs(V[1, 0]), abs(V[0, 0]))
    s, t = np.angle(V[1, 1]), np.angle(V[1, 0])
    return float(s + t), c, float(s - t)


def _sector_rz(qc, angle, a, b):
    # RZ(angle) on the weight-1 pair (e_a, e_b): diag(exp(-i angle/2), exp(i angle/2))
    qc.p(-angle / 2, a)
    qc.p(angle / 2, b)


def solar_circuit(E: float, h: float, X: float, r: int, encoding: str = "unary", compiled: bool = False,
                  grid: str = "theta", rho_ye: float = SOLAR_RHO_YE, params: dict = None, measure: bool = True):
    """Circuit for a nu_e born at the solar core, read out in the vacuum mass basis.

    encoding="unary": 2 qubits, weight 1; qubit 1 excited at the end = nu_2. Each slab is RZ RBS RZ on the
    pair (phase gates and one XX+YY gate), so every gate conserves the weight and the qg filter applies.
    encoding="qubit": 1 qubit, |1> = nu_2 at the end; every slab is one single-qubit rotation.
    compiled=True: only the final nu_2 population matters (the phases are lost on the way to the Earth),
    so the whole Sun is replaced by one rotation that loads it: one RBS (unary) or one RY (qubit).
    """
    from qiskit import QuantumCircuit

    t12, _, _ = _solar_params(params)
    mats = solar_slab_unitaries(E, h, X, r, grid, rho_ye, params)
    if compiled:
        ang = math.asin(math.sqrt(min(1.0, max(0.0, _solar_p2(mats, t12)))))
        if encoding == "qubit":
            qc = QuantumCircuit(1)
            qc.ry(2 * ang, 0)
        else:
            qc = QuantumCircuit(2)
            qc.x(0)
            _rbs(qc, ang, 0, 1)
    else:
        to_mass = np.array([[math.cos(t12), -math.sin(t12)], [math.sin(t12), math.cos(t12)]], complex)
        mats = mats + [to_mass]
        qc = QuantumCircuit(1 if encoding == "qubit" else 2)
        if encoding != "qubit":
            qc.x(0)
        for S in mats:
            b, c, d = _zyz(S)
            if encoding == "qubit":
                qc.rz(d, 0)
                qc.ry(c, 0)
                qc.rz(b, 0)
            else:
                _sector_rz(qc, d, 0, 1)
                _rbs(qc, c / 2, 0, 1)
                _sector_rz(qc, b, 0, 1)
    if measure:
        qc.measure_all()
    return qc


# ---------------------------------------------------------------- what oscillation circuits see (§163)
#
# Mass ordering: inverted ordering (IO) has dm31 < 0. In vacuum P(nu_mu -> nu_e) is nearly blind to the sign of
# dm31; in matter the sign of A dm31 decides whether nu or anti-nu is enhanced, so the ordering is read from
# matter_probabilities / matter_trotter_circuit with NUFIT_IO.
# Majorana phases: U_M = U diag(1, e^{i a21/2}, e^{i a31/2}). They are diagonal in the mass basis, so they commute
# with the propagation phases and cancel in U_M D U_M^dagger: no oscillation probability, in vacuum or in matter,
# depends on them (``majorana_invisible``). Neutrinoless double beta decay does: m_bb = |sum_i U_ei^2 m_i|.
# Sterile neutrino (3+1): a fourth flavour that does not interact, one more qubit in the weight-1 sector.

NUFIT_IO = {"theta12": math.radians(33.4), "theta13": math.radians(8.6), "theta23": math.radians(49.3),
            "dm21": 7.42e-5, "dm31": -2.42e-3}


def pmns_majorana(theta12: float, theta13: float, theta23: float, delta_cp: float, a21: float,
                  a31: float) -> np.ndarray:
    """U diag(1, e^{i a21 / 2}, e^{i a31 / 2})."""
    return pmns(theta12, theta13, theta23, delta_cp) @ np.diag([1, np.exp(1j * a21 / 2), np.exp(1j * a31 / 2)])


def majorana_invisible(L: float, E: float, delta_cp: float, a21: float, a31: float, rho: float = 0.0,
                       params: dict = None) -> float:
    """max |P(a21, a31) - P(0, 0)| over the 3x3 transition probabilities (vacuum, or constant matter rho)."""
    from scipy.linalg import expm

    p = dict(NUFIT_NO, **(params or {}))

    def probs(U):
        M = U @ np.diag([0.0, p["dm21"], p["dm31"]]) @ U.conj().T + np.diag([matter_potential(rho, E), 0, 0])
        return np.abs(expm(-1j * 2 * K_OSC * L / E * M)) ** 2

    U0 = pmns_majorana(p["theta12"], p["theta13"], p["theta23"], delta_cp, 0, 0)
    U1 = pmns_majorana(p["theta12"], p["theta13"], p["theta23"], delta_cp, a21, a31)
    return float(np.abs(probs(U1) - probs(U0)).max())


def masses(m_lightest: float, ordering: str = "NO", params: dict = None) -> np.ndarray:
    """(m1, m2, m3) in eV from the lightest mass and the splittings."""
    p = dict(NUFIT_NO if ordering == "NO" else NUFIT_IO, **(params or {}))
    if ordering == "NO":
        m1 = m_lightest
        return np.array([m1, math.sqrt(m1 ** 2 + p["dm21"]), math.sqrt(m1 ** 2 + p["dm31"])])
    m3 = m_lightest
    m1 = math.sqrt(m3 ** 2 - p["dm31"])
    return np.array([m1, math.sqrt(m1 ** 2 + p["dm21"]), m3])


def m_bb(m_lightest: float, a21: float, a31: float, ordering: str = "NO", delta_cp: float = 0.0,
         params: dict = None) -> float:
    """Effective Majorana mass |sum_i U_ei^2 m_i| (eV) of neutrinoless double beta decay."""
    p = dict(NUFIT_NO if ordering == "NO" else NUFIT_IO, **(params or {}))
    U = pmns_majorana(p["theta12"], p["theta13"], p["theta23"], delta_cp, a21, a31)
    return float(abs(np.sum(U[0] ** 2 * masses(m_lightest, ordering, p))))


def majorana_circuit(initial: str, L: float, E: float, delta_cp: float, a21: float, a31: float,
                     params: dict = None, measure: bool = True):
    """three_flavour_circuit with the Majorana phases inserted as phase gates in the mass basis (U_M D U_M^dagger)."""
    from qiskit import QuantumCircuit

    p = dict(NUFIT_NO, **(params or {}))
    d = delta_cp
    qc = QuantumCircuit(3)
    qc.x(FLAVOURS.index(initial))
    _apply_rotation(qc, 1, 2, -p["theta23"])
    qc.p(-d, 2)
    _apply_rotation(qc, 0, 2, -p["theta13"])
    qc.p(d, 2)
    _apply_rotation(qc, 0, 1, -p["theta12"])
    qc.p(-a21 / 2, 1)  # Majorana phases (U_M^dagger side)
    qc.p(-a31 / 2, 2)
    qc.p(-phase(p["dm21"], L, E), 1)
    qc.p(-phase(p["dm31"], L, E), 2)
    qc.p(a21 / 2, 1)  # U_M side
    qc.p(a31 / 2, 2)
    _apply_rotation(qc, 0, 1, p["theta12"])
    qc.p(-d, 2)
    _apply_rotation(qc, 0, 2, p["theta13"])
    qc.p(d, 2)
    _apply_rotation(qc, 1, 2, p["theta23"])
    if measure:
        qc.measure_all()
    return qc


STERILE = ("e", "mu", "tau", "s")


def _r4(i: int, j: int, t: float) -> np.ndarray:
    R = np.eye(4)
    R[i, i] = R[j, j] = math.cos(t)
    R[i, j] = math.sin(t)
    R[j, i] = -math.sin(t)
    return R


def sterile_angles(params: dict = None, theta14: float = 0.15, theta24: float = 0.15, theta34: float = 0.0) -> list:
    """Rotation sequence (i, j, angle) of the real 3+1 mixing U = R34 R24 R14 R23 R13 R12 (no CP phases)."""
    p = dict(NUFIT_NO, **(params or {}))
    return [(2, 3, theta34), (1, 3, theta24), (0, 3, theta14), (1, 2, p["theta23"]), (0, 2, p["theta13"]),
            (0, 1, p["theta12"])]


def sterile_mixing(params: dict = None, **angles) -> np.ndarray:
    U = np.eye(4)
    for i, j, t in sterile_angles(params, **angles):
        U = U @ _r4(i, j, t)
    return U


def sterile_probabilities(L: float, E: float, dm41: float = 1.0, params: dict = None, **angles) -> np.ndarray:
    """P[alpha, beta], 4x4, in vacuum for 3+1 (masses 0, dm21, dm31, dm41)."""
    p = dict(NUFIT_NO, **(params or {}))
    U = sterile_mixing(p, **angles)
    D = np.diag(np.exp(-1j * np.array([0, phase(p["dm21"], L, E), phase(p["dm31"], L, E), phase(dm41, L, E)])))
    A = U @ D @ U.conj().T
    return (np.abs(A) ** 2).T


def sterile_circuit(initial: str, L: float, E: float, dm41: float = 1.0, params: dict = None, measure: bool = True,
                    **angles):
    """Weight-1 circuit on 4 qubits (e, mu, tau, s): U^dagger, mass phases, U; RBS and phase gates only."""
    from qiskit import QuantumCircuit

    p = dict(NUFIT_NO, **(params or {}))
    seq = sterile_angles(p, **angles)
    qc = QuantumCircuit(4)
    qc.x(STERILE.index(initial))
    for i, j, t in seq:  # U^dagger = R12^T ... R34^T: R34^T acts first
        _apply_rotation(qc, i, j, -t)
    for k, dm in ((1, p["dm21"]), (2, p["dm31"]), (3, dm41)):
        qc.p(-phase(dm, L, E), k)
    for i, j, t in reversed(seq):  # U = R34 ... R12: R12 acts first
        _apply_rotation(qc, i, j, t)
    if measure:
        qc.measure_all()
    return qc
