"""
The four quantum numbers of an electron in qg terms (RESEARCH_NOTES §155). NumPy only.

qg_Z is <sigma_z> of one qubit. For angular momentum j it generalizes to

    qg_Z = <J_z> / j  in [-1, 1],

and a register of n qubits in its permutation-symmetric subspace IS a spin
j = n/2: qubit |0> carries m = +1/2, |1> carries -1/2, so

    m = n/2 - w     (w = Hamming weight of the outcome),
    qg_Z(register mean) = <J_z> / j.

So the magnetic quantum number m of the collective spin is read off each shot
from its Hamming weight, and the qg weight filter (keep weight w) is the
projection onto one value of m. The total spin j (the azimuthal quantum
number l for orbital momentum) is the permutation symmetry: j = n/2 is the
symmetric subspace, and lower j need antisymmetric pairs (swap test, §149).

The four quantum numbers of hydrogen:
  * m_s: one qubit; qg_Z = 2 m_s.
  * m_l: the weight of a 2l-qubit symmetric register, m_l = l - w. The three
    complex p orbitals (l = 1) are the triplet |00>, (|01> + |10>)/sqrt2, |11>;
    the real p_x, p_y are superpositions of m_l = +1 and -1, i.e. of weights 0
    and 2, which is why the weight filter needs the complex orbitals (§153).
  * l: the total spin of that register.
  * n: hydrogen's hidden SO(4) symmetry makes shell n two spins
    j1 = j2 = (n - 1)/2 (two symmetric registers of n - 1 qubits each);
    coupling them gives l = 0 .. n - 1 with m_l = m1 + m2, n^2 states in all.
    So m_l is (n - 1) minus the total Hamming weight of the 2(n - 1) qubits,
    and the weight filter fixes m_l in every shell.

The angle. A spin-coherent state (every qubit at polar angle theta) has
qg_Z = cos theta for any j: the qang angle arccos(qg_Z) is the direction of
the spin, exactly as for one qubit, and P(m) is binomial. An eigenstate
|l, m> has qang angle arccos(m / l), while the textbook vector-model angle is
arccos(m / sqrt(l (l + 1))): the two differ because |l, m> has transverse
spread <J_x^2 + J_y^2> = l (l + 1) - m^2 > 0, so even m = l does not point
along z in the vector model. qg_Z measures the direction of <J>; the vector
model draws |J| = sqrt(l (l + 1)).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------- m and the Hamming weight

def m_from_weight(n_qubits: int, weight: int) -> float:
    """Magnetic quantum number of an n-qubit outcome of Hamming weight w: m = n/2 - w."""
    return n_qubits / 2 - weight


def weight_from_m(n_qubits: int, m: float) -> int:
    w = n_qubits / 2 - m
    if abs(w - round(w)) > 1e-9 or not 0 <= round(w) <= n_qubits:
        raise ValueError(f"m = {m} is not reachable with {n_qubits} qubits.")
    return int(round(w))


def m_distribution(probs: Sequence[float], n_qubits: int) -> np.ndarray:
    """P(m) for m = j, j - 1, ..., -j (j = n/2) from a distribution over the 2^n outcomes (bit i = qubit i)."""
    p = np.asarray(probs, float)
    w = ((np.arange(2 ** n_qubits)[:, None] >> np.arange(n_qubits)) & 1).sum(axis=1)
    return np.bincount(w, weights=p, minlength=n_qubits + 1)


# ---------------------------------------------------------------- qg for spin j

def qg_z_spin(pm: Sequence[float]) -> float:
    """qg_Z = <J_z> / j from P(m), m = j, j - 1, ..., -j (len(pm) = 2j + 1)."""
    pm = np.asarray(pm, float)
    two_j = len(pm) - 1
    if two_j == 0:
        raise ValueError("j = 0 has no direction.")
    m = two_j / 2 - np.arange(two_j + 1)
    return float(np.dot(m, pm) / (two_j / 2) / pm.sum())


def qg_s_spin(pm: Sequence[float]) -> float:
    """Entropy of P(m) in units of log2(2j + 1), in [0, 1]; for j = 1/2 it is qg_S."""
    pm = np.asarray(pm, float)
    pm = pm / pm.sum()
    nz = pm[pm > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(len(pm)))


def coherent_m_distribution(j: float, theta: float) -> np.ndarray:
    """P(m) of the spin-coherent state at polar angle theta (2j qubits each RY(theta)|0>): binomial."""
    n = int(round(2 * j))
    c2, s2 = math.cos(theta / 2) ** 2, math.sin(theta / 2) ** 2
    return np.array([math.comb(n, w) * c2 ** (n - w) * s2 ** w for w in range(n + 1)])


def qang_angle_of_eigenstate(l: float, m: float) -> float:
    """arccos(qg_Z) of |l, m>: arccos(m / l)."""
    return math.acos(m / l)


def vector_model_angle(l: float, m: float) -> float:
    """The textbook 'space quantization' angle arccos(m / sqrt(l (l + 1)))."""
    return math.acos(m / math.sqrt(l * (l + 1)))


def transverse_spread(l: float, m: float) -> float:
    """<J_x^2 + J_y^2> of |l, m> = l (l + 1) - m^2: why the vector-model angle is never 0."""
    return l * (l + 1) - m * m


# ---------------------------------------------------------------- states

def dicke_state(n_qubits: int, weight: int) -> np.ndarray:
    """|j = n/2, m = n/2 - w>: the equal superposition of all outcomes of weight w."""
    idx = np.arange(2 ** n_qubits)
    w = ((idx[:, None] >> np.arange(n_qubits)) & 1).sum(axis=1)
    v = (w == weight).astype(complex)
    return v / np.linalg.norm(v)


def p_orbital(name: str) -> np.ndarray:
    """The l = 1 orbitals as 2-qubit symmetric states: 'p+1', 'p0', 'p-1' (complex, sharp m_l) and
    'px', 'py', 'pz' (real; pz = p0, px = (p-1 - p+1)/sqrt2, py = i (p-1 + p+1)/sqrt2)."""
    p1, p0, pm1 = dicke_state(2, 0), dicke_state(2, 1), dicke_state(2, 2)
    table = {"p+1": p1, "p0": p0, "p-1": pm1, "pz": p0,
             "px": (pm1 - p1) / math.sqrt(2), "py": 1j * (pm1 + p1) / math.sqrt(2)}
    return table[name]


def spin_operators(n_qubits: int):
    """J_x, J_y, J_z of n qubits (dense; bit i of the index = qubit i)."""
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.diag([1.0, -1.0]).astype(complex)

    def total(P):
        out = np.zeros((2 ** n_qubits, 2 ** n_qubits), complex)
        for q in range(n_qubits):
            op = np.ones((1, 1), complex)
            for k in reversed(range(n_qubits)):
                op = np.kron(op, P if k == q else np.eye(2))
            out += op / 2
        return out

    return total(X), total(Y), total(Z)


# ---------------------------------------------------------------- hydrogen shells

def shell_quantum_numbers(n: int):
    """All (l, m_l) of shell n (spin excluded), n^2 of them."""
    return [(l, m) for l in range(n) for m in range(l, -l - 1, -1)]


def shell_register_qubits(n: int) -> int:
    """Qubits of the SO(4) register of shell n: two symmetric registers of n - 1 qubits."""
    return 2 * (n - 1)


def shell_l_spectrum(n: int) -> np.ndarray:
    """Eigenvalues l of the total angular momentum on the symmetric(n - 1) x symmetric(n - 1) register,
    each repeated 2l + 1 times: exactly l = 0 .. n - 1 (checked numerically, not assumed)."""
    k = n - 1
    if k == 0:
        return np.array([0.0])
    sym = np.array([dicke_state(k, w) for w in range(k + 1)]).T  # 2^k x (k + 1)
    P = np.kron(sym, sym)  # basis of the symmetric x symmetric space (register B = high bits)
    Jx, Jy, Jz = spin_operators(2 * k)
    J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
    ev = np.linalg.eigvalsh(P.conj().T @ J2 @ P)
    return np.round((-1 + np.sqrt(1 + 4 * ev)) / 2, 9)


# ---------------------------------------------------------------- l from qg values of SWAP

def j_squared_from_swaps(n_qubits: int, swaps: Sequence[float]) -> float:
    """<J^2> of n qubits from the qg values <SWAP_ij> of all pairs i < j (each in [-1, 1]):
    J^2 = sum_{i<j} SWAP_ij + n (4 - n) / 4. For two qubits <J^2> = 1 + qg_SWAP: l (l + 1) from one qang number."""
    swaps = np.asarray(swaps, float)
    if swaps.size != n_qubits * (n_qubits - 1) // 2:
        raise ValueError("one SWAP value per pair i < j is needed.")
    return float(swaps.sum() + n_qubits * (4 - n_qubits) / 4)


def l_from_j_squared(j2: float) -> float:
    """Solve l (l + 1) = <J^2> for l >= 0."""
    return (-1 + math.sqrt(1 + 4 * max(j2, 0.0))) / 2
