"""
qang.formulation -- the 15 main gates and 14 main algorithms in qg language.

The state of n qubits is described by its qg values, qg_P = Tr(rho P) for
every Pauli string P, and a gate U acts as

    qg'_P = qg_{U^dag P U}.

Clifford gates permute qg values with signs; non-Clifford gates (T, generic
rotations, Toffoli, Fredkin) mix them. Each algorithm becomes a readout
identity: which qg values hold the answer. This is the Pauli representation
(known); the module gives it in the units used throughout qang, with the
classification of gates by Hamming-weight conservation (where the qg witness
and filter apply) and of algorithms by what has to be read out.

Conventions: in Pauli strings the leftmost letter is qubit 0, the most
significant bit of a basis index, and the control of CX/CZ (the controls of
Toffoli, the control of Fredkin).

NumPy only (``walk_register_mean`` also needs SciPy). Theory and every
identity: manuscript/teoria_es (RESEARCH_NOTES), pinned by
tests/test_formulation.py.
"""

from __future__ import annotations

import itertools

import numpy as np

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])


def kron(*mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def pauli(label):
    return kron(*[PAULI[c] for c in label])


def rx(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X


def ry(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Y


def rz(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Z


def phase(p):
    return np.diag([1, np.exp(1j * p)])


CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[[6, 7]] = TOFFOLI[[7, 6]]
FREDKIN = np.eye(8, dtype=complex)
FREDKIN[[5, 6]] = FREDKIN[[6, 5]]

GATES_1Q = {"X": X, "Y": Y, "Z": Z, "H": H, "S": S, "T": T}
GATES_MULTI = {"CX": CX, "CZ": CZ, "SWAP": SWAP, "iSWAP": ISWAP, "Toffoli": TOFFOLI, "Fredkin": FREDKIN}


def conjugate(U, label, tol=1e-9):
    """U^dag P U as {Pauli string: real coefficient}: qg'_label = sum c * qg_string."""
    n = len(label)
    M = U.conj().T @ pauli(label) @ U
    out = {}
    for t in itertools.product("IXYZ", repeat=n):
        t = "".join(t)
        c = np.trace(pauli(t).conj().T @ M) / 2**n
        if abs(c) > tol:
            assert abs(c.imag) < tol
            out[t] = float(c.real)
    return out


def conserves_weight(U, n):
    """True if U maps every Hamming-weight sector into itself."""
    w = np.array([bin(i).count("1") for i in range(2**n)])
    return bool(np.all(np.abs(U[w[:, None] != w[None, :]]) < 1e-12))


def z_parity(probs, n, subset):
    """qg_{Z_S} = sum_y p(y) (-1)^{S.y} (qubit 0 = most significant bit)."""
    ys = np.arange(2**n)
    bits = (ys[:, None] >> np.arange(n)[::-1]) & 1
    return float(np.sum(np.asarray(probs) * (-1.0) ** (bits @ np.asarray(subset))))


def single_qg_z(probs, n):
    return [z_parity(probs, n, [int(i == j) for j in range(n)]) for i in range(n)]


def hadamard_n(n):
    return kron(*[H] * n)


def phase_oracle_output(f_signs):
    """Deutsch-Jozsa / Bernstein-Vazirani: H^n O_f H^n |0>, returns probabilities."""
    n = int(np.log2(len(f_signs)))
    psi = hadamard_n(n) @ (np.asarray(f_signs) * (hadamard_n(n) @ np.eye(2**n)[0]))
    return np.abs(psi) ** 2


def simon_output(s):
    """Ideal Simon output: y uniform over {y : y.s = 0 mod 2}."""
    n = len(s)
    ys = (np.arange(2**n)[:, None] >> np.arange(n)[::-1]) & 1
    return np.where((ys @ np.asarray(s)) % 2 == 0, 2.0 ** (1 - n), 0.0)


def grover_state(n, marked, k):
    N = 2**n
    s0 = np.ones(N) / np.sqrt(N)
    psi = s0.copy()
    for _ in range(k):
        psi[marked] *= -1
        psi = 2 * s0 * (s0 @ psi) - psi
    return psi


def grover_qg_z_prediction(n, marked, P):
    """qg_Z of qubit i after Grover with success probability P (one marked item)."""
    N = 2**n
    bits = [(marked >> (n - 1 - i)) & 1 for i in range(n)]
    return [float((-1) ** b * (N * P - 1) / (N - 1)) for b in bits]


def qft_matrix(n):
    w = np.exp(2j * np.pi / 2**n)
    return np.array([[w ** (j * k) for k in range(2**n)] for j in range(2**n)]) / np.sqrt(2**n)


def local_qg(psi, n, q):
    rho = np.outer(psi, psi.conj())
    lab = lambda c: "".join(c if j == q else "I" for j in range(n))  # noqa: E731
    return tuple(float(np.real(np.trace(rho @ pauli(lab(c))))) for c in "XYZ")


def kickback_control(phi):
    """Control in |+>, target an eigenstate of U with eigenvalue e^{i phi}: control's (qg_X, qg_Y)."""
    U = np.diag([np.exp(1j * phi), 1])
    CU = np.kron(np.diag([1, 0]), I2) + np.kron(np.diag([0, 1]), U)
    psi = CU @ np.kron(H @ np.array([1, 0]), np.array([1, 0]))
    rho = np.outer(psi, psi.conj())
    return float(np.real(np.trace(rho @ pauli("XI")))), float(np.real(np.trace(rho @ pauli("YI"))))


def counting_qg_x(n, marked):
    """Hadamard test of the Grover operator on the uniform state: control qg_X = 1 - 2M/N."""
    N = 2**n
    O = np.diag([-1.0 if i in marked else 1.0 for i in range(N)])
    s0 = np.ones(N) / np.sqrt(N)
    G = (2 * np.outer(s0, s0) - np.eye(N)) @ O
    CG = np.kron(np.diag([1, 0]), np.eye(N)) + np.kron(np.diag([0, 1]), G)
    psi = CG @ np.kron(H @ np.array([1, 0]), s0)
    rho = np.outer(psi, psi.conj())
    return float(np.real(np.trace(rho @ np.kron(X, np.eye(N)))))


def order_finding_output(r, m):
    """Ideal QPE output of order finding with period r on m counting qubits."""
    Q = 2**m
    p = np.zeros(Q)
    for s_ in range(r):
        amp = np.array([np.sum(np.exp(2j * np.pi * (s_ / r - y / Q) * np.arange(Q))) / Q for y in range(Q)])
        p += np.abs(amp) ** 2 / r
    return p


def purity_from_qg(rho, n):
    return sum(float(np.real(np.trace(rho @ pauli("".join(t))))) ** 2 for t in itertools.product("IXYZ", repeat=n)) / 2**n


def walk_register_mean(n, start, times):
    from scipy.linalg import expm

    def hop(j, c):
        return pauli("".join(c if k in (j, j + 1) else "I" for k in range(n)))

    Hw = sum((hop(j, "X") + hop(j, "Y")) / 2 for j in range(n - 1))
    psi = np.zeros(2**n, dtype=complex)
    psi[start] = 1
    out = []
    for t in times:
        ph = expm(-1j * Hw * t) @ psi
        out.append(float(np.mean([local_qg(ph, n, j)[2] for j in range(n)])))
    return out


# --------------------------------------------------------------------- #
# working with qg values directly
# --------------------------------------------------------------------- #
GATES_PARAMETRIC = {"Rx": rx, "Ry": ry, "Rz": rz, "P": phase}


def qg_values(state, n=None, tol=1e-12):
    """All qg values of a statevector or density matrix: {Pauli string: qg_P}
    (zero values omitted, identity included)."""
    s = np.asarray(state, dtype=complex)
    rho = np.outer(s, s.conj()) if s.ndim == 1 else s
    n = int(np.log2(rho.shape[0])) if n is None else n
    out = {}
    for t in itertools.product("IXYZ", repeat=n):
        t = "".join(t)
        v = float(np.real(np.trace(rho @ pauli(t))))
        if abs(v) > tol:
            out[t] = v
    return out


def state_from_qg(qg, n):
    """Density matrix rho = 2^-n sum_P qg_P P (inverse of qg_values)."""
    rho = np.zeros((2**n, 2**n), dtype=complex)
    for p, v in qg.items():
        rho += v * pauli(p)
    return rho / 2**n


def apply_gate(qg, U, tol=1e-12):
    """Transform a dict of qg values by gate U on all n qubits: qg'_P = qg_{U^dag P U}."""
    n = int(np.log2(U.shape[0]))
    out = {}
    for t in itertools.product("IXYZ", repeat=n):
        t = "".join(t)
        v = sum(c * qg.get(q, 0.0) for q, c in conjugate(U, t).items())
        if abs(v) > tol:
            out[t] = v
    return out


def gate_table(U):
    """The qg action of U on every single-qubit Pauli of every qubit:
    {local Pauli string: {string: coefficient}}."""
    n = int(np.log2(U.shape[0]))
    labels = ["".join(c if j == q else "I" for j in range(n)) for q in range(n) for c in "XYZ"]
    return {p: conjugate(U, p) for p in labels}


def is_clifford(U):
    """True if U maps every single-qubit Pauli to a single signed Pauli string."""
    return all(len(v) == 1 and abs(abs(next(iter(v.values()))) - 1) < 1e-9 for v in gate_table(U).values())


def born_p0(qg_z):
    """P(0) = (1 + qg_Z) / 2."""
    return (1.0 + qg_z) / 2.0


# --------------------------------------------------------------------- #
# readout rules
# --------------------------------------------------------------------- #
def deutsch_jozsa_is_constant(probs, n, tol=1e-9):
    """Deutsch-Jozsa: f is constant iff every local qg_Z is +1."""
    return all(q > 1 - tol for q in single_qg_z(probs, n))


def bernstein_vazirani_secret(probs, n):
    """Bernstein-Vazirani: bit i of the secret is the sign of qubit i's qg_Z."""
    return [0 if q > 0 else 1 for q in single_qg_z(probs, n)]


def simon_nonzero_parities(probs, n, tol=1e-9):
    """Simon: the Z-parities with qg != 0; ideally exactly {0, s}."""
    return [S for S in itertools.product([0, 1], repeat=n) if abs(z_parity(probs, n, S)) > tol]


def grover_marked_from_signs(probs, n):
    """Grover: the marked string read as the sign pattern of the local qg_Z (needs P > 1/N)."""
    return bernstein_vazirani_secret(probs, n)


def count_from_qg_x(qg_x, N):
    """Quantum counting from a single Hadamard test: M = N (1 - qg_X) / 2."""
    return N * (1.0 - qg_x) / 2.0


def hhl_readout(A, b, M=None, C=None):
    """Exact HHL readout for a Hermitian A: (ancilla qg_Z, <x|M|x>) with |x> ~ A^-1 |b>.
    The ancilla is flagged |1> on success with amplitude C / lambda (C = min |lambda|)."""
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    b = b / np.linalg.norm(b)
    lam, V = np.linalg.eigh(A)
    C = float(np.min(np.abs(lam))) if C is None else C
    beta = V.conj().T @ b
    p_success = float(np.sum(np.abs(beta) ** 2 * (C / lam) ** 2))
    x = np.linalg.solve(A, b)
    x = x / np.linalg.norm(x)
    M = np.eye(len(b)) if M is None else np.asarray(M)
    return 1.0 - 2.0 * p_success, float(np.real(x.conj() @ M @ x))


def energy_from_qg(hamiltonian, qg):
    """VQE: E = sum_P c_P qg_P for a Hamiltonian given as {Pauli string: c_P}."""
    return float(sum(c * qg.get(p, 1.0 if set(p) == {"I"} else 0.0) for p, c in hamiltonian.items()))


def maxcut_from_qg(edges, qg, n):
    """QAOA MaxCut: C = sum_(i,j) w_ij (1 - qg_{Z_i Z_j}) / 2, edges = [(i, j, w), ...]."""
    tot = 0.0
    for i, j, w in edges:
        lab = "".join("Z" if k in (i, j) else "I" for k in range(n))
        tot += w * (1.0 - qg.get(lab, 0.0)) / 2.0
    return tot


def product_kernel(qvecs_a, qvecs_b):
    """Fidelity kernel of two product states from their local qg vectors: prod_i (1 + q_i . q_i') / 2."""
    out = 1.0
    for a, b in zip(qvecs_a, qvecs_b):
        out *= (1.0 + float(np.dot(a, b))) / 2.0
    return out
