"""
Numerical checks for the qg formulation of the main gates and algorithms
(manuscript/teoria_es, "Formulación qg de las compuertas y los algoritmos
cuánticos principales").

Conventions. For an n-qubit state rho, qg_P = Tr(rho P) for every Pauli string
P (letters left to right = qubits 0, 1, ...; qubit 0 is the most significant
bit and the control of CX/CZ, the controls of Toffoli, the control of
Fredkin). A gate U maps the qg values as

    qg'_P = qg_{U^dag P U},

so each gate is fully described by U^dag P U for the single-qubit Paulis of
each qubit (``conjugate``). Clifford gates give a signed Pauli string (a signed
permutation of the qg values); non-Clifford gates give a combination.

Everything here is exact linear algebra on small statevectors (numpy only);
tests/test_qg_formulation_checks.py pins every identity quoted in the
document.
"""

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
    return [(-1) ** b * (N * P - 1) / (N - 1) for b in bits]


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
        out.append(np.mean([local_qg(ph, n, j)[2] for j in range(n)]))
    return out


def main():
    print("Gates: qg'_P = qg_{U^dag P U}")
    for name, U in GATES_1Q.items():
        print(f"  {name:7s}", {p: conjugate(U, p) for p in "XYZ"})
    for name, U in (("Rx(t)", rx(0.7)), ("Ry(t)", ry(0.7)), ("Rz(t)", rz(0.7))):
        print(f"  {name:7s} t=0.7", {p: conjugate(U, p) for p in "XYZ"})
    for name, U in GATES_MULTI.items():
        n = int(np.log2(U.shape[0]))
        labels = ["".join(c if j == q else "I" for j in range(n)) for q in range(n) for c in "XZ"]
        print(f"  {name:7s}", {p: conjugate(U, p) for p in labels}, "| conserves Hamming weight:", conserves_weight(U, n))


if __name__ == "__main__":
    main()
