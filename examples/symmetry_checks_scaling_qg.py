"""
Do the electron-number checks of §52 scale beyond LiH?

§52 showed that a parity ancilla plus an N mod 4 ancilla reach every
measurement group of LiH (6 qubits, N = 2) and cut the hardware error
6-8x, close to the full number-projection ceiling. Here the same checks
on larger, strongly correlated molecules (STO-3G, Jordan-Wigner, built
with PySCF + OpenFermion at run time):

    H4 chain, 1.5 A        8 qubits, N = 4   (185 Pauli terms)
    H2O, 4e in 4 orbitals   8 qubits, N = 4   (105 terms)
    H6 chain, 1.5 A        12 qubits, N = 6   (919 terms)

Ansatz: the §21/§52 number-conserving brick (XX+YY rotation and a
controlled phase on each neighbouring pair), 3 layers, parameters
optimised classically on the statevector, with the spin-orbitals ordered
spin-blocked (all up, then all down) so that neighbouring qubits share a
spin (with OpenFermion's interleaved order the brick cannot leave
Hartree-Fock). Noise: the generic all-to-all
model of §29. Errors are against the noiseless energy of the same circuit.

  A. Exact density matrix of the noisy circuit, ideal projections onto
     even parity, N mod 4 and N (the ceiling). For N = 4 in 8 qubits the
     mod-4 check accepts N = 0, 4, 8; for N = 6 in 12 qubits it accepts
     N = 2, 6, 10.
  B. (8-qubit molecules) The checks built with ancillas: parity (8 CNOTs)
     and N mod 4 (Hadamard test of prod_j S_j, 8 controlled phases), exact
     outcome probabilities of the noisy 10-qubit circuits, no readout
     error, post-selection in every measurement group.

Findings (python examples/symmetry_checks_scaling_qg.py):

  Error vs the noiseless circuit (mHa; kept fraction in brackets). LiH
  (§52, 60 CX) for comparison:

                     CX   no check  parity        N mod 4       N (ceiling)
    LiH  6 q  N=2    60    76.1     49.4 (0.83)    6.9 (0.77)    6.2 (0.77)
    H4   8 q  N=4    84   238.5    146.4 (0.77)   72.2 (0.72)   70.5 (0.72)
    H2O  8 q  N=4    84   292.2    187.5 (0.77)   48.8 (0.71)   44.0 (0.71)
    H6  12 q  N=6   132   407.6    279.9 (0.69)  143.8 (0.60)  135.2 (0.59)

    with ancillas (B):  raw    parity         parity + mod 4
    H4                  272.8  183.8 (0.76)   100.0 (0.67)
    H2O                 324.5  220.8 (0.76)    77.0 (0.66)

  * The checks keep working, with a smaller factor. The N mod 4 check is
    within 2-10 % of the full number-projection ceiling in every case, so
    two ancillas suffice up to 12 qubits; parity alone recovers 35-40 %.
  * What shrinks is the ceiling itself: the share of the error that
    changes N falls from 92 % (LiH, 60 CX) to 70-85 % (8 qubits, 84 CX)
    and 67 % (H6, 132 CX). The rest is number-conserving (two-qubit
    errors on a hopping pair, dephasing) and no number check can see it.
    With ancillas the cut is 2.7x (H4) and 4.2x (H2O), against 6x for
    LiH.
  * Against Hartree-Fock: the noiseless H4 circuit is 50.8 mHa above the
    exact energy (HF 167.0); with ideal mod-4 checks the noisy estimate
    is about 123 mHa above exact and with the ancillas about 151, both
    below HF, because stretched hydrogen chains are where mean field fails.
    H6: 88 + 144 = 232 vs HF 245 (ideal checks only). H2O at equilibrium
    (HF 7.4 mHa above exact, and the ansatz does not improve on it) is far
    from HF in every case.

Scaling reading. The electron-number checks do not stop working as the
molecule grows; they catch a shrinking share of the error because the
number-conserving errors grow with depth. Beyond ~100 two-qubit gates
they must be combined with a method for in-sector errors (ZNE, §48, or
other symmetries such as S_z and S^2).

Honest scope. One generic noise model, one ansatz family, 3 layers; the
12-qubit case with ideal projections only (the 14-qubit ancilla circuits
were not simulated). The checks are standard symmetry verification; the
content is how their reach and ceiling scale.
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LAYERS = 3


# --------------------------------------------------------------------- #
# molecules
# --------------------------------------------------------------------- #
def _to_qiskit_plain(qop, n):
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = [], []
    for term, c in qop.terms.items():
        lab = ["I"] * n
        for q, p in term:
            lab[n - 1 - q] = p
        labels.append("".join(lab))
        coeffs.append(complex(c).real)
    return SparsePauliOp(labels, coeffs).simplify()


def _blocked(q, n):
    """OpenFermion interleaves spins (2i = up, 2i+1 = down); put all up
    spin-orbitals first so that neighbouring qubits carry the same spin and
    the nearest-neighbour ansatz can move electrons within a spin sector."""
    return q // 2 + (n // 2) * (q % 2)


MOLECULES = {
    "H4": dict(geometry=[("H", (0, 0, 1.5 * i)) for i in range(4)], occ=None, act=None),
    "H2O": dict(geometry=[("O", (0, 0, 0)), ("H", (0.757, 0.586, 0)), ("H", (-0.757, 0.586, 0))],
                occ=[0, 1, 2], act=[3, 4, 5, 6]),
    "H6": dict(geometry=[("H", (0, 0, 1.5 * i)) for i in range(6)], occ=None, act=None),
}
_CACHE = {}


def molecule(name):
    """(H as SparsePauliOp, sparse matrix, n qubits, N electrons, E_exact, E_HF)."""
    if name in _CACHE:
        return _CACHE[name]
    from openfermion import MolecularData, get_fermion_operator, jordan_wigner
    from openfermionpyscf import run_pyscf

    spec = MOLECULES[name]
    m = run_pyscf(MolecularData(spec["geometry"], "sto-3g", 1, 0), run_scf=True, run_fci=False)
    h = m.get_molecular_hamiltonian(occupied_indices=spec["occ"], active_indices=spec["act"])
    n = 2 * (len(spec["act"]) if spec["act"] else m.n_orbitals)
    ne = m.n_electrons - 2 * (len(spec["occ"]) if spec["occ"] else 0)
    from openfermion import reorder

    fop = reorder(get_fermion_operator(h), lambda q, nm: _blocked(q, nm))
    op = _to_qiskit_plain(jordan_wigner(fop), n)
    mat = op.to_matrix(sparse=True).tocsr()
    w = np.array([bin(i).count("1") for i in range(2**n)])
    sector = np.where(w == ne)[0]
    sub = mat[sector][:, sector]
    e_exact = float(eigsh(sub, k=1, which="SA")[0][0].real)
    hf_index = sum(1 << q for q in hf_qubits(n, ne))
    e_hf = float(mat[hf_index, hf_index].real)
    _CACHE[name] = (op, mat, n, ne, e_exact, e_hf)
    return _CACHE[name]


def hf_qubits(n, ne):
    """Occupied qubits of the Hartree-Fock state in the spin-blocked order."""
    return [i for i in range(ne // 2)] + [n // 2 + i for i in range(ne - ne // 2)]


# --------------------------------------------------------------------- #
# ansatz: fast numpy statevector for the optimisation, Qiskit circuit for noise
# --------------------------------------------------------------------- #
def _pairs(n, layers=LAYERS):
    out = []
    for _ in range(layers):
        for start in (0, 1):
            for a in range(start, n - 1, 2):
                out.append(a)
    return out


def _xxyy(theta, beta):
    from qiskit.circuit.library import XXPlusYYGate

    return XXPlusYYGate(theta, beta).to_matrix()


def _cp(phi):
    return np.diag([1, 1, 1, np.exp(1j * phi)])


def _apply(psi, U, a, n):
    """Two-qubit U on qubits (a, a+1), little-endian (qubit q is axis n-1-q)."""
    t = psi.reshape([2] * n)
    ax_lo, ax_hi = n - 1 - a, n - 2 - a  # qubit a, qubit a+1
    t = np.moveaxis(t, [ax_hi, ax_lo], [0, 1]).reshape(4, -1)
    t = U @ t
    t = np.moveaxis(t.reshape([2, 2] + [2] * (n - 2)), [0, 1], [ax_hi, ax_lo])
    return t.reshape(-1)


def statevector(params, n, ne, layers=LAYERS):
    psi = np.zeros(2**n, dtype=complex)
    psi[sum(1 << q for q in hf_qubits(n, ne))] = 1.0
    k = 0
    for a in _pairs(n, layers):
        psi = _apply(psi, _xxyy(params[k], params[k + 1]), a, n)
        psi = _apply(psi, _cp(params[k + 2]), a, n)
        k += 3
    return psi


def circuit(params, n, ne, layers=LAYERS):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    qc = QuantumCircuit(n)
    for q in hf_qubits(n, ne):
        qc.x(q)
    k = 0
    for a in _pairs(n, layers):
        qc.append(XXPlusYYGate(params[k], params[k + 1]), [a, a + 1])
        qc.cp(params[k + 2], a, a + 1)
        k += 3
    return qc


def optimise(name, layers=LAYERS, seed=0, scales=(0.1, 0.6)):
    op, mat, n, ne, e_exact, e_hf = molecule(name)
    npar = 3 * len(_pairs(n, layers))
    best = None
    rng = np.random.default_rng(seed)
    maxiter = 400 if n <= 8 else 120
    for scale in scales:
        x0 = scale * rng.standard_normal(npar)
        f = lambda x: float(np.real(np.vdot(statevector(x, n, ne, layers), mat @ statevector(x, n, ne, layers))))  # noqa: E731
        res = minimize(f, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)


# --------------------------------------------------------------------- #
# A. ideal projections on the noisy density matrix
# --------------------------------------------------------------------- #
def _noisy_rho(qc):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    import hubbard_trotter_qg_filters as HB

    tc = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
    tc.save_density_matrix()
    res = AerSimulator(method="density_matrix", noise_model=HB.all_to_all_noise_model()).run(tc).result()
    return np.asarray(res.data(0)["density_matrix"]), tc.count_ops().get("cx", 0)


def _energy(mat, rho):
    coo = mat.tocoo()
    return float(np.real(np.sum(coo.data * rho[coo.col, coo.row])))


def level_a(name, params):
    op, mat, n, ne, e_exact, e_hf = molecule(name)
    psi = statevector(params, n, ne)
    e0 = float(np.real(np.vdot(psi, mat @ psi)))
    rho, cx = _noisy_rho(circuit(params, n, ne))
    w = np.array([bin(i).count("1") for i in range(2**n)])
    out = {"cx": cx, "raw": 1e3 * (_energy(mat, rho) - e0)}
    for key, mask in (("parity", (w % 2) == (ne % 2)), ("mod4", (w % 4) == (ne % 4)), ("number", w == ne)):
        d = mask.astype(float)
        r = rho * np.outer(d, d)
        k = float(np.real(np.trace(r)))
        out[key] = 1e3 * (_energy(mat, r / k) - e0)
        out[f"kept_{key}"] = k
    out.update({"ansatz_vs_exact": 1e3 * (e0 - e_exact), "hf_vs_exact": 1e3 * (e_hf - e_exact)})
    return out


# --------------------------------------------------------------------- #
# B. ancilla checks (exact probabilities of the noisy circuits)
# --------------------------------------------------------------------- #
def level_b(name, params):
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    import filter_scaling_lih_qg as F
    import hubbard_trotter_qg_filters as HB

    op, mat, n, ne, _, _ = molecule(name)
    groups, bases = F._groups(op, n)
    circs, ideal = [], []
    for b in bases:
        qc = QuantumCircuit(n + 2)
        qc.compose(circuit(params, n, ne), qubits=range(n), inplace=True)
        for q in range(n):
            qc.cx(q, n)
        qc.h(n + 1)
        for q in range(n):
            qc.cp(math.pi / 2, n + 1, q)
        qc.h(n + 1)
        for q, ch in enumerate(b):
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.sdg(q)
                qc.h(q)
        circs.append(qc)
    tc = transpile(circs, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
    for c in tc:
        c.save_probabilities()
    res = AerSimulator(method="density_matrix", noise_model=HB.all_to_all_noise_model()).run(tc).result()
    idx = np.arange(2 ** (n + 2))
    a1, a2 = (idx >> n) & 1, (idx >> (n + 1)) & 1
    sysidx = idx & (2**n - 1)
    masks = {"raw": np.ones_like(idx, bool), "parity": a1 == 0,
             "parity_mod4": (a1 == 0) & (a2 == ((ne // 2) % 2))}
    psi = statevector(params, n, ne)
    e0 = float(np.real(np.vdot(psi, mat @ psi)))
    tot = {k: 0.0 for k in masks}
    kept = {k: [] for k in masks}
    for i, g in enumerate(groups):
        p = np.asarray(res.data(i)["probabilities"])
        for k, m in masks.items():
            s = np.bincount(sysidx, weights=p * m, minlength=2**n)
            kept[k].append(s.sum())
            tot[k] += F._group_energy(g, s / s.sum(), n)
    out = {k: 1e3 * (v - e0) for k, v in tot.items()}
    out.update({f"kept_{k}": float(np.mean(v)) for k, v in kept.items()})
    out["groups"] = len(groups)
    return out


def make_figure(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rows)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    keys = [("raw", "#8c8c8c", "no check"), ("parity", "#1f6fb2", "parity"), ("mod4", "#e0a030", "N mod 4"),
            ("number", "#8c2d04", "N projection (ceiling)")]
    for i, (k, c, lab) in enumerate(keys):
        ax.bar(x + (i - 1.5) * 0.2, [abs(rows[m][k]) for m in names], 0.2, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{rows[m]['n']} qubits, {rows[m]['cx']} CX" for m in names], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("|energy error| vs noiseless circuit (mHa)")
    ax.set_title("Ideal symmetry checks on larger molecules (all-to-all noise)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import time

    rows = {}
    for name in ("H4", "H2O", "H6"):
        t = time.time()
        params, e = optimise(name)
        a = level_a(name, params)
        a["n"] = molecule(name)[2]
        rows[name] = a
        print(f"{name}: n = {a['n']}, CX {a['cx']}, ansatz {a['ansatz_vs_exact']:.1f} mHa above exact "
              f"(HF {a['hf_vs_exact']:.1f}); A: raw {a['raw']:.1f}  parity {a['parity']:.1f} ({a['kept_parity']:.3f})"
              f"  mod4 {a['mod4']:.1f} ({a['kept_mod4']:.3f})  number {a['number']:.1f} ({a['kept_number']:.3f})"
              f"   [{time.time() - t:.0f} s]")
        if a["n"] <= 8:
            b = level_b(name, params)
            print(f"     B (ancillas, {b['groups']} groups): raw {b['raw']:.1f}  parity {b['parity']:.1f} "
                  f"({b['kept_parity']:.3f})  parity+mod4 {b['parity_mod4']:.1f} ({b['kept_parity_mod4']:.3f})")
            rows[name]["b"] = b
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), rows)
