"""
LiH Hamiltonian with the pi orbitals as m_l = +1 / -1 complex orbitals (§153).

Writes examples/data/lih_lambda_jw.json. Needs PySCF and OpenFermion, which do
not install on Windows; run it in Docker:

    docker run --rm -v "$PWD":/w -w /w python:3.11-slim \
        sh -c "pip install -q pyscf openfermion && python tools/make_lih_lambda_hamiltonian.py"

LiH along z, STO-3G, Li 1s frozen. Active space: the 5 remaining spatial
orbitals, 3 sigma (m_l = 0) and the two pi orbitals. In STO-3G the pi MOs are
the bare Li 2px and 2py AOs (H has no p functions); they are replaced by
pi(+/-) = (px +/- i py)/sqrt(2), so that L_z, and with it
Lambda = sum_i m_l(i) n_i, is diagonal in the computational basis.

Qubit order, spin-blocked (as in examples/spin_checks_qg.py):
q0..q4 = sigma0, sigma1, sigma2, pi+, pi- spin up; q5..q9 the same, spin down.
Jordan-Wigner, little-endian (qubit 0 is the rightmost character of a label).
"""

import json
import os

import numpy as np
from openfermion import FermionOperator, jordan_wigner, get_sparse_operator
from pyscf import ao2mo, fci, gto, mcscf, scf

R = 1.6  # Angstrom, near the STO-3G equilibrium
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "examples", "data", "lih_lambda_jw.json")

mol = gto.M(atom=f"Li 0 0 0; H 0 0 {R}", basis="sto-3g", unit="Angstrom", verbose=0)
mf = scf.RHF(mol).run()

labels = mol.ao_labels()
ipx = [i for i, l in enumerate(labels) if "Li 2px" in l][0]
ipy = [i for i, l in enumerate(labels) if "Li 2py" in l][0]
C = mf.mo_coeff
pi_cols = [k for k in range(C.shape[1]) if abs(C[ipx, k]) + abs(C[ipy, k]) > 1e-6]
assert len(pi_cols) == 2, pi_cols
sigma_cols = [k for k in range(1, C.shape[1]) if k not in pi_cols]  # MO 0 = Li 1s, frozen
assert len(sigma_cols) == 3

# Real pi MOs aligned with px and py (any rotation of the degenerate pair is allowed).
S = mol.intor("int1e_ovlp")
px = np.zeros(C.shape[0]); px[ipx] = 1 / np.sqrt(S[ipx, ipx])
py = np.zeros(C.shape[0]); py[ipy] = 1 / np.sqrt(S[ipy, ipy])
Cact = np.column_stack([C[:, sigma_cols], px, py])
Cfull = np.column_stack([C[:, 0], Cact])  # CASCI slices off the frozen core itself
assert np.allclose(Cfull.T @ S @ Cfull, np.eye(6), atol=1e-8)

mc = mcscf.CASCI(mf, 5, 2)
h1_real, ecore = mc.get_h1eff(Cfull)
h2_real = ao2mo.restore(1, mc.get_h2eff(Cfull), 5)

e_casci_real = fci.direct_spin1.kernel(h1_real, h2_real, 5, (1, 1))[0] + ecore

# Real -> complex orbitals: columns are the new orbitals in the old basis.
U = np.eye(5, dtype=complex)
U[3:, 3:] = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)  # pi+ = (px + i py)/sqrt2, pi- = (px - i py)/sqrt2
h1 = U.conj().T @ h1_real @ U
h2 = np.einsum("ap,bq,cr,ds,abcd->pqrs", U.conj(), U, U.conj(), U, h2_real)  # chemist (pq|rs)
M_L = [0, 0, 0, +1, -1]

H = FermionOperator((), ecore)
so = lambda p, s: p + 5 * s  # noqa: E731
for s in (0, 1):
    for p in range(5):
        for q in range(5):
            if abs(h1[p, q]) > 1e-12:
                H += FermionOperator(((so(p, s), 1), (so(q, s), 0)), h1[p, q])
for s in (0, 1):
    for t in (0, 1):
        for p in range(5):
            for q in range(5):
                for r in range(5):
                    for u in range(5):
                        v = h2[p, q, r, u]
                        if abs(v) > 1e-12:
                            H += FermionOperator(
                                ((so(p, s), 1), (so(r, t), 1), (so(u, t), 0), (so(q, s), 0)), 0.5 * v
                            )

Q = jordan_wigner(H)
Q.compress(1e-10)
terms = {}
for ops, c in Q.terms.items():
    assert abs(c.imag) < 1e-9, (ops, c)
    lab = ["I"] * 10
    for q, P in ops:
        lab[9 - q] = P
    terms["".join(lab)] = float(c.real)

# Checks: same CASCI energy as the real basis; Lambda, N and S_z commute with H.
Hs = get_sparse_operator(Q, n_qubits=10).toarray()
n = lambda q: np.array([(b >> (9 - q)) & 1 for b in range(1024)])  # noqa: E731  (OpenFermion: qubit 0 = most significant bit)
N = sum(n(q) for q in range(10))
Sz = sum(n(q) for q in range(5)) - sum(n(q) for q in range(5, 10))
Lam = sum(M_L[p] * (n(p) + n(p + 5)) for p in range(5))
for name, d in (("N", N), ("S_z", Sz), ("Lambda", Lam)):
    comm = Hs * d[None, :] - d[:, None] * Hs
    assert np.abs(comm).max() < 1e-9, name
idx = np.where((N == 2) & (Sz == 0))[0]
w, v = np.linalg.eigh(Hs[np.ix_(idx, idx)])
e0 = w[0]
assert abs(e0 - e_casci_real) < 1e-8, (e0, e_casci_real)
gs = v[:, 0]
lam_gs = float(np.sum(np.abs(gs) ** 2 * Lam[idx]))
pi_occ = float(np.sum(np.abs(gs) ** 2 * sum(n(q)[idx] for q in (3, 4, 8, 9))))
sector = {}
for lam in (-2, -1, 0, 1, 2):
    sel = np.where((N == 2) & (Sz == 0) & (Lam == lam))[0]
    if len(sel):
        sector[str(lam)] = float(np.linalg.eigvalsh(Hs[np.ix_(sel, sel)])[0])

data = {
    "geometry": f"Li-H {R} Angstrom along z, STO-3G, frozen Li 1s, active 2 electrons in 5 spatial orbitals "
    "(3 sigma, pi+, pi-), pi orbitals complex (m_l = +1, -1), Jordan-Wigner, spin-blocked",
    "qubit_order": "q0..q4 = sigma0, sigma1, sigma2, pi+, pi- (up); q5..q9 the same (down)",
    "m_l": M_L + M_L,
    "e_hf": float(mf.e_tot),
    "fci_active": float(e0),
    "fci_lambda_sectors": sector,
    "ground_lambda": lam_gs,
    "ground_pi_occupation": pi_occ,
    "terms": terms,
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=1)
print(f"{len(terms)} Pauli terms; HF {mf.e_tot:.6f}; CASCI {e0:.6f}; <Lambda> {lam_gs:.2e}; "
      f"pi occupation {pi_occ:.4f}; lowest energy per Lambda sector {sector}")
