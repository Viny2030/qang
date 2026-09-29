"""
Beyond diagonal symmetries: projecting on total spin S^2 after the N and
S_z checks.

§54-§55 checked the electron number N and the spin projections (N_up,
N_down), all diagonal in Z, so they cost nothing beyond the Z shots (or a few
ancillas for rotated terms). The ground states of H4 and H2O are singlets:
they also have S^2 = 0, a symmetry that is NOT diagonal in Z. An error that
keeps N_up and N_down but mixes singlet and triplet (for example a ZZ-type
error on two spin-orbitals of different spatial orbitals) passes every
diagonal check.

Here: the §55 ansatz (conserves N_up and N_down), 3 layers, optimised
classically, under the generic all-to-all noise model of §29; exact
density matrices of 8 qubits. Energies with ideal projections:
  N              total number (Z-diagonal)
  N_up, N_down   spin-resolved number (Z-diagonal, §55)
  + S^2 = 0      projection on the singlet inside that sector
Because H commutes with the projector P, Tr(P rho P H)/Tr(P rho) =
Tr(rho P H)/Tr(rho P): the S^2 projection can be done in post-processing
(symmetry expansion, Bonet-Monroig et al. 2018) by measuring P and PH. The
cost is the number of Pauli strings in P and in PH, reported below, and the
shot variance, which grows as 1/Tr(rho P)^2 for the ratio.

Prediction, written before the run: the S^2 projection removes at least a
further 10 % of the error left after the N_up, N_down check, and it costs at
least 10x more Pauli strings to measure than H itself.

Needs pyscf, openfermion, openfermionpyscf (as §54-§55).

Findings (python examples/spin_squared_check_qg.py):

  Error vs the noiseless circuit (mHa; kept fraction), 8 qubits, 96 CX:

              raw     N              N_up, N_down    + S^2 = 0       further share
    H2O       363.9   54.6 (0.67)    43.3 (0.66)     26.7 (0.64)     38 %
    H4        287.5   79.9 (0.69)    70.5 (0.67)     44.7 (0.58)     37 %

  Pauli strings to measure: H2O 105 (H) vs 640 (P) and 2064 (PH); H4 185 vs
  640 and 3456.

  * The prediction holds. After the diagonal checks, the singlet projection
    removes a further 37-38 % of the error, the largest single gain of any
    check since §52 (N_up, N_down added 12-21 % in §55). Noise that keeps
    N_up and N_down but mixes spin multiplets is a large part of what the Z
    checks cannot see: the noisy <S^2> is 0.40 (H2O) and 0.60 (H4) where the
    ideal state has 0.
  * The price is measurement, not shots: the kept fraction barely drops
    (0.66 -> 0.64 for H2O), but the post-processed projection needs P and
    PH, about 20x more Pauli strings than H (2064 vs 105).
  * H2O is the clean case: its ideal state is a singlet to 3e-6. The H4
    ansatz is itself spin-contaminated (singlet weight 0.91, 34 mHa above
    the exact ground state); projecting even the noiseless state lowers it
    by 13.7 mHa, so part of the H4 gain corrects the ansatz, not the noise.
  * Optimisation is run single-threaded: with threaded BLAS the L-BFGS
    optimum of the H4 ansatz changed between runs (different local minima).

  What is new and what is not. S^2 symmetry verification by post-processing
  is known (Bonet-Monroig et al. 2018; McArdle et al. 2019). Ours: how much
  it adds on top of the qg (Z-diagonal) checks for the same circuits, and
  its measurement cost in Pauli strings. Limitations: ideal projections
  (infinite shots), one noise model, 8 qubits, and the Pauli count is an
  upper bound (no grouping); a measured S^2 check with ancillas would need
  a controlled-S^2 circuit, not built here.
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # reproducible optimisation (threaded BLAS changes the local minimum)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import spin_checks_qg as SC  # noqa: E402
import symmetry_checks_scaling_qg as S  # noqa: E402


def s_squared_matrix(n):
    """S^2 in the spin-blocked Jordan-Wigner order used by S.molecule."""
    from openfermion import jordan_wigner, reorder, s_squared_operator

    fop = reorder(s_squared_operator(n // 2), lambda q, nm: S._blocked(q, nm))
    op = S._to_qiskit_plain(jordan_wigner(fop), n)
    return op.to_matrix()


def singlet_projector(n, nu, nd):
    """Projector on S^2 = 0 inside the (N_up = nu, N_down = nd) sector (dense)."""
    S2 = s_squared_matrix(n)
    up, dn = SC._spin_counts(n)
    idx = np.where((up == nu) & (dn == nd))[0]
    sub = S2[np.ix_(idx, idx)]
    w, v = np.linalg.eigh(sub)
    zero = np.abs(w) < 1e-8
    P = np.zeros((2**n, 2**n), dtype=complex)
    V = v[:, zero]
    P[np.ix_(idx, idx)] = V @ V.conj().T
    return P, S2


def pauli_count(mat, tol=1e-9):
    from qiskit.quantum_info import SparsePauliOp

    op = SparsePauliOp.from_operator(mat).simplify(atol=tol)
    return len(op)


def study(name, params=None, count_paulis=True):
    op, mat, n, ne, e_exact, e_hf = S.molecule(name)
    if params is None:
        params, _ = SC.optimise(name)
    psi = SC.statevector(params, n, ne)
    H = mat.toarray()
    e0 = float(np.real(np.vdot(psi, H @ psi)))
    rho, cx = S._noisy_rho(SC.circuit(params, n, ne))
    nu, nd = ne // 2, ne - ne // 2
    P, S2 = singlet_projector(n, nu, nd)
    up, dn = SC._spin_counts(n)
    out = {"cx": cx, "S2_ideal": float(np.real(np.vdot(psi, S2 @ psi))),
           "singlet_weight_ideal": float(np.real(np.vdot(psi, P @ psi))),
           "ansatz_vs_exact": 1e3 * (e0 - e_exact)}

    def proj_energy(Pm):
        r = Pm @ rho @ Pm
        k = float(np.real(np.trace(r)))
        return 1e3 * (float(np.real(np.trace(r @ H))) / k - e0), k

    ppsi = P @ psi
    out["ideal_projected"] = 1e3 * (float(np.real(np.vdot(ppsi, H @ ppsi) / np.vdot(ppsi, ppsi))) - e0)
    out["raw"] = 1e3 * (float(np.real(np.trace(rho @ H))) - e0)
    out["S2_noisy"] = float(np.real(np.trace(rho @ S2)))
    for key, mask in (("N", (up + dn) == ne), ("spin", (up == nu) & (dn == nd))):
        out[key], out["kept_" + key] = proj_energy(np.diag(mask.astype(float)))
    out["S2"], out["kept_S2"] = proj_energy(P)
    if count_paulis:
        out["paulis_H"] = len(op.simplify())
        out["paulis_P"] = pauli_count(P)
        out["paulis_PH"] = pauli_count(P @ H)
    return out


def main():
    for name in ("H4", "H2O"):
        r = study(name)
        print(f"{name}: {r['cx']} CX; ideal state <S^2> = {r['S2_ideal']:.2e} (singlet weight {r['singlet_weight_ideal']:.6f});"
              f" noisy <S^2> = {r['S2_noisy']:.3f}")
        print(f"  error vs noiseless circuit (mHa, kept fraction): raw {r['raw']:.1f} | N {r['N']:.1f} ({r['kept_N']:.2f})"
              f" | N_up,N_down {r['spin']:.1f} ({r['kept_spin']:.2f}) | + S^2=0 {r['S2']:.1f} ({r['kept_S2']:.2f})")
        print(f"  further share removed by S^2 after the spin check: {1 - r['S2'] / r['spin']:.1%}"
              f" | projecting the NOISELESS state shifts it by {r['ideal_projected']:+.1f} mHa;"
              f" ansatz vs exact ground state {r['ansatz_vs_exact']:.1f} mHa")
        print(f"  Pauli strings to measure: H {r['paulis_H']}, P {r['paulis_P']}, PH {r['paulis_PH']}")


if __name__ == "__main__":
    main()
