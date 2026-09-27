"""
Spin-resolved number checks: does checking N_up and N_down separately
catch the errors that the total-number checks of §54 cannot see?

§54 found that the share of the error a total electron-number check can
remove shrinks with circuit size (92 % for LiH at 60 two-qubit gates, 67 %
for H6 at 132): the rest conserves N. But the Hamiltonian conserves more:
N_up and N_down separately (S_z and N). An error that moves an electron
from an up to a down spin-orbital keeps N and breaks S_z. In qg terms the
witness splits in two, one register-mean qg per spin block (as the spin
filters of §28 did for Hubbard).

The ansatz must respect the symmetry being checked, so §54's brick is
replaced by one that conserves N_up and N_down: XX+YY rotations and
controlled phases on neighbouring qubits inside each spin block (spin-
blocked order: all up spin-orbitals, then all down), plus a controlled
phase between the up and down spin-orbital of each spatial orbital (the
only coupling between the blocks, an on-site-like correlator). 3 layers,
optimised classically. Noise: the generic all-to-all model of §29.
Errors against the noiseless energy of the same circuit.

  A. Ideal projections on the noisy density matrix:
       N           total number (the §54 ceiling)
       N mod 4     §52/§54 check
       spin par.   parities of N_up and N_down (two ancillas)
       spin        N_up and N_down exactly (for these sizes the per-spin
                   N mod 4 check, four ancillas, is already exact)
  B. (8-qubit molecules) spin checks with ancillas: per spin block a
     parity ancilla and an N mod 4 ancilla (Hadamard test of prod S_j over
     the block), 4 ancillas and 16 extra two-qubit gates, the same gate
     count as the total parity + mod 4 check of §54 (2 ancillas, 16 gates);
     exact probabilities of the noisy 12-qubit circuits.

Findings (python examples/spin_checks_qg.py):

  The ansatz keeps the whole ideal state in the (N_up, N_down) sector
  (weight 1.000000). Error vs the noiseless circuit (mHa; kept fraction):

                    CX   no check  N (§54 ceiling)  N mod 4  spin parities  N_up and N_down
    H4   8 q, N=4   96    285.9     75.7 (0.68)      78.0    135.1 (0.71)    66.8 (0.67)
    H2O  8 q, N=4   96    366.4     57.8 (0.67)      63.1    173.1 (0.71)    46.2 (0.66)
    H6  12 q, N=6  156    404.3    113.9 (0.54)     128.8    222.8 (0.60)    90.4 (0.51)

  With ancillas (exact noisy probabilities, +24 two-qubit gates either way):

                    total parity + mod 4 (2 anc.)   spin parity + mod 4 per block (4 anc.)
    H4              106.4 (kept 0.64)               89.3 (kept 0.62)
    H2O              95.4 (kept 0.63)               67.5 (kept 0.61)

  * Checking N_up and N_down separately removes a further 12-21 % of the
    error beyond the total-number ceiling (ideal projections), and 16-29 %
    with real ancillas, at the same two-qubit-gate cost and 1-2 points
    more shots discarded. The share of the error a symmetry check can
    remove on H6 rises from 72 % (N) to 78 % (N_up, N_down).
  * The gain is real but modest: most of the residual error is both
    number- and spin-conserving (for example ZZ-type two-qubit errors),
    which no check of these symmetries sees.
  * Parities alone are weak here too (135-223 mHa): the per-spin mod-4
    ancillas carry the benefit, as for the total number (§52).

Honest scope. One noise model, one ansatz family (built to conserve N_up
and N_down, so the numbers are not directly comparable with §54's
S_z-breaking ansatz), ideal projections for the 12-qubit case. Spin and
number checks are standard symmetry verification; the content is how
much a second conserved quantity adds.
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import symmetry_checks_scaling_qg as S  # noqa: E402

LAYERS = 3


def ops_list(n, layers=LAYERS):
    """[(kind, a, b)] with kind 'xy' (XX+YY, 2 params) or 'cp' (1 param)."""
    h = n // 2
    ops = []
    for _ in range(layers):
        for block in (0, h):
            for start in (0, 1):
                for a in range(block + start, block + h - 1, 2):
                    ops.append(("xy", a, a + 1))
                    ops.append(("cp", a, a + 1))
        for i in range(h):
            ops.append(("cp", i, h + i))
    return ops


def n_params(n, layers=LAYERS):
    return sum(2 if k == "xy" else 1 for k, _, _ in ops_list(n, layers))


def _apply2(psi, U, a, b, n):
    """Two-qubit U (little-endian: qubit a least significant) on qubits a, b."""
    t = psi.reshape([2] * n)
    ax_a, ax_b = n - 1 - a, n - 1 - b
    t = np.moveaxis(t, [ax_b, ax_a], [0, 1]).reshape(4, -1)
    t = U @ t
    t = np.moveaxis(t.reshape([2, 2] + [2] * (n - 2)), [0, 1], [ax_b, ax_a])
    return t.reshape(-1)


def statevector(params, n, ne, layers=LAYERS):
    psi = np.zeros(2**n, dtype=complex)
    psi[sum(1 << q for q in S.hf_qubits(n, ne))] = 1.0
    k = 0
    for kind, a, b in ops_list(n, layers):
        if kind == "xy":
            psi = _apply2(psi, S._xxyy(params[k], params[k + 1]), a, b, n)
            k += 2
        else:
            psi = _apply2(psi, S._cp(params[k]), a, b, n)
            k += 1
    return psi


def circuit(params, n, ne, layers=LAYERS):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    qc = QuantumCircuit(n)
    for q in S.hf_qubits(n, ne):
        qc.x(q)
    k = 0
    for kind, a, b in ops_list(n, layers):
        if kind == "xy":
            qc.append(XXPlusYYGate(params[k], params[k + 1]), [a, b])
            k += 2
        else:
            qc.cp(params[k], a, b)
            k += 1
    return qc


def optimise(name, layers=LAYERS, seed=0, scales=(0.1, 0.6)):
    op, mat, n, ne, e_exact, e_hf = S.molecule(name)
    rng = np.random.default_rng(seed)
    best = None
    maxiter = 400 if n <= 8 else 120

    def f(x):
        psi = statevector(x, n, ne, layers)
        return float(np.real(np.vdot(psi, mat @ psi)))

    for sc in scales:
        res = minimize(f, sc * rng.standard_normal(n_params(n, layers)), method="L-BFGS-B",
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)


def _spin_counts(n):
    idx = np.arange(2**n)
    h = n // 2
    up = np.array([bin(i & ((1 << h) - 1)).count("1") for i in idx])
    dn = np.array([bin(i >> h).count("1") for i in idx])
    return up, dn


def level_a(name, params):
    op, mat, n, ne, e_exact, e_hf = S.molecule(name)
    psi = statevector(params, n, ne)
    e0 = float(np.real(np.vdot(psi, mat @ psi)))
    rho, cx = S._noisy_rho(circuit(params, n, ne))
    up, dn = _spin_counts(n)
    w = up + dn
    nu, nd = ne // 2, ne - ne // 2
    masks = {"N": w == ne, "N_mod4": (w % 4) == (ne % 4),
             "spin_parity": ((up % 2) == (nu % 2)) & ((dn % 2) == (nd % 2)),
             "spin": (up == nu) & (dn == nd)}
    out = {"cx": cx, "raw": 1e3 * (S._energy(mat, rho) - e0), "ansatz_vs_exact": 1e3 * (e0 - e_exact),
           "hf_vs_exact": 1e3 * (e_hf - e_exact), "ideal_in_spin_sector": float(np.sum(np.abs(psi[masks["spin"]]) ** 2))}
    for key, m in masks.items():
        d = m.astype(float)
        r = rho * np.outer(d, d)
        k = float(np.real(np.trace(r)))
        out[key] = 1e3 * (S._energy(mat, r / k) - e0)
        out[f"kept_{key}"] = k
    return out


def level_b(name, params):
    """Spin checks on 4 ancillas vs total checks on 2 ancillas, exact noisy probabilities."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    import filter_scaling_lih_qg as F
    import hubbard_trotter_qg_filters as HB

    op, mat, n, ne, _, _ = S.molecule(name)
    h = n // 2
    nu, nd = ne // 2, ne - ne // 2
    groups, bases = F._groups(op, n)

    def rotate(qc, b):
        for q, ch in enumerate(b):
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.sdg(q)
                qc.h(q)

    def build(b, spin):
        na = 4 if spin else 2
        qc = QuantumCircuit(n + na)
        qc.compose(circuit(params, n, ne), qubits=range(n), inplace=True)
        blocks = [range(0, h), range(h, n)] if spin else [range(0, n)]
        for j, blk in enumerate(blocks):
            par, m4 = n + 2 * j, n + 2 * j + 1
            for q in blk:
                qc.cx(q, par)
            qc.h(m4)
            for q in blk:
                qc.cp(math.pi / 2, m4, q)
            qc.h(m4)
        rotate(qc, b)
        return qc

    psi = statevector(params, n, ne)
    e0 = float(np.real(np.vdot(psi, mat @ psi)))
    out = {}
    for spin in (False, True):
        na = 4 if spin else 2
        tc = transpile([build(b, spin) for b in bases], basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                       seed_transpiler=1)
        for c in tc:
            c.save_probabilities()
        res = AerSimulator(method="density_matrix", noise_model=HB.all_to_all_noise_model()).run(tc).result()
        idx = np.arange(2 ** (n + na))
        sysidx = idx & (2**n - 1)
        bit = lambda k: (idx >> (n + k)) & 1  # noqa: E731
        if spin:
            # the Hadamard test reads N_s mod 4 only on the even sector; for odd N_s keep parity only
            accept = (bit(0) == nu % 2) & (bit(2) == nd % 2)
            if nu % 2 == 0:
                accept &= bit(1) == (nu // 2) % 2
            if nd % 2 == 0:
                accept &= bit(3) == (nd // 2) % 2
        else:
            accept = (bit(0) == ne % 2) & (bit(1) == (ne // 2) % 2)
        tot, kept = 0.0, []
        for i, g in enumerate(groups):
            p = np.asarray(res.data(i)["probabilities"])
            s = np.bincount(sysidx, weights=p * accept, minlength=2**n)
            kept.append(s.sum())
            tot += F._group_energy(g, s / s.sum(), n)
        key = "spin_checks" if spin else "total_checks"
        out[key] = 1e3 * (tot - e0)
        out[f"kept_{key}"] = float(np.mean(kept))
        out[f"extra_2q_{key}"] = tc[0].count_ops().get("cx", 0) - S._noisy_rho(circuit(params, n, ne))[1]
    return out


def make_figure(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rows)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    keys = [("raw", "#8c8c8c", "no check"), ("N", "#1f6fb2", "total N (ceiling of §54)"),
            ("spin_parity", "#e0a030", "N_up, N_down parities"), ("spin", "#8c2d04", "N_up and N_down")]
    for i, (k, c, lab) in enumerate(keys):
        ax.bar(x + (i - 1.5) * 0.2, [abs(rows[m][k]) for m in names], 0.2, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{rows[m]['n']} qubits, {rows[m]['cx']} CX" for m in names], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("|energy error| vs noiseless circuit (mHa)")
    ax.set_title("Spin-resolved checks vs total-number checks (ideal projections)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import time

    rows = {}
    for name in ("H4", "H2O", "H6"):
        t = time.time()
        params, _ = optimise(name)
        a = level_a(name, params)
        a["n"] = S.molecule(name)[2]
        rows[name] = a
        print(f"{name}: n {a['n']}, CX {a['cx']}, ansatz {a['ansatz_vs_exact']:.1f} mHa above exact (HF "
              f"{a['hf_vs_exact']:.1f}), ideal weight in spin sector {a['ideal_in_spin_sector']:.6f}")
        print(f"   A: raw {a['raw']:.1f} | N {a['N']:.1f} ({a['kept_N']:.3f})  N mod4 {a['N_mod4']:.1f}"
              f"  spin parities {a['spin_parity']:.1f} ({a['kept_spin_parity']:.3f})  spin {a['spin']:.1f}"
              f" ({a['kept_spin']:.3f})   [{time.time() - t:.0f} s]")
        if a["n"] <= 8:
            b = level_b(name, params)
            rows[name]["b"] = b
            print(f"   B: total checks {b['total_checks']:.1f} ({b['kept_total_checks']:.3f}, +{b['extra_2q_total_checks']} CX)"
                  f"   spin checks {b['spin_checks']:.1f} ({b['kept_spin_checks']:.3f}, +{b['extra_2q_spin_checks']} CX)")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), rows)
