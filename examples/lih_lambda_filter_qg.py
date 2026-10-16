"""
LiH with the pi orbitals: Lambda (the m_l of the molecule) as a third qg filter (§153)

The weight filter keeps the shots with the right electron number N (and, spin
resolved, N_up and N_down: §28, §55). A linear molecule conserves one more
quantum number, the projection of the orbital angular momentum on its axis,
Lambda = sum_i m_l(i) n_i, the molecular version of the magnetic quantum number
m_l. With the pi orbitals written as complex pi(+) and pi(-) (m_l = +1 and -1)
instead of px and py, Lambda is diagonal in the computational basis, so it can
be checked shot by shot, like N.

Two differences from the N filter make it worth testing:
  1. It catches errors that keep N: one electron moved between a sigma and a pi
     orbital changes Lambda by 1 and leaves N, N_up and N_down untouched.
  2. It needs only the four pi qubits in the Z basis. The N filter needs all
     ten, so it acts on the all-Z measurement group alone (§21 and the
     filter-scaling study: that is why it did not help LiH); Lambda can also
     be checked in every off-diagonal group whose pi qubits are I or Z.

Set-up (fixed before the run): LiH at 1.6 Angstrom, STO-3G, Li 1s frozen,
2 electrons in 5 spatial orbitals (3 sigma, pi+, pi-), 10 qubits,
Hamiltonian from tools/make_lih_lambda_hamiltonian.py (PySCF; data in
examples/data/lih_lambda_jw.json). Ansatz: Hartree-Fock, then the double
excitations sigma0^2 -> sigma1^2, sigma2^2 and sigma0^2 -> pi+ pi- (both spin
arrangements, one angle), then the single excitations sigma0 -> sigma1,
sigma2 (same angle for both spins); every gate conserves N, N_up, N_down and
Lambda. 5 angles optimised on the noiseless energy (COBYLA, fixed start).
Groups: the 1 all-Z group plus qubit-wise commuting groups of the other
terms; a qubit with I in a group is measured in Z. Noisy runs: FakeBrisbane,
FakeTorino, FakeFez, optimization_level 1, seed_transpiler 137, 20000 shots
per group. Errors are against the NOISELESS energy of the same circuit, so the
ansatz error is not mixed with the hardware error.

Read-outs:
  raw     every shot.
  NS      keep a shot when N_up = N_down = 1 can be checked (all five qubits
          of that spin in Z) and holds; groups where it cannot are left raw.
  L       keep a shot when Lambda = 0 can be checked (the four pi qubits in Z)
          and holds.
  NSL     both.

python examples/lih_lambda_filter_qg.py --dev   (noiseless and structural checks only)
python examples/lih_lambda_filter_qg.py

Checked during development, before the predictions (not predictions):
  * The Hamiltonian commutes with N, S_z and Lambda (to 1e-9); its lowest
    state (N = 2, S_z = 0) is the CASCI energy, -7.882097 Ha, with
    Lambda = 0 and pi occupation 0.003; HF is -7.861865 Ha. Lowest energy per
    Lambda sector: 0 -> -7.8821, +-1 -> -7.7161, +-2 -> -7.2139.
  * A first ansatz of Jordan-Wigner excitation exponentials needed 414-450
    two-qubit gates after transpilation and was replaced, before any noisy
    run, by the pair loader above: 28 two-qubit gates on all three backends,
    4 angles, noiseless energy 5.0 mHa above CASCI (it recovers 75% of the
    20.2 mHa correlation energy), pi occupation 0.0020, output exactly in the
    N_up = N_down = 1, Lambda = 0 sector.
  * 317 measurement groups. N_up/N_down can be checked in 1 (the all-Z group,
    79.1% of the sum of |coefficients| without the identity), Lambda in 29
    (91.5%). Noiseless samples (200,000 shots) reproduce the exact group
    energies with every read-out, and no filter removes any shot.
  * Global depolarizing floor (§24): 25 of 1024 bitstrings pass N_up/N_down,
    11 pass N_up/N_down plus Lambda.

Pre-registered predictions (sha256 of this file recorded in
examples/data/lih_lambda_filter_qg.prereg before the run). Errors in mHa
against the noiseless energy of the same circuit, readout errors not
corrected:
  L1  where N_up/N_down can already be checked (the all-Z group) Lambda adds
      little: NSL keeps at least 98% of the shots NS keeps, on all 3
      backends. Errors that keep N_up and N_down but change Lambda need an
      electron to move between a sigma and a pi orbital of the same spin
      without being caught, which takes two errors.
  L2  the filter still works where it applies: the all-Z group error falls
      at least 3x from raw to NS (median over the 3 backends).
  L3  in the 28 off-diagonal groups where only Lambda can be checked, the L
      read-out keeps between 85% and 97% of the shots (median over groups),
      on every backend: it does catch errors there, mostly read-out flips on
      the four pi qubits.
  L4  the extra reach pays: the total energy error of NSL is at least 1 mHa
      below that of NS on at least 2 of the 3 backends.
  L5  noise puts electrons into the pi orbitals: the raw pi occupation of the
      all-Z group is at least 10x the ideal one on every backend, and NSL
      lowers it below NS on every backend.
  L6  none of this reaches chemical accuracy: every read-out has a total
      error above 1.6 mHa on every backend (the 288 groups no filter can
      check are left raw).

Findings (pre-registration sha256 5fb77126...ba6f, recorded 2026-10-16; the
run used that exact file, then only this block was added):

Noiseless circuit energy -7.877097 Ha (5.0 mHa above CASCI), ideal pi
occupation 0.0020, 28 two-qubit gates on every backend. Errors in mHa against
that energy; "off L" = the 28 off-diagonal groups where only Lambda applies.

  backend       read-out  total   all-Z   off L  kept all-Z  kept off L  pi occ.
  FakeBrisbane  raw       227.4   218.6    8.54     1.000       1.000     0.235
                NS         13.9     5.16   8.54     0.721       1.000     0.0051
                L         199.8   191.3    8.27     0.902       0.902     0.127
                NSL        13.3     4.79   8.27     0.720       0.902     0.0037
  FakeTorino    raw       163.8   154.4    8.64     1.000       1.000     0.138
                NS         11.9     2.55   8.64     0.741       1.000     0.0037
                L         147.4   138.0    8.76     0.938       0.903     0.073
                NSL        11.7     2.26   8.76     0.740       0.903     0.0023
  FakeFez       raw       112.0   108.2    3.54     1.000       1.000     0.104
                NS          8.18    4.39   3.54     0.853       1.000     0.0085
                L         107.0   103.3    3.46     0.974       0.960     0.079
                NSL         8.02    4.31   3.46     0.853       0.960     0.0081

  L1 PASS: where N_up/N_down already applies, Lambda removes almost nothing
     more (NSL keeps 99.9%, 99.9%, 99.95% of what NS keeps).
  L2 PASS: the all-Z error falls 42x, 60x, 25x from raw to NS (median 42).
  L3 PASS: in the Lambda-only groups L keeps 0.902, 0.903, 0.960.
  L4 FAIL: NSL is only 0.64, 0.18, 0.16 mHa below NS (needed >= 1 mHa on 2
     of 3). Lambda does reach the 28 off-diagonal groups and does discard
     4-10% of their shots, but their error barely moves (8.54 -> 8.27,
     8.64 -> 8.76, 3.54 -> 3.46): what it discards there is mostly read-out
     flips on the pi qubits, which do not change terms that are I on the pi
     qubits, while the error that matters is on the sigma qubits measured in
     X/Y, which Lambda cannot see.
  L5 PASS: noise puts 52-118x the ideal pi occupation into the pi orbitals
     (raw 0.10-0.24); NSL lowers it below NS on all 3 (0.0037 vs 0.0051,
     0.0023 vs 0.0037, 0.0081 vs 0.0085).
  L6 PASS: the best total error is 8.0-13.3 mHa.
  * Lambda alone is a weak filter (all-Z error 191 vs raw 219 on Brisbane):
    it checks 4 of 10 qubits. Its value is as a third check and in the pi
    occupation, not in the energy.
  * After NS, most of the remaining error (3.5-8.8 of 8.0-13.3 mHa) sits in
    the Lambda-checkable off-diagonal groups, not in the 288 groups no
    filter reaches (whose terms nearly vanish on this pair state).
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

DATA = os.path.join(HERE, "data", "lih_lambda_jw.json")
OUT = os.path.join(HERE, "data", "lih_lambda_filter_qg.json")
PREREG = os.path.join(HERE, "data", "lih_lambda_filter_qg.prereg")
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
SHOTS = 20000
NQ = 5  # spatial orbitals; qubit p = orbital p spin up, p + NQ spin down
PI = (3, 4)
PI_QUBITS = (3, 4, 8, 9)
M_L = (0, 0, 0, 1, -1)
KINDS = ("raw", "NS", "L", "NSL")

with open(DATA, encoding="utf-8") as _fh:
    _D = json.load(_fh)


def hamiltonian():
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_list(list(_D["terms"].items()))


# --- ansatz: a unary loader over the five electron pairs ---------------------------------------------
#
# The exact ground state is almost entirely a combination of five closed-shell pairs: sigma0^2,
# sigma1^2, sigma2^2 and the two pi pairs (pi+ up, pi- down) and (pi- up, pi+ down); all five have
# N_up = N_down = 1 and Lambda = 0. One qubit of each pair (the spin-up one, q0..q4) carries a unary
# loader (an X and four RBS = XX+YY gates, §146), and a CNOT copies it onto the pair's spin-down
# partner. The final state is in the sector of all three filters; the CNOTs leave it in between,
# which does not matter for post-selection on the output.

PARTNER = {0: 5, 1: 6, 2: 7, 3: 9, 4: 8}  # pi+ up pairs with pi- down, pi- up with pi+ down
N_PARAMS = 4


def ansatz(theta, measure_basis=None):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    qc = QuantumCircuit(2 * NQ)
    qc.x(0)
    for k, t in enumerate(theta):
        qc.append(XXPlusYYGate(2 * float(t)), [k, k + 1])
    for q, p in PARTNER.items():
        qc.cx(q, p)
    if measure_basis is not None:
        for q, b in enumerate(reversed(measure_basis)):
            if b == "X":
                qc.h(q)
            elif b == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
    return qc


def in_sector(psi, tol=1e-12):
    """True when the state has N_up = N_down = 1 and Lambda = 0 exactly."""
    b = np.arange(psi.size)
    bit = lambda q: (b >> q) & 1  # noqa: E731
    nup = sum(bit(q) for q in range(NQ))
    ndn = sum(bit(q) for q in range(NQ, 2 * NQ))
    lam = bit(3) + bit(8) - bit(4) - bit(9)
    bad = (nup != 1) | (ndn != 1) | (lam != 0)
    return float(np.sum(np.abs(psi[bad]) ** 2)) < tol


def state(theta):
    from qiskit.quantum_info import Statevector

    return Statevector(ansatz(theta)).data


def optimise():
    from scipy.optimize import minimize

    Hm = hamiltonian().to_matrix(sparse=True)
    f = lambda th: float(np.real(np.vdot(state(th), Hm @ state(th))))  # noqa: E731
    best = None
    for x0 in ([0.1, 0.1, 0.1, 0.1], [-0.1, -0.1, -0.1, -0.1], [0.1, -0.1, 0.1, -0.1], [-0.1, 0.1, -0.1, 0.1]):
        r = minimize(f, np.array(x0), method="COBYLA", options={"maxiter": 4000, "rhobeg": 0.3, "tol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


# --- measurement groups and read-outs -----------------------------------------------------------------

def groups(H):
    diag, rest = [], []
    for lab, c in zip(H.paulis.to_labels(), H.coeffs):
        (diag if set(lab) <= set("IZ") else rest).append((lab, float(np.real(c))))
    from qiskit.quantum_info import SparsePauliOp

    # Terms whose pi qubits are all I or Z are grouped apart, so that every group they form keeps the four
    # pi qubits in the Z basis and Lambda can be checked there.
    pi_z = lambda lab: all(lab[2 * NQ - 1 - q] in "IZ" for q in PI_QUBITS)  # noqa: E731
    out = [diag]
    for part in ([t for t in rest if pi_z(t[0])], [t for t in rest if not pi_z(t[0])]):
        if part:
            for g in SparsePauliOp.from_list(part).group_commuting(qubit_wise=True):
                out.append([(l, float(np.real(c))) for l, c in zip(g.paulis.to_labels(), g.coeffs)])
    return out


def basis_of(group):
    b = ["Z"] * (2 * NQ)
    for lab, _ in group:
        for i, ch in enumerate(lab):
            if ch != "I":
                b[i] = ch
    return "".join(b)


def checks(basis):
    z = [basis[2 * NQ - 1 - q] == "Z" for q in range(2 * NQ)]
    return {"NS": all(z[:NQ]) and all(z[NQ:]), "L": all(z[q] for q in PI_QUBITS)}


def estimate(counts, group, basis, kind):
    """Energy of one group from counts {bitstring: n} (qubit 0 rightmost)."""
    can = checks(basis)
    use_ns = kind in ("NS", "NSL") and can["NS"]
    use_l = kind in ("L", "NSL") and can["L"]
    tot, acc = 0, np.zeros(len(group))
    for s, c in counts.items():
        bit = [int(s[2 * NQ - 1 - q]) for q in range(2 * NQ)]
        if use_ns and (sum(bit[:NQ]) != 1 or sum(bit[NQ:]) != 1):
            continue
        if use_l and bit[3] + bit[8] - bit[4] - bit[9] != 0:
            continue
        tot += c
        for k, (lab, _) in enumerate(group):
            par = sum(bit[q] for q in range(2 * NQ) if lab[2 * NQ - 1 - q] != "I") % 2
            acc[k] += c * (1 - 2 * par)
    vals = acc / max(tot, 1)
    return float(sum(cf * v for (_, cf), v in zip(group, vals))), tot


def pi_occupation(counts, kind):
    basis = "Z" * (2 * NQ)
    can = checks(basis)
    tot = occ = 0
    for s, c in counts.items():
        bit = [int(s[2 * NQ - 1 - q]) for q in range(2 * NQ)]
        if kind in ("NS", "NSL") and can["NS"] and (sum(bit[:NQ]) != 1 or sum(bit[NQ:]) != 1):
            continue
        if kind in ("L", "NSL") and bit[3] + bit[8] - bit[4] - bit[9] != 0:
            continue
        tot += c
        occ += c * sum(bit[q] for q in PI_QUBITS)
    return occ / max(tot, 1), tot


def development():
    """Noiseless and structural checks only: no noisy circuit is run here."""
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    import qiskit_ibm_runtime.fake_provider as fp

    H = hamiltonian()
    th, e = optimise()
    sv = Statevector(ansatz(th))
    probs = sv.probabilities_dict()
    pi = sum(p * sum(int(s[2 * NQ - 1 - q]) for q in PI_QUBITS) for s, p in probs.items())
    G = groups(H)
    cov = {"NS": 0, "L": 0}
    wcov = {"NS": 0.0, "L": 0.0}
    wtot = sum(abs(c) for g in G for lab, c in g if set(lab) != {"I"})
    for g in G:
        ch = checks(basis_of(g))
        for k in cov:
            if ch[k]:
                cov[k] += 1
                wcov[k] += sum(abs(c) for lab, c in g if set(lab) != {"I"})
    ts = {}
    for name in BACKENDS:
        b = getattr(fp, name)()
        t = transpile(ansatz(th, basis_of(G[0])), b, optimization_level=1, seed_transpiler=137)
        ts[name] = sum(1 for x in t.data if x.operation.num_qubits == 2)
    print(json.dumps({"in_sector": in_sector(sv.data), "theta": [round(x, 5) for x in th],
                      "E_circuit": round(e, 6), "E_casci": _D["fci_active"], "E_hf": _D["e_hf"],
                      "above_casci_mHa": round(1000 * (e - _D["fci_active"]), 3), "pi_occupation": round(pi, 5),
                      "groups": len(G), "groups_NS": cov["NS"], "groups_L": cov["L"],
                      "weight_share_NS": round(wcov["NS"] / wtot, 3), "weight_share_L": round(wcov["L"] / wtot, 3),
                      "two_qubit_gates": ts}, indent=1))




# --- noisy runs ----------------------------------------------------------------------------------------

def exact_by_group(G, th):
    from qiskit.quantum_info import SparsePauliOp, Statevector

    sv = Statevector(ansatz(th))
    return [float(np.real(sv.expectation_value(SparsePauliOp.from_list(g)))) for g in G]


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name, G, th, ex):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    bases = [basis_of(g) for g in G]
    ts = [transpile(ansatz(th, bs), b, optimization_level=1, seed_transpiler=137) for bs in bases]
    res = sampler(b).run(ts, shots=SHOTS).result()
    counts = [r.data.meas.get_counts() for r in res]
    lgroups = [i for i, bs in enumerate(bases) if checks(bs)["L"] and not checks(bs)["NS"]]
    out = {"backend": name, "two_qubit_gates": int(np.mean([sum(1 for x in t.data if x.operation.num_qubits == 2)
                                                             for t in ts]))}
    for k in KINDS:
        est = [estimate(c, g, bs, k) for c, g, bs in zip(counts, G, bases)]
        e = np.array([v for v, _ in est])
        kept = [n / SHOTS for _, n in est]
        out[k] = {"total_mHa": 1000 * float(abs(e.sum() - sum(ex))),
                  "diag_mHa": 1000 * float(abs(e[0] - ex[0])),
                  "offdiag_L_mHa": 1000 * float(abs(sum(e[i] - ex[i] for i in lgroups))),
                  "kept_diag": kept[0],
                  "kept_offdiag_L_median": float(np.median([kept[i] for i in lgroups])),
                  "pi_occupation": pi_occupation(counts[0], k)[0]}
    return out


def main():
    with open(__file__, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(json.dumps({"sha256": digest}), flush=True)
    H = hamiltonian()
    G = groups(H)
    th, e_circ = optimise()
    ex = exact_by_group(G, th)
    from qiskit.quantum_info import Statevector

    probs = Statevector(ansatz(th)).probabilities_dict()
    pi_ideal = sum(p * sum(int(s[2 * NQ - 1 - q]) for q in PI_QUBITS) for s, p in probs.items())
    print(json.dumps({"E_circuit": e_circ, "sum_groups": sum(ex), "pi_ideal": pi_ideal, "groups": len(G)}), flush=True)
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name, G, th, ex)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(r, default=float), flush=True)
        rows.append(r)
    ok = [r for r in rows if "raw" in r]
    full = len(ok) == 3
    v = {
        "L1": full and all(r["NSL"]["kept_diag"] >= 0.98 * r["NS"]["kept_diag"] for r in ok),
        "L2": full and float(np.median([r["raw"]["diag_mHa"] / max(r["NS"]["diag_mHa"], 1e-9) for r in ok])) >= 3,
        "L3": full and all(0.85 <= r["L"]["kept_offdiag_L_median"] <= 0.97 for r in ok),
        "L4": full and sum(r["NSL"]["total_mHa"] <= r["NS"]["total_mHa"] - 1.0 for r in ok) >= 2,
        "L5": full and all(r["raw"]["pi_occupation"] >= 10 * pi_ideal and r["NSL"]["pi_occupation"]
                           < r["NS"]["pi_occupation"] for r in ok),
        "L6": full and all(r[k]["total_mHa"] > 1.6 for r in ok for k in KINDS),
    }
    print("predictions: " + ", ".join(f"{k} {'PASS' if x else 'FAIL'}" for k, x in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sha256": digest, "theta": list(map(float, th)), "E_circuit": e_circ, "exact_by_group": ex,
                   "pi_ideal": pi_ideal, "rows": rows, "verdict": v}, fh, indent=1, default=float)
    return rows, v


if __name__ == "__main__":
    if "--dev" in sys.argv:
        development()
    elif "--prereg" in sys.argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        from datetime import datetime, timezone

        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/lih_lambda_filter_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
    else:
        main()
