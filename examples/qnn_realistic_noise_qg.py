"""
The qg filter for QNNs under realistic noise: unequal T1 across qubits,
and dephasing (§78)

§76-§77 showed that for a weight-conserving QNN (model E) the qg filter
makes T1 exact, and that training under T1 with the filter is the same
problem as training without noise. Both rest on one assumption: every qubit
has the same damping. Real devices have a spread of T1 across qubits, and
dephasing, which conserves weight and so cannot be filtered. This section
tests both, and checks that the exactness extends beyond weight 1.

A fact stated before the run (F4). With equal damping gamma on every
qubit, the filter is exact in any fixed-weight sector, not only weight 1:
the no-jump Kraus operator multiplies every weight-k basis state by the same
factor (1 - gamma)^(k/2), and every jump lowers the weight and leaves the
sector. The kept fraction is (1 - gamma)^(k * depth). With unequal damping
gamma_q the no-jump operator is diag(prod sqrt(1 - gamma_q)) over the
excited qubits, which is not uniform, so the filtered state is distorted.

Noise conditions (per qubit, after every two-qubit sublayer):
  H  unequal T1: gamma_q = 0.08 (1 + 0.5 s_q), s_q evenly spaced in [-1, 1]
     (E, 5 qubits: 0.04, 0.06, 0.08, 0.10, 0.12; A, 4 qubits: 0.04 to 0.12);
     mean 0.08 as in §77.
  D  equal T1 0.08 plus dephasing (phase flip) with probability 0.03.

Models (the §75/§76 architectures, data and splits; seed 78, 5 splits,
120 epochs; same initializations):
  E exact      E trained and evaluated without noise
  E filter     E trained without noise, evaluated under the condition with the filter
  E noisy+f    E trained and evaluated under the condition with the filter
  A noisy      standard QNN trained and evaluated under the condition
               (noise-aware training, the remedy that worked in §77)

Pre-registered predictions (written and committed before the seed-78 run;
code debugged on seed 1 with few epochs):
  S1  F4 holds numerically: under equal T1 the filtered distribution equals
      the noiseless one in the weight-1 to weight-4 sectors of 5 qubits
      (max error below 1e-12), and the kept fraction is (1 - gamma)^(k depth).
  S2  unequal T1 costs the filter little: under H, mean accuracy of E filter
      >= E exact - 1 point.
  S3  noise-aware training with the filter recovers what is lost: under H
      and under D, mean E noisy+f >= mean E filter.
  S4  dephasing hurts the filtered model: under D, mean E filter <= E exact
      - 1 point.
  S5  E with the filter stays competitive with the noise-aware standard QNN:
      under D, mean E noisy+f >= A noisy - 1 point.

The simulators are exact. For E a block simulator (weight-1 block plus the
|00000> population) with per-qubit damping and dephasing; for A the full
density matrix with per-qubit Kraus operators. Both are checked against the
full density matrix in the tests. Uses the installed library
(pip install qang) for the filter and the Hamming weights (qang.sectors).
Needs scikit-learn. Single-threaded BLAS is faster here:
OMP_NUM_THREADS=1 python examples/qnn_realistic_noise_qg.py [datasets]

Findings (python examples/qnn_realistic_noise_qg.py):

FINDINGS_PLACEHOLDER
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_noise_aware_qg as N77  # noqa: E402
import qnn_unary_norm_qg as E  # noqa: E402
from qang.sectors import filter_distribution  # noqa: E402

GAMMA = 0.08
SPREAD = 0.5
PHI = 0.03


def gammas(n, spread=SPREAD, gamma=GAMMA):
    return gamma * (1 + spread * np.linspace(-1, 1, n))


CONDITIONS = {
    "H": {"E": (gammas(E.N), 0.0), "A": (gammas(Q.N), 0.0)},
    "D": {"E": (np.full(E.N, GAMMA), PHI), "A": (np.full(Q.N, GAMMA), PHI)},
}


def kraus_t1(g):
    return [np.array([[1, 0], [0, math.sqrt(1 - g)]]), np.array([[0, math.sqrt(g)], [0, 0]])]


def kraus_phi(p):
    return [math.sqrt(1 - p) * np.eye(2), math.sqrt(p) * np.diag([1.0, -1.0])]


# --------------------------------------------------------------------- #
# exact simulators
# --------------------------------------------------------------------- #
def probs_E_block(theta, psi0, gam, phi):
    """Model E (weight-1 inputs) under per-qubit T1 gam[q] and dephasing phi."""
    U = N77.UNARY
    a = psi0[:, U]
    M = np.einsum("si,sj->sij", a, a.conj())
    p0 = np.zeros(len(psi0))
    for L in E.sublayers(theta):
        B = L[np.ix_(U, U)]
        M = B[None] @ M @ B.T[None]
        for q in range(E.N):
            p0 = p0 + gam[q] * np.real(M[:, q, q])
            d = math.sqrt(1.0 - gam[q])
            M[:, q, :] *= d
            M[:, :, q] *= d
            if phi:
                diag = M[:, q, q].copy()
                M[:, q, :] *= 1 - 2 * phi
                M[:, :, q] *= 1 - 2 * phi
                M[:, q, q] = diag
    probs = np.zeros((len(psi0), E.DIM))
    probs[:, U] = np.real(np.einsum("sii->si", M))
    probs[:, 0] = p0
    return probs


def probs_E_full(theta, psi0, gam, phi):
    """Reference: full 32 x 32 density matrix for E (any input weight)."""
    rho = np.einsum("si,sj->sij", psi0, psi0.conj()).astype(complex)
    for L in E.sublayers(theta):
        rho = L[None] @ rho @ L.T[None]
        for q in range(E.N):
            rho = E._channel(rho, kraus_t1(gam[q]), q)
            if phi:
                rho = E._channel(rho, kraus_phi(phi), q)
    return np.real(np.einsum("sii->si", rho))


def probs_A(theta, psi0, gam, phi):
    """Standard QNN A: full 16 x 16 density matrix with per-qubit noise."""
    rho = np.einsum("si,sj->sij", psi0, psi0.conj())
    KT = [kraus_t1(g) for g in gam]
    KP = kraus_phi(phi) if phi else None
    for L in Q.sublayers("A", theta):
        rho = L[None] @ rho @ L.conj().T[None]
        for q in range(Q.N):
            rho = Q._apply_1q_channel(rho, KT[q], q)
            if KP is not None:
                rho = Q._apply_1q_channel(rho, KP, q)
    return np.real(np.einsum("sii->si", rho))


def check_F4(gamma=GAMMA, seed=0, n_states=3):
    """Max error of the filtered distribution against the noiseless one in the
    weight-1..4 sectors under equal T1, and the kept fractions."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, E.NP)
    depth = len(E.sublayers(th))
    out = {}
    for k in range(1, E.N):
        idx = np.where(E.WEIGHT == k)[0]
        psi = np.zeros((n_states, E.DIM))
        psi[:, idx] = rng.normal(size=(n_states, len(idx)))
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
        exact = E.probs_exact(th, psi)
        noisy = probs_E_full(th, psi, np.full(E.N, gamma), 0.0)
        filt = np.array([filter_distribution(p, E.N, k)[0] for p in noisy])
        kept = noisy[:, idx].sum(axis=1)
        out[k] = (float(np.abs(filt - exact).max()), float(np.abs(kept - (1 - gamma) ** (k * depth)).max()))
    return out


# --------------------------------------------------------------------- #
# readouts and evaluation
# --------------------------------------------------------------------- #
def readout_E(psi0, cond=None, filt=False):
    def f(theta):
        probs = E.probs_exact(theta, psi0) if cond is None else probs_E_block(theta, psi0, *cond)
        return E.local_z(probs, filt)
    return f


def readout_A(psi0, cond):
    def f(theta):
        return Q.readout_from_probs("A", probs_A(theta, psi0, *cond))
    return f


def eval_E(params, Xte, yte, cond=None, filt=False):
    psi0 = E.encode(Xte)
    probs = E.probs_exact(params[:E.NP], psi0) if cond is None else probs_E_block(params[:E.NP], psi0, *cond)
    kept = float(np.mean(probs[:, E.WEIGHT == 1].sum(axis=1)))
    z = E.local_z(probs, filt) @ params[E.NP:E.NP + E.N] + params[-1]
    return N77.accuracy(z, yte), kept


def eval_A(params, Xte, yte, cond):
    P = Q.n_params("A")
    r = Q.readout_from_probs("A", probs_A(params[:P], Q.encode("A", Xte), *cond))
    return N77.accuracy(params[P] * r + params[P + 1], yte)


def run_dataset(name, seed=78, splits=5, epochs=120):
    rng = np.random.default_rng(seed)
    Xa, ya, use_pca = Q.load(name, rng)
    rows = []
    for s in range(splits):
        Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, seed * 100 + s)
        pA, pE = Q.encode("A", Xtr), E.encode(Xtr)
        nA = Q.n_params("A")
        sd = seed * 1000 + 10 * s
        E_clean = N77.adam_train(E.NP, E.N, readout_E(pE), ytr, np.random.default_rng(sd + 2), epochs)
        rec = {"E exact": eval_E(E_clean, Xte, yte)[0]}
        for c, cond in CONDITIONS.items():
            E_nf = N77.adam_train(E.NP, E.N, readout_E(pE, cond["E"], filt=True), ytr, np.random.default_rng(sd + 2), epochs)
            A_n = N77.adam_train(nA, 1, readout_A(pA, cond["A"]), ytr, np.random.default_rng(sd + 1), epochs)
            acc, kept = eval_E(E_clean, Xte, yte, cond["E"], filt=True)
            rec[f"{c} E filter"] = acc
            rec[f"{c} kept"] = kept
            rec[f"{c} E noisy+f"] = eval_E(E_nf, Xte, yte, cond["E"], filt=True)[0]
            rec[f"{c} A noisy"] = eval_A(A_n, Xte, yte, cond["A"])
        rows.append(rec)
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out, rows


def verdict(res, f4=None):
    ds = list(res)
    m = lambda k: float(np.mean([res[d][k] for d in ds]))  # noqa: E731
    f4 = check_F4() if f4 is None else f4
    depth_ok = all(err < 1e-12 and kerr < 1e-12 for err, kerr in f4.values())
    return {
        "S1": depth_ok,
        "S2": m("H E filter") >= m("E exact") - 0.01,
        "S3": m("H E noisy+f") >= m("H E filter") and m("D E noisy+f") >= m("D E filter"),
        "S4": m("D E filter") <= m("E exact") - 0.01,
        "S5": m("D E noisy+f") >= m("D A noisy") - 0.01,
    }


def main(seed=78, splits=5, epochs=120, datasets=None):
    f4 = check_F4()
    for k, (err, kerr) in f4.items():
        print(f"F4 weight {k}: filtered vs noiseless max error {err:.1e}, kept-fraction error {kerr:.1e}")
    res = {}
    for name in datasets or Q.DATASETS:
        res[name], _ = run_dataset(name, seed=seed, splits=splits, epochs=epochs)
        r = res[name]
        print(f"{name}: E exact {r['E exact']:.3f} | H: E filter {r['H E filter']:.3f}, E noisy+f {r['H E noisy+f']:.3f}, "
              f"A noisy {r['H A noisy']:.3f}, kept {r['H kept']:.3f} | D: E filter {r['D E filter']:.3f}, "
              f"E noisy+f {r['D E noisy+f']:.3f}, A noisy {r['D A noisy']:.3f}, kept {r['D kept']:.3f}", flush=True)
    if datasets:
        print(json.dumps(res))
        return res, None
    v = verdict(res, f4)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, v


if __name__ == "__main__":
    main(datasets=sys.argv[1:] or None)
