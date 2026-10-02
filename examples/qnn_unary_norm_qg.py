"""
A weight-conserving QNN that keeps the norm of the data: the qg filter's
protection without the accuracy cost? (§76)

§75: the only QNN that admits the qg filter (unary amplitude encoding, Givens
/ RBS layers, Hamming weight 1) was fully protected against T1 by the filter
but was the weakest classifier (0.765 vs about 0.95), because amplitude
encoding drops the norm of x and the readout was a single <Z_0>.

Here, model E (5 qubits):
  * encoding: unary amplitudes (x_1, x_2, x_3, x_4, 1) / norm on 5 qubits,
    so the ratio to the constant fifth component keeps |x|;
  * 3 layers of real Givens (RBS) rotations on the pairs (0,1), (2,3) |
    (1,2), (3,4) | (4,0): 15 parameters, 9 two-qubit sublayers;
  * readout z = sum_i c_i qg_Z^(i) + b with trainable c_i, head sigmoid(z).

Two facts, stated before the run (they follow from the construction):
  F1  T1 exactness. Every weight-1 basis state has exactly one excitation,
      so the no-jump part of amplitude damping multiplies all of them by the
      same factor and the jump part leaves weight 1. Keeping the weight-1
      shots (the qg filter) therefore returns the noiseless distribution
      exactly, for any damping strength.
  F2  Classical simulability. In the weight-1 sector the network is a 5 x 5
      orthogonal matrix O acting on v = (x, 1)/|(x, 1)|; with u = O v,
      qg_Z^(i) = 1 - 2 u_i^2, so z = sum_i c_i - 2 v^T O^T diag(c) O v + b, a
      quadratic form in v. The model is a quadratic classifier on normalized
      augmented features, computable classically in O(n^2). It cannot give a
      quantum advantage.

Comparison on the §75 data and splits (iris, cancer, wine, digits; 5 splits,
seed 76): E; the §75 standard QNN A; the §75 weight-conserving QNN D; and
classical models: RBF-SVM, MLP, and the classical twin of E (logistic
regression on the 15 quadratic features v_i v_j of the same v).
Evaluation as in §75: exact, 200 shots, T1 gamma = 0.03 and depolarizing
p = 0.01 after every two-qubit sublayer, E with and without the filter.

Pre-registered predictions (written and committed before the seed-76 run;
the code was debugged on seed 1 with few epochs):
  Q1  E closes the gap: its mean exact accuracy is at least that of A minus
      2 points.
  Q2  F1 holds numerically: E with the filter under T1 matches its exact
      accuracy within 0.5 points, and under T1 E + filter is at least as
      accurate as A on average.
  Q3  no advantage (F2): on every dataset E is at most 1 point above its
      classical quadratic twin, and the best classical model is at least as
      accurate as E minus 1 point.
  Q4  the filter's limit: under depolarizing noise the filter improves E by
      at most 1 point on average.

Uses the installed library (pip install qang): the qg filter is
qang.sectors.filter_distribution and Hamming weights come from
qang.sectors.hamming_weights. Needs scikit-learn.

Findings (python examples/qnn_unary_norm_qg.py):

FINDINGS_PLACEHOLDER
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.sectors import filter_distribution, hamming_weights  # noqa: E402

N = 5
DIM = 2**N
LAYERS = 3
SUBLAYERS = [[(0, 1), (2, 3)], [(1, 2), (3, 4)], [(4, 0)]]
BITS = (np.arange(DIM)[:, None] >> np.arange(N)[::-1]) & 1
ZSIGN = 1.0 - 2.0 * BITS
WEIGHT = hamming_weights(N)  # qang: weight of every basis index (order-independent)
NP = LAYERS * sum(len(s) for s in SUBLAYERS)  # 15


def rbs(a, b, t):
    U = np.eye(DIM)
    c, s = math.cos(t), math.sin(t)
    for i in range(DIM):
        if BITS[i, a] == 1 and BITS[i, b] == 0:
            j = i ^ (1 << (N - 1 - a)) ^ (1 << (N - 1 - b))
            U[i, i], U[j, j], U[i, j], U[j, i] = c, c, -s, s
    return U


def sublayers(theta):
    out, k = [], 0
    for _ in range(LAYERS):
        for pairs in SUBLAYERS:
            U = np.eye(DIM)
            for a, b in pairs:
                U = rbs(a, b, theta[k]) @ U
                k += 1
            out.append(U)
    return out


def augmented(Xs):
    V = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def encode(Xs):
    V = augmented(Xs)
    psi = np.zeros((len(Xs), DIM))
    for i in range(N):
        psi[:, 1 << (N - 1 - i)] = V[:, i]
    return psi


def local_z(probs, filt=False):
    if filt:  # the qg filter from the library: keep the weight-1 outcomes and renormalize
        probs = np.array([filter_distribution(pr, N, 1)[0] for pr in probs])
    return probs @ ZSIGN  # (S, N)


def probs_exact(theta, psi0):
    U = np.eye(DIM)
    for L in sublayers(theta):
        U = L @ U
    return np.abs(psi0 @ U.T) ** 2


def _channel(rho, kraus, q):
    S = rho.shape[0]
    r = rho.reshape((S,) + (2,) * (2 * N))
    out = np.zeros_like(r)
    for K in kraus:
        t = np.moveaxis(np.tensordot(K, r, axes=([1], [1 + q])), 0, 1 + q)
        t = np.moveaxis(np.tensordot(t, K.conj(), axes=([1 + N + q], [1])), -1, 1 + N + q)
        out += t
    return out.reshape(S, DIM, DIM)


def probs_noisy(theta, psi0, kind, p):
    rho = np.einsum("si,sj->sij", psi0, psi0.conj()).astype(complex)
    K = Q.kraus_set(kind, p)
    for L in sublayers(theta):
        rho = L[None] @ rho @ L.T[None]
        for q in range(N):
            rho = _channel(rho, K, q)
    return np.real(np.einsum("sii->si", rho))


def train(Xtr, ytr, rng, epochs=120, lr=0.1, h=1e-4):
    params = np.concatenate([rng.uniform(-np.pi, np.pi, NP), rng.normal(0, 0.5, N), [0.0]])
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    psi0 = encode(Xtr)
    for t in range(1, epochs + 1):
        R = local_z(probs_exact(params[:NP], psi0))
        z = R @ params[NP:NP + N] + params[-1]
        dz = (Q.sigmoid(z) - ytr) / len(ytr)
        g = np.zeros_like(params)
        g[NP:NP + N] = dz @ R
        g[-1] = dz.sum()
        for k in range(NP):
            e = np.zeros(NP)
            e[k] = h
            Rp = local_z(probs_exact(params[:NP] + e, psi0))
            Rm = local_z(probs_exact(params[:NP] - e, psi0))
            g[k] = np.sum(dz * ((Rp - Rm) @ params[NP:NP + N])) / (2 * h)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        params -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return params


def predict(params, Xte, mode="exact", rng=None, shots=200, noise=None, filt=False):
    psi0 = encode(Xte)
    probs = probs_exact(params[:NP], psi0) if noise is None else probs_noisy(params[:NP], psi0, *noise)
    if mode == "shots":
        probs = np.array([rng.multinomial(shots, pr / pr.sum()) / shots for pr in probs])
    z = local_z(probs, filt) @ params[NP:NP + N] + params[-1]
    return (Q.sigmoid(z) > 0.5).astype(int)


def quadratic_twin(Xtr, Xte, ytr, yte):
    from sklearn.linear_model import LogisticRegression

    def feats(Xs):
        V = augmented(Xs)
        iu = np.triu_indices(N)
        return np.einsum("si,sj->sij", V, V)[:, iu[0], iu[1]]

    return LogisticRegression(max_iter=5000).fit(feats(Xtr), ytr).score(feats(Xte), yte)


def run_dataset(name, seed=76, splits=5, epochs=120):
    rng = np.random.default_rng(seed)
    Xa, ya, use_pca = Q.load(name, rng)
    rows = []
    for s in range(splits):
        Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, seed * 100 + s)
        cl = Q.classical_scores(Xtr, Xte, ytr, yte, seed)
        rec = {"RBF-SVM": cl["RBF-SVM"], "MLP": cl["MLP"], "quadratic twin": quadratic_twin(Xtr, Xte, ytr, yte)}
        acc = lambda yp: float(np.mean(yp == yte))  # noqa: E731
        for mdl in ("A", "D"):
            par = Q.train(mdl, Xtr, ytr, np.random.default_rng(seed * 1000 + 10 * s + ord(mdl)), epochs=epochs)
            rec[mdl] = acc(Q.predict(mdl, par, Xte))
            rec[mdl + " T1"] = acc(Q.predict(mdl, par, Xte, noise=Q.T1))
        parE = train(Xtr, ytr, np.random.default_rng(seed * 1000 + 10 * s + 69), epochs=epochs)
        rec["E"] = acc(predict(parE, Xte))
        rec["E shots"] = acc(predict(parE, Xte, mode="shots", rng=np.random.default_rng(seed + s)))
        for tag, noise in (("T1", Q.T1), ("depol", Q.DEPOL)):
            rec[f"E {tag}"] = acc(predict(parE, Xte, noise=noise))
            rec[f"E {tag} filter"] = acc(predict(parE, Xte, noise=noise, filt=True))
        rows.append(rec)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows


def verdict(res):
    ds = list(res)
    mean = lambda k: float(np.mean([res[d][k] for d in ds]))  # noqa: E731
    q1 = mean("E") >= mean("A") - 0.02
    q2 = abs(mean("E T1 filter") - mean("E")) <= 0.005 and mean("E T1 filter") >= mean("A T1")
    q3 = all(res[d]["E"] <= res[d]["quadratic twin"] + 0.01 for d in ds) and \
        all(max(res[d]["RBF-SVM"], res[d]["MLP"], res[d]["quadratic twin"]) >= res[d]["E"] - 0.01 for d in ds)
    q4 = mean("E depol filter") - mean("E depol") <= 0.01
    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}


def main(seed=76, splits=5, epochs=120):
    res = {}
    for name in Q.DATASETS:
        res[name], _ = run_dataset(name, seed=seed, splits=splits, epochs=epochs)
        r = res[name]
        print(f"{name}: RBF-SVM {r['RBF-SVM']:.3f}, MLP {r['MLP']:.3f}, quadratic twin {r['quadratic twin']:.3f} | "
              f"A {r['A']:.3f}, D {r['D']:.3f}, E {r['E']:.3f} (200 shots {r['E shots']:.3f})")
        print(f"   T1: A {r['A T1']:.3f}, D {r['D T1']:.3f}, E {r['E T1']:.3f}, E + filter {r['E T1 filter']:.3f} | "
              f"depol: E {r['E depol']:.3f}, E + filter {r['E depol filter']:.3f}", flush=True)
    v = verdict(res)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, v


if __name__ == "__main__":
    main()
