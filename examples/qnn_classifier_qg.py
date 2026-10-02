"""
Quantum neural network classifiers with and without qang, against classical
baselines (§75).

Two different questions, reported separately:
  (a) the qang gain INSIDE quantum ML: does the same 4-qubit QNN do better
      with qg tools (arccos encoding, register-mean qg readout, the qg
      symmetry filter under noise)?
  (b) the gain OVER classical ML: does any QNN beat logistic regression, an
      RBF-SVM or an MLP on the same data?

Data (binary, 4 features, real datasets from scikit-learn):
  iris      versicolor vs virginica (100 samples)
  cancer    breast cancer, PCA to 4 (200 stratified samples of 569)
  wine      class 0 vs 1, PCA to 4 (130 samples)
  digits    3 vs 8, PCA to 4 (200 stratified samples)
Five stratified 70/30 splits per dataset; scaler and PCA fitted on the
training part only; features min-max scaled to [-1, 1] on the training part
(test values clipped).

Quantum models (4 qubits, exact statevector / density-matrix simulation):
  A  standard QNN: angle encoding Ry(pi (x+1)/2), 3 layers of Ry, Rz on every
     qubit and a CZ ring, readout <Z_0>
  B  as A with arccos encoding Ry(arccos x), so qg_Z of qubit i = x_i
  C  as B with the register-mean readout (1/n) sum_i qg_Z^(i) (local cost)
  D  weight-conserving QNN: unary amplitude encoding sum_i (x_i/|x|) |e_i>
     (Hamming weight 1), 3 layers of real Givens (RBS) rotations on the pairs
     (0,1), (2,3), (1,2), (3,0), readout <Z_0>; under noise it can use the qg
     filter (keep weight-1 shots)
All heads: p = sigmoid(w r + b) on the readout r, trained jointly (Adam,
lr 0.1, 120 full-batch epochs, cross-entropy, central-difference gradients).
A, B, C have 24 circuit parameters, D has 12 (same depth: 6 two-qubit
sublayers).

Classical baselines: logistic regression; RBF-SVM; MLP (16 hidden units);
and the classical twin of the arccos encoding, logistic regression on the
Chebyshev features T_1..T_3 of each x_i.

Evaluation of the trained models (trained noiselessly):
  exact     ideal expectation values
  shots     200 shots per test sample
  T1        amplitude damping gamma = 0.03 on every qubit after each two-qubit
            sublayer (density matrices)
  depol     depolarizing p = 0.01 on every qubit after each sublayer
  D under noise: with and without the qg filter

Pre-registered predictions (written and committed before the seed-75 run;
the code was debugged on seed 1 with few epochs):
  P1  encoding: B is not worse than A by more than 2 points of test accuracy
      on any dataset, and B >= A on the average over the four datasets.
  P2  no quantum advantage: on every dataset the best classical baseline is
      at least as accurate as the best quantum model minus 1 point.
  P3  filter: under T1, D with the qg filter loses at most 2 points on
      average (vs its exact accuracy), and loses less than D without filter
      and less than A.
  P4  shots: at 200 shots, the accuracy loss of C (register-mean readout) is
      at most that of B (single-qubit readout), averaged over datasets.

Needs scikit-learn.

Findings (python examples/qnn_classifier_qg.py):

FINDINGS_PLACEHOLDER
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

N = 4
DIM = 2**N
LAYERS = 3
PAIRS_RING = [(0, 1), (2, 3), (1, 2), (3, 0)]  # two sublayers: {(0,1),(2,3)}, {(1,2),(3,0)}
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Zm = np.diag([1.0, -1.0])
BITS = (np.arange(DIM)[:, None] >> np.arange(N)[::-1]) & 1  # qubit 0 = MSB
ZSIGN = 1.0 - 2.0 * BITS  # (DIM, N): eigenvalue of Z_i on each basis state
WEIGHT = BITS.sum(axis=1)


# --------------------------------------------------------------------- #
# small statevector helpers
# --------------------------------------------------------------------- #
def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def on_qubit(g, q):
    mats = [I2] * N
    mats[q] = g
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def cz(a, b):
    d = np.ones(DIM, dtype=complex)
    d[(BITS[:, a] == 1) & (BITS[:, b] == 1)] = -1
    return np.diag(d)


def rbs(a, b, t):
    """Real Givens rotation between |..1_a..0_b..> and |..0_a..1_b..> (weight conserving)."""
    U = np.eye(DIM, dtype=complex)
    c, s = math.cos(t), math.sin(t)
    for i in range(DIM):
        if BITS[i, a] == 1 and BITS[i, b] == 0:
            j = i ^ (1 << (N - 1 - a)) ^ (1 << (N - 1 - b))
            U[i, i], U[j, j], U[i, j], U[j, i] = c, c, -s, s
    return U


CZ_SUB = [cz(0, 1) @ cz(2, 3), cz(1, 2) @ cz(3, 0)]


def sublayers(model, theta):
    """List of unitaries, one per two-qubit sublayer (noise is applied after each)."""
    out = []
    if model == "D":
        k = 0
        for _ in range(LAYERS):
            out.append(rbs(0, 1, theta[k]) @ rbs(2, 3, theta[k + 1]))
            out.append(rbs(1, 2, theta[k + 2]) @ rbs(3, 0, theta[k + 3]))
            k += 4
        return out
    k = 0
    for _ in range(LAYERS):
        U = np.eye(DIM, dtype=complex)
        for q in range(N):
            U = on_qubit(rz(theta[k + 2 * q + 1]) @ ry(theta[k + 2 * q]), q) @ U
        k += 2 * N
        out.append(CZ_SUB[0] @ U)
        out.append(CZ_SUB[1])
    return out


def n_params(model):
    return 4 * LAYERS if model == "D" else 2 * N * LAYERS


def encode(model, Xs):
    """Initial statevectors (S, DIM) for features in [-1, 1]."""
    S = len(Xs)
    if model == "D":
        psi = np.zeros((S, DIM), dtype=complex)
        nrm = np.linalg.norm(Xs, axis=1)
        nrm[nrm == 0] = 1.0
        for i in range(N):
            psi[:, 1 << (N - 1 - i)] = Xs[:, i] / nrm
        return psi
    ang = np.pi * (Xs + 1) / 2 if model == "A" else np.arccos(np.clip(Xs, -1, 1))
    c, s = np.cos(ang / 2), np.sin(ang / 2)  # per-qubit amplitudes of |0>, |1>
    psi = np.ones((S, DIM), dtype=complex)
    for q in range(N):
        psi *= np.where(BITS[None, :, q] == 0, c[:, q:q + 1], s[:, q:q + 1])
    return psi


def readout_from_probs(model, probs, filt=False):
    """Readout r in [-1, 1] from outcome distributions (S, DIM)."""
    if filt:
        keep = (WEIGHT == 1)
        pk = probs * keep[None, :]
        tot = pk.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        probs = pk / tot
    if model == "C":
        return probs @ ZSIGN.mean(axis=1)
    return probs @ ZSIGN[:, 0]


def forward_exact(model, theta, psi0):
    U = np.eye(DIM, dtype=complex)
    for L in sublayers(model, theta):
        U = L @ U
    psi = psi0 @ U.T
    return readout_from_probs(model, np.abs(psi) ** 2)


# --------------------------------------------------------------------- #
# noise (density matrices, evaluation only)
# --------------------------------------------------------------------- #
def _apply_1q_channel(rho, kraus, q):
    S = rho.shape[0]
    r = rho.reshape((S,) + (2,) * (2 * N))
    out = np.zeros_like(r)
    for K in kraus:
        t = np.tensordot(K, r, axes=([1], [1 + q]))  # new axis 0
        t = np.moveaxis(t, 0, 1 + q)
        t = np.tensordot(t, K.conj(), axes=([1 + N + q], [1]))
        t = np.moveaxis(t, -1, 1 + N + q)
        out += t
    return out.reshape(S, DIM, DIM)


def kraus_set(kind, p):
    if kind == "T1":
        return [np.array([[1, 0], [0, math.sqrt(1 - p)]]), np.array([[0, math.sqrt(p)], [0, 0]])]
    Y = np.array([[0, -1j], [1j, 0]])
    return [math.sqrt(1 - 3 * p / 4) * I2, math.sqrt(p / 4) * X, math.sqrt(p / 4) * Y, math.sqrt(p / 4) * Zm]


def probs_noisy(model, theta, psi0, kind, p):
    rho = np.einsum("si,sj->sij", psi0, psi0.conj())
    K = kraus_set(kind, p)
    for L in sublayers(model, theta):
        rho = L[None] @ rho @ L.conj().T[None]
        for q in range(N):
            rho = _apply_1q_channel(rho, K, q)
    return np.real(np.einsum("sii->si", rho))


# --------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------- #
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(model, Xtr, ytr, rng, epochs=120, lr=0.1, h=1e-4):
    P = n_params(model)
    params = np.concatenate([rng.uniform(-np.pi, np.pi, P), [2.0, 0.0]])  # theta, w, b
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    psi0 = encode(model, Xtr)

    def loss_and_r(par):
        r = forward_exact(model, par[:P], psi0)
        p = np.clip(sigmoid(par[P] * r + par[P + 1]), 1e-9, 1 - 1e-9)
        return -np.mean(ytr * np.log(p) + (1 - ytr) * np.log(1 - p)), r

    for t in range(1, epochs + 1):
        _, r = loss_and_r(params)
        p = sigmoid(params[P] * r + params[P + 1])
        dz = (p - ytr) / len(ytr)  # dL/dz
        g = np.zeros_like(params)
        g[P] = np.sum(dz * r)
        g[P + 1] = np.sum(dz)
        for k in range(P):
            e = np.zeros_like(params)
            e[k] = h
            rp = forward_exact(model, (params + e)[:P], psi0)
            rm = forward_exact(model, (params - e)[:P], psi0)
            g[k] = np.sum(dz * params[P] * (rp - rm) / (2 * h))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        params -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return params


def predict(model, params, Xte, mode="exact", rng=None, shots=200, noise=None, filt=False):
    P = n_params(model)
    psi0 = encode(model, Xte)
    if noise is None:
        U = np.eye(DIM, dtype=complex)
        for L in sublayers(model, params[:P]):
            U = L @ U
        probs = np.abs(psi0 @ U.T) ** 2
    else:
        probs = probs_noisy(model, params[:P], psi0, *noise)
    if mode == "shots":
        probs = np.array([rng.multinomial(shots, pr / pr.sum()) / shots for pr in probs])
    r = readout_from_probs(model, probs, filt)
    return (sigmoid(params[P] * r + params[P + 1]) > 0.5).astype(int)


# --------------------------------------------------------------------- #
# data and classical baselines
# --------------------------------------------------------------------- #
DATASETS = ("iris", "cancer", "wine", "digits")


def load(name, rng):
    from sklearn import datasets

    if name == "iris":
        d = datasets.load_iris()
        mask = d.target > 0
        return d.data[mask], (d.target[mask] == 2).astype(int), False
    if name == "cancer":
        d = datasets.load_breast_cancer()
        Xa, ya = d.data, d.target
    elif name == "wine":
        d = datasets.load_wine()
        mask = d.target < 2
        return d.data[mask], d.target[mask], True
    else:
        d = datasets.load_digits()
        mask = (d.target == 3) | (d.target == 8)
        Xa, ya = d.data[mask], (d.target[mask] == 8).astype(int)
    idx = np.concatenate([rng.choice(np.where(ya == c)[0], 100, replace=False) for c in (0, 1)])
    return Xa[idx], ya[idx], True


def split_prepare(Xa, ya, use_pca, seed):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if use_pca:
        pca = PCA(n_components=N, random_state=seed).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    f = lambda Z: np.clip(2 * (Z - lo) / span - 1, -1, 1)  # noqa: E731
    return f(Xtr), f(Xte), ytr, yte


def chebyshev(Xs):
    return np.concatenate([np.cos(k * np.arccos(Xs)) for k in (1, 2, 3)], axis=1)


def classical_scores(Xtr, Xte, ytr, yte, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    out = {}
    out["logistic"] = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
    out["RBF-SVM"] = SVC().fit(Xtr, ytr).score(Xte, yte)
    out["MLP"] = MLPClassifier((16,), max_iter=3000, random_state=seed).fit(Xtr, ytr).score(Xte, yte)
    out["Chebyshev+logistic"] = LogisticRegression(max_iter=2000).fit(chebyshev(Xtr), ytr).score(chebyshev(Xte), yte)
    return out


QMODELS = ("A", "B", "C", "D")
CLASSICAL = ("logistic", "RBF-SVM", "MLP", "Chebyshev+logistic")
T1 = ("T1", 0.03)
DEPOL = ("depol", 0.01)


def run_dataset(name, seed=75, splits=5, epochs=120):
    rng = np.random.default_rng(seed)
    Xa, ya, use_pca = load(name, rng)
    rows = []
    for s in range(splits):
        Xtr, Xte, ytr, yte = split_prepare(Xa, ya, use_pca, seed * 100 + s)
        rec = dict(classical_scores(Xtr, Xte, ytr, yte, seed))
        for mdl in QMODELS:
            par = train(mdl, Xtr, ytr, np.random.default_rng(seed * 1000 + 10 * s + ord(mdl)), epochs=epochs)
            acc = lambda yp: float(np.mean(yp == yte))  # noqa: E731
            rec[mdl] = acc(predict(mdl, par, Xte))
            rec[mdl + " shots"] = acc(predict(mdl, par, Xte, mode="shots", rng=np.random.default_rng(seed + s)))
            rec[mdl + " T1"] = acc(predict(mdl, par, Xte, noise=T1))
            rec[mdl + " depol"] = acc(predict(mdl, par, Xte, noise=DEPOL))
            if mdl == "D":
                rec["D T1 filter"] = acc(predict(mdl, par, Xte, noise=T1, filt=True))
                rec["D depol filter"] = acc(predict(mdl, par, Xte, noise=DEPOL, filt=True))
        rows.append(rec)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows


def verdict(res):
    ds = list(res)
    mean = lambda k: float(np.mean([res[d][k] for d in ds]))  # noqa: E731
    p1 = all(res[d]["B"] >= res[d]["A"] - 0.02 for d in ds) and mean("B") >= mean("A")
    p2 = all(max(res[d][c] for c in CLASSICAL) >= max(res[d][q] for q in QMODELS) - 0.01 for d in ds)
    loss = lambda k, base: mean(base) - mean(k)  # noqa: E731
    p3 = loss("D T1 filter", "D") <= 0.02 and loss("D T1 filter", "D") < loss("D T1", "D") and \
        loss("D T1 filter", "D") < loss("A T1", "A")
    p4 = loss("C shots", "C") <= loss("B shots", "B")
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4}


def main(seed=75, splits=5, epochs=120):
    res = {}
    for name in DATASETS:
        res[name], _ = run_dataset(name, seed=seed, splits=splits, epochs=epochs)
        r = res[name]
        print(f"{name}: classical " + ", ".join(f"{c} {r[c]:.3f}" for c in CLASSICAL))
        print("   quantum exact " + ", ".join(f"{q} {r[q]:.3f}" for q in QMODELS))
        print("   200 shots     " + ", ".join(f"{q} {r[q + ' shots']:.3f}" for q in QMODELS))
        print("   T1 0.03       " + ", ".join(f"{q} {r[q + ' T1']:.3f}" for q in QMODELS) + f", D+filter {r['D T1 filter']:.3f}")
        print("   depol 0.01    " + ", ".join(f"{q} {r[q + ' depol']:.3f}" for q in QMODELS) + f", D+filter {r['D depol filter']:.3f}", flush=True)
    v = verdict(res)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, v


if __name__ == "__main__":
    main()
