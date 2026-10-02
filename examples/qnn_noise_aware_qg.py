"""
Training QNNs under relaxation noise: does the qg filter make noise-aware
training unnecessary? (§77)

§76 showed that a weight-conserving QNN that keeps the norm (model E) is
exactly immune to T1 when its readout uses the qg filter, when trained
noiselessly and only evaluated with noise. The usual remedy for noise in
QNNs is noise-aware training: train under the same noise you will meet.
Here both are compared at a stronger damping, gamma = 0.08 per qubit per
two-qubit sublayer.

Models (the §75/§76 architectures, data and splits; seed 77):
  A clean      standard QNN trained noiselessly, evaluated under T1
  A noisy      standard QNN trained under T1, evaluated under T1
  E filter     E trained noiselessly, evaluated under T1 with the qg filter
  E noisy+f    E trained under T1 with the filtered readout, evaluated the same way
  E noisy      E trained under T1 without the filter, evaluated without it
  E clean      E trained noiselessly, evaluated under T1 without the filter
Also: 200-shot evaluation of A noisy and E filter (the filter discards the
shots that left weight 1; the kept fraction is reported).

A consequence of §76 F1, stated before the run:
  F3  with the filter the T1-noisy readout equals the noiseless one for
      every parameter value, so the training loss and every gradient are
      identical: training under T1 with the filter gives exactly the
      parameters of noiseless training (same initialization).

Pre-registered predictions (written and committed before the seed-77 run;
code debugged on seed 1 with few epochs):
  R1  F3 holds numerically: E noisy+f and E filter have the same parameters
      (max difference below 1e-6) and the same accuracy.
  R2  noise-aware training helps the standard QNN: mean accuracy of A noisy
      >= A clean under T1.
  R3  the filter beats noise-aware training: mean accuracy of E filter >= A
      noisy, and >= E noisy.
  R4  even paying the discarded shots: at 200 shots under T1, E filter is
      at least as accurate as A noisy minus 1 point (mean).

Training under T1 uses an exact block simulator (weight-1 block plus the
|00000> population), equal to the full density matrix of §76 to machine
precision and about 100 times faster. Uses the installed library
(pip install qang) through the §76 model, whose filter is
qang.sectors.filter_distribution. Needs scikit-learn.

Run: python examples/qnn_noise_aware_qg.py            (all datasets)
     python examples/qnn_noise_aware_qg.py iris wine  (a subset; prints JSON)

Findings (python examples/qnn_noise_aware_qg.py):

FINDINGS_PLACEHOLDER
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_unary_norm_qg as E  # noqa: E402

GAMMA = 0.08
T1 = ("T1", GAMMA)


def adam_train(n_theta, n_head, readout, ytr, rng, epochs=120, lr=0.1, h=1e-4):
    """Generic trainer: readout(theta) -> features R (S, n_head) or r (S,);
    head z = R @ c + b (or w r + b). Central-difference gradients on theta."""
    if n_head == 1:
        params = np.concatenate([rng.uniform(-np.pi, np.pi, n_theta), [2.0, 0.0]])
    else:
        params = np.concatenate([rng.uniform(-np.pi, np.pi, n_theta), rng.normal(0, 0.5, n_head), [0.0]])
    m = np.zeros_like(params)
    v = np.zeros_like(params)

    def feats(theta):
        R = readout(theta)
        return R[:, None] if R.ndim == 1 else R

    for t in range(1, epochs + 1):
        th, c, b = params[:n_theta], params[n_theta:n_theta + n_head], params[-1]
        R = feats(th)
        dz = (Q.sigmoid(R @ c + b) - ytr) / len(ytr)
        g = np.zeros_like(params)
        g[n_theta:n_theta + n_head] = dz @ R
        g[-1] = dz.sum()
        for k in range(n_theta):
            e = np.zeros(n_theta)
            e[k] = h
            g[k] = np.sum(dz * ((feats(th + e) - feats(th - e)) @ c)) / (2 * h)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        params -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return params


def readout_A(psi0, noise=None):
    def f(theta):
        if noise is None:
            return Q.forward_exact("A", theta, psi0)
        return Q.readout_from_probs("A", Q.probs_noisy("A", theta, psi0, *noise))
    return f


# Weight-1 inputs, weight-conserving gates and T1 (which only lowers weight):
# the state stays block-diagonal, a 5x5 block on the weight-1 states plus the
# population of |00000>. probs_t1 propagates exactly that, and equals
# E.probs_noisy(theta, psi0, "T1", gamma) to machine precision (checked in the
# tests); it is about 100 times faster, which makes training under T1 feasible.
UNARY = np.array([1 << (E.N - 1 - i) for i in range(E.N)])


def _block(L):
    return L[np.ix_(UNARY, UNARY)]


def probs_t1(theta, psi0, gamma):
    a = psi0[:, UNARY]
    M = np.einsum("si,sj->sij", a, a.conj())
    p0 = np.zeros(len(psi0))
    d = np.sqrt(1.0 - gamma)
    for L in E.sublayers(theta):
        B = _block(L)
        M = B[None] @ M @ B.T[None]
        for q in range(E.N):
            p0 = p0 + gamma * np.real(M[:, q, q])
            M[:, q, :] *= d
            M[:, :, q] *= d
    probs = np.zeros((len(psi0), E.DIM))
    probs[:, UNARY] = np.real(np.einsum("sii->si", M))
    probs[:, 0] = p0
    return probs


def probs_E(theta, psi0, noise=None):
    if noise is None:
        return E.probs_exact(theta, psi0)
    if noise[0] == "T1":
        return probs_t1(theta, psi0, noise[1])
    return E.probs_noisy(theta, psi0, *noise)


def readout_E(psi0, noise=None, filt=False):
    def f(theta):
        return E.local_z(probs_E(theta, psi0, noise), filt)
    return f


def accuracy(z, y):
    return float(np.mean((Q.sigmoid(z) > 0.5).astype(int) == y))


def eval_A(params, Xte, yte, noise=None, shots=None, rng=None):
    psi0 = Q.encode("A", Xte)
    P = Q.n_params("A")
    probs = np.abs(psi0 @ _unitary_A(params[:P]).T) ** 2 if noise is None else Q.probs_noisy("A", params[:P], psi0, *noise)
    if shots:
        probs = np.array([rng.multinomial(shots, pr / pr.sum()) / shots for pr in probs])
    r = Q.readout_from_probs("A", probs)
    return accuracy(params[P] * r + params[P + 1], yte)


def _unitary_A(theta):
    U = np.eye(Q.DIM, dtype=complex)
    for L in Q.sublayers("A", theta):
        U = L @ U
    return U


def eval_E(params, Xte, yte, noise=None, filt=False, shots=None, rng=None):
    psi0 = E.encode(Xte)
    probs = probs_E(params[:E.NP], psi0, noise)
    kept = float(np.mean(probs[:, E.WEIGHT == 1].sum(axis=1)))
    if shots:
        probs = np.array([rng.multinomial(shots, pr / pr.sum()) / shots for pr in probs])
    z = E.local_z(probs, filt) @ params[E.NP:E.NP + E.N] + params[-1]
    return accuracy(z, yte), kept


def run_dataset(name, seed=77, splits=5, epochs=120):
    rng = np.random.default_rng(seed)
    Xa, ya, use_pca = Q.load(name, rng)
    rows = []
    for s in range(splits):
        Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, seed * 100 + s)
        pA, pE = Q.encode("A", Xtr), E.encode(Xtr)
        nA, nE = Q.n_params("A"), E.NP
        sd = seed * 1000 + 10 * s
        A_clean = adam_train(nA, 1, readout_A(pA), ytr, np.random.default_rng(sd + 1), epochs)
        A_noisy = adam_train(nA, 1, readout_A(pA, T1), ytr, np.random.default_rng(sd + 1), epochs)
        E_clean = adam_train(nE, E.N, readout_E(pE), ytr, np.random.default_rng(sd + 2), epochs)
        E_noisyf = adam_train(nE, E.N, readout_E(pE, T1, filt=True), ytr, np.random.default_rng(sd + 2), epochs)
        E_noisy = adam_train(nE, E.N, readout_E(pE, T1), ytr, np.random.default_rng(sd + 2), epochs)
        r = np.random.default_rng(seed + s)
        rec = {
            "A clean": eval_A(A_clean, Xte, yte, T1),
            "A noisy": eval_A(A_noisy, Xte, yte, T1),
            "A exact": eval_A(A_clean, Xte, yte),
            "E filter": eval_E(E_clean, Xte, yte, T1, filt=True)[0],
            "E noisy+f": eval_E(E_noisyf, Xte, yte, T1, filt=True)[0],
            "E noisy": eval_E(E_noisy, Xte, yte, T1)[0],
            "E clean": eval_E(E_clean, Xte, yte, T1)[0],
            "E exact": eval_E(E_clean, Xte, yte)[0],
            "A noisy 200": eval_A(A_noisy, Xte, yte, T1, shots=200, rng=r),
            "E filter 200": eval_E(E_clean, Xte, yte, T1, filt=True, shots=200, rng=r)[0],
            "kept": eval_E(E_clean, Xte, yte, T1)[1],
            "max dparam F3": float(np.max(np.abs(E_noisyf - E_clean))),
        }
        rows.append(rec)
    out = {k: float(np.mean([rr[k] for rr in rows])) for k in rows[0]}
    out["max dparam F3"] = float(max(rr["max dparam F3"] for rr in rows))
    return out, rows


def verdict(res):
    ds = list(res)
    mean = lambda k: float(np.mean([res[d][k] for d in ds]))  # noqa: E731
    r1 = max(res[d]["max dparam F3"] for d in ds) < 1e-6 and abs(mean("E noisy+f") - mean("E filter")) < 1e-12
    r2 = mean("A noisy") >= mean("A clean")
    r3 = mean("E filter") >= mean("A noisy") and mean("E filter") >= mean("E noisy")
    r4 = mean("E filter 200") >= mean("A noisy 200") - 0.01
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4}


def main(seed=77, splits=5, epochs=120, datasets=None):
    res = {}
    for name in datasets or Q.DATASETS:
        res[name], _ = run_dataset(name, seed=seed, splits=splits, epochs=epochs)
        r = res[name]
        print(f"{name}: exact A {r['A exact']:.3f}, E {r['E exact']:.3f} | under T1 {GAMMA}: A clean {r['A clean']:.3f}, "
              f"A noisy {r['A noisy']:.3f}, E clean {r['E clean']:.3f}, E noisy {r['E noisy']:.3f}, "
              f"E filter {r['E filter']:.3f}, E noisy+f {r['E noisy+f']:.3f}")
        print(f"   200 shots: A noisy {r['A noisy 200']:.3f}, E filter {r['E filter 200']:.3f} (kept fraction {r['kept']:.2f}) | "
              f"F3 max parameter difference {r['max dparam F3']:.1e}", flush=True)
    if datasets:
        import json
        print(json.dumps(res))
        return res, None
    v = verdict(res)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, v


if __name__ == "__main__":
    main(datasets=sys.argv[1:] or None)
