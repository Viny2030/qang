"""
A weight-2 QNN with and without qang (§79)

§76-§78 used a weight-1 QNN (model E). Its sector has dimension n, which
makes it a classical quadratic form (§76 F2). §78 F4 showed that, under
equal T1, the qg filter is exact in every fixed-weight sector. This section
moves the classifier to weight 2 and reports every result twice: with qang
(the readout keeps only the weight-2 shots, qang.sectors.filter_distribution)
and without qang (the raw readout over all shots). The difference is the
effect of qang.

Model W (5 qubits, weight 2). With v = (x1, x2, x3, x4, 1)/norm as in §76,
the input state has amplitude v_i v_j on the basis state with qubits i and j
excited (i < j), normalized: products of features and the features
themselves (pairs with the constant component). The same 3 layers of RBS
rotations as E (15 parameters), readout z = sum_i c_i <Z_i> + b. The sector
has dimension C(5, 2) = 10 instead of 5. It is still classically simulable
(C(n, 2) grows polynomially), but the output is no longer a quadratic form
in v: it is quartic.

Noise conditions (per qubit after every sublayer, as in §77-§78):
  T1  equal damping gamma = 0.08
  H   unequal damping 0.04-0.12 (mean 0.08)
  D   equal damping 0.08 plus dephasing 0.03
For each model (E and W) and condition, four readings:
  clean qang / clean raw   trained without noise, evaluated under the
                           condition with / without the filter
  aware qang / aware raw   trained and evaluated under the condition with /
                           without the filter (noise-aware training)
Kept fraction under equal T1 (F4): E (1 - gamma)^9 = 0.472, W (1 - gamma)^18
= 0.223.

Pre-registered predictions (written and committed before the seed-79 run;
code debugged on seed 1 with few epochs):
  P1  F4 at weight 2: under equal T1 the clean-trained W with qang has
      exactly its noiseless accuracy on every split.
  P2  qang helps more at weight 2: under equal T1, clean-trained, the gain
      with qang (qang - raw, mean) is at least 1 point for E and at least as
      large for W as for E.
  P3  weight 2 does not cost accuracy: W exact >= E exact - 1 point (mean).
  P4  with noise-aware training, qang still does not hurt W: mean aware qang
      >= aware raw under T1, H and D.
  P5  the shot cost: at 200 shots under equal T1, W aware qang >= W aware raw
      - 1 point (mean), although the filter keeps only 22% of the shots.

Exact block simulator: T1 and dephasing keep the state block-diagonal in
the Hamming weight, so one block per weight is propagated (checked against
the full density matrix in the tests). Uses the installed library
(pip install qang) for the filter and the Hamming weights. Needs
scikit-learn. OMP_NUM_THREADS=1 python examples/qnn_weight2_qg.py [datasets]

Findings (python examples/qnn_weight2_qg.py):

Seed 79, 5 splits per dataset, 120 epochs. Means over the four datasets;
"diff" is with qang minus without qang (one test sample is 2-3 points):

  exact (no noise): E 0.951, W 0.914
                   trained without noise          trained under the noise
                   qang   without  diff           qang   without  diff
  T1  E            0.951  0.929   +0.021          0.951  0.947   +0.004
  T1  W            0.914  0.780   +0.134          0.914  0.920   -0.006
  H   E            0.949  0.911   +0.039          0.952  0.947   +0.005
  H   W            0.915  0.750   +0.164          0.921  0.918   +0.003
  D   E            0.911  0.796   +0.115          0.945  0.948   -0.003
  D   W            0.870  0.664   +0.206          0.915  0.908   +0.007
  200 shots, trained under the noise: diff between -0.006 and 0.000 for
  every model and condition. Kept fraction: E 0.472 (H 0.452), W 0.223
  (H 0.215) = (1 - 0.08)^18.
  Without qang per dataset (T1, W, trained without noise): iris 0.813,
  cancer 0.887, wine 0.615, digits 0.803; with qang 0.913, 0.913, 0.892,
  0.937 (= exact).

  * P1 PASS. F4 at weight 2: with qang the clean-trained W keeps exactly
    its noiseless accuracy on all 20 splits.
  * P2 PASS. The effect of qang grows with the weight: +2.1 points for E,
    +13.4 for W under equal T1 (trained without noise); +16.4 under unequal
    T1 and +20.6 with dephasing. Without qang the weight-2 model falls to
    0.66-0.78, because 78% of its shots have decayed.
  * P3 FAILS. Weight 2 costs 3.7 points (W 0.914 against E 0.951), worst on
    wine (5.7). The pair-product encoding with a sum-of-<Z> readout does
    not use the larger sector well on these datasets.
  * P4 FAILS, narrowly. Trained under the noise, the model without qang
    learns to read the decayed shots and the two readings meet: -0.6
    points under T1 (the failure), +0.3 under H, +0.7 under D, all well
    under one test sample.
  * P5 PASS. At 200 shots the filter costs 0.5 points while keeping 22% of
    the shots.
  * Not predicted: under unequal T1 the clean-trained W with qang (0.915)
    is within 0.6 points of noise-aware training (0.921); E loses 0.3. The
    §78 S2 loss (1.5 points) came from a different seed and splits, so the
    size of that loss is not stable across runs. Dephasing still requires
    noise-aware training (0.870 vs 0.915 for W).
  Verdict. The difference made by qang depends on how the model is
  trained. Trained on a simulator and run under noise, qang is decisive
  and more so at higher weight (+13 to +21 points for W). Trained under the
  calibrated noise, a model without qang learns the decay and catches up,
  and qang changes nothing measurable (within +-0.7 points). The value of
  qang is to make noise-free training valid for T1, at a shot cost of
  1 - (1 - gamma)^(k depth). No quantum advantage: the weight-2 model is
  less accurate than the weight-1 model here and still classically
  simulable (sector dimension C(n, 2)). Limitations: one encoding and
  readout for weight 2, 5 qubits, simulated noise, small test sets.
"""

import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_noise_aware_qg as N77  # noqa: E402
import qnn_realistic_noise_qg as R  # noqa: E402
import qnn_unary_norm_qg as E  # noqa: E402
from qang.sectors import filter_distribution, hamming_weights  # noqa: E402

N = E.N
DIM = E.DIM
WEIGHT = hamming_weights(N)
GAMMA = 0.08
CONDITIONS = {
    "T1": (np.full(N, GAMMA), 0.0),
    "H": R.CONDITIONS["H"]["E"],
    "D": R.CONDITIONS["D"]["E"],
}
MODELS = {"E": 1, "W": 2}

IDX = {k: np.where(WEIGHT == k)[0] for k in range(N + 1)}
POS = {k: {s: i for i, s in enumerate(IDX[k])} for k in IDX}


def _bit(q):
    return 1 << (N - 1 - q)


# per weight k and qubit q: excitation mask, jump map to weight k-1, Z signs
EXC, JUMP, ZS = {}, {}, {}
for _k in range(1, N + 1):
    for _q in range(N):
        exc = np.array([(s & _bit(_q)) != 0 for s in IDX[_k]])
        J = np.zeros((len(IDX[_k - 1]), len(IDX[_k])))
        for i, s in enumerate(IDX[_k]):
            if s & _bit(_q):
                J[POS[_k - 1][s ^ _bit(_q)], i] = 1.0
        EXC[_k, _q], JUMP[_k, _q] = exc, J
for _k in range(N + 1):
    for _q in range(N):
        ZS[_k, _q] = np.array([1.0 if s & _bit(_q) else 0.0 for s in IDX[_k]])


def encode_w2(Xs):
    v = E.augmented(Xs)
    psi = np.zeros((len(Xs), DIM))
    for i, j in itertools.combinations(range(N), 2):
        psi[:, _bit(i) | _bit(j)] = v[:, i] * v[:, j]
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def encode(model, Xs):
    return E.encode(Xs) if model == "E" else encode_w2(Xs)


def probs_block(theta, psi0, k0, gam=None, phi=0.0):
    """Exact probabilities for a weight-k0 input under per-qubit T1 gam and
    dephasing phi (gam None: noiseless). The state stays block-diagonal in
    the Hamming weight."""
    S = len(psi0)
    a = psi0[:, IDX[k0]]
    M = {k: np.zeros((S, len(IDX[k]), len(IDX[k]))) for k in range(k0 + 1)}
    M[k0] = np.einsum("si,sj->sij", a, a)
    for L in E.sublayers(theta):
        for k in range(1, k0 + 1):
            B = L[np.ix_(IDX[k], IDX[k])]
            M[k] = B[None] @ M[k] @ B.T[None]
        if gam is None:
            continue
        for q in range(N):
            g = gam[q]
            for k in range(1, k0 + 1):  # jumps from k to k-1 use M[k] before damping
                if g:
                    J = JUMP[k, q]
                    M[k - 1] = M[k - 1] + g * (J[None] @ M[k] @ J.T[None])
                    d = np.where(EXC[k, q], math.sqrt(1 - g), 1.0)
                    M[k] = M[k] * d[None, :, None] * d[None, None, :]
            if phi:
                for k in range(1, k0 + 1):
                    z = ZS[k, q]
                    f = np.where(z[:, None] == z[None, :], 1.0, 1 - 2 * phi)
                    M[k] = M[k] * f[None]
    probs = np.zeros((S, DIM))
    for k in range(k0 + 1):
        probs[:, IDX[k]] = np.einsum("sii->si", M[k])
    return probs


def local_z(probs, k, qang=False):
    if qang:  # the qg filter from the library: keep the weight-k shots, renormalize
        probs = np.array([filter_distribution(pr, N, k)[0] for pr in probs])
    return probs @ E.ZSIGN


def readout(model, psi0, cond=None, qang=False):
    k = MODELS[model]

    def f(theta):
        probs = probs_block(theta, psi0, k) if cond is None else probs_block(theta, psi0, k, *cond)
        return local_z(probs, k, qang)
    return f


def evaluate(model, params, Xte, yte, cond=None, qang=False, shots=None, rng=None):
    k = MODELS[model]
    psi0 = encode(model, Xte)
    th = params[:E.NP]
    probs = probs_block(th, psi0, k) if cond is None else probs_block(th, psi0, k, *cond)
    kept = float(np.mean(probs[:, IDX[k]].sum(axis=1)))
    if shots:
        probs = np.array([rng.multinomial(shots, pr / pr.sum()) / shots for pr in probs])
    z = local_z(probs, k, qang) @ params[E.NP:E.NP + N] + params[-1]
    return N77.accuracy(z, yte), kept


def run_dataset(name, seed=79, splits=5, epochs=120):
    rng = np.random.default_rng(seed)
    Xa, ya, use_pca = Q.load(name, rng)
    rows = []
    for s in range(splits):
        Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, seed * 100 + s)
        sd = seed * 1000 + 10 * s
        rec = {}
        for m in MODELS:
            psi = encode(m, Xtr)
            train = lambda ro: N77.adam_train(E.NP, N, ro, ytr, np.random.default_rng(sd + 2), epochs)  # noqa: E731
            clean = train(readout(m, psi))
            rec[f"{m} exact"] = evaluate(m, clean, Xte, yte)[0]
            for c, cond in CONDITIONS.items():
                aq = train(readout(m, psi, cond, qang=True))
                ar = train(readout(m, psi, cond, qang=False))
                acc, kept = evaluate(m, clean, Xte, yte, cond, qang=True)
                rec[f"{c} {m} clean qang"] = acc
                rec[f"{c} {m} kept"] = kept
                rec[f"{c} {m} clean raw"] = evaluate(m, clean, Xte, yte, cond)[0]
                rec[f"{c} {m} aware qang"] = evaluate(m, aq, Xte, yte, cond, qang=True)[0]
                rec[f"{c} {m} aware raw"] = evaluate(m, ar, Xte, yte, cond)[0]
                r2 = np.random.default_rng(seed + s)
                rec[f"{c} {m} aware qang 200"] = evaluate(m, aq, Xte, yte, cond, True, 200, r2)[0]
                r2 = np.random.default_rng(seed + s)
                rec[f"{c} {m} aware raw 200"] = evaluate(m, ar, Xte, yte, cond, False, 200, r2)[0]
        rec["P1 split ok W"] = float(rec["T1 W clean qang"] == rec["W exact"])
        rows.append(rec)
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out, rows


def verdict(res):
    ds = list(res)
    m = lambda k: float(np.mean([res[d][k] for d in ds]))  # noqa: E731
    gain = lambda c, mod, t="clean": m(f"{c} {mod} {t} qang") - m(f"{c} {mod} {t} raw")  # noqa: E731
    return {
        "P1": m("P1 split ok W") == 1.0,
        "P2": gain("T1", "E") >= 0.01 and gain("T1", "W") >= gain("T1", "E"),
        "P3": m("W exact") >= m("E exact") - 0.01,
        "P4": all(gain(c, "W", "aware") >= 0 for c in CONDITIONS),
        "P5": m("T1 W aware qang 200") >= m("T1 W aware raw 200") - 0.01,
    }


def table(res):
    """Mean over datasets of every reading, with the qang - raw difference."""
    ds = list(res)
    m = lambda k: float(np.mean([res[d][k] for d in ds]))  # noqa: E731
    lines = [f"exact: E {m('E exact'):.3f}, W {m('W exact'):.3f}"]
    for c in CONDITIONS:
        for mod in MODELS:
            parts = []
            for t in ("clean", "aware", "aware 200"):
                key = t.split()[0]
                suf = " 200" if "200" in t else ""
                q, r = m(f"{c} {mod} {key} qang{suf}"), m(f"{c} {mod} {key} raw{suf}")
                parts.append(f"{t}: qang {q:.3f} / without {r:.3f} (diff {q - r:+.3f})")
            lines.append(f"{c} {mod} | " + " | ".join(parts) + f" | kept {m(f'{c} {mod} kept'):.3f}")
    return "\n".join(lines)


def main(seed=79, splits=5, epochs=120, datasets=None):
    res = {}
    for name in datasets or Q.DATASETS:
        res[name], _ = run_dataset(name, seed=seed, splits=splits, epochs=epochs)
        print(name + "\n" + table({name: res[name]}), flush=True)
    if datasets:
        print(json.dumps(res))
        return res, None
    print("all datasets\n" + table(res))
    v = verdict(res)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, v


if __name__ == "__main__":
    main(datasets=sys.argv[1:] or None)
