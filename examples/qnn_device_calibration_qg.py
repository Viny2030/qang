"""
Calibrating with the device itself, not with T1/T2 (§120)

Section 119 found that a calibration from the published T1 and T2 adds
nothing on IBM device noise models, where gate errors dominate. Here the
calibration comes from runs on the device, at the cost of a few circuits:

  E  echo-aware training: the 5 echo circuits of the noiselessly trained
     model A measure the in-sector mixing M; a model is retrained with the
     filter and the forward half M^1/2 applied to its sector distribution, so
     it learns to classify through the mixing; on the device it is read with
     the filter alone.
  H  head recalibration: 40 training inputs are run on the device with model
     A; the linear readout (5 weights and a bias) is refit by logistic
     regression on their device features, read with the filter (H) or with
     the filter + echo (H echo). The angles stay those of A.

Baselines: A raw, A with the filter, A with the filter + echo. Same models,
data (iris, cancer, wine, digits; 189 test inputs), fixed layout and IBM
fake backends (brisbane, sherbrooke, torino) as section 119; 1000 shots per
input, echoes at 4000.

Pre-registered predictions (committed before any noisy run; code checked in
noiseless --mode local only):
  G1  pooled over the three backends, H is at least as accurate as A with
      the filter.
  G2  pooled, E is at least as accurate as A with the filter.
  G3  pooled, the best of E, H and H echo is within 0.5 points of A's
      noiseless accuracy.
  G4  pooled, H echo is at least as accurate as A with the filter + echo.
  G5  on every backend, at least one of E, H, H echo is at least as accurate
      as A with the filter.

python examples/qnn_device_calibration_qg.py --mode local
python examples/qnn_device_calibration_qg.py --mode fake --out fake.json

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qg_radius_hardware_qg as R  # noqa: E402
import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import (echo_transfer_matrix, filter_distribution, matrix_power_stochastic,  # noqa: E402
                          sector_states, unmix_sector)

N = 5
SHOTS, SHOTS_CAL, N_CAL, EPOCHS = 1000, 4000, 40, 120
ONE = [1 << (N - 1 - q) for q in range(N)]
SECTOR = sector_states(N, 1)
KEYS = ("noiseless", "A raw", "A qang", "A qang + echo", "E", "H", "H echo")


class MixedQNN(WeightQNN):
    """WeightQNN whose filtered readout passes through a measured in-sector mixing A."""

    def __init__(self, mix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mix = np.asarray(mix, float)

    def qg_z(self, probs, qang=True):
        probs = np.atleast_2d(probs)
        if qang is not True:
            return super().qg_z(probs, qang)
        f = np.array([filter_distribution(p, self.n, self.weight)[0] for p in probs])
        f[:, SECTOR] = f[:, SECTOR] @ self.mix.T
        return f @ self.features


def feats(m, probs, how, M=None):
    if how == "raw":
        return np.array([WeightQNN.qg_z(m, p, qang=False)[0] for p in probs])
    if how == "qang":
        return np.array([WeightQNN.qg_z(m, p, qang=True)[0] for p in probs])
    return np.array([WeightQNN.qg_z(m, unmix_sector(p, M, N, 1, power=0.5), qang=False)[0] for p in probs])


def refit_head(F, y):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=10.0, max_iter=2000).fit(F, y)
    return lr.coef_[0], lr.intercept_[0]


def run_backend(mode, name, epochs=EPOCHS, datasets=Q.DATASETS):
    backend = H.get_backend("fake", name or "fake_brisbane")
    out = {"backend": name if mode == "fake" else "local", "per dataset": {}}
    layout = None
    for ds in datasets:
        Xtr, Xte, ytr, yte = P.data(ds)
        A = WeightQNN(N, 1).fit(Xtr, ytr, epochs, seed=DS.TRAIN_SEED)
        if layout is None:
            layout, _ = P.calibration(backend, H.build_circuit(A, Xte[0]))
            out["layout"] = layout
        cal = np.random.default_rng(1200).choice(len(Xtr), min(N_CAL, len(Xtr)), replace=False)
        Xc, yc = Xtr[cal], np.asarray(ytr)[cal]
        n, k = len(Xte), len(Xc)
        circ = [H.build_circuit(A, x) for x in Xte] + [R.echo_circuit(A, j) for j in range(N)] + [H.build_circuit(A, x) for x in Xc]
        shots = [SHOTS] * n + [SHOTS_CAL] * N + [SHOTS] * k
        pr = [H.counts_to_probs(c) for c in P.counts_for(mode, backend, circ, shots, layout)]
        test, echo, calp = pr[:n], pr[n:n + N], pr[n + N:]
        M = echo_transfer_matrix(echo, N, 1, prepared=ONE)
        w, b = A.params_[A.n_theta:A.n_theta + A.n_head], A.params_[-1]
        acc = lambda F, w_, b_: int(np.sum(((F @ w_ + b_) > 0).astype(int) == yte))  # noqa: E731
        r = {"inputs": n, "noiseless": int(np.sum(A.predict(Xte) == yte)),
             "A raw": acc(feats(A, test, "raw"), w, b), "A qang": acc(feats(A, test, "qang"), w, b),
             "A qang + echo": acc(feats(A, test, "echo", M), w, b)}
        wh, bh = refit_head(feats(A, calp, "qang"), yc)
        r["H"] = acc(feats(A, test, "qang"), wh, bh)
        wh2, bh2 = refit_head(feats(A, calp, "echo", M), yc)
        r["H echo"] = acc(feats(A, test, "echo", M), wh2, bh2)
        Em = MixedQNN(matrix_power_stochastic(M, 0.5), N, 1).fit(Xtr, ytr, epochs, seed=DS.TRAIN_SEED)
        pe = [H.counts_to_probs(c) for c in P.counts_for(mode, backend, [H.build_circuit(Em, x) for x in Xte], [SHOTS] * n, layout)]
        r["E"] = acc(feats(Em, pe, "qang"), Em.params_[Em.n_theta:Em.n_theta + Em.n_head], Em.params_[-1])
        r["echo diagonal"] = [float(x) for x in np.diag(M)]
        out["per dataset"][ds] = r
    tot = sum(r["inputs"] for r in out["per dataset"].values())
    out["pooled"] = {k: sum(r[k] for r in out["per dataset"].values()) / tot for k in KEYS}
    out["pooled"]["inputs"] = tot
    return out


def verdict(results):
    P_ = [r["pooled"] for r in results]
    mean = lambda k: float(np.mean([p[k] for p in P_]))  # noqa: E731
    return {
        "G1": mean("H") >= mean("A qang"),
        "G2": mean("E") >= mean("A qang"),
        "G3": mean("noiseless") - max(mean("E"), mean("H"), mean("H echo")) <= 0.005,
        "G4": mean("H echo") >= mean("A qang + echo"),
        "G5": all(max(p["E"], p["H"], p["H echo"]) >= p["A qang"] for p in P_),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "fake"], default="fake")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = P.BACKENDS if args.mode == "fake" else [None]
    rs = []
    for nm in names:
        r = run_backend(args.mode, nm)
        rs.append(r)
        print(r["backend"], {k: round(r["pooled"][k], 4) for k in KEYS}, flush=True)
    if args.mode == "fake":
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(rs).items()))
    if args.out:
        json.dump(rs, open(args.out, "w"), indent=1)
    return rs


if __name__ == "__main__":
    main()
