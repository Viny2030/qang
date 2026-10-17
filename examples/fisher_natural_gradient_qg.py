"""
The qang angle as the Fisher-flat coordinate, shift rules, and the natural gradient from qg_Y correlators (§159)

Claim tested: does writing a variational circuit in probability (qg) coordinates speed up training, or is the
angle the right coordinate? Theory (qang.fisher): per shot, the Fisher information is 1 in the angle and
1 / (1 - qg^2) in qg_Z; no shift rule needs fewer than 2 evaluations per parameter in any coordinate; for a
real RY + CZ ansatz every diagonal entry of the Fubini-Study metric is exactly 1/4 and the curvature lives in the
off-diagonal terms, which inside a layer are qg_Y correlators <Y_i Y_j> / 4, read from one Y-basis circuit.

Checked during development, before the predictions (not a prediction): n = 4, L = 3: diag(g) = 0.25 exactly,
the block metric equals the diagonal blocks of the exact metric to 1e-16, off-diagonal entries up to 0.24.
Adam (lr 0.05, 800 steps, 3 seeds) ends 0.0245 above the exact ground energy, the floor of this ansatz.

Set-up (fixed before the run):
  Task: open transverse-field Ising chain, n = 4, h = 1, E0 = -4.7588; ansatz L = 3 layers of RY on every qubit
  then a CZ chain (12 parameters), initial angles uniform in [0, 2 pi). Success: E - E0 <= 0.03 within 300
  steps. Exact gradients (parameter shift on the state vector).
  Methods: GD-theta (plain descent in the angles); GD-qg (descent in q = cos theta with the branch sign,
  qang.gradients.signed_qg_step, clipped Jacobian eps = 0.05); QNG-block (theta -= lr (g_block + 1e-3 I)^-1
  grad, g_block from the qg_Y correlators); QNG-full (the same with the exact metric).
  Learning rate per method chosen on tuning seeds 0-5 from {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}: the lr with
  the lowest median steps (failures count as 301); reported on test seeds 6-25.
  Part V: 20 random angle vectors (seed 159), the block metric of each layer from 1000 Y-basis shots (Aer,
  noiseless) against the exact block.

Pre-registered predictions (sha256 of this file recorded in examples/data/fisher_natural_gradient_qg.prereg
before the run):
  V1  the shot-estimated block metric is within 0.04 (max entry) of the exact one in at least 19 of 20 points.
  G1  median steps of GD-qg >= 1.5 x median steps of GD-theta.
  G2  median steps of QNG-block <= median steps of GD-theta / 1.5.
  G3  median steps of QNG-full <= median steps of QNG-block.
  G4  successes of QNG-block >= successes of GD-theta (out of 20).

python examples/fisher_natural_gradient_qg.py

Findings (pre-registration sha256 ca4c4ff5...dd90, recorded 2026-10-17 19:04 UTC; the run used that exact file,
then only this block was added; data/fisher_natural_gradient_qg_run.log):

  Part V: max entry error of the 1000-shot block metric 0.013-0.022 in all 20 points.
  Test seeds 6-25 (lr from tuning), steps to E - E0 <= 0.03:

  method      lr     median steps  successes  final gap (median)
  GD-theta    0.2    103           19 / 20    0.0298
  GD-qg       0.005  301 (never)    0 / 20    0.351
  QNG-block   0.05   79.5          19 / 20    0.0298
  QNG-full    0.1    22.5          20 / 20    0.0294

  V1 PASS (20 of 20 within 0.04).
  G1 PASS: descent in qg coordinates never reaches the tolerance at any lr of the grid (tuning medians 301 for all
     seven), while descent in the angle does.
  G2 FAIL: the block metric speeds descent up by 1.3x (103 -> 79.5), not 1.5x.
  G3 PASS: the full metric is fastest, 4.6x fewer steps than GD-theta.
  G4 PASS: 19 = 19.
  * Post hoc, circuit evaluations to the tolerance (parameter shift 24 per step; block metric +3 Y-basis
    circuits; full metric about p(p+1)/2 = 78 overlap circuits on hardware): GD-theta 103 x 24 = 2472,
    QNG-block 79.5 x 27 = 2147, QNG-full 22.5 x 102 = 2295. Per evaluation the three are within 15%.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import fisher as F  # noqa: E402
from qang import gradients as GR  # noqa: E402

N, L, HFIELD = 4, 3, 1.0
TOL, MAX_STEPS = 0.03, 300
LRS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
TUNE, TEST = range(0, 6), range(6, 26)
METHODS = ("GD-theta", "GD-qg", "QNG-block", "QNG-full")
OUT = os.path.join(HERE, "data", "fisher_natural_gradient_qg.json")

H = F.tfim_hamiltonian(N, HFIELD)
E0 = float(np.linalg.eigvalsh(H)[0])


def run(method, lr, seed):
    """Steps to E - E0 <= TOL (MAX_STEPS + 1 if never) and the final gap."""
    th = np.random.default_rng(seed).uniform(0, 2 * math.pi, N * L)
    q = np.cos(th)
    s = np.where(np.sin(th) >= 0, 1, -1)
    for step in range(MAX_STEPS + 1):
        gap = F.energy(th, H, N, L) - E0
        if gap <= TOL:
            return step, gap
        if step == MAX_STEPS:
            return MAX_STEPS + 1, gap
        g = F.gradient(th, H, N, L)
        if method == "GD-theta":
            th = th - lr * g
        elif method == "GD-qg":
            for i in range(th.size):
                q[i], s[i] = GR.signed_qg_step(q[i], int(s[i]), g[i], lr)
            th = np.array([GR.signed_theta(q[i], int(s[i])) for i in range(th.size)])
        else:
            G = F.block_metric(th, N, L) if method == "QNG-block" else F.full_metric(th, N, L)
            th = th - lr * np.linalg.solve(G + 1e-3 * np.eye(th.size), g)


def part_v():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(seed_simulator=159)
    rng = np.random.default_rng(159)
    errs = []
    for _ in range(20):
        th = rng.uniform(0, 2 * math.pi, N * L)
        exact = F.block_metric(th, N, L)
        worst = 0.0
        for layer in range(L):
            qc = transpile(F.y_basis_circuit(th, N, L, layer), sim)
            c = sim.run(qc, shots=1000).result().get_counts()
            est = F.block_metric_from_counts(c, N)
            blk = exact[layer * N:(layer + 1) * N, layer * N:(layer + 1) * N]
            worst = max(worst, float(np.abs(est - blk).max()))
        errs.append(worst)
    return errs


def main():
    errs = part_v()
    print(json.dumps({"V max entry errors": [round(e, 4) for e in errs]}), flush=True)
    tuning, best_lr = {}, {}
    for m in METHODS:
        tuning[m] = {}
        for lr in LRS:
            tuning[m][lr] = [run(m, lr, sd)[0] for sd in TUNE]
        best_lr[m] = min(LRS, key=lambda lr: (np.median(tuning[m][lr]), lr))
        print(json.dumps({"method": m, "tuning medians": {lr: float(np.median(v)) for lr, v in tuning[m].items()},
                          "lr": best_lr[m]}), flush=True)
    test = {m: [run(m, best_lr[m], sd) for sd in TEST] for m in METHODS}
    steps = {m: [s for s, _ in v] for m, v in test.items()}
    med = {m: float(np.median(v)) for m, v in steps.items()}
    succ = {m: sum(s <= MAX_STEPS for s in v) for m, v in steps.items()}
    for m in METHODS:
        print(json.dumps({"method": m, "lr": best_lr[m], "median steps": med[m], "successes": succ[m],
                          "steps": steps[m]}), flush=True)
    v = {"V1": sum(e < 0.04 for e in errs) >= 19,
         "G1": med["GD-qg"] >= 1.5 * med["GD-theta"],
         "G2": med["QNG-block"] <= med["GD-theta"] / 1.5,
         "G3": med["QNG-full"] <= med["QNG-block"],
         "G4": succ["QNG-block"] >= succ["GD-theta"]}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"E0": E0, "V_errors": errs, "best_lr": best_lr,
                   "tuning": {m: {str(k): x for k, x in d.items()} for m, d in tuning.items()},
                   "test": {m: [[int(a), float(b)] for a, b in t] for m, t in test.items()},
                   "median": med, "successes": succ, "verdict": v}, fh, indent=1)
    return v


if __name__ == "__main__":
    main()
