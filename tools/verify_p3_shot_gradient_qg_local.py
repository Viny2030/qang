"""Pre-registered check for P3 follow-up: shot-estimated gradient magnitude advantage.

Prediction (SHA-256 recorded in examples/data/shot_gradient_snr_qg_local.prereg):
  qg-local cost median |shot-grad| / global Z^n median |shot-grad| >= 3.0
  for n=6, L=2 HEA, 300 random inits, 100 shots per parameter-shift circuit,
  first parameter, theta-space.

This is a measurement-level ideal-simulator check, not a hardware claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from shot_gradient_snr_qg_local import run, summarize  # noqa: E402

PREDICTION = {
    "claim": "qg-local cost yields >=3x median |shot-grad| vs global Z^n under 100 shots at n=6,L=2",
    "direction": ">=",
    "eps": 0.05,
    "id": "P3-SHOT-GRAD-QG-LOCAL",
    "metric": "median_abs_ratio_local_over_global",
    "n_qubits": 6,
    "n_samples": 300,
    "param_index": 0,
    "reps": 2,
    "script": "examples/shot_gradient_snr_qg_local.py",
    "seed": 7,
    "shots": 100,
    "threshold": 3.0,
}


def canonical(prediction):
    return json.dumps(prediction, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(prediction=PREDICTION):
    return hashlib.sha256(canonical(prediction).encode("utf-8")).hexdigest()


def main():
    d = digest()
    prereg = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "shot_gradient_snr_qg_local.prereg")
    text = open(prereg, encoding="utf-8").read()
    assert d in text, f"prereg digest mismatch: {d}"
    rows = run(
        n_qubits=PREDICTION["n_qubits"],
        reps=PREDICTION["reps"],
        n_samples=PREDICTION["n_samples"],
        shots=PREDICTION["shots"],
        seed=PREDICTION["seed"],
        param_index=PREDICTION["param_index"],
        eps=PREDICTION["eps"],
    )
    summary = summarize(rows)
    ratio = summary["theta"]["ratio_local_over_global"]
    clipped_ratio = summary["qg_clipped"]["ratio_local_over_global"]
    print(f"[0] prereg SHA-256 = {d[:16]}...")
    print(f"[1] theta ratio local/global = {ratio:.4f} (threshold {PREDICTION['threshold']})")
    print(f"[2] clipped qg ratio local/global = {clipped_ratio:.4f}")
    print(f"[3] raw qg max abs qg_local = {summary['qg_raw']['max_abs_qg_local']:.4f}")
    assert ratio >= PREDICTION["threshold"]
    print("verify_p3_shot_gradient_qg_local: OK")
    return summary


if __name__ == "__main__":
    main()
