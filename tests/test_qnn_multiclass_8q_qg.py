"""Tests for examples/qnn_multiclass_8q_qg.py (§114): data preparation and verdict."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_prepare_and_verdict():
    pytest.importorskip("sklearn")
    import qnn_multiclass_8q_qg as Q

    Xtr, Xte, ytr, yte = Q.prepare(8, 1, per_class=20)
    assert Xtr.shape[1] == 7 and np.all(np.abs(Xte) <= 1) and set(ytr) == set(range(8))
    rows = []
    for C in (3, 5, 8):
        for ro, f in (("qubit", 0.5), ("head", 1.0)):
            loss = f * 0.03 * C
            rows.append({"classes": C, "readout": ro, "seed": 0, "noiseless": 0.9, "T1 with qang": 0.9,
                         "T1 without qang": 0.9 - loss, "unequal T1 with qang": 0.89,
                         "unequal T1 without qang": 0.8, "noise-aware without qang": 0.89})
    assert Q.verdict(rows) == {"Q1": True, "Q2": True, "Q3": True, "Q4": True, "Q5": True, "Q6": True}


def test_noiseless_run_small():
    pytest.importorskip("sklearn")
    import qnn_multiclass_8q_qg as Q

    rows = Q.run(0, classes=(3,), epochs=1, noisy=False)
    assert len(rows) == 2 and all(0 <= r["noiseless"] <= 1 for r in rows)
