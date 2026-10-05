"""Tests for examples/qnn_scaling_qg.py (§111): data preparation and verdict."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_prepare_and_verdict():
    pytest.importorskip("sklearn")
    import qnn_scaling_qg as S

    Xtr, Xte, ytr, yte = S.prepare("wine", 8, 1)
    assert Xtr.shape[1] == 7 and np.all(np.abs(Xte) <= 1)
    rows = []
    for n in (5, 8):
        for w in (1, 2):
            loss = 0.01 * n * w
            rows.append({"n": n, "weight": w, "dataset": "x", "kept": 0.5, "noiseless": 0.9, "T1 with qang": 0.9,
                         "T1 without qang": 0.9 - loss, "unequal with qang": 0.89, "unequal without qang": 0.85,
                         "noise-aware without qang": 0.89})
    assert S.verdict(rows) == {"SC1": True, "SC2": True, "SC3": True, "SC4": True, "SC5": True}
