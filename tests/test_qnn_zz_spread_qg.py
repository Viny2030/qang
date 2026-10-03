"""Tests for examples/qnn_zz_readout_qg.py (§81) and
examples/qnn_t1_spread_qg.py (§82): small runs and verdict logic."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

import qnn_t1_spread_qg as V  # noqa: E402
import qnn_zz_readout_qg as Z  # noqa: E402


def test_zz_small_run():
    rows = Z.run_seed(1, splits=1, epochs=2, datasets=["iris"])
    q = Z.quantities(rows)
    assert q["F4 ok"] == 1.0
    assert set(Z.verdict({k: (v, v, v) for k, v in q.items()})) == {"U1", "U2", "U3", "U4", "U5"}


def test_spread_small_run():
    rows = V.run_seed(1, splits=1, epochs=2, datasets=["iris"])
    q = V.quantities(rows)
    assert q["E exact at s=0"] == 1.0 and q["W exact at s=0"] == 1.0
    assert q["E loss 0.0"] == 0.0
    assert set(V.verdict({k: (v, v, v) for k, v in q.items()})) == {"V1", "V2", "V3", "V4", "V5"}


def test_encoding_study_small_run():
    import qnn_weight2_encoding_qg as X

    rows = X.run_seed(1, splits=1, epochs=2, datasets=["iris"])
    q = X.quantities(rows)
    assert q["Wring exact at T1 with qang"] == 1.0
    assert set(X.verdict({k: (v, v, v) for k, v in q.items()})) == {"X1", "X2", "X3", "X4"}
