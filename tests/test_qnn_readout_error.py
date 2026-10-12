"""Tests for examples/qnn_readout_error_qg.py (§129): the per-qubit readout confusion."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_readout_confusion_acts_on_the_right_bit():
    import qnn_readout_error_qg as R

    N = R.N
    p = np.zeros((1, 2 ** N))
    p[0, 0] = 1.0                                      # |00000>
    errs = [(0.0, 0.0)] * N
    assert np.allclose(R.apply_readout(p, errs), p)
    for v in range(N):
        e = [(0.0, 0.0)] * N
        e[v] = (0.1, 0.0)                              # only Qiskit qubit v can flip 0 -> 1
        out = R.apply_readout(p, e)[0]
        assert out[0] == pytest.approx(0.9) and out[1 << v] == pytest.approx(0.1)
    q = np.random.default_rng(0).dirichlet(np.ones(2 ** N), size=3)
    out = R.apply_readout(q, [(0.02, 0.05)] * N)
    assert np.allclose(out.sum(axis=1), 1) and np.all(out >= 0)


def test_verdict():
    import qnn_readout_error_qg as R

    dev = {"b": {"kept device": 0.70, "kept predicted 124": 0.80, "A qang device": 184}}
    rows = [{"backend": "b", "kept predicted (readout)": 0.72, "A qang predicted (readout)": 183}]
    assert R.verdict(rows, dev) == {"K1": True, "K2": True, "K3": True}
    rows[0]["kept predicted (readout)"] = 0.76
    assert R.verdict(rows, dev)["K1"] is False and R.verdict(rows, dev)["K2"] is False
