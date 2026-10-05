"""Tests for examples/qnn_echo_narrow_margin_qg.py (§108): input selection and verdict."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_selection_and_verdict():
    pytest.importorskip("sklearn")
    import qnn_echo_narrow_margin_qg as M

    m, X, y, d0 = M.model_and_inputs(epochs=2)
    assert X.shape == (M.N_TEST, 4) and np.all(np.abs(X) <= 1)
    assert np.all(np.diff(np.abs(d0)) >= 0)  # sorted by margin, smallest first
    m, X, y, d0 = M.model_and_inputs()
    assert np.all((np.abs(d0) >= M.BAND[0]) & (np.abs(d0) <= M.BAND[1]))
    r = {"inputs": 10, "without qang": {"differ": 5, "accuracy": 0.5, "decision error": 0.3},
         "qang": {"differ": 3, "accuracy": 0.6, "decision error": 0.2},
         "qang + echo": {"differ": 1, "accuracy": 0.7, "decision error": 0.1}}
    assert M.verdict([r]) == {"M1": True, "M2": True, "M3": True, "M4": True}
