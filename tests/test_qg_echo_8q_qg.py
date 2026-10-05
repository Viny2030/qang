"""Tests for examples/qg_echo_8q_qg.py (§115): sector bookkeeping and verdict."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_sector_and_verdict():
    import qg_echo_8q_qg as E

    assert len(E.SECTOR) == 28 and len({E.state_index(p) for p in E.SECTOR}) == 28
    assert all(bin(E.state_index(p)).count("1") == 2 for p in E.SECTOR)
    r = {"qg_Z error, without qang": 0.2, "qg_Z error, qang": 0.1, "qg_Z error, qang + echo": 0.065,
         "qg_ZZ error, without qang": 0.2, "qg_ZZ error, qang": 0.1, "qg_ZZ error, qang + echo": 0.06}
    assert E.verdict([r, r]) == {"X1": True, "X2": True, "X3": True, "X4": True, "X5": True}
    assert abs(E.reduction(r) - 0.35) < 1e-12


def test_noiseless_local():
    pytest.importorskip("qiskit_aer")
    import qg_echo_8q_qg as E

    r = E.run_backend("local", None, None, None)
    assert np.allclose(r["echo diagonal"], 1.0)
    assert r["qg_Z error, qang"] < 0.05
