"""
Tests for examples/qnn_seeds_qg.py (RESEARCH_NOTES §80): the paired
quantities, the confidence intervals and the verdict logic.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

import qnn_seeds_qg as S  # noqa: E402


def test_ci95_known_values():
    mean, lo, hi = S.ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    assert hi - mean == pytest.approx(2.776445 * np.sqrt(2.5) / np.sqrt(5), rel=1e-5)
    assert mean - lo == pytest.approx(hi - mean)


def test_small_split_and_quantities():
    rec = S.run_split("iris", 1, 0, epochs=2)
    for m in ("E", "W"):
        assert rec[f"{m} T1 clean qang"] == rec[f"{m} exact"]  # F4: exact with qang
    q = S.quantities([rec])
    assert q["G_W-G_E"] == pytest.approx(q["G_W"] - q["G_E"])
    assert set(S.verdict(S.summarize([q, q]))) == {"T1", "T2", "T3", "T4", "T5"}
