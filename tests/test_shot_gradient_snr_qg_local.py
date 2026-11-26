import os
import sys

import pytest

pytest.importorskip("qiskit")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shot_gradient_snr_qg_local import run, summarize  # noqa: E402


def test_shot_gradient_summary_structure():
    rows = run(n_qubits=4, reps=1, n_samples=30, shots=50, seed=0)
    s = summarize(rows)
    assert set(s.keys()) == {"theta", "qg_clipped", "qg_raw"}
    for v in s.values():
        assert v["median_abs_global"] >= 0
        assert v["median_abs_qg_local"] >= 0
        assert v["ratio_local_over_global"] >= 0


def test_qg_local_has_nonzero_shot_gradient_signal():
    rows = run(n_qubits=4, reps=1, n_samples=40, shots=100, seed=1)
    s = summarize(rows)
    assert s["theta"]["median_abs_qg_local"] > 0
