"""Tests for examples/qnn_multiclass_qpu_qg.py (§116): verdict and the QPU cost guard."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_verdict():
    import qnn_multiclass_qpu_qg as Q

    r = {"without qang": {"accuracy": 0.80, "differ": 9}, "qang": {"accuracy": 0.86, "differ": 4},
         "qang + echo": {"accuracy": 0.88, "differ": 3}, "echo diagonal": [0.8] * 5}
    assert Q.verdict(r) == {"H1": True, "H2": True, "H3": True, "H4": True}


def test_qpu_refused_without_flag(monkeypatch, capsys):
    import qnn_multiclass_qpu_qg as Q

    monkeypatch.setattr(Q, "package", lambda n: (None, [0] * n, [0] * n, [None] * (n + 5), [1000] * (n + 5)))
    monkeypatch.setattr(Q, "ionq_backend", lambda *a: (_ for _ in ()).throw(AssertionError("no backend call")))
    assert Q.main(["--mode", "qpu"]) is None
    assert "Nothing submitted" in capsys.readouterr().out
