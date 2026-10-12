"""Tests for examples/ionq_qpu_h2_qg.py (§128): circuits, analysis and verdict without any IonQ access."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_seven_circuits_and_noiseless_analysis():
    pytest.importorskip("qiskit_aer")
    import ionq_qpu_h2_qg as Q
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    labels, circs = Q.circuits()
    assert len(circs) == 7 and labels[0] == "ZZZZ" and labels[-2:] == ["cal0", "cal1"]
    sim = AerSimulator(seed_simulator=1)
    counts = [sim.run(transpile(c, sim), shots=20000).result().get_counts() for c in circs]
    r = Q.analyse(counts)
    assert abs(r["mean_qg_z"]) < 0.02 and r["kept_fraction"] > 0.99
    assert abs(r["readout_qg_filter"]) < 0.005          # noiseless: close to FCI (Ha)
    assert r["hf_error"] == pytest.approx(0.0203, abs=0.001)


def test_verdict():
    import ionq_qpu_h2_qg as Q

    r = {"mean_qg_z": 0.01, "kept_fraction": 0.95, "hf_error": 0.0203, "raw": 0.033,
         "readout": 0.030, "readout_qg_filter": 0.012}
    assert Q.verdict(r) == {"H1": True, "H2": True, "H3": True, "H4": True}
    r["readout_qg_filter"] = 0.028
    assert Q.verdict(r)["H3"] is False and Q.verdict(r)["H4"] is False


def test_submit_refuses_without_cost_flag(capsys):
    import ionq_qpu_h2_qg as Q

    assert Q.submit(False) is None
    assert "NOT submitted" in capsys.readouterr().out
