"""
Tests for examples/ionq_sim_lih_mod4.py (RESEARCH_NOTES §53): the local
dry run and the recorded forte-1 prediction.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

import ionq_sim_lih_mod4 as M  # noqa: E402

REC = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "ionq_sim_results.json")


def test_local_dry_run():
    out = M.run("local")
    assert out["parity_mod4"] < 0.3 * out["raw"]
    assert out["parity_mod4"] < out["parity"] < out["raw"]
    assert 0.5 < out["kept_parity_mod4"] < out["kept_parity"] < 1.0


def test_recorded_forte1_prediction():
    with open(REC, encoding="utf-8") as fh:
        rec = json.load(fh)
    runs = [rec[k] for k in sorted(rec) if k.startswith("ionq_sim|forte-1|lih_mod4|")]
    if len(runs) < 5:
        pytest.skip("forte-1 LiH runs not recorded")
    raw = np.array([r["raw"] for r in runs])
    par = np.array([r["parity"] for r in runs])
    m4 = np.array([r["parity_mod4"] for r in runs])
    assert np.all(raw / m4 > 2.5)
    assert np.all(raw / par < 1.5)
    assert m4.mean() == pytest.approx(38.6, abs=0.1)
    assert m4.mean() + M.ANSATZ_MHA > M.HF_MHA  # does not beat Hartree-Fock on forte-1
