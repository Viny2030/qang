"""
Tests for examples/ionq_sim_xxz_filter.py (RESEARCH_NOTES §61): native
IonQ circuits of the §60 chain and the recorded forte-1 result.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ionq")
from qiskit.quantum_info import Statevector  # noqa: E402

import ionq_sim_xxz_filter as S  # noqa: E402
import xxz_trotter_filter_qg as X  # noqa: E402

REC = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "ionq_sim_results.json")


@pytest.mark.parametrize("comp", S.COMPILATIONS)
@pytest.mark.parametrize("steps", [1, 3])
def test_native_circuits_are_exact(comp, steps):
    p = Statevector(S.build(steps, comp, measure=False)).probabilities()
    assert X.imbalance(p, 6)[0] == pytest.approx(S.ideal_imbalance(steps, S.DELTA[comp]), abs=1e-10)


def test_recorded_forte1_result():
    with open(REC, encoding="utf-8") as fh:
        rec = json.load(fh)
    runs = [rec[k] for k in sorted(rec) if k.startswith("ionq_sim|forte-1|xxz_filter|")]
    if len(runs) < 5:
        pytest.skip("forte-1 XXZ runs not recorded")

    def share(r, c):
        return 1 - sum(r[f"{c}|{s}"]["err_filter"] for s in (4, 6)) / sum(r[f"{c}|{s}"]["err_raw"] for s in (4, 6))

    ms = np.mean([share(r, "xy_ms") for r in runs])
    zz = np.mean([share(r, "xy_zz") for r in runs])
    assert 0.8 <= ms / zz <= 1.25
    for c in ("xy_ms", "xy_zz"):
        raw = abs(np.mean([r[f"{c}|4"]["raw"] for r in runs]) - runs[0][f"{c}|4"]["ideal"])
        filt = abs(np.mean([r[f"{c}|4"]["filter"] for r in runs]) - runs[0][f"{c}|4"]["ideal"])
        assert filt < 0.7 * raw
