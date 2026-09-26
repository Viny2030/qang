"""
Tests for examples/ionq_sim_qrng.py: the native circuits measure the
intended axes, a local dry run, and the recorded IonQ noisy-simulator
findings (no key needed).
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ionq")

from qiskit.quantum_info import Statevector  # noqa: E402

import ionq_sim_qrng as R  # noqa: E402

REC = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "ionq_sim_results.json")


def _ideal_qg(theta, setting):
    qc = R._circuit(theta, setting)
    qc.remove_final_measurements()
    p = np.abs(Statevector(qc).data) ** 2
    return 1 - 2 * sum(p[i] for i in range(len(p)) if i & 1)  # copy 0


@pytest.mark.parametrize("theta", R.THETAS)
def test_native_circuits_measure_the_intended_axes(theta):
    c = math.cos(2 * math.pi * theta)
    assert _ideal_qg(theta, "output") == pytest.approx(0, abs=1e-9)
    assert _ideal_qg(theta, "z+") == pytest.approx(c, abs=1e-9)
    assert _ideal_qg(theta, "z-") == pytest.approx(-c, abs=1e-9)
    assert _ideal_qg(theta, "y+") == pytest.approx(0, abs=1e-9)
    assert _ideal_qg(theta, "y-") == pytest.approx(0, abs=1e-9)


def test_truth_matches_the_section_41_formula():
    assert R.truth(0.0) == pytest.approx(1.0)
    assert R.truth(0.25) == pytest.approx(0.0, abs=1e-12)
    assert R.truth(0.08) == pytest.approx(0.433, abs=1e-3)


def test_local_dry_run():
    out = R.run(None, "local")
    for t in R.THETAS:
        r = out["rows"][str(t)]
        assert r["naive"] > 0.9
        for e in ("qg one-sided", "qg +/- pairs", "qg calibrated"):
            assert r[e] <= r["truth"] + 1e-9


def _entries(noise):
    with open(REC, encoding="utf-8") as fh:
        rec = json.load(fh)
    return [rec[k] for k in sorted(rec) if k.startswith(f"ionq_sim|{noise}|qrng|")]


@pytest.mark.parametrize("noise", ("aria-1", "forte-1"))
def test_recorded_naive_unsafe_qg_safe(noise):
    runs = _entries(noise)
    assert len(runs) >= 3
    for run in runs:
        for t in R.THETAS:
            r = run["rows"][str(t)]
            for e in ("qg one-sided", "qg +/- pairs", "qg calibrated"):
                assert r[e] <= r["truth"] + 1e-9
            if t > 0:
                assert r["naive"] > r["truth"] + 0.25
    s = R.summarize(runs)
    for t in R.THETAS[1:]:
        assert s[t]["unsafe"]["naive"] == 1.0
        assert 0.8 < s[t]["qg +/- pairs"] / s[t]["truth"] < 1.0
    assert 0.7 < s[0.0]["qg +/- pairs"] < 0.75  # pole at r = 1


@pytest.mark.parametrize("noise", ("aria-1", "forte-1"))
def test_recorded_readout_nearly_perfect_and_gate_noise_small(noise):
    runs = _entries(noise)
    for run in runs:
        assert run["readout_b"] > 0.999
        assert abs(run["readout_a"]) < 1e-3
    s = R.summarize(runs)
    for t in R.THETAS[1:]:
        ideal = abs(math.cos(2 * math.pi * t))
        assert 0.97 * ideal < s[t]["r_calibrated"] < ideal
