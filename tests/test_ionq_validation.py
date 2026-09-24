"""
Tests for examples/ionq_validation.py in local mode (Qiskit Aer, no IonQ
account needed). The IonQ runs themselves are recorded in RESEARCH_NOTES
§20; they need an API key and are not part of the test suite.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from ionq_validation import (  # noqa: E402
    circuit_sizes,
    experiment_noise_benchmark,
    experiment_rzz_qg,
    get_backend,
    main,
)


@pytest.fixture(scope="module")
def backend():
    return get_backend("local")


def test_noiseless_benchmark_matches_ideal_within_shot_noise(backend):
    ideal, meas = experiment_noise_benchmark(backend, shots=20000)
    assert ideal["qg_s"] == pytest.approx(0.9193, abs=1e-4)
    assert ideal["mean_qg_z"] == pytest.approx(-0.0289, abs=1e-4)
    assert abs(meas["mean_qg_z"] - ideal["mean_qg_z"]) < 4 * meas["mean_qg_z_stderr"]
    assert meas["xeb"] == pytest.approx(ideal["xeb"], abs=0.05)


def test_rzz_probe_reads_cos_theta(backend):
    for row in experiment_rzz_qg(backend, shots=20000):
        assert row["x0_measured"] == pytest.approx(math.cos(row["theta"]), abs=0.03)


def test_qpu_mode_refuses_without_explicit_cost_acceptance(capsys):
    main(["--mode", "ionq_qpu"])
    out = capsys.readouterr().out
    assert "NOT submitted" in out
    one_q, two_q = circuit_sizes()["qv"]
    assert two_q > 0 and one_q > 0
