"""
Tests for examples/nisq_hardware_validation.py, run on the calibration-
based fake_brisbane backend (a local Aer simulation of a real IBM device's
published noise data -- no account or network needed).

Checked:
  1. choose_layout returns a connected line of qubits whose calibration
     data is physically valid (T2 <= 2*T1).
  2. Experiment 1: across the idle-delay (T1) sweep, heavy output
     probability falls and mean qg_Z rises at every step, while qg_S is
     NOT monotonic -- it rises and then falls, ending below even the
     noiseless value (Findings C/D reproduced with device-level noise).
  3. Experiment 2: the raw device-noise error of a LiH energy evaluation
     is far above chemical accuracy and far above the HF-to-optimum
     energy difference.
  4. get_backend rejects an unknown mode.

Skipped automatically if qiskit-ibm-runtime is not installed
(pip install ".[hardware]").
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from nisq_hardware_validation import (  # noqa: E402
    choose_layout,
    get_backend,
    run_lih_energy_experiment,
    run_relaxation_experiment,
)

CHEMICAL_ACCURACY = 1.6e-3


@pytest.fixture(scope="module")
def backend():
    return get_backend("fake", "fake_brisbane")


@pytest.fixture(scope="module")
def relaxation_rows(backend):
    return run_relaxation_experiment(backend, shots=4000, seed=42)


def test_choose_layout_is_a_valid_connected_line(backend):
    layout = choose_layout(backend, 4)
    assert layout is not None and len(set(layout)) == 4
    edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    for a, b in zip(layout, layout[1:]):
        assert tuple(sorted((a, b))) in edges
    for q in layout:
        props = backend.qubit_properties(q)
        assert props.t2 <= 2 * props.t1


def test_ideal_reference_row_matches_the_rest_of_the_codebase(relaxation_rows):
    ideal = relaxation_rows[0]
    assert ideal["delay_us"] is None
    assert ideal["qg_s"] == pytest.approx(0.9193, abs=1e-4)
    assert ideal["mean_qg_z"] == pytest.approx(-0.0289, abs=1e-4)
    assert ideal["hop"] == pytest.approx(0.7722, abs=1e-4)
    assert ideal["xeb"] == pytest.approx(0.3662, abs=1e-4)


def test_standard_benchmarks_degrade_monotonically_with_idle_relaxation(relaxation_rows):
    measured = relaxation_rows[1:]
    hops = [r["hop"] for r in measured]
    xebs = [r["xeb"] for r in measured]
    assert np.all(np.diff(hops) < 0)
    assert np.all(np.diff(xebs) < 0)


def test_mean_qg_z_rises_monotonically_with_idle_relaxation(relaxation_rows):
    biases = [r["mean_qg_z"] for r in relaxation_rows[1:]]
    assert np.all(np.diff(biases) > 0)
    assert biases[-1] - biases[0] > 0.3


def test_qg_s_is_not_monotonic_under_device_relaxation(relaxation_rows):
    ideal_qg_s = relaxation_rows[0]["qg_s"]
    qg_s = [r["qg_s"] for r in relaxation_rows[1:]]
    # device noise at zero delay raises qg_S above the noiseless value ...
    assert qg_s[0] > ideal_qg_s + 0.02
    # ... it keeps rising for short delays, then falls ...
    peak = int(np.argmax(qg_s))
    assert 0 < peak < len(qg_s) - 1
    assert qg_s[-1] < qg_s[peak] - 0.05
    # ... and the most-relaxed point ends below even the noiseless value.
    assert qg_s[-1] < ideal_qg_s


def test_raw_lih_energy_error_dwarfs_chemical_accuracy(backend):
    rows = run_lih_energy_experiment(backend, shots=8000, seed=42)
    by_point = {r["point"]: r for r in rows}
    hf, opt = by_point["hartree_fock"], by_point["ansatz_optimum"]
    assert hf["ideal"] == pytest.approx(-0.04378130564745453, abs=1e-9)
    assert opt["ideal"] == pytest.approx(-0.044015242088863316, abs=1e-6)
    gap = hf["ideal"] - opt["ideal"]
    for r in rows:
        assert r["error"] > 10 * CHEMICAL_ACCURACY
        assert r["error"] > 50 * gap


def test_get_backend_rejects_unknown_mode():
    with pytest.raises(ValueError):
        get_backend("simulator")
