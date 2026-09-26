"""
Tests for examples/ionq_sim_zne_grover.py: native MS folding, the 3-qubit
Grover circuit, a local dry run, and the recorded IonQ noisy-simulator
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
pytest.importorskip("scipy")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402

import ionq_sim_zne_grover as Z  # noqa: E402

REC = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "ionq_sim_results.json")


def test_native_ms_folding_preserves_the_unitary():
    qiskit_ionq = pytest.importorskip("qiskit_ionq")
    qc = QuantumCircuit(2)
    qc.append(qiskit_ionq.GPI2Gate(0.1), [0])
    qc.append(qiskit_ionq.MSGate(0.13, 0.41, 0.25), [0, 1])
    qc.append(qiskit_ionq.GPIGate(0.3), [1])
    for s in (3, 5):
        f = Z.fold_ms(qc, s)
        assert f.count_ops()["ms"] == s
        assert Operator(f).equiv(Operator(qc))


def test_grover_circuit_gives_chebyshev_flag():
    t = math.asin(math.sqrt(Z.A_TRUE))
    for k in Z.GROVER_DEPTHS:
        qc = Z.grover_circuit(t, k)
        qc.remove_final_measurements()
        p = np.abs(Statevector(qc).data) ** 2
        qg = 1 - 2 * sum(p[i] for i in range(8) if i & 1)
        assert qg == pytest.approx(Z.G.flag_qg(1 - 2 * Z.A_TRUE, k), abs=1e-9)


def test_local_dry_run():
    h = Z.h2_zne(None, "local")
    assert h["raw_by_scale_mHa"][0] < h["raw_by_scale_mHa"][2]  # folding amplifies noise
    assert abs(h["qg_filter_mHa"]) < abs(h["raw_mHa"])
    g = Z.grover(None, "local")
    assert abs(g["a_aware"] - Z.A_TRUE) < 0.01


def _rec():
    with open(REC, encoding="utf-8") as fh:
        return json.load(fh)


def test_recorded_h2_filter_beats_raw_and_zne_is_noisy():
    r = _rec()
    for noise in ("aria-1", "forte-1"):
        runs = [r[k] for k in sorted(r) if k.startswith(f"ionq_sim|{noise}|h2zne")]
        assert len(runs) >= 6
        raw = np.array([x["raw_mHa"] for x in runs])
        filt = np.array([x["qg_filter_mHa"] for x in runs])
        zne = np.array([x["zne_mHa"] for x in runs])
        assert np.all(np.abs(filt) < np.abs(raw))
        assert filt.mean() < 20.3  # below Hartree-Fock on average
        assert zne.std() > 2 * filt.std()
        assert np.sqrt((filt**2).mean()) < np.sqrt((zne**2).mean())
        scale = np.mean([x["raw_by_scale_mHa"] for x in runs], axis=0)
        assert scale[0] < scale[1] < scale[2]  # native folding amplifies


def test_recorded_grover_mlae_beats_noisy_monte_carlo():
    r = _rec()
    for noise in ("aria-1", "forte-1"):
        g = [r[k] for k in sorted(r) if k.startswith(f"ionq_sim|{noise}|grover")]
        for est in ("a_naive", "a_aware"):
            assert max(abs(x[est] - x["a_true"]) for x in g) < 0.003
        k0 = np.sqrt(np.mean([(x["a_k0_only"] - x["a_true"]) ** 2 for x in g]))
        assert k0 > 3 * g[0]["mc_rmse_same_queries"]
