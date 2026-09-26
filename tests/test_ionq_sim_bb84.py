"""
Tests for examples/ionq_sim_bb84.py: the memory and Eve circuits do what
they claim, a local dry run, and the recorded IonQ noisy-simulator
findings (no key needed).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import bb84_qg_eve_vs_noise as B  # noqa: E402
import ionq_sim_bb84 as R  # noqa: E402

REC = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "ionq_sim_results.json")


@pytest.mark.parametrize("gamma", (0.02, 0.06))
@pytest.mark.parametrize("eve", ("none", "Z", "X"))
def test_circuits_match_the_analytic_channel(gamma, eve):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import DensityMatrix, partial_trace

    f = 0.0 if eve == "none" else 1.0
    for i, s in enumerate(R.STATES):
        qc = R.bb84_circuit(s, gamma, eve, copies=1)
        qc.remove_final_measurements()
        rho = partial_trace(DensityMatrix(qc), [1, 2]).data
        p1 = float(np.real(rho[1, 1]))
        err = p1 if s in ("0", "+") else 1 - p1
        if eve == "none":
            expect = B.error_probabilities(gamma=gamma, p=0.0, e01=0.0, e10=0.0)[i]
        else:  # Eve always attacks in one basis: error 1/2 in the other, channel only in hers
            other = (eve == "Z") != (s in ("0", "1"))
            expect = None if other else B.error_probabilities(gamma=gamma, p=0.0, e01=0.0, e10=0.0)[i]
            if other:
                assert err == pytest.approx(0.5, abs=gamma / 2 + 1e-9)
                continue
        assert err == pytest.approx(expect, abs=1e-9), (s, eve)
    assert f in (0.0, 1.0)


def test_model_table_adds_symmetric_flips():
    t = R.model_table(0.02, 0.03)
    i0 = int(np.argmin(np.abs(R.G_GRID - 0.02)))
    ideal = B.error_probabilities(gamma=0.02, p=0.0, e01=0.0, e10=0.0)
    eps = np.array([0.02, 0.02, 0.03, 0.03])
    assert np.allclose(t[i0, 0], ideal * (1 - eps) + (1 - ideal) * eps)


def test_blocks_mix_eve_rounds_without_replacement():
    data = {}
    for s in R.STATES:
        for g in R.GAMMAS:
            data["|".join((s, f"{g:.2f}", "none"))] = {"errors": [40] * R.COPIES, "shots": 2000}
        for g in R.EVE_GAMMAS:
            for b in ("Z", "X"):
                data["|".join((s, f"{g:.2f}", b))] = {"errors": [1000] * R.COPIES, "shots": 2000}
    rng = np.random.default_rng(0)
    k0, n0 = R.make_block(data, 0, 0.02, 0.0, rng)
    assert list(k0) == [40] * 4 and list(n0) == [2000] * 4
    k, n = R.make_block(data, 0, 0.02, 0.10, rng)
    assert list(n) == [2000] * 4
    assert np.all(k > 100)  # ~36 + ~100 errors from the 200 attacked rounds


def test_local_dry_run():
    pytest.importorskip("qiskit_aer")
    d = R.run(None, "local")
    rates = R.error_rates([d])
    assert rates["1|0.06|none"] > rates["1|0.02|none"] + 0.02  # T1 shows up as 1 -> 0 errors
    assert rates["0|0.02|Z"] < 0.1 and rates["0|0.02|X"] > 0.4


def _runs(noise):
    if not os.path.exists(REC):
        pytest.skip("no recorded results")
    runs = R.load_runs(noise, REC)
    if len(runs) < 4:
        pytest.skip("IonQ BB84 runs not recorded")
    return runs


@pytest.mark.parametrize("noise", ("aria-1", "forte-1"))
def test_recorded_fit_recovers_the_healthy_memory(noise):
    runs = _runs(noise)
    rng = np.random.default_rng(0)
    base = [R.make_block(d, c, R.GAMMA0, 0.0, rng) for d in runs for c in range(R.COPIES)]
    ez, ex, i_g0 = R.fit_baseline(base)
    assert 0.018 <= R.G_GRID[i_g0] <= 0.025
    assert 0.015 < ez < 0.04 and 0.015 < ex < 0.04


@pytest.mark.parametrize("noise", ("aria-1", "forte-1"))
def test_recorded_attribution_does_not_cross(noise):
    s = R.analyse(_runs(noise))
    assert s["blocks"] == 16
    for name in ("T1 drift 0.03", "T1 drift 0.04", "T1 drift 0.06"):
        assert s[name]["attack"] <= 1 / 16 + 1e-9
    for name in ("intercept f = 0.05", "intercept f = 0.10"):
        assert s[name]["drift"] <= 1 / 16 + 1e-9
        assert s[name]["attack"] >= 0.75
    assert s["T1 drift 0.06"]["drift"] == 1.0
    assert s["intercept f = 0.10"]["attack"] == 1.0
    assert s["baseline"]["attack"] <= 1 / 16 + 1e-9 and s["baseline"]["drift"] <= 1 / 16 + 1e-9


def test_recorded_model_extrapolates_to_large_blocks():
    runs = _runs("aria-1")
    p = R.predicted(runs, reps=150, scale=25)
    for name, g, f in R.SCENARIOS:
        if g > R.GAMMA0:
            assert p[name]["drift"] > 0.95
        if f > 0:
            assert p[name]["attack"] > 0.95
        if f == 0:
            assert p[name]["attack"] < 0.05
        if g == R.GAMMA0:
            assert p[name]["drift"] < 0.05
    assert math.isclose(p["baseline"]["qber"], 0.0, abs_tol=0.05)
