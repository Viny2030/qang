"""
Tests for examples/ionq_sim_hubbard_qaoa.py: local dry run, and the
findings of the recorded IonQ noisy-simulator runs (no key needed).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("scipy")

import ionq_sim_hubbard_qaoa as I  # noqa: E402

REC = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "ionq_sim_results.json")


def _rec():
    with open(REC, encoding="utf-8") as fh:
        return json.load(fh)


def test_local_hubbard_dry_run():
    rows = I.hubbard(None, None, "local")
    assert set(rows) == {str(s) for s in I.HUBBARD_STEPS}
    assert rows["1"]["spin_leak"] < 0.02 and rows["6"]["kept_n"] < rows["1"]["kept_n"]


def test_recorded_hubbard_spin_filter_pays_at_depth():
    r = _rec()
    for noise in ("aria-1", "forte-1"):
        runs = [r[f"ionq_sim|{noise}|hubbard|run{t}"] for t in range(3)]
        err = {m: np.mean([abs(x[s][f"double_occ_{m}"] - x[s]["double_occ_ideal"]) for x in runs
                           for s in ("1", "2", "4", "6")]) for m in ("raw", "n_filter", "spin_filter")}
        assert err["spin_filter"] < err["n_filter"] < err["raw"]
        leak = {s: np.mean([x[s]["spin_leak"] for x in runs]) for s in ("1", "6")}
        assert leak["1"] < 0.01 and leak["6"] > 0.05
        for s in ("1", "2", "4", "6"):
            for q in ("qg_up", "qg_down"):
                assert abs(np.mean([x[s][q] for x in runs])) < 0.015  # unital: no drift of the witnesses


def test_recorded_qaoa_filter_doubles_xy_mixer():
    r = _rec()
    for noise in ("aria-1", "forte-1"):
        for p in (1, 2):
            s = I.summarize_qaoa(r[f"ionq_sim|{noise}|qaoa{p}|run0"])
            assert s["xy"]["noisy_f"] > 1.7 * s["xy"]["noisy"]
            assert abs(s["standard"]["noisy_f"] - s["random_feasible"]) < 0.1
            assert s["xy"]["ratio_f"] < 0.9  # greedy reaches 0.985
