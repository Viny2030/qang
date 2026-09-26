"""
Pins the pre-registered simulator predictions for the Forte-1 hardware
plan (RESEARCH_NOTES §46) to the recorded IonQ noisy-simulator data.
"""

import json
import os

import numpy as np
import pytest

REC = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "ionq_sim_results.json")


@pytest.fixture(scope="module")
def rec():
    with open(REC, encoding="utf-8") as fh:
        return json.load(fh)


def _runs(rec, prefix):
    runs = [rec[k] for k in sorted(rec) if k.startswith(prefix)]
    if not runs:
        pytest.skip(f"no recorded runs for {prefix}")
    return runs


def test_a1_decisive_prediction(rec):
    runs = _runs(rec, "ionq_sim|forte-1|h2zne|")
    raw = np.array([r["raw_mHa"] for r in runs])
    filt = np.array([r["qg_filter_mHa"] for r in runs])
    assert raw.mean() == pytest.approx(34.8, abs=0.1)
    assert filt.mean() == pytest.approx(13.9, abs=0.1)
    assert filt.mean() < 20.3 < raw.mean()
    assert np.all(filt < 0.6 * raw + 5)


def test_a2_stretched_bonds(rec):
    runs = _runs(rec, "ionq_sim|forte-1|h2stretched|")
    assert len(runs) == 3
    for r in runs:
        for b in ("1.5", "2.5"):
            assert r[b]["qg_filter_mHa"] < r[b]["readout_mHa"] < r[b]["hf_mHa"]
    gain = {b: 1 - np.mean([r[b]["qg_filter_mHa"] for r in runs]) / np.mean([r[b]["readout_mHa"] for r in runs])
            for b in ("1.5", "2.5")}
    assert gain["1.5"] == pytest.approx(0.44, abs=0.02)
    assert gain["2.5"] == pytest.approx(0.21, abs=0.02)


def test_b_folding_amplifies(rec):
    runs = _runs(rec, "ionq_sim|forte-1|h2zne|")
    s = np.mean([r["raw_by_scale_mHa"] for r in runs], axis=0)
    assert s[0] < s[1] < s[2]


def test_c_filter_doubles_xy_qaoa(rec):
    q = rec.get("ionq_sim|forte-1|qaoa1|run0") or pytest.skip("no QAOA run")
    xs = [v for k, v in q.items() if k.endswith("_xy")]
    ratios = [v["noisy_f"]["p_opt"] / v["noisy"]["p_opt"] for v in xs]
    assert len(xs) == 5 and min(ratios) > 2.0
    assert np.mean([v["noisy"]["p_opt"] for v in xs]) == pytest.approx(0.120, abs=0.002)
    assert np.mean([v["noisy_f"]["p_opt"] for v in xs]) == pytest.approx(0.256, abs=0.002)


def test_d_spin_leak_grows_and_filters_help(rec):
    runs = _runs(rec, "ionq_sim|forte-1|hubbard|")
    leak = [np.mean([r[s]["spin_leak"] for r in runs]) for s in ("1", "2", "4")]
    assert leak[0] < leak[1] < leak[2]
    for s in ("1", "2", "4"):
        raw = np.mean([abs(r[s]["imbalance_raw"] - r[s]["imbalance_ideal"]) for r in runs])
        spin = np.mean([abs(r[s]["imbalance_spin_filter"] - r[s]["imbalance_ideal"]) for r in runs])
        assert spin < raw


def test_e_qrng_readout_ideal_in_simulator(rec):
    runs = _runs(rec, "ionq_sim|forte-1|qrng|")
    assert all(r["readout_b"] > 0.999 for r in runs)
    for r in runs:
        for t in ("0.0", "0.08"):
            row = r["rows"][t]
            assert row["qg +/- pairs"] <= row["truth"] + 1e-9
