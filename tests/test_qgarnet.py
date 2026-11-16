import os
import numpy as np
import pytest
from qang.qgarnet import (raw_mqg, readout_from_poles, contrast, predicted_raw_mqg, correct_mqg, binomial_sigma_mqg,
                          equator_dispersion_ratio, fit_slope_origin, neutrino_summary, load, budget_summary)

D = os.environ.get("QANG_DATA", "examples/data")


def test_readout_roundtrip_and_poles():
    e0, e1 = readout_from_poles({"0": 3978, "1": 22}, {"1": 3884, "0": 116})
    assert abs(e0 - 0.0055) < 1e-12 and abs(e1 - 0.029) < 1e-12 and abs(contrast(e0, e1) - 0.9655) < 1e-12
    for t in (-1000.0, -707.0, 0.0, 500.0):
        assert abs(correct_mqg(predicted_raw_mqg(t, e0, e1), e0, e1) - t) < 1e-9
    assert abs(raw_mqg({"0": 50, "1": 50})) < 1e-12


def test_equator_ratio_near_one_for_binomial():
    rng = np.random.default_rng(3)
    copies = [{"1": int(k), "0": 4000 - int(k)} for k in rng.binomial(4000, 0.5, 400)]
    assert 0.9 < equator_dispersion_ratio(copies) < 1.1
    assert abs(binomial_sigma_mqg(0.5, 4000) - 15.811) < 1e-3


def test_slope_through_origin():
    x = np.array([2.0, 4.0, 10.0]); a, chi = fit_slope_origin(x, 1.5 * x, np.ones(3))
    assert abs(a - 1.5) < 1e-12 and chi < 1e-12


def test_budget_arithmetic():
    B = budget_summary([("t", 0.5), ("r", 16.0)], 30.0, ("t",))
    assert abs(B["spent"] - 16.5) < 1e-12 and abs(B["tests_frac"] - 0.5 / 30) < 1e-12


@pytest.mark.skipif(not os.path.exists(f"{D}/iqm_neutrino_qg_run.json"), reason="corridas guardadas no disponibles")
def test_stored_neutrino_run():
    S = neutrino_summary(load(f"{D}/iqm_neutrino_qg_run.json"))
    assert abs(S["factor"] - 1.5929) < 1e-3 and S["emin_filtered"] == 2.6 and S["emin_raw"] == 0.8
