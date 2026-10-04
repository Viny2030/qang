"""Tests for examples/qec_syndrome_dephasing_target_qg.py (§99)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import qec_syndrome_dephasing_target_qg as T  # noqa: E402


def test_equal_qubits_give_equal_means_and_spread_raises_the_product_mean():
    pm, pp = T.targets([200] * 4, [150] * 4, 80.0, 1 - np.exp(-80.0 / 200))
    assert abs(pm - pp) < 1e-12
    pm, pp = T.targets([200] * 4, [30, 300, 300, 300], 80.0, 1 - np.exp(-80.0 / 200))
    assert pp > 1.2 * pm
