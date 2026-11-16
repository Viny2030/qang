import pytest; jax = pytest.importorskip("jax")
from qang.qecho import readout_affine, drag_free_point, mse_raw_vs_inverted, sector_readout_error_order
import numpy as np
def test_fixed_point(): m = drag_free_point(0.02, 0.06); assert abs(readout_affine(m, 0.02, 0.06) - m) < 1e-9
def test_108_rule(): r, i, _, _ = mse_raw_vs_inverted(600, 0.02, 0.06, 200, 2000); assert r < i
def test_second_order(): _, b, c = sector_readout_error_order(np.array([.5, .3, .15, .05]), 0.02, 0.06); assert b < 3 * c
