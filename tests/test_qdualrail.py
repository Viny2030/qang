import pytest; jax = pytest.importorskip("jax")
from qang.qdualrail import dual_rail_state, apply_loss, readout_unfiltered, readout_sector_filtered

def test_bias_and_filter():
    rho = apply_loss(dual_rail_state(-0.6), 0.1)
    assert abs(float(readout_unfiltered(rho)) - (0.1 * -0.6 + 0.9)) < 1e-14
    q, acc = readout_sector_filtered(rho); assert abs(float(q) + 0.6) < 1e-13 and abs(float(acc) - 0.1) < 1e-14
