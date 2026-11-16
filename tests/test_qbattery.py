import pytest; jax = pytest.importorskip("jax")
from qang.qbattery import ergotropy_qg, max_links, chain_two_links_numeric
def test_plus_state_half(): assert abs(float(ergotropy_qg(1.0, 0.0)) - 0.5) < 1e-15
def test_links(): assert max_links(0.9) == 10
def test_chain(): assert abs(float(chain_two_links_numeric(0.9, 0.8)[2]) - 0.72) < 1e-13
