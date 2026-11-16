import pytest; jax = pytest.importorskip("jax")
from qang.qkd import qber, sifted_bits_for_qber_sigma
def test_ir_quarter(): assert abs(qber(1.0, 0.0) - 0.25) < 1e-14
def test_budget(): assert abs(sifted_bits_for_qber_sigma(0.02, 0.005) - 784) < 1
