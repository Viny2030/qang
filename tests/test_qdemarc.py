import pytest; jax = pytest.importorskip("jax")
from qang.qdemarc import bb84_threshold_qz, qrng_min_entropy_from_qz, shots_to_resolve_bias

def test_bb84_threshold(): assert abs(bb84_threshold_qz() - 0.7799) < 1e-3
def test_qrng_unbiased(): assert abs(float(qrng_min_entropy_from_qz(0.0)) - 1.0) < 1e-15
def test_pqc_uncountable(): import math; assert math.log2(shots_to_resolve_bias(2.0 ** -138.8)) > 270
