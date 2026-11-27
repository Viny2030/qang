import pytest; jax = pytest.importorskip("jax")
from qang.qrng import certified_min_entropy, point_min_entropy, toeplitz_matrix, toeplitz_extract, bell_chsh
import numpy as np
def test_cert_below_point(): assert certified_min_entropy(5000, 10000) < point_min_entropy(5000, 10000)
def test_cert_grows_with_N(): assert certified_min_entropy(500000, 1000000) > certified_min_entropy(5000, 10000)
def test_toeplitz_extract_is_seeded_and_linear():
    x = np.random.default_rng(1).integers(0, 2, 512, dtype=np.uint8)
    y = np.random.default_rng(2).integers(0, 2, 512, dtype=np.uint8)
    assert np.array_equal(toeplitz_extract(x, 128, seed=7), toeplitz_extract(x, 128, seed=7))
    assert np.array_equal(toeplitz_extract(x ^ y, 128, seed=7),
                          toeplitz_extract(x, 128, seed=7) ^ toeplitz_extract(y, 128, seed=7))
    T = toeplitz_matrix(512, 128, seed=7)
    assert T.shape == (128, 512) and np.array_equal(T[1:, 1:], T[:-1, :-1])
def test_bell_chsh_bounds():
    q = np.sqrt(0.5)
    assert not bell_chsh([0.0, 0.0, 0.0, 0.0])["violates_local"]
    result = bell_chsh([-q, -q, -q, q])
    assert result["violates_local"] and abs(abs(result["S"]) - result["tsirelson"]) < 1e-12
