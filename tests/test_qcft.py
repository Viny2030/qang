import numpy as np
from qang.qcft import correlation_matrix, connected_qgzz, exact_state_xx, spin_zz_connected, block_entropy_bits, spin_block_entropy_bits, fit_central_charge, local_qgz
def test_identities():
    n = 6; C = correlation_matrix(n); psi = exact_state_xx(n)
    assert abs(connected_qgzz(C, 1, 3) - spin_zz_connected(psi, n, 1, 3)) < 1e-12
    assert abs(block_entropy_bits(C, 3) - spin_block_entropy_bits(psi, n, 3)) < 1e-10
def test_central_charge(): assert abs(fit_central_charge(128)[0] - 1.0) < 0.05
def test_local_flat(): assert np.max(np.abs(local_qgz(correlation_matrix(64)))) < 1e-10
