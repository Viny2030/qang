import pytest; jax = pytest.importorskip("jax"); pytest.importorskip("scipy")
import numpy as np
from qang.qchem import hubbard_dimer, ground_state_sector, witness_mqg, pca_unary, givens_decomposition, apply_givens
def test_hubbard_energy():
    H, HZ, Hh, N, Sz = hubbard_dimer(1.0, 2.0); E0, _ = ground_state_sector(H, N, 2)
    assert abs(E0 - (2 - np.sqrt(20)) / 2) < 1e-12
def test_witness(): assert abs(witness_mqg(np.array([0.05, 0.05, 0.05, 0.05]), 2) - 50.0) < 1e-9
def test_pca_rbs():
    X = np.random.default_rng(3).normal(size=(16, 5)) + 2.0; lam, V, rho = pca_unary(X); r, D = givens_decomposition(V)
    assert len(r) == 10 and np.max(np.abs(apply_givens(rho, r, D) - np.diag(lam))) < 1e-12
