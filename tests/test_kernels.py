"""Tests for qang.kernels (§161)."""

import numpy as np
import pytest

from qang import kernels as KR


def test_embedding_and_exact_kernel():
    X = np.random.default_rng(0).normal(size=(6, 4))
    E = KR.unary_embedding(X)
    assert np.allclose(np.linalg.norm(E, axis=1), 1)
    K = KR.exact_kernel(E, E)
    assert np.allclose(np.diag(K), 1)
    assert np.all((K >= 0) & (K <= 1 + 1e-12))
    phi = np.arccos(np.clip(np.abs(E @ E.T), 0, 1))
    assert np.allclose(KR.kernel_qang_angle(K), 2 * phi, atol=1e-6)


def test_normalization_undoes_a_global_shrink():
    K = KR.exact_kernel(*(2 * [KR.unary_embedding(np.random.default_rng(1).normal(size=(5, 3)))]))
    noisy = 0.8 * K
    assert np.allclose(KR.normalize_gram(noisy, np.diag(noisy), np.diag(noisy)), K)


def test_alignment_of_ideal_kernel_is_one():
    y = np.array([1, 1, -1, -1])
    assert KR.alignment(np.outer(y, y), y) == pytest.approx(1.0)


def test_inversion_test_counts_give_the_kernel():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qang import ibm

    E = KR.unary_embedding(np.random.default_rng(2).normal(size=(3, 4)))
    for i, j in ((0, 1), (1, 2), (2, 2)):
        pr = Statevector(ibm.unary_overlap_circuit(E[i], E[j], measure=False)).probabilities_dict()
        r = KR.kernel_from_counts({k: int(round(v * 1e9)) for k, v in pr.items()})
        assert r["raw"] == pytest.approx((E[i] @ E[j]) ** 2, abs=1e-6)
        assert r["filtered"] == pytest.approx(r["raw"], abs=1e-6)
        assert r["kept"] == pytest.approx(1.0, abs=1e-6)
