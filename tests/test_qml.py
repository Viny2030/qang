"""Tests for qang.qml (RESEARCH_NOTES §75-§80): the exact block simulator
against a full density matrix, F4 (filter exact in any weight sector under
equal T1), F3 (training under T1 with the filter = noiseless training), and
the with/without-qang comparison."""

import math

import numpy as np
import pytest

from qang.qml import WeightQNN, compare_qang, kept_fraction


def full_probs(model, theta, psi, gamma, dephasing):
    n = model.n
    gam = np.broadcast_to(np.asarray(gamma, float), (n,))
    rho = np.einsum("si,sj->sij", psi, psi)

    def channel(rho, kraus, q):
        out = np.zeros_like(rho)
        for K in kraus:
            op = np.eye(1)
            for j in range(n):
                op = np.kron(op, K if j == q else np.eye(2))
            out += op[None] @ rho @ op.T[None]
        return out

    for L in model.unitaries(theta):
        rho = L[None] @ rho @ L.T[None]
        for q in range(n):
            g = gam[q]
            rho = channel(rho, [np.array([[1, 0], [0, math.sqrt(1 - g)]]), np.array([[0, math.sqrt(g)], [0, 0]])], q)
            if dephasing:
                rho = channel(rho, [math.sqrt(1 - dephasing) * np.eye(2), math.sqrt(dephasing) * np.diag([1.0, -1.0])], q)
    return np.einsum("sii->si", rho)


@pytest.mark.parametrize("weight", [1, 2])
@pytest.mark.parametrize("noise", [(0.08, 0.0), ([0.04, 0.06, 0.08, 0.10], 0.0), (0.08, 0.03)])
def test_block_simulator_matches_full_density_matrix(weight, noise):
    rng = np.random.default_rng(1)
    m = WeightQNN(4, weight)
    th = rng.uniform(-3, 3, m.n_theta)
    psi = m.encode(rng.uniform(-1, 1, (3, 3)))
    assert np.allclose(m.probs(th, psi, *noise), full_probs(m, th, psi, *noise), atol=1e-12)


@pytest.mark.parametrize("weight", [1, 2])
def test_encoding_in_sector(weight):
    m = WeightQNN(5, weight)
    psi = m.encode(np.random.default_rng(2).uniform(-1, 1, (4, 4)))
    assert np.allclose(np.linalg.norm(psi, axis=1), 1)
    assert np.allclose((psi**2)[:, m.wt != weight].sum(axis=1), 0)


@pytest.mark.parametrize("weight", [1, 2])
def test_F4_filter_exact_and_kept_fraction(weight):
    rng = np.random.default_rng(3)
    m = WeightQNN(5, weight)
    th = rng.uniform(-3, 3, m.n_theta)
    psi = m.encode(rng.uniform(-1, 1, (4, 4)))
    exact = m.qg_z(m.probs(th, psi), qang=False)
    noisy = m.probs(th, psi, 0.1)
    assert np.allclose(m.qg_z(noisy, qang=True), exact, atol=1e-12)
    assert np.abs(m.qg_z(noisy, qang=False) - exact).max() > 1e-2
    assert np.allclose(noisy[:, m.idx[weight]].sum(axis=1), kept_fraction(0.1, weight, m.depth))


def test_unequal_t1_and_dephasing_are_not_corrected():
    rng = np.random.default_rng(4)
    m = WeightQNN(5, 1)
    th = rng.uniform(-3, 3, m.n_theta)
    psi = m.encode(rng.uniform(-1, 1, (4, 4)))
    exact = m.qg_z(m.probs(th, psi))
    assert np.abs(m.qg_z(m.probs(th, psi, [0.04, 0.06, 0.08, 0.1, 0.12])) - exact).max() > 1e-3
    assert np.abs(m.qg_z(m.probs(th, psi, 0.0, 0.05)) - exact).max() > 1e-3


def test_F3_training_under_t1_with_qang_equals_noiseless():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, (12, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = WeightQNN(5, 2)
    clean = m.fit(X, y, epochs=5, seed=1).params_
    noisy = m.fit(X, y, epochs=5, gamma=0.1, qang=True, seed=1).params_
    raw = m.fit(X, y, epochs=5, gamma=0.1, qang=False, seed=1).params_
    assert np.max(np.abs(clean - noisy)) < 1e-8
    assert np.max(np.abs(clean - raw)) > 1e-4


def test_fit_score_and_compare():
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, (40, 4))
    y = (X[:, 0] > 0).astype(int)
    m = WeightQNN(5, 1).fit(X[:30], y[:30], epochs=30, seed=0)
    assert m.score(X[30:], y[30:]) >= 0.7
    shots = m.score(X[30:], y[30:], gamma=0.08, qang=True, shots=200, seed=1)
    assert 0 <= shots <= 1
    r = compare_qang(WeightQNN(5, 2), X[:30], y[:30], X[30:], y[30:], gamma=0.08, epochs=5)
    assert r["with qang"] == r["exact"]
    assert set(r) == {"exact", "with qang", "without qang", "difference", "kept fraction"}
    assert r["difference"] == pytest.approx(r["with qang"] - r["without qang"])


def test_bad_inputs():
    with pytest.raises(ValueError):
        WeightQNN(5, 5)
    with pytest.raises(ValueError):
        WeightQNN(5, 3).encode(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        WeightQNN(5, 1).encode(np.zeros((2, 3)))


def test_zz_readout_rank_and_exactness():
    for weight, rank in ((1, 5), (2, 10)):
        m = WeightQNN(5, weight, readout="zz")
        rows = m.idx[weight]
        assert m.n_head == 15
        assert np.linalg.matrix_rank(np.c_[np.ones(len(rows)), m.zsign[rows]]) == 5
        assert np.linalg.matrix_rank(np.c_[np.ones(len(rows)), m.features[rows]]) == rank
    rng = np.random.default_rng(7)
    m = WeightQNN(5, 2, readout="zz")
    th = rng.uniform(-3, 3, m.n_theta)
    psi = m.encode(rng.uniform(-1, 1, (3, 4)))
    assert np.allclose(m.qg_z(m.probs(th, psi, 0.1), qang=True), m.qg_z(m.probs(th, psi), qang=False), atol=1e-12)
    with pytest.raises(ValueError):
        WeightQNN(5, 1, readout="xx")


def test_ring_encoding_in_sector_and_exact():
    rng = np.random.default_rng(9)
    m = WeightQNN(5, 2, encoding="ring")
    X = rng.uniform(-1, 1, (4, 4))
    psi = m.encode(X)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1)
    assert np.allclose((psi**2)[:, m.wt != 2].sum(axis=1), 0)
    assert np.count_nonzero(np.abs(psi[0]) > 0) == 5
    th = rng.uniform(-3, 3, m.n_theta)
    assert np.allclose(m.qg_z(m.probs(th, psi, 0.1), qang=True), m.qg_z(m.probs(th, psi), qang=False), atol=1e-12)
    with pytest.raises(ValueError):
        WeightQNN(5, 2, encoding="star")


def test_dual_encoding_fills_the_sector_and_is_exact():
    rng = np.random.default_rng(10)
    m = WeightQNN(5, 2, encoding="dual")
    X = rng.uniform(-1, 1, (4, 4))
    psi = m.encode(X)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1)
    assert np.allclose((psi**2)[:, m.wt != 2].sum(axis=1), 0)
    assert np.all(np.count_nonzero(np.abs(psi) > 1e-15, axis=1) == 10)
    th = rng.uniform(-3, 3, m.n_theta)
    assert np.allclose(m.qg_z(m.probs(th, psi, 0.1), qang=True), m.qg_z(m.probs(th, psi), qang=False), atol=1e-12)
    with pytest.raises(ValueError):
        WeightQNN(5, 1, encoding="dual")


def test_two_channel_readout_contains_raw_and_filtered():
    rng = np.random.default_rng(11)
    m = WeightQNN(5, 2, encoding="dual", readout="zz")
    X = rng.uniform(-1, 1, (6, 4))
    y = (X[:, 0] > 0).astype(int)
    th = rng.uniform(-3, 3, m.n_theta)
    pr = m.probs(th, m.encode(X), 0.08)
    both = m.qg_z(pr, "both")
    assert np.allclose(both[:, :m.n_head], m.qg_z(pr, True))
    assert np.allclose(both[:, m.n_head:], m.qg_z(pr, False))
    p = m.fit(X, y, epochs=2, gamma=0.08, qang="both", seed=1).params_
    assert len(p) == m.n_theta + 2 * m.n_head + 1
    assert 0 <= m.score(X, y, p, gamma=0.08, qang="both") <= 1
    with pytest.raises(ValueError):
        m.qg_z(pr, "all")


def test_multiclass_qnn_filter_exact_and_qubit_readout():
    from qang.qml import MultiClassQNN

    rng = np.random.default_rng(12)
    X = rng.uniform(-1, 1, (15, 4))
    y = np.argmax(X[:, :3], axis=1)
    for ro in ("qubit", "head"):
        m = MultiClassQNN(5, 3, readout=ro).fit(X, y, epochs=3, seed=1)
        p = m.params_
        assert np.allclose(m.logits(p, X, gamma=0.1, qang=True), m.logits(p, X), atol=1e-10)
        assert m.predict(X).shape == (15,) and 0 <= m.score(X, y) <= 1
    with pytest.raises(ValueError):
        MultiClassQNN(5, 6, readout="qubit")
    with pytest.raises(ValueError):
        MultiClassQNN(5, 3, readout="softmax")
