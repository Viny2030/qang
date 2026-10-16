"""Tests for qang.angular: the quantum numbers in qg terms (§155)."""

import math

import numpy as np
import pytest

from qang import angular as A
from qang.core import qg_s_from_qg_z


def test_m_is_read_from_the_hamming_weight():
    for n in (1, 2, 5):
        for w in range(n + 1):
            assert A.weight_from_m(n, A.m_from_weight(n, w)) == w
    with pytest.raises(ValueError):
        A.weight_from_m(3, 1.0)  # 3 qubits give half-integer m only


def test_dicke_states_are_sharp_m_and_maximal_j():
    for n in (2, 3, 4):
        Jx, Jy, Jz = A.spin_operators(n)
        J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
        j = n / 2
        for w in range(n + 1):
            v = A.dicke_state(n, w)
            assert np.allclose(Jz @ v, A.m_from_weight(n, w) * v)
            assert np.allclose(J2 @ v, j * (j + 1) * v)
            pm = A.m_distribution(np.abs(v) ** 2, n)
            assert pm[w] == pytest.approx(1)


def test_coherent_state_qg_z_is_cos_theta_for_every_j():
    for n in (1, 2, 3, 6):
        for th in (0.0, 0.4, 1.3, math.pi / 2, 2.9):
            single = np.array([math.cos(th / 2), math.sin(th / 2)])
            psi = np.ones(1)
            for _ in range(n):
                psi = np.kron(psi, single)
            pm = A.m_distribution(psi ** 2, n)
            assert np.allclose(pm, A.coherent_m_distribution(n / 2, th))
            assert A.qg_z_spin(pm) == pytest.approx(math.cos(th), abs=1e-12)


def test_spin_half_reduces_to_qg():
    for z in (-0.8, 0.0, 0.3, 1.0):
        pm = [(1 + z) / 2, (1 - z) / 2]
        assert A.qg_z_spin(pm) == pytest.approx(z)
        assert A.qg_s_spin(pm) == pytest.approx(qg_s_from_qg_z(z), abs=1e-12)


def test_qang_angle_and_vector_model_angle():
    for l in (1, 2, 3):
        Jx, Jy, Jz = A.spin_operators(2 * l)
        for m in range(-l, l + 1):
            v = A.dicke_state(2 * l, A.weight_from_m(2 * l, m))
            spread = np.real(np.vdot(v, (Jx @ Jx + Jy @ Jy) @ v))
            assert spread == pytest.approx(A.transverse_spread(l, m))
            # the vector model always tilts further from the poles than the qang angle
            assert abs(math.cos(A.vector_model_angle(l, m))) <= abs(math.cos(A.qang_angle_of_eigenstate(l, m)))
        assert A.qang_angle_of_eigenstate(l, l) == 0.0
        assert A.vector_model_angle(l, l) == pytest.approx(math.acos(math.sqrt(l / (l + 1))))


def test_p_orbitals_real_ones_mix_m_sectors():
    Jx, Jy, Jz = A.spin_operators(2)
    for name, pm_expected in (("p+1", [1, 0, 0]), ("p0", [0, 1, 0]), ("p-1", [0, 0, 1]),
                              ("px", [0.5, 0, 0.5]), ("py", [0.5, 0, 0.5]), ("pz", [0, 1, 0])):
        v = A.p_orbital(name)
        assert np.allclose(A.m_distribution(np.abs(v) ** 2, 2), pm_expected)
    # p_x and p_y are the m = 0 states along x and y
    assert np.allclose(Jx @ A.p_orbital("px"), 0)
    assert np.allclose(Jy @ A.p_orbital("py"), 0)
    for name in ("px", "py", "pz"):
        assert A.qg_z_spin(A.m_distribution(np.abs(A.p_orbital(name)) ** 2, 2)) == pytest.approx(0)


def test_hydrogen_shell_is_two_spins_n_minus_1_over_2():
    for n in (1, 2, 3, 4):
        ls = A.shell_l_spectrum(n)
        expected = sorted(l for l, _ in A.shell_quantum_numbers(n))
        assert len(ls) == n * n
        assert np.allclose(sorted(ls), expected)
        assert A.shell_register_qubits(n) == 2 * (n - 1)


def test_j_squared_is_a_sum_of_swap_qg_values():
    rng = np.random.default_rng(5)
    for n in (2, 3, 4):
        Jx, Jy, Jz = A.spin_operators(n)
        J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        v /= np.linalg.norm(v)
        idx = np.arange(2 ** n)
        swaps = []
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = (idx >> i) & 1, (idx >> j) & 1
                perm = idx ^ ((bi ^ bj) << i) ^ ((bi ^ bj) << j)
                swaps.append(np.real(np.vdot(v, v[perm])))
        assert A.j_squared_from_swaps(n, swaps) == pytest.approx(np.real(np.vdot(v, J2 @ v)), abs=1e-12)
    assert A.l_from_j_squared(A.j_squared_from_swaps(2, [1.0])) == pytest.approx(1)
    assert A.l_from_j_squared(A.j_squared_from_swaps(2, [-1.0])) == pytest.approx(0)
