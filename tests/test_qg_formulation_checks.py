"""
Tests for examples/qg_formulation_checks.py: every identity quoted in the qg
formulation of the main gates and algorithms (manuscript/teoria_es).
"""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import qg_formulation_checks as F  # noqa: E402

C, Sn = math.cos(0.7), math.sin(0.7)
R2 = 1 / math.sqrt(2)


@pytest.mark.parametrize("gate,expected", [
    ("X", {"X": {"X": 1}, "Y": {"Y": -1}, "Z": {"Z": -1}}),
    ("Y", {"X": {"X": -1}, "Y": {"Y": 1}, "Z": {"Z": -1}}),
    ("Z", {"X": {"X": -1}, "Y": {"Y": -1}, "Z": {"Z": 1}}),
    ("H", {"X": {"Z": 1}, "Y": {"Y": -1}, "Z": {"X": 1}}),
    ("S", {"X": {"Y": -1}, "Y": {"X": 1}, "Z": {"Z": 1}}),
    ("T", {"X": {"X": R2, "Y": -R2}, "Y": {"X": R2, "Y": R2}, "Z": {"Z": 1}}),
])
def test_single_qubit_gates(gate, expected):
    for p, exp in expected.items():
        got = F.conjugate(F.GATES_1Q[gate], p)
        assert got.keys() == exp.keys()
        for k in exp:
            assert abs(got[k] - exp[k]) < 1e-12


def test_rotations():
    assert F.conjugate(F.rx(0.7), "Z") == pytest.approx({"Z": C, "Y": Sn})
    assert F.conjugate(F.rx(0.7), "Y") == pytest.approx({"Y": C, "Z": -Sn})
    assert F.conjugate(F.ry(0.7), "Z") == pytest.approx({"Z": C, "X": -Sn})
    assert F.conjugate(F.ry(0.7), "X") == pytest.approx({"X": C, "Z": Sn})
    assert F.conjugate(F.rz(0.7), "X") == pytest.approx({"X": C, "Y": -Sn})
    assert F.conjugate(F.rz(0.7), "Y") == pytest.approx({"X": Sn, "Y": C})
    # the qang identity: Ry(theta)|0> has qg_Z = cos(theta), qg_X = sin(theta)
    psi = F.ry(0.7) @ np.array([1, 0])
    assert F.local_qg(psi, 1, 0) == pytest.approx((Sn, 0.0, C))
    # P(phi) and Rz(phi) act identically on qg
    for p in "XYZ":
        assert F.conjugate(F.phase(0.7), p) == pytest.approx(F.conjugate(F.rz(0.7), p))


def test_two_qubit_gates():
    cj = F.conjugate
    assert cj(F.CX, "XI") == {"XX": 1} and cj(F.CX, "IZ") == {"ZZ": 1}
    assert cj(F.CX, "ZI") == {"ZI": 1} and cj(F.CX, "IX") == {"IX": 1}
    assert cj(F.CZ, "XI") == {"XZ": 1} and cj(F.CZ, "IX") == {"ZX": 1} and cj(F.CZ, "ZI") == {"ZI": 1}
    assert cj(F.SWAP, "XI") == {"IX": 1} and cj(F.SWAP, "ZI") == {"IZ": 1}
    assert cj(F.ISWAP, "ZI") == {"IZ": 1} and cj(F.ISWAP, "XI") == {"ZY": -1} and cj(F.ISWAP, "YI") == {"ZX": 1}


def test_three_qubit_gates():
    assert F.conjugate(F.TOFFOLI, "IIZ") == pytest.approx({"IIZ": 0.5, "IZZ": 0.5, "ZIZ": 0.5, "ZZZ": -0.5})
    assert F.conjugate(F.TOFFOLI, "IIX") == {"IIX": 1}
    assert F.conjugate(F.FREDKIN, "IZI") == pytest.approx({"IZI": 0.5, "ZZI": 0.5, "IIZ": 0.5, "ZIZ": -0.5})


def test_weight_conservation_classification():
    keep = {"CZ", "SWAP", "iSWAP", "Fredkin"}
    for name, U in F.GATES_MULTI.items():
        assert F.conserves_weight(U, int(np.log2(U.shape[0]))) == (name in keep)
    for name, U in F.GATES_1Q.items():
        assert F.conserves_weight(U, 1) == (name in {"Z", "S", "T"})


def test_deutsch_jozsa_and_bv():
    n = 4
    const = F.phase_oracle_output(np.ones(16))
    assert F.single_qg_z(const, n) == pytest.approx([1, 1, 1, 1])
    rng = np.random.default_rng(0)
    for _ in range(20):
        f = np.array([1] * 8 + [-1] * 8)
        rng.shuffle(f)
        p = F.phase_oracle_output(f)
        assert min(F.single_qg_z(p, n)) < 1 - 1e-9  # balanced: some qg_Z < 1
        avg = sum(F.z_parity(p, n, S) for S in itertools.product([0, 1], repeat=n)) / 2**n
        assert abs(avg - p[0]) < 1e-12  # P(0^n) = mean of all Z-parities
    s = np.array([1, 0, 1, 1])
    xs = (np.arange(16)[:, None] >> np.arange(4)[::-1]) & 1
    p = F.phase_oracle_output((-1.0) ** (xs @ s))
    assert F.single_qg_z(p, n) == pytest.approx([(-1) ** b for b in s])


def test_simon_single_nonzero_parity():
    s = [1, 0, 1]
    p = F.simon_output(s)
    nz = {S for S in itertools.product([0, 1], repeat=3) if abs(F.z_parity(p, 3, S)) > 1e-12}
    assert nz == {(0, 0, 0), tuple(s)}


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_grover_qg_z_formula(k):
    psi = F.grover_state(5, 19, k)
    p = psi**2
    assert F.single_qg_z(p, 5) == pytest.approx(F.grover_qg_z_prediction(5, 19, p[19]), abs=1e-12)


@pytest.mark.parametrize("x", [3, 5])
def test_qft_basis_state_is_equatorial_product(x):
    n = 3
    psi = F.qft_matrix(n) @ np.eye(2**n)[x]
    for l in range(1, n + 1):
        qx, qy, qz = F.local_qg(psi, n, l - 1)
        a = 2 * math.pi * x / 2**l
        assert (qx, qy, qz) == pytest.approx((math.cos(a), math.sin(a), 0.0), abs=1e-12)


def test_kickback_and_counting():
    assert F.kickback_control(0.9) == pytest.approx((math.cos(0.9), math.sin(0.9)))
    qx = F.counting_qg_x(4, [2, 7, 11])
    assert qx == pytest.approx(1 - 2 * 3 / 16)
    assert 16 * (1 - qx) / 2 == pytest.approx(3)


def test_order_finding_marginals():
    q4 = F.single_qg_z(F.order_finding_output(4, 6), 6)
    assert q4 == pytest.approx([0, 0, 1, 1, 1, 1], abs=1e-9)  # r = 4 visible in the marginals
    q6 = F.single_qg_z(F.order_finding_output(6, 6), 6)
    assert max(abs(v) for v in q6) < 0.5  # r = 6 is not


def test_purity_and_walk():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    assert F.purity_from_qg(rho, 2) == pytest.approx(float(np.real(np.trace(rho @ rho))))
    pytest.importorskip("scipy")
    assert F.walk_register_mean(5, 4, (0.5, 2.0)) == pytest.approx([0.6, 0.6])
