"""
Tests for qang.formulation: every identity quoted in the qg formulation of the
main gates and algorithms (manuscript/teoria_es), plus the working helpers.
"""

import itertools
import math
import numpy as np
import pytest

import qang.formulation as F

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


def test_qg_values_roundtrip_and_apply_gate():
    psi = np.array([1, 0, 0, 0], dtype=complex)
    qg = F.qg_values(psi)
    bell = F.apply_gate(F.apply_gate(qg, np.kron(F.H, F.I2)), F.CX)
    assert bell == pytest.approx({"II": 1, "XX": 1, "YY": -1, "ZZ": 1})
    assert bell == {"II": 1.0, "XX": 1.0, "YY": -1.0, "ZZ": 1.0}  # exact after clean()
    assert F.clean(-1e-17) == 0.0 and str(F.clean(-1e-17)) == "0.0"
    rho = F.state_from_qg(bell, 2)
    target = np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2
    assert np.allclose(rho, target)


def test_clifford_classification_and_born():
    assert F.is_clifford(F.H) and F.is_clifford(F.CX) and F.is_clifford(F.ISWAP)
    assert not F.is_clifford(F.T) and not F.is_clifford(F.TOFFOLI) and not F.is_clifford(F.FREDKIN)
    assert F.born_p0(0.5) == 0.75


def test_readout_helpers():
    n = 4
    assert F.deutsch_jozsa_is_constant(F.phase_oracle_output(np.ones(16)), n)
    f = np.array([1, -1] * 8)
    assert not F.deutsch_jozsa_is_constant(F.phase_oracle_output(f), n)
    s = np.array([1, 0, 1, 1])
    xs = (np.arange(16)[:, None] >> np.arange(4)[::-1]) & 1
    assert F.bernstein_vazirani_secret(F.phase_oracle_output((-1.0) ** (xs @ s)), n) == [1, 0, 1, 1]
    assert set(F.simon_nonzero_parities(F.simon_output([1, 0, 1]), 3)) == {(0, 0, 0), (1, 0, 1)}
    p = F.grover_state(5, 19, 4) ** 2
    assert F.grover_marked_from_signs(p, 5) == [1, 0, 0, 1, 1]
    assert F.count_from_qg_x(F.counting_qg_x(4, [2, 7, 11]), 16) == pytest.approx(3)


def test_hhl_vqe_maxcut_kernel():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    qz, ez = F.hhl_readout(A, [1, 0], F.Z)
    x = np.linalg.solve(A, [1, 0])
    x = x / np.linalg.norm(x)
    assert ez == pytest.approx(x[0] ** 2 - x[1] ** 2)
    lam = np.linalg.eigvalsh(A)
    assert -1 < qz < 1 and qz == pytest.approx(1 - 2 * float(np.sum(np.abs(np.linalg.eigh(A)[1].T @ [1, 0]) ** 2 * (min(abs(lam)) / lam) ** 2)))
    bell = {"II": 1, "XX": 1, "YY": -1, "ZZ": 1}
    assert F.energy_from_qg({"ZZ": 1.0, "XX": 0.5, "II": -0.2}, bell) == pytest.approx(1.3)
    assert F.maxcut_from_qg([(0, 1, 1.0)], {"ZZ": -1.0}, 2) == pytest.approx(1.0)
    assert F.product_kernel([(0, 0, 1)], [(1, 0, 0)]) == pytest.approx(0.5)


# ---- the local radius (surface) metric, RESEARCH_NOTES §97 ----
def test_radius_basics_and_tangle():
    bell = np.array([1, 0, 0, 1]) / math.sqrt(2)
    assert F.radius_profile(bell) == [0.0, 0.0]
    assert F.radius_profile(F.qg_values(bell)) == [0.0, 0.0]
    assert F.meyer_wallach(bell) == 1.0
    assert F.sphere_area(1.0) == pytest.approx(4 * math.pi)
    rng = np.random.default_rng(0)
    for _ in range(5):
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        conc = 2 * abs(psi[0] * psi[3] - psi[1] * psi[2])
        assert F.radius_deficit(psi) == pytest.approx([conc**2, conc**2])
        rho = np.outer(psi, psi.conj())
        rho_a = rho.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)
        assert 1 - F.radius_profile(psi)[0] == pytest.approx(4 * np.linalg.det(rho_a).real)
        assert F.radius2(*F.local_qg(psi, 2, 0)) == pytest.approx(2 * np.trace(rho_a @ rho_a).real - 1)


def test_gate_radius_classes():
    cls = {name: F.gate_radius_class(U) for name, U in {**F.GATES_1Q, **F.GATES_MULTI}.items()}
    for name in "XYZHST":
        assert cls[name]["class"] == "preserves" and cls[name]["max deficit"] == 0.0
    for name in ("Rx", "Ry", "Rz", "P"):
        assert F.gate_radius_class(F.GATES_PARAMETRIC[name](0.7))["class"] == "preserves"
    assert cls["SWAP"]["class"] == "permutes"
    for name in ("CX", "CZ", "iSWAP", "Toffoli", "Fredkin"):
        assert cls[name]["class"] == "entangling" and cls[name]["max deficit"] == 1.0
    # a one-qubit gate keeps the radius of a mixed qubit too
    rho = F.state_from_qg({"I": 1, "X": 0.3, "Z": -0.4}, 1)
    out = F.T @ rho @ F.T.conj().T
    assert F.radius_profile(out, 1) == pytest.approx(F.radius_profile(rho, 1))


def test_algorithm_radii():
    a = F.algorithm_radii()
    assert a["Deutsch-Jozsa constant"] == [1.0] * 4
    assert a["Bernstein-Vazirani"] == [1.0] * 4 and a["QFT"] == [1.0] * 4
    assert max(a["Deutsch-Jozsa balanced"]) < 1
    assert a["Simon"] == [0.0, 0.0, 0.0]
    assert a["Shor r=4"] == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert max(a["Shor r=6"]) < 0.12
    g = a["Grover"]
    assert g[0] == [1.0] * 5 and min(g[2]) < 0.67 and min(g[4]) > 0.998
    assert a["kickback eigenstate"] == 1.0
    assert a["kickback non-eigenstate"] == pytest.approx(math.cos(0.45) ** 2)
    assert a["counting"] == pytest.approx((1 - 2 * 3 / 16) ** 2)
    assert 0 < a["HHL ancilla"] < 1
    assert a["VQE singlet"] == [0.0, 0.0] and a["VQE Ising h=1"][0] < 1
    assert a["kernel product encoding"] == [1.0] * 3


def test_weight_sector_radius_is_qg_z():
    from scipy.linalg import expm
    n = 5
    hop = lambda j, c: F.pauli("".join(c if k in (j, j + 1) else "I" for k in range(n)))  # noqa: E731
    Hw = sum((hop(j, "X") + hop(j, "Y")) / 2 for j in range(n - 1))
    psi = expm(-2j * Hw) @ np.eye(2**n)[4]
    qz = [F.local_qg(psi, n, q)[2] for q in range(n)]
    assert F.radius_profile(psi) == pytest.approx([z**2 for z in qz], abs=1e-12)
    assert F.weight_sector_radius(qz) == pytest.approx(F.radius_deficit(psi), abs=1e-12)


def test_direction_from_qg_is_invariant_under_depolarizing():
    for theta, phi in ((0.3, 0.2), (1.2, -2.0), (2.9, 1.0)):
        q = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
        for shrink in (1.0, 0.7, 0.2):
            t, f, r = F.direction_from_qg(*(shrink * q))
            assert t == pytest.approx(theta) and f == pytest.approx(phi) and r == pytest.approx(shrink)
