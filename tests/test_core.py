import math

import pytest

from qang.core import Qang, MILLIQANG_PER_QANG, qg_s_from_qg_z

PI = math.pi


# --------------------------------------------------------------------- #
# Table 1 (qg_Z) and Table 2 (qg_S) anchor values, from the paper
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "theta, expected_qg_z",
    [
        (0.0, 1.0),
        (PI / 2, 0.0),
        (PI, -1.0),
        (PI / 4, math.sqrt(2) / 2),
    ],
)
def test_table1_qg_z_anchors(theta, expected_qg_z):
    q = Qang.from_angles(theta, mode="polar")
    assert q.value == pytest.approx(expected_qg_z, abs=1e-9)


@pytest.mark.parametrize(
    "theta, expected_qg_s",
    [
        (0.0, 0.0),
        (PI / 2, 1.0),
        (PI, 0.0),
        (PI / 4, 0.6009, ),  # paper Table 2, ~0.601
    ],
)
def test_table2_qg_s_anchors(theta, expected_qg_s):
    q = Qang.from_angles(theta, mode="entropic")
    assert q.value == pytest.approx(expected_qg_s, abs=1e-3)


def test_qg_s_symmetry_theta_vs_pi_minus_theta():
    """Section 2.2: qg_S(theta) = qg_S(pi - theta)."""
    for theta in (0.3, 1.0, 1.5, 2.0, 2.8):
        a = Qang.from_angles(theta, mode="entropic").value
        b = Qang.from_angles(PI - theta, mode="entropic").value
        assert a == pytest.approx(b, abs=1e-9)


# --------------------------------------------------------------------- #
# validation / range checks
# --------------------------------------------------------------------- #
def test_polar_out_of_range_raises():
    with pytest.raises(ValueError):
        Qang(1.5, mode="polar")
    with pytest.raises(ValueError):
        Qang(-1.5, mode="polar")


def test_entropic_out_of_range_raises():
    with pytest.raises(ValueError):
        Qang(-0.1, mode="entropic")
    with pytest.raises(ValueError):
        Qang(1.1, mode="entropic")


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        Qang(0.5, mode="bogus")


# --------------------------------------------------------------------- #
# inversion relation theta = arccos(qg_Z), and full round trips
# --------------------------------------------------------------------- #
def test_theta_inversion_round_trip():
    for theta in (0.1, 0.7854, 1.2, 2.5, 3.0):
        q = Qang.from_angles(theta, mode="polar")
        assert q.to_theta() == pytest.approx(theta, abs=1e-9)


def test_bloch_vector_round_trip():
    for theta, phi in ((0.9, 1.1), (2.0, 4.2), (PI / 3, PI / 6)):
        q = Qang.from_angles(theta, phi=phi, mode="polar")
        x, y, z = q.to_bloch_vector()
        q2 = Qang.from_bloch_vector(x, y, z)
        assert q2.value == pytest.approx(q.value, abs=1e-6)
        assert q2.phi == pytest.approx(phi % (2 * PI), abs=1e-6)


def test_statevector_round_trip():
    for theta, phi in ((0.9, 1.1), (2.0, 4.2), (PI / 3, PI / 6)):
        q = Qang.from_angles(theta, phi=phi, mode="polar")
        alpha, beta = q.to_statevector()
        q2 = Qang.from_statevector(alpha, beta)
        assert q2.value == pytest.approx(q.value, abs=1e-6)
        assert q2.phi == pytest.approx(phi % (2 * PI), abs=1e-6)


def test_probability_round_trip():
    q = Qang.from_probability(p0=0.85, mode="polar")
    assert q.value == pytest.approx(0.7, abs=1e-9)
    p0, p1 = q.to_probabilities()
    assert p0 == pytest.approx(0.85, abs=1e-9)
    assert p1 == pytest.approx(0.15, abs=1e-9)


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        Qang.from_probability(p0=0.3, p1=0.3, mode="polar")


# --------------------------------------------------------------------- #
# entropic mode: entropic instances cannot do bijective conversions
# --------------------------------------------------------------------- #
def test_entropic_mode_rejects_bloch_conversions():
    q = Qang(0.9, mode="entropic")
    with pytest.raises(NotImplementedError):
        q.to_theta()
    with pytest.raises(NotImplementedError):
        q.to_bloch_vector()
    with pytest.raises(NotImplementedError):
        q.to_statevector()


# --------------------------------------------------------------------- #
# explicit half-domain inversion of qg_S (Future Research Direction #2)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.2, 0.7, 1.0, 1.5])
def test_theta_from_entropic_lower_branch(theta):
    s = Qang.from_angles(theta, mode="entropic").value
    recovered = Qang.theta_from_entropic(s, branch="lower")
    assert recovered == pytest.approx(theta, abs=1e-4)


@pytest.mark.parametrize("theta", [1.6, 2.1, 2.6, 3.0])
def test_theta_from_entropic_upper_branch(theta):
    s = Qang.from_angles(theta, mode="entropic").value
    recovered = Qang.theta_from_entropic(s, branch="upper")
    assert recovered == pytest.approx(theta, abs=1e-4)


# --------------------------------------------------------------------- #
# milliqang subunit
# --------------------------------------------------------------------- #
def test_milliqang_conversion():
    q = Qang(0.5, mode="polar")
    assert q.milliqang == pytest.approx(500.0)
    q2 = Qang.from_milliqang(500.0, mode="polar")
    assert q2.value == pytest.approx(0.5)
    assert MILLIQANG_PER_QANG == 1000.0


def test_repr_smoke():
    assert "qg_Z" in repr(Qang(0.5, mode="polar"))
    assert "qg_S" in repr(Qang(0.5, mode="entropic"))


def test_equality():
    a = Qang.from_angles(1.0, phi=0.5, mode="polar")
    b = Qang.from_angles(1.0, phi=0.5, mode="polar")
    c = Qang.from_angles(1.0, phi=0.6, mode="polar")
    assert a == b
    assert a != c


# --------------------------------------------------------------------- #
# the direct qg_Z <-> qg_S relationship (this module's docstring, "The
# direct qg_Z <-> qg_S relationship (no theta, no branch)")
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, 2.0, 2.9])
def test_qg_s_from_qg_z_matches_the_theta_roundtrip(theta):
    """qg_s_from_qg_z(qg_Z(theta)) must agree exactly with computing qg_S
    directly from theta, for every theta -- these are two routes to the
    same number, not two different definitions."""
    qg_z = Qang.from_angles(theta, mode="polar").value
    direct = qg_s_from_qg_z(qg_z)
    via_theta = Qang.from_angles(theta, mode="entropic").value
    assert direct == pytest.approx(via_theta, abs=1e-9)


@pytest.mark.parametrize("qg_z", [-1.0, -0.5, 0.0, 0.3, 1.0])
def test_qg_s_from_qg_z_anchors(qg_z):
    """qg_Z = +-1 (a computational basis state) has zero measurement
    entropy; qg_Z = 0 (the equator) has maximal entropy."""
    s = qg_s_from_qg_z(qg_z)
    if abs(qg_z) == 1.0:
        assert s == pytest.approx(0.0, abs=1e-9)
    elif qg_z == 0.0:
        assert s == pytest.approx(1.0, abs=1e-9)
    else:
        assert 0.0 < s < 1.0


def test_qg_s_from_qg_z_rejects_out_of_range_input():
    with pytest.raises(ValueError):
        qg_s_from_qg_z(1.5)
    with pytest.raises(ValueError):
        qg_s_from_qg_z(-1.5)


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, 2.0, 2.9])
def test_to_entropic_matches_qg_s_from_qg_z(theta):
    q = Qang.from_angles(theta, mode="polar")
    qe = q.to_entropic()
    assert qe.mode == "entropic"
    assert qe.value == pytest.approx(qg_s_from_qg_z(q.value), abs=1e-12)


def test_to_entropic_requires_polar_mode():
    q = Qang(0.5, mode="entropic")
    with pytest.raises(NotImplementedError):
        q.to_entropic()
