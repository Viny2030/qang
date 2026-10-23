"""Tests for qang.probability (§167): the qang probability equation and the milli-qang."""

import math

import pytest

from qang import probability as PB
from qang.core import Qang, qg_s_from_qg_z
from qang.proportions import theta_from_proportion


def test_four_readings_of_one_probability():
    for p0 in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        r = PB.readings_from_p0(p0)
        assert math.isclose(math.cos(r["theta"] / 2) ** 2, p0, abs_tol=1e-12)  # Born's rule
        assert math.isclose(r["qg_z"], math.cos(r["theta"]), abs_tol=1e-12)
        assert math.isclose(r["qg_s"], qg_s_from_qg_z(r["qg_z"]), abs_tol=1e-12)
        assert math.isclose(r["milliqang"], Qang(r["qg_z"]).milliqang, abs_tol=1e-9)
        assert math.isclose(r["theta"], theta_from_proportion(p0), abs_tol=1e-12)


def test_milliqang_is_linear_in_probability():
    assert PB.milliqang_from_p0(0.0) == -1000.0
    assert PB.milliqang_from_p0(0.5) == 0.0
    assert PB.milliqang_from_p0(1.0) == 1000.0
    assert math.isclose(PB.milliqang_from_p0(0.5005), 1.0, abs_tol=1e-9)  # 1 m-qg = 0.0005 of p0
    for m in (-1000.0, -501.8, 0.0, 1.8, 1000.0):
        assert math.isclose(PB.milliqang_from_p0(PB.p0_from_milliqang(m)), m, abs_tol=1e-9)
        assert math.isclose(PB.milliqang_from_theta(PB.theta_from_milliqang(m)), m, abs_tol=1e-9)


def test_one_milliqang_is_one_milliradian_on_the_equator_and_grows_at_the_poles():
    assert math.isclose(PB.angle_per_milliqang(math.pi / 2), 1e-3, rel_tol=1e-12)
    assert PB.angle_per_milliqang(math.pi / 6) > PB.angle_per_milliqang(math.pi / 3)
    assert PB.angle_per_milliqang(0.0) == math.inf
    assert math.isclose(PB.pole_angle(1.0), 0.04473, abs_tol=1e-5)
    for d in (0.1, 1.0, 10.0):
        assert math.isclose(PB.pole_angle(d), math.sqrt(2 * d / 1000), rel_tol=0.01)


def test_shot_noise_and_shot_planning():
    assert math.isclose(PB.shot_noise_milliqang(math.pi / 2, 10_000), 10.0, rel_tol=1e-12)
    assert PB.shots_for_resolution(10.0, math.pi / 2) == 10_000
    assert PB.shots_for_resolution(1.0, math.pi / 2) == 1_000_000
    assert PB.shots_for_resolution(10.0, math.pi / 6) == 2_500  # sin(30 deg) = 1/2: four times fewer shots
    assert PB.shot_noise_milliqang(0.0, 100) == 0.0


def test_pulse_error_of_an_x_gate_is_twelve_milliqang():
    e = PB.pulse_error_milliqang(math.pi, 0.05)
    assert math.isclose(e, 12.26, abs_tol=0.01)  # qg_Z = -0.988 instead of -1
    assert PB.pulse_error_milliqang(math.pi / 2, 0.05) == pytest.approx(0.0, abs=1e-12)
    assert PB.pulse_error_milliqang(0.0, 0.05) == 0.0


def test_statistical_angle_reduces_to_half_the_qang_angle():
    for p0 in (0.0, 0.2, 0.5, 0.83, 1.0):
        th = PB.readings_from_p0(p0)["theta"]
        assert math.isclose(PB.statistical_angle([p0, 1 - p0], [1.0, 0.0]), th / 2, abs_tol=1e-7)
    assert PB.statistical_angle([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0, abs=1e-7)
    assert math.isclose(PB.statistical_angle([1, 0, 0], [0, 1, 0]), math.pi / 2)
    with pytest.raises(ValueError):
        PB.statistical_angle([0.5, 0.6], [0.5, 0.5])


def test_analyze_maps_one_interval_through_every_reading():
    r = PB.analyze(750, 1000, target_qg_z=0.5)
    assert r.qg_z == pytest.approx((0.5, 0.444, 0.552), abs=1e-3)
    assert r.milliqang == pytest.approx(tuple(1000 * x for x in r.qg_z))
    assert r.p0 == pytest.approx(tuple((1 + x) / 2 for x in r.qg_z))
    est, lo, hi = r.theta
    assert lo < est < hi and math.isclose(est, math.pi / 3, abs_tol=1e-12)
    assert r.qg_s[1] <= r.qg_s[0] <= r.qg_s[2]
    assert r.deviation_milliqang == pytest.approx(0.0, abs=1e-9)
    edge = PB.analyze(0, 50)
    assert edge.qg_z[1] >= -1.0 and edge.theta[2] <= math.pi


def test_invalid_inputs():
    with pytest.raises(ValueError):
        PB.milliqang_from_p0(1.2)
    with pytest.raises(ValueError):
        PB.theta_from_milliqang(1500)
    with pytest.raises(ValueError):
        PB.shots_for_resolution(0.0, 1.0)
    with pytest.raises(ValueError):
        PB.analyze(5, 10, target_qg_z=2.0)
