"""Tests for qang.qga, the angular qga scale (book, chapter 1)."""

import math

import numpy as np
import pytest

from qang import probability as pb
from qang import qga


def test_turn_and_conversions():
    assert qga.QGA_PER_TURN == 1000.0
    assert qga.radians_from_qga(1000) == pytest.approx(2 * math.pi)
    assert qga.qga_from_radians(math.pi) == pytest.approx(500.0)
    assert qga.qga_from_degrees(60) == pytest.approx(1000 / 6)
    assert qga.degrees_from_qga(250) == pytest.approx(90.0)
    for a in np.linspace(-1500, 1500, 31):
        assert qga.qga_from_radians(qga.radians_from_qga(a)) == pytest.approx(a)


@pytest.mark.parametrize("a, p0, m", [(0, 1.0, 1000.0), (125, 0.8535533905932737, 707.1067811865476),
                                      (250, 0.5, 0.0), (1000 / 6, 0.75, 500.0), (500, 0.0, -1000.0)])
def test_anchors(a, p0, m):
    assert qga.p0_from_qga(a) == pytest.approx(p0, abs=1e-12)
    assert qga.milliqang_from_qga(a) == pytest.approx(m, abs=1e-9)
    assert qga.qga_from_p0(p0) == pytest.approx(a, abs=1e-6)
    assert qga.qga_from_milliqang(m) == pytest.approx(a, abs=1e-6)


def test_agrees_with_probability_module():
    for a in np.linspace(0, 500, 41):
        p0 = qga.p0_from_qga(a)
        assert qga.milliqang_from_qga(a) == pytest.approx(pb.milliqang_from_p0(p0), abs=1e-9)
        assert qga.radians_from_qga(a) == pytest.approx(pb.theta_from_milliqang(qga.milliqang_from_qga(a)), abs=1e-7)


def test_p0_at_125_is_not_one_half():
    # the Foundation's claim P = 0.5 at 125 qg does not follow from its own equation
    assert qga.p0_from_qga(125) == pytest.approx(0.854, abs=1e-3)
    assert qga.qga_from_p0(0.5) == pytest.approx(250.0)


def test_azimuth():
    rng = np.random.default_rng(1)
    for _ in range(50):
        th, ph = rng.uniform(0.1, math.pi - 0.1), rng.uniform(0, 2 * math.pi)
        x, y = math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph)
        assert qga.azimuth_qga(x, y) == pytest.approx(ph * 1000 / (2 * math.pi), abs=1e-9)
        assert qga.azimuth_qga(0.3 * x, 0.3 * y) == pytest.approx(qga.azimuth_qga(x, y), abs=1e-9)
    with pytest.raises(ValueError):
        qga.azimuth_qga(0.0, 0.0)


def test_fisher_and_sigma():
    assert qga.FISHER_PER_QGA2 == pytest.approx(4 * math.pi ** 2 * 1e-6)
    assert qga.angle_sigma_qga(1000) == pytest.approx(5.0329, abs=1e-4)
    assert qga.angle_sigma_qga(1) == pytest.approx(1 / math.sqrt(qga.FISHER_PER_QGA2))
    # sampled spread of the estimated angle does not depend on the state
    rng = np.random.default_rng(2)
    n = 2000
    for p in (0.1, 0.5, 0.9):
        k = rng.binomial(n, p, 20000)
        est = [qga.qga_from_p0(x / n) for x in k]
        assert np.std(est) == pytest.approx(qga.angle_sigma_qga(n), rel=0.05)
    with pytest.raises(ValueError):
        qga.angle_sigma_qga(0)


def test_fubini_study():
    g_tt, g_pp = qga.fubini_study_metric_qga(250)
    assert g_tt == pytest.approx(math.pi ** 2 / 1e6)
    assert g_pp == pytest.approx(math.pi ** 2 / 1e6)
    assert qga.fubini_study_metric_qga(0)[1] == pytest.approx(0.0, abs=1e-20)


def test_cone_phase():
    assert qga.cone_phase_qga(0.5) == pytest.approx(750.0)
    assert qga.cone_phase_qga(0.0) == pytest.approx(500.0)
    assert qga.cone_phase_qga(1.0) == pytest.approx(0.0)
    # against the discrete Pancharatnam phase of a loop at constant height
    for qz in (0.8, -0.3, -0.9):
        th = math.acos(qz)
        n = 4000
        prod = 1.0 + 0j
        for k in range(n):
            a = np.array([math.cos(th / 2), np.exp(2j * math.pi * k / n) * math.sin(th / 2)])
            b = np.array([math.cos(th / 2), np.exp(2j * math.pi * (k + 1) / n) * math.sin(th / 2)])
            prod *= a.conj() @ b
        phase = (-np.angle(prod) * 1000 / (2 * math.pi)) % 1000
        assert phase == pytest.approx(qga.cone_phase_qga(qz), abs=1e-3)   # discretization error ~3e-5 qga


def test_input_checks():
    with pytest.raises(ValueError):
        qga.qga_from_p0(1.2)
    with pytest.raises(ValueError):
        qga.qga_from_milliqang(1001)
    with pytest.raises(ValueError):
        qga.cone_phase_qga(-1.5)
