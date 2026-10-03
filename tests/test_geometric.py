"""Tests for qang.geometric (RESEARCH_NOTES §84): the geometric phase is minus
half the enclosed solid angle (Stokes' theorem on the Bloch sphere), checked
from state overlaps, from the solid angle and from the curvature flux."""

import math

import numpy as np
import pytest

from qang import geometric as G
from qang import polarization as P


def cone(qg_z, n=400, sign=1):
    s = math.sqrt(1 - qg_z**2)
    t = sign * np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([s * np.cos(t), s * np.sin(t), np.full(n, qg_z)], axis=1)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@pytest.mark.parametrize("qg_z", [0.9, 0.5, 0.0, -0.3])
def test_discrete_loop_phase_is_half_its_geodesic_solid_angle(qg_z):
    path = cone(qg_z, n=50)
    states = [G.state_from_bloch(r) for r in path]
    gamma = G.pancharatnam_phase(states)
    assert gamma == pytest.approx(G.berry_phase_from_solid_angle(G.solid_angle(path, reference=[0, 0, 1])), abs=1e-10)


@pytest.mark.parametrize("qg_z", [0.9, 0.5, 0.0, -0.3])
def test_cone_phase_is_fixed_by_qg_z(qg_z):
    path = cone(qg_z, n=4000)
    phi = G.geometric_phase_qg(path).phi
    expect = G.cone_phase_turns(qg_z)
    d = (phi - expect + 0.5) % 1.0 - 0.5
    assert abs(d) < 1e-5
    assert G.pancharatnam_phase([G.state_from_bloch(r) for r in path]) == pytest.approx(
        wrap(-math.pi * (1 - qg_z)), abs=1e-5)


@pytest.mark.parametrize("qg_z", [0.8, 0.2, -0.6])
def test_stokes_theorem_curvature_flux_equals_line_integral(qg_z):
    flux = G.curvature_flux(qg_z)
    assert flux == pytest.approx(math.pi * (1 - qg_z), rel=1e-5)
    line = -G.pancharatnam_phase([G.state_from_bloch(r) for r in cone(qg_z, n=4000)])
    assert wrap(line - flux) == pytest.approx(0, abs=1e-5)


def test_gauge_invariance_and_orientation():
    rng = np.random.default_rng(1)
    path = cone(0.3, n=60)
    states = [G.state_from_bloch(r) for r in path]
    g = G.pancharatnam_phase(states)
    rephased = [np.exp(1j * rng.uniform(0, 2 * math.pi)) * s for s in states]
    assert G.pancharatnam_phase(rephased) == pytest.approx(g, abs=1e-12)
    assert G.pancharatnam_phase(states[::-1]) == pytest.approx(-g, abs=1e-12)


def test_bloch_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(10):
        r = rng.normal(size=3)
        r /= np.linalg.norm(r)
        assert np.allclose(G.bloch_from_state(G.state_from_bloch(r)), r, atol=1e-12)


def test_pancharatnam_octant_for_polarized_light():
    # H -> D -> R -> H on the Poincare sphere: an octant, solid angle pi/2
    H, D, R = np.array([1, 0]), np.array([1, 1]) / math.sqrt(2), np.array([1, 1j]) / math.sqrt(2)
    path = [[P.jones_to_qg(E)[k] for k in "XYZ"] for E in (H, D, R)]
    assert G.solid_angle(path) == pytest.approx(math.pi / 2)
    assert G.pancharatnam_phase([H, D, R]) == pytest.approx(-math.pi / 4)
