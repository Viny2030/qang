import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qgeom import state_from_qga, bloch_from_state, bargmann_phase, solid_angle_triangle, berry_phase_latitude

def test_bargmann_equals_minus_half_solid_angle():
    ps = [state_from_qga(a, p) for a, p in ((40., 10.), (200., 300.), (330., 700.))]
    Om = float(solid_angle_triangle(*[bloch_from_state(q) for q in ps]))
    assert abs(float(bargmann_phase(*ps)) + Om / 2) < 1e-12

def test_latitude_equator():
    assert abs(float(berry_phase_latitude(0.0)) + np.pi) < 1e-15
