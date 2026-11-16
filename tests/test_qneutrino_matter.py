import pytest; jax = pytest.importorskip("jax")
import numpy as np
from qang.qneutrino import cos2theta_m, resonance_A, adiabatic_survival_qz

def test_resonance_is_equator():
    th, dm2 = 0.583, 7.42e-5
    assert abs(float(cos2theta_m(th, dm2, resonance_A(th, dm2)))) < 1e-14

def test_vacuum_limit():
    th, dm2 = 0.583, 7.42e-5
    assert abs(float(adiabatic_survival_qz(th, dm2, 0.0)) - np.cos(2 * th) ** 2) < 1e-14
