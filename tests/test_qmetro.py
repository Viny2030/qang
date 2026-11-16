import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qmetro import ramsey_fisher_phi, ghz_fisher_omega, product_fisher_omega, optimal_T, thermal_fisher_x

def test_equator_optimum():
    assert abs(float(ramsey_fisher_phi(jnp.pi / 2, 0.7)) - 0.49) < 1e-12

def test_ghz_no_gain_per_time():
    _, vg = optimal_T(ghz_fisher_omega, 1.0, 4.0); _, vp = optimal_T(product_fisher_omega, 1.0, 4.0)
    assert abs(vg - vp) < 1e-9

def test_thermal_fisher_closed_form():
    x = jnp.asarray(1.3)
    assert abs(float(thermal_fisher_x(x)) - 0.25 / np.cosh(0.65) ** 2) < 1e-12
