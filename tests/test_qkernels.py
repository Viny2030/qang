import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qkernels import product_fidelity_kernel, unary_inversion_test
def test_same_point_is_one(): r = jnp.array([[0., 0.6, 0.8], [1., 0., 0.]]); assert abs(float(product_fidelity_kernel(r, r)) - 1) < 1e-14
def test_unary(): x = jnp.array([0.6, 0.8, 0., 0.]); y = jnp.array([0., 0., 1., 0.]); assert float(unary_inversion_test(x, y, 4)) < 1e-14
