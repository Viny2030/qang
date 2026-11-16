import pytest; jax = pytest.importorskip("jax")
import jax.numpy as jnp
from qang.qscramble import yoshida_kitaev, haar_unitary

def test_pf_law():
    P, F, Fq = yoshida_kitaev(haar_unitary(jax.random.PRNGKey(0), 16), 3, 2)
    assert abs(P * F - 0.25) < 1e-12 and abs(F - Fq) < 1e-12
