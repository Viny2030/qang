import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qdynamics import exact_walk, trotter_walk_rbs, AffineChannel, imaginary_time_qz_exact, varqte_ry_mclachlan
def test_trotter_converges(): assert float(np.max(np.abs(np.asarray(trotter_walk_rbs(4, 1.0, 200)[0]) - np.asarray(exact_walk(4, 1.0))))) < 2e-3
def test_affine_compose(): ch = AffineChannel.depolarizing(0.5).then(AffineChannel.depolarizing(0.5)); assert np.allclose(np.asarray(ch(jnp.array([1., 0, 0]))), [0.25, 0, 0])
def test_varqte(): assert abs(float(varqte_ry_mclachlan(1.0, 2.0, 2000)[-1]) - float(imaginary_time_qz_exact(1.0, 2.0))) < 2e-3
