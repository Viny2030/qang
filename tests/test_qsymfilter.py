import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qweightqnn import unary_loader
from qang.qsymfilter import run_weight_circuit, kraus_amplitude_damping, apply_local_channel, filtered_sector_state

def test_filter_exact_under_uniform_T1():
    n, pairs = 4, ((0, 1), (2, 3), (1, 2)); th = jnp.array([100., 300., 700.])
    psi = unary_loader(jnp.array([0.6, 0.0, 0.8, 0.0]), n); rho0 = jnp.outer(psi, jnp.conj(psi))
    S0, _ = filtered_sector_state(run_weight_circuit(rho0, th, pairs, n), n)
    S1, acc = filtered_sector_state(run_weight_circuit(rho0, th, pairs, n, lambda r: apply_local_channel(r, kraus_amplitude_damping(0.1), n)), n)
    assert float(jnp.max(jnp.abs(S0 - S1))) < 1e-13 and abs(float(acc) - 0.9 ** 3) < 1e-13
