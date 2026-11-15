"""tools/verify_cap27.py — XY/caminata en sector peso 1 con RBS, canales afines, varQTE. python tools/verify_cap27.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qdynamics import (exact_walk, trotter_walk_rbs, walk_qz, AffineChannel, dissipative_deficit,
                            imaginary_time_qz_exact, varqte_ry_mclachlan)
from qang.qsymfilter import kraus_amplitude_damping

# 1. Caminata cuántica XY (n=6, J=1, t=2): exacta vs Trotter par-impar de RBS; conservación del peso exacta
n, t = 6, 2.0
p_ex = np.asarray(exact_walk(n, t))
for steps in (1, 4, 16, 64):
    p_tr, ang_qga = trotter_walk_rbs(n, t, steps)
    err = float(np.max(np.abs(np.asarray(p_tr) - p_ex)))
    print(f"[1] Trotter {steps:3d} pasos × {n-1} RBS({ang_qga:6.1f} qga): error máx={err:.2e}; Σp={float(p_tr.sum()):.12f}")
e16 = float(np.max(np.abs(np.asarray(trotter_walk_rbs(n, t, 16)[0]) - p_ex))); e64 = float(np.max(np.abs(np.asarray(trotter_walk_rbs(n, t, 64)[0]) - p_ex)))
assert e64 < 1e-2 and e64 < e16 / 3     # Trotter de primer orden: error ∝ dt
print(f"    lectura m-qg por sitio (t=2): {np.round(1000*np.asarray(walk_qz(jnp.asarray(p_ex)))).astype(int)}")
print("    sector de peso 1 conservado por construcción ⇒ filtro de simetría exacto bajo T1 (Cap. 24) también en dinámica")

# 2. Canales afines: composición y comparación con Kraus (T1); déficit disipativo
r0 = jnp.array([0.6, 0.0, 0.5])
ch = AffineChannel.amplitude_damping(0.2).then(AffineChannel.dephasing(0.05)).then(AffineChannel.readout(0.02, 0.06))
r1 = ch(r0)
# Kraus T1 sobre ρ(r0) para comprobar la forma afín
I2 = jnp.eye(2); X = jnp.array([[0, 1], [1, 0]], dtype=complex); Y = jnp.array([[0, -1j], [1j, 0]]); Z = jnp.diag(jnp.array([1., -1.]))
rho = 0.5 * (I2 + r0[0] * X + r0[1] * Y + r0[2] * Z)
K = kraus_amplitude_damping(0.2); rho1 = sum(k @ rho @ jnp.conj(k).T for k in K)
r_kraus = jnp.real(jnp.array([jnp.trace(rho1 @ X), jnp.trace(rho1 @ Y), jnp.trace(rho1 @ Z)]))
assert np.allclose(np.asarray(AffineChannel.amplitude_damping(0.2)(r0)), np.asarray(r_kraus), atol=1e-14)
print(f"[2] T1 afín = Kraus a 1e-14; cadena T1(0.2)∘desfase(0.05)∘lectura(0.02,0.06): r={np.asarray(r0)} → {np.round(np.asarray(r1),4)}; "
      f"m-qg {1000*float(r0[2]):+.0f} → {1000*float(r1[2]):+.0f}; qg_τ {float(dissipative_deficit(r0)):.3f} → {float(dissipative_deficit(r1)):.3f}")
# la composición es un solo (M, c): 1 calibración de 12 números sustituye 3 modelos
M, c = ch.M, ch.c
assert np.allclose(np.asarray(M @ r0 + c), np.asarray(r1), atol=1e-14)

# 3. varQTE de 1 qubit: McLachlan con RY(θ) en qga reproduce qg_Z(τ) = −tanh(ωτ)
omega, tau_max = 1.0, 3.0
for steps in (30, 300, 3000):
    q = varqte_ry_mclachlan(omega, tau_max, steps)
    taus = jnp.linspace(tau_max / steps, tau_max, steps)
    err = float(jnp.max(jnp.abs(q - imaginary_time_qz_exact(omega, taus))))
    print(f"[3] varQTE Euler {steps:4d} pasos: error máx en qg_Z = {err:.2e} (final {1000*float(q[-1]):+.0f} m-qg vs {1000*float(imaginary_time_qz_exact(omega, tau_max)):+.0f})")
assert float(jnp.max(jnp.abs(varqte_ry_mclachlan(omega, tau_max, 3000) - imaginary_time_qz_exact(omega, jnp.linspace(tau_max/3000, tau_max, 3000))))) < 2e-3
print("    el estado térmico del Cap. 11 (qg_Z = tanh) es el punto fijo de esta ecuación: 1 parámetro, 0 ancillas, sin mesetas (P=1)")
print("verify_cap27: OK")
