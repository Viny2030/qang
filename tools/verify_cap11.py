"""tools/verify_cap11.py — Metrología en qga: Ramsey, GHZ, termometría, relojes (JAX). python tools/verify_cap11.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qmetro import (ramsey_fisher_phi, ramsey_fisher_omega, ghz_fisher_omega, product_fisher_omega,
                         optimal_T, thermal_qz, thermal_fisher_x, thermal_fisher_T, clock_redshift_phase,
                         shots_to_resolve_height, RAD_PER_QGA)

# 1. Punto de operación Ramsey: C<1 ⇒ máximo de I_F(φ) en φ=π/2 (qg_Z=0), valor C²
phis = jnp.linspace(0.01, np.pi - 0.01, 2001)
for C in (1.0, 0.8, 0.5):
    I = ramsey_fisher_phi(phis, jnp.asarray(C))
    k = int(jnp.argmax(I))
    assert abs(float(I[k]) - C ** 2) < 1e-6 and (C == 1.0 or abs(float(phis[k]) - np.pi / 2) < 2e-3)
    I_pole = float(ramsey_fisher_phi(jnp.asarray(0.2), jnp.asarray(C)))
    print(f"[1] C={C}: I_F máx={float(I[k]):.4f} en φ=π/2 (qg_Z=0, 250 qga); en φ=0.2 rad: {I_pole:.4f} "
          f"→ pérdida {100*(1-I_pole/C**2):.0f}% por operar cerca del polo")
# C=1: uniforme (coincide con Cap. 9: I_F = 1 por disparo, σ_qga = 159.15/√N)
assert float(jnp.std(ramsey_fisher_phi(phis, jnp.asarray(1.0)))) < 1e-12

# 2. Tiempo óptimo Ramsey: max_T I_F(ω)/T ⇒ T* = T2/2  (= argmax T e^{−2T/T2})
T2 = 1.0
T_star, val = optimal_T(lambda n, T, T2_: ramsey_fisher_omega(T, T2_) * n, T2, 1.0)
assert abs(T_star - T2 / 2) < 1e-6
print(f"[2] Ramsey: T* = {T_star:.6f} T2 (I_F(ω)/T máx = {val:.6f} T2)")

# 3. GHZ vs producto bajo desfase markoviano independiente: misma Fisher por unidad de tiempo (no gain)
for n in (2, 5, 10):
    Tg, vg = optimal_T(ghz_fisher_omega, T2, n)
    Tp, vp = optimal_T(product_fisher_omega, T2, n)
    assert abs(Tg - T2 / (2 * n)) < 1e-6 and abs(Tp - T2 / 2) < 1e-6 and abs(vg - vp) < 1e-9
    print(f"[3] n={n:2d}: GHZ T*={Tg:.4f} T2, I/T={vg:.5f} | producto T*={Tp:.4f} T2, I/T={vp:.5f} → iguales")
# a T fijo = T2/2 el GHZ pierde
n = 5
r = float(ghz_fisher_omega(jnp.asarray(n), jnp.asarray(0.5), jnp.asarray(T2)) /
          product_fisher_omega(jnp.asarray(n), jnp.asarray(0.5), jnp.asarray(T2)))
print(f"    a T=T2/2 fijo, n=5: I_GHZ/I_prod = {r:.4f}  (GHZ sólo gana si T ≪ T2/n: T<{T2*np.log(n)/(2*(n-1)):.3f} T2 = T2·ln n/(2(n−1)))")

# 4. Termometría tanh: I_F(x) = ¼ sech²(x/2); óptimo relativo x* (qg_Z*)
xs = jnp.linspace(0.05, 8.0, 4001)
assert float(jnp.max(jnp.abs(jax.vmap(thermal_fisher_x)(xs) - 0.25 / jnp.cosh(xs / 2) ** 2))) < 1e-12
k = int(jnp.argmax(jax.vmap(thermal_fisher_T)(xs)))
x_star, q_star = float(xs[k]), float(thermal_qz(xs[k]))
assert abs(x_star - 2.3994) < 5e-3
print(f"[4] termometría: x*=ħω/kT*={x_star:.3f}, qg_Z*={q_star:.3f} ({1000*q_star:.0f} m-qg), "
      f"I_F(ln T)*={float(thermal_fisher_T(xs[k])):.4f}/disparo")

# 5. Relojes: Sr (ω = 2π·429.228 THz), T = 1 s, Δh = 1 cm
om = 2 * np.pi * 429.228e12
dphi = clock_redshift_phase(om, 1.0, 0.01)
N3 = shots_to_resolve_height(om, 1.0, 0.01, z=3.0)
print(f"[5] Δφ = {dphi:.3e} rad = {dphi/RAD_PER_QGA:.3f} qga por cm; N(3σ, C=1) = {N3:.3e} disparos; "
      f"con C=0.5: {shots_to_resolve_height(om, 1.0, 0.01, 3.0, 0.5):.3e}")
assert abs(dphi - om * 9.80665 * 0.01 / 299792458.0 ** 2) < 1e-18
print("verify_cap11: OK")
