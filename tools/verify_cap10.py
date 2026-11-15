"""tools/verify_cap10.py — Fases geométricas y Stokes en qga (JAX). python tools/verify_cap10.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qgeom import (state_from_qga, bloch_from_state, berry_phase_latitude, solid_angle_triangle,
                        bargmann_phase, solid_angle_polygon, shots_for_berry_latitude, RAD_PER_QGA)

wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

# 1. Lazo de latitud α = 150 qga, K puntos: fase de Pancharatnam discreta → −Ω/2 = −π(1−qg_Z)
alpha = 150.0
qz = float(np.cos(alpha * RAD_PER_QGA))
for K in (24, 96, 384):
    phis = jnp.linspace(0.0, 1000.0, K, endpoint=False)
    psis = jax.vmap(lambda ph: state_from_qga(alpha, ph))(phis)
    z = 1.0 + 0j
    for k in range(K):
        z = z * jnp.vdot(psis[(k + 1) % K], psis[k])   # γ = arg ∏<ψ_{k+1}|ψ_k> = i∮<ψ|dψ>
    gam_K = float(jnp.angle(z))
    err = abs(wrap(gam_K - float(berry_phase_latitude(qz))))
    print(f"[1] K={K:4d}: γ_disc={gam_K:+.6f}  γ_Stokes={float(berry_phase_latitude(qz)):+.6f}  |Δ|={err:.2e}")
assert err < 1e-4
rs = jax.vmap(bloch_from_state)(psis)
Om = float(solid_angle_polygon(rs))
assert abs(Om - 2 * np.pi * (1 - qz)) < 1e-4
print(f"    Ω polígono={Om:.6f} = 2π(1−qg_Z)={2*np.pi*(1-qz):.6f}  (qg_Z={qz:.4f}, α={alpha:.0f} qga)")

# 2. Triángulo geodésico aleatorio: fase de Bargmann = −Ω/2 (invariante de gauge)
rng = np.random.default_rng(3)
for _ in range(5):
    a = rng.uniform(0, 500, 3); p = rng.uniform(0, 1000, 3)
    ps = [state_from_qga(a[i], p[i]) for i in range(3)]
    ps_g = [ps[0] * np.exp(1j * 0.7), ps[1] * np.exp(-1j * 2.1), ps[2]]   # gauge arbitrario
    rr = [bloch_from_state(q) for q in ps]
    g1, g2 = float(bargmann_phase(*ps)), float(bargmann_phase(*ps_g))
    Om = float(solid_angle_triangle(*rr))
    assert abs(wrap(g1 - g2)) < 1e-12 and abs(wrap(g1 + Om / 2)) < 1e-12, (g1, g2, Om)
print("[2] arg(<ψ2|ψ1><ψ3|ψ2><ψ1|ψ3>) = −Ω/2 en 5/5 triángulos, invariante de gauge a 1e-12")

# 3. Stokes numérico: ∬ ½ sinθ dθ dφ sobre el casquete = ½·2π(1−cosθ) = −γ
th = alpha * RAD_PER_QGA
tt = jnp.linspace(0.0, th, 20001)
F = float(2 * np.pi * jnp.trapezoid(0.5 * jnp.sin(tt), tt))
assert abs(F + float(berry_phase_latitude(qz))) < 1e-9
print(f"[3] ∬F = {F:.9f} = −γ ✔ (monopolo de carga ½)")

# 4. Área de pulso para una fase Z de Ω rad: virtual / física / geométrica (lazo polo-(θ,0)-(θ,Δφ))
dphi = 0.6
Om = dphi * (1 - np.cos(th))
d = np.arccos(np.cos(th) ** 2 + np.sin(th) ** 2 * np.cos(dphi))
area_geo = (2 * th + d) / RAD_PER_QGA
print(f"[4] R_Z(Ω={Om:.4f} rad = {Om/RAD_PER_QGA:.1f} qga): virtual 0 qga | física {Om/RAD_PER_QGA:.1f} qga (1 pulso) | "
      f"geométrica {area_geo:.1f} qga (3 pulsos) → sin ganancia de área (no gain); robustez: unverified")
assert area_geo > Om / RAD_PER_QGA

# 5. Disparos para σ_γ = 0.01 rad desde qg_Z (1 base, 0 ancillas) en lazo de latitud
for a_ in (50.0, 150.0, 250.0):
    q_ = float(np.cos(a_ * RAD_PER_QGA))
    print(f"[5] α={a_:3.0f} qga (qg_Z={q_:+.3f}): N = {shots_for_berry_latitude(q_, 0.01):.0f}  "
          f"(tomografía 3 estados×3 bases: ≥9 configuraciones vs 1)")
print("verify_cap10: OK")
