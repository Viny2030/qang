"""tools/verify_cap14.py — Majorana, 0νββ, estériles 3+1 y proyecciones angulares en qga. python tools/verify_cap14.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qneutrino import pmns, p3_vacuum, embed_two_qubits, depolarize, sector_filter
from qang.qmajorana import (pmns_majorana, m_betabeta, masses_from_lightest, p4_vacuum, u4_minimal,
                            sterile_sensitivity_floor, qz_spin_projection, spin_coherent_qfi_theta, RAD_PER_QGA)

deg = np.pi / 180
t12, t13, t23, dcp = 33.4 * deg, 8.57 * deg, 49.0 * deg, 195 * deg
dm21, dm31 = 7.42e-5, 2.51e-3
m2 = jnp.array([0.0, dm21, dm31])

# 1. Fases de Majorana: invisibles en oscilación (R_Z en base de masa que conmuta con la propagación)
U0 = pmns(t12, t13, t23, dcp)
for a21, a31 in ((0.0, 0.0), (1.0, 2.5), (np.pi, 0.3)):
    UM = pmns_majorana(t12, t13, t23, dcp, a21, a31)
    for L, E in ((1300.0, 2.5), (295.0, 0.6), (810.0, 2.0)):
        assert np.allclose(np.asarray(p3_vacuum(UM, m2, L, E, 1)), np.asarray(p3_vacuum(U0, m2, L, E, 1)), atol=1e-14)
print("[1] P(ν_α→ν_β) independiente de (α21, α31) a 1e-14 en 9 configuraciones: fase virtual que nunca llega a la lectura")

# 2. |m_ββ|: suma de 3 fasores; rango NO/IO barriendo α21, α31 ∈ [0, 2π) (aritmética clásica: no gain)
grid = np.linspace(0, 2 * np.pi, 73)
for ordering, ml in (("NO", 0.0), ("NO", 0.05), ("IO", 0.0)):
    m = masses_from_lightest(ml, dm21, dm31, ordering)
    vals = np.array([[float(m_betabeta(pmns_majorana(t12, t13, t23, dcp, a, b), m)) for b in grid] for a in grid])
    print(f"[2] {ordering} m_light={ml:.2f} eV: |m_ββ| ∈ [{1e3*vals.min():.1f}, {1e3*vals.max():.1f}] meV "
          f"(3 fasores: {np.asarray(jnp.abs(U0[0,:])**2*m).round(5)} eV)")
mNO = masses_from_lightest(0.0, dm21, dm31, "NO")
assert abs(float(m_betabeta(U0, mNO)) - float(jnp.abs(jnp.sum(U0[0, :] ** 2 * mNO)))) < 1e-15

# 3. 3+1: el sector |11> pasa de diagnóstico de ruido a señal estéril; umbral de sensibilidad con ε
U4 = u4_minimal(U0, t14=np.arcsin(np.sqrt(0.1)) / 1)   # sin²θ14 = 0.1
m2_4 = jnp.array([0.0, dm21, dm31, 1.0])                 # Δm²41 = 1 eV²
L, E = 0.5, 0.003                                        # 500 m, 3 MeV (reactor SBL)
p4 = p4_vacuum(U4, m2_4, L, E, alpha=0)
assert abs(float(p4.sum()) - 1) < 1e-12
P_s = float(p4[3])
phi41 = 2 * 1.267 * 1.0 * L / E
print(f"[3] SBL L=500 m, E=3 MeV, Δm²41=1 eV², sin²θ14=0.1: φ41={phi41:.3f} rad={phi41/RAD_PER_QGA:.0f} qga; "
      f"P(ν_e→ν_s)=P(|11>)={P_s:.4f} → {1000*(1-2*P_s):.0f} m-qg en el eje (físico, s)")
for eps in (0.01, 0.05, 0.10):
    floor = sterile_sensitivity_floor(eps)
    print(f"    ε={eps:.2f}: fondo espurio |11> = {eps/4:.4f}; umbral 3σ sin calibrar = {floor:.4f} → "
          f"{'detectable' if P_s > floor else 'NO detectable'} (filtro de sector PROHIBIDO en 3+1)")
assert P_s > sterile_sensitivity_floor(0.01)

# 4. Proyecciones angulares: qg_Z = m/j (paso 1000/j m-qg); QFI de estado coherente de espín = 2j
for j in (0.5, 1.0, 1.5, 2.0):
    ms = np.arange(-j, j + 1)
    qs = np.asarray(jax.vmap(lambda m: qz_spin_projection(m, j))(jnp.asarray(ms)))
    assert qs[0] == -1 and qs[-1] == 1 and abs(np.diff(qs)[0] - 1 / j) < 1e-15
    print(f"[4] j={j}: qg_Z ∈ {np.round(1000*qs).astype(int)} m-qg; QFI(θ)={float(spin_coherent_qfi_theta(j)):.1f} rad⁻² "
          f"(={float(spin_coherent_qfi_theta(j))*RAD_PER_QGA**2:.3e} qga⁻²)")
assert float(spin_coherent_qfi_theta(0.5)) == 1.0
print("    identidad exacta; ventaja de qudits en hardware: unverified (THEORY_CORE §2.7)")
print("verify_cap14: OK")
