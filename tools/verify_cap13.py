"""tools/verify_cap13.py — Efectos de materia (MSW) en qga: resonancia, lectura adiabática, Trotter. python tools/verify_cap13.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qneutrino import (matter_potential_A, sin2_2theta_m, cos2theta_m, resonance_A, adiabatic_survival_qz,
                            trotter_survival_qz, RAD_PER_QGA, KAPPA)

deg = np.pi / 180
t12, dm21 = 33.4 * deg, 7.42e-5
NA = 6.02214076e23

# 1. Resonancia: A_res = Δm² cos2θ ⇒ θ_m = π/4 ⇒ eje efectivo en el ecuador (qg_Z = 0, 250 qga)
A_res = float(resonance_A(t12, dm21))
assert abs(float(cos2theta_m(t12, dm21, A_res))) < 1e-14 and abs(float(sin2_2theta_m(t12, dm21, A_res)) - 1) < 1e-14
E_res_core = A_res / (1.52e-4 * 100.0)   # GeV, para N_e = 100 N_A cm⁻³
print(f"[1] A_res = {A_res:.3e} eV² → cos2θ_m = 0 (250 qga); E_res(N_e=100 N_A) = {1000*E_res_core:.2f} MeV; "
      f"2θ_m en qga: vacío {2*t12/RAD_PER_QGA:.1f} → resonancia 250.0")

# 2. Lectura adiabática solar: qg_Z^{ee} = cos2θ_m(A_prod)·cos2θ  (producto de dos qg); pp (0.4 MeV) y 8B (10 MeV)
for name, E in (("pp 0.4 MeV", 4e-4), ("7Be 0.86 MeV", 8.6e-4), ("8B 10 MeV", 1e-2)):
    A = float(matter_potential_A(100.0 * NA, E))
    q = float(adiabatic_survival_qz(t12, dm21, A))
    print(f"[2] {name:13s}: A/A_res={A/A_res:5.2f}  cos2θ_m={float(cos2theta_m(t12, dm21, A)):+.3f}  "
          f"P_ee={(1+q)/2:.3f}  qg_Z={q:+.3f} → {1000*q:+.0f} m-qg")
q_pp = float(adiabatic_survival_qz(t12, dm21, float(matter_potential_A(100.0 * NA, 4e-4))))
q_B = float(adiabatic_survival_qz(t12, dm21, float(matter_potential_A(100.0 * NA, 1e-2))))
assert abs((1 + q_pp) / 2 - (1 - 0.5 * np.sin(2 * t12) ** 2)) < 0.02 and abs((1 + q_B) / 2 - np.sin(t12) ** 2) < 0.03

# 3. Trotter por tramos vs adiabático: perfil exponencial N_e = 100 N_A e^{−r/r0}, r0 = R_sun/10.54, salida en R_sun
E = 1e-2; R_sun, r0 = 6.957e5, 6.957e5 / 10.54
A_prod = float(matter_potential_A(100.0 * NA, E))
# poblaciones de masa a la salida: |<ν1|ψ>|² debe ser cos²θ_m(prod) si la evolución es adiabática
def mass1_population(N):
    r = (jnp.arange(N) + 0.5) * (R_sun / N)
    A = A_prod * jnp.exp(-r / r0)
    q = trotter_survival_qz(t12, dm21, E, A, R_sun / N)         # qg_Z en la salida (base de sabor)
    return q
# Comparación directa de P_ee medida a la salida promediada sobre una banda de energía (lava la fase de vacío)
def pee_trotter_avg(N, nE=41):
    Es = jnp.linspace(0.9 * E, 1.1 * E, nE)
    def one(Ei):
        r = (jnp.arange(N) + 0.5) * (R_sun / N)
        A = float(matter_potential_A(100.0 * NA, 1.0)) * Ei * jnp.exp(-r / r0)
        return (1 + trotter_survival_qz(t12, dm21, Ei, A, R_sun / N)) / 2
    return float(jnp.mean(jax.vmap(one)(Es)))
P_ad = (1 + q_B) / 2
for N in (50, 200, 1000, 4000):
    P_tr = pee_trotter_avg(N)
    print(f"[3] Trotter N={N:5d} tramos (≈{N} rotaciones de eje arbitrario = {2*N} pulsos RY + RZ virtuales): "
          f"<P_ee>={P_tr:.4f} | adiabático (0 tramos, 2 qg): {P_ad:.4f} | Δ={abs(P_tr-P_ad):.4f}")
assert abs(pee_trotter_avg(4000) - P_ad) < 0.01
# parámetro de adiabaticidad γ = Δm² sin²2θ / (2E cos2θ |d ln N_e/dr|) en la resonancia (unidades: κ)
gamma = 2 * KAPPA * dm21 * np.sin(2 * t12) ** 2 / (E * np.cos(2 * t12)) * r0
print(f"    γ_adiab(res) = {gamma:.0f} ≫ 1 → P_LZ ≈ e^(−πγ/2) ≈ 0; fórmula de 2 qg válida (exact bajo adiabaticidad)")
print("verify_cap13: OK")
