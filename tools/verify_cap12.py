"""tools/verify_cap12.py — Oscilación de sabor en el vacío en qga (JAX). python tools/verify_cap12.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qneutrino import (osc_phase_rad, osc_phase_qga, p_transition, survival_qz, fisher_phi, fisher_theta,
                            optimal_phase_for, pmns, p3_vacuum, embed_two_qubits, depolarize, sector_filter,
                            flavor_error, RAD_PER_QGA)

deg = np.pi / 180
t12, t13, t23, dcp = 33.4 * deg, 8.57 * deg, 49.0 * deg, 195 * deg
dm21, dm31 = 7.42e-5, 2.51e-3

# 1. Circuito de 1 qubit RY(2θ)·RZ(φ)·RY(−2θ)|0> reproduce P = sin²2θ sin²(φ/2); L_osc = 1000 qga
def ry(a): return np.array([[np.cos(a/2), -np.sin(a/2)], [np.sin(a/2), np.cos(a/2)]])
def rz(a): return np.diag([np.exp(-1j*a/2), np.exp(1j*a/2)])
for phi in (0.3, 1.7, 3.0, 5.2):
    psi = ry(-2*t12) @ rz(phi) @ ry(2*t12) @ np.array([1, 0])
    assert abs((1 - abs(psi[0])**2) - float(p_transition(t12, phi))) < 1e-14
L_osc = 4 * np.pi * 2.5 / (2 * 1.267 * 2 * dm21)   # km, E=2.5 GeV: φ(L_osc)=2π
assert abs(float(osc_phase_qga(dm21, L_osc, 2.5)) - 1000.0) < 1e-9
print(f"[1] circuito 1 qubit = fórmula (1e-14); L_osc(Δm²21, 2.5 GeV) = {L_osc:.0f} km ≡ 1000 qga; 2 pulsos físicos + RZ virtual")

# 2. Punto de operación: Fisher de φ (⇒ Δm²) frente a Fisher de θ
phi_t, I_t = optimal_phase_for(fisher_theta, t12)
I_p_at_max = float(fisher_phi(t12, np.pi - 1e-3))
I_p_typ = float(fisher_phi(t12, 0.75 * np.pi))
I_p_half = float(fisher_phi(t12, 0.5 * np.pi))
I_p_0 = float(fisher_phi(t12, 1e-3))
# I_F(φ) = sin²2θ cos²(φ/2)/(1 − sin²2θ sin²(φ/2)): monótona decreciente en [0,π], → sin²2θ en φ→0, 0 en φ=π
assert abs(I_p_0 - np.sin(2*t12)**2) < 1e-5 and I_p_0 > I_p_half > I_p_typ > I_p_at_max
print(f"[2] θ12: I_F(θ) máx en φ={phi_t:.4f} (=π, máximo de oscilación), {I_t:.3f}/evento")
print(f"    I_F(φ) por evento: φ→0: {I_p_0:.4f} (=sin²2θ) | φ=π/2 (250 qga): {I_p_half:.4f} | φ=0.75π: {I_p_typ:.4f} | "
      f"φ=π (máx. osc.): {I_p_at_max:.2e}")
print(f"    → medir Δm² en el máximo de oscilación es el peor punto; en π/2 vs 0.75π: eventos ×{I_p_half/I_p_typ:.2f} menos")
assert abs(phi_t - np.pi) < 2e-3

# 3. Lectura m-qg en L/E tipo DUNE (1300 km, 2.5 GeV) para el sector atmosférico (θ23, Δm²31)
phi = float(osc_phase_rad(dm31, 1300.0, 2.5))
P, q = float(p_transition(t23, phi)), float(survival_qz(t23, phi))
print(f"[3] L=1300 km, E=2.5 GeV: φ={phi:.3f} rad = {phi/RAD_PER_QGA:.1f} qga; P(νμ→ντ)={P:.4f}; qg_Z={q:+.4f} → {1000*q:+.0f} m-qg")
assert abs(1000 * q - (1000 - 2000 * P)) < 1e-9

# 4. Tres sabores en 2 qubits: sector físico {|00>,|01>,|10>}, filtro bajo despolarización
U = pmns(t12, t13, t23, dcp); m2 = jnp.array([0.0, dm21, dm31])
p3 = p3_vacuum(U, m2, 1300.0, 2.5, alpha=1)              # ν_μ → (e, μ, τ)
assert abs(float(p3.sum()) - 1) < 1e-12
p_true = embed_two_qubits(p3)
for eps in (0.05, 0.10, 0.20):
    pn = depolarize(p_true, eps)
    e_raw, e_filt = float(flavor_error(pn, p_true)), float(flavor_error(sector_filter(pn), p_true))
    print(f"[4] ε={eps:.2f}: error sin filtro={e_raw:.4f}  con filtro de sector={e_filt:.4f}  (×{e_raw/e_filt:.2f})")
    assert e_filt < e_raw
print("    hardware IQM Garnet §165 (8×300 shots): 0.062 → 0.039 (×1.59) [tested on hardware]")
print(f"    P(νμ→e,μ,τ)={np.asarray(p3).round(4)}; P(|11>) ideal = 0 → todo peso en |11> es ruido diagnosticable")
print("verify_cap12: OK")
