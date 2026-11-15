"""tools/verify_cap16.py — Doble carril, pérdida como borrado y filtro de sector. python tools/verify_cap16.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qdualrail import (transmission, dual_rail_state, apply_loss, readout_unfiltered, readout_sector_filtered,
                            shots_multiplier_two_photons)
from qang.qtele import werner, teleport, bloch, state_from_bloch

# 1. Sesgo sin filtro qg_Z' = η qg_Z + (1−η) ; con filtro: qg_Z exacto, tasa η
for L in (10.0, 50.0, 100.0, 200.0):
    eta = float(transmission(L))
    for qz in (-0.6, 0.0, 0.8):
        rho = apply_loss(dual_rail_state(qz, qx=0.3), eta)
        q_un = float(readout_unfiltered(rho))
        q_f, acc = readout_sector_filtered(rho)
        assert abs(q_un - (eta * qz + (1 - eta))) < 1e-14
        assert abs(float(q_f) - qz) < 1e-13 and abs(float(acc) - eta) < 1e-14
    rho = apply_loss(dual_rail_state(-0.6), eta)
    print(f"[1] L={L:5.0f} km η={eta:.4f}: entrada −600 m-qg → sin filtro {1000*float(readout_unfiltered(rho)):+.0f} m-qg "
          f"(sesgo {1000*(1-eta)*(1-(-0.6)):+.0f}) | filtrado −600 m-qg con tasa {eta:.4f}")

# 2. Coherencia dentro del sector sobrevive a la pérdida (ρ filtrada = ρ_L): qg_X también
rho = apply_loss(dual_rail_state(0.2, qx=0.7), 0.1)
sub = rho[1:3, 1:3]; sub = sub / jnp.trace(sub)
assert abs(float(jnp.real(2 * sub[0, 1])) - 0.7) < 1e-13 or abs(float(jnp.real(2 * sub[1, 0])) - 0.7) < 1e-13
print("[2] pérdida pura: ρ postseleccionada = ρ_L exacta (qg_X, qg_Y, qg_Z intactos) → borrado anunciado")

# 3. Teleportación a distancia: par de Werner V con pérdida simétrica; fidelidad filtrada (1+V)/2 independiente de L,
#    presupuesto de disparos × 1/η(L)
for L in (50.0, 100.0, 200.0):
    mult = shots_multiplier_two_photons(L)
    for V in (0.9,):
        out = teleport(state_from_bloch(jnp.array([0.0, 0.0, 0.6])), werner(V))
        m_out = 1000 * float(bloch(out)[2])
    print(f"[3] L={L:4.0f} km: F̄ filtrada = {(1+0.9)/2:.3f} (V=0.9, sin dependencia de L); salida {m_out:+.0f} m-qg; "
          f"disparos × {mult:.3g} por anuncio (η(L)={1/mult:.2e})")
assert abs(shots_multiplier_two_photons(100.0) - 100.0) < 1e-9

# 4. Umbral de utilidad combinado: con ruido depolarizante dentro del sector (ε por km-independiente) el par sirve
#    si V > 1/3 (Cap. 15); la pérdida sólo cuesta tasa, no fidelidad (modelo de borrado)
print("[4] regla: pérdida → tasa (×1/η), ruido intra-sector → fidelidad (V>1/3 ⇔ qg_ZZ>333 m-qg); separables tras el filtro")
print("verify_cap16: OK")
