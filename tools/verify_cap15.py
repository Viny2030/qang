"""tools/verify_cap15.py — Teleportación con par de Werner: canal r→Vr, umbral 333 m-qg. python tools/verify_cap15.py"""
import numpy as np
import jax, jax.numpy as jnp
from qang.qtele import (werner, resource_correlators, singlet_fraction_from_qg, teleport_fidelity_from_singlet,
                        teleport_fidelity_werner_qzz, teleport, bloch, state_from_bloch, shots_for_fidelity)

# 1. Correladores del recurso: qg_XX = V, qg_YY = −V, qg_ZZ = V; F_s = (1+3V)/4; F̄ = (1+V)/2
for V in (1.0, 0.8, 1 / 3, 0.1):
    qxx, qyy, qzz = resource_correlators(werner(V))
    assert abs(float(qxx) - V) < 1e-14 and abs(float(qyy) + V) < 1e-14 and abs(float(qzz) - V) < 1e-14
    Fs = float(singlet_fraction_from_qg(qxx, qyy, qzz))
    assert abs(Fs - (1 + 3 * V) / 4) < 1e-14
    assert abs(float(teleport_fidelity_from_singlet(Fs)) - (1 + V) / 2) < 1e-14
print("[1] Werner: qg_ZZ = V, F_s = (1+3V)/4, F̄ = (2F_s+1)/3 = (1+V)/2  (1e-14)")

# 2. Teleportación explícita: Bloch de salida = V · Bloch de entrada (canal depolarizante), m-qg' = V·m-qg
rng = np.random.default_rng(0)
for V in (1.0, 0.7, 0.4):
    for _ in range(3):
        r = rng.normal(size=3); r = r / np.linalg.norm(r) * rng.uniform(0.3, 1.0)
        out = teleport(state_from_bloch(jnp.asarray(r)), werner(V))
        assert np.allclose(np.asarray(bloch(out)), V * r, atol=1e-13)
r = jnp.array([0.0, 0.0, 0.6])
for V in (1.0, 0.7, 1 / 3):
    m_out = 1000 * float(bloch(teleport(state_from_bloch(r), werner(V)))[2])
    print(f"[2] V={V:.3f}: entrada +600 m-qg → salida {m_out:+.0f} m-qg  (lineal en lectura: m' = V·m)")

# 3. Fidelidad media sobre el 2-diseño de 6 autoestados de Pauli = (1+V)/2; umbral clásico 2/3 ⇔ 333 m-qg
axes = [np.array(v, float) for v in ([1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1])]
for V in (0.9, 1 / 3, 0.2):
    Fs = []
    for a in axes:
        rin = state_from_bloch(jnp.asarray(a)); out = teleport(rin, werner(V))
        Fs.append(float(jnp.real(jnp.trace(rin @ out))))
    Fbar = float(np.mean(Fs))
    assert abs(Fbar - (1 + V) / 2) < 1e-13
    print(f"[3] V={V:.3f} (qg_ZZ={1000*V:.0f} m-qg): F̄={Fbar:.4f} → {'cuántico' if Fbar > 2/3 + 1e-12 else 'límite clásico (2/3)'}")
assert abs(float(teleport_fidelity_werner_qzz(1 / 3)) - 2 / 3) < 1e-15

# 4. Presupuesto: σ_F = 0.01 desde qg_ZZ (1 base) frente a tomografía de proceso (4 entradas × 3 bases = 12 config.)
for V in (0.9, 0.6):
    N = shots_for_fidelity(V, 0.01)
    print(f"[4] V={V}: N(σ_F=0.01) = {N:.0f} disparos en 1 configuración (Z⊗Z del recurso) | "
          f"tomografía de proceso: 12 configuraciones ≥ {12*N:.0f} disparos para la misma σ por celda")
# 5. Estado no Werner: F̄ por 3 bases (sin suposición) vs 1 base (suposición de Werner): sesgo
rho = 0.8 * jnp.outer(jnp.array([1,0,0,1])/np.sqrt(2), jnp.array([1,0,0,1])/np.sqrt(2)) + 0.2 * jnp.diag(jnp.array([0.5,0,0,0.5]))
qxx, qyy, qzz = resource_correlators(rho)
F3 = float(teleport_fidelity_from_singlet(singlet_fraction_from_qg(qxx, qyy, qzz)))
F1 = float(teleport_fidelity_werner_qzz(qzz))
print(f"[5] recurso con desfase (no Werner): qg_XX={float(qxx):.2f}, qg_ZZ={float(qzz):.2f}; F̄(3 bases)={F3:.4f}, "
      f"F̄(1 base, supuesto Werner)={F1:.4f} → sesgo +{F1-F3:.4f}: la lectura de 1 base es cota superior (mixed)")
assert F1 >= F3
print("verify_cap15: OK")
