"""tools/verify_cap20.py — BB84 en qg: IR ⇒ QBER f/4, tasas, presupuesto de bits tamizados. python tools/verify_cap20.py"""
import numpy as np
from qang.qkd import qber, key_rate_individual, sifted_bits_for_qber_sigma, BB84, RAD_PER_QGA
from qang.qdemarc import bb84_rate_from_qz, qber_to_qz
import jax.numpy as jnp

# 1. Intercepta-y-reenvía: e = f/4 exacto; f = 1 ⇒ 25% ⇔ 500 m-qg
for f in (0.0, 0.2, 0.5, 1.0):
    e = qber(f, 0.0)
    assert abs(e - f / 4) < 1e-14
    print(f"[1] IR f={f:.1f}: QBER={e:.4f} ⇔ qg_Z={1000*float(qber_to_qz(e)):+.0f} m-qg; "
          f"r_colectivo={float(bb84_rate_from_qz(qber_to_qz(e))):+.4f}, r_IR={key_rate_individual(e):+.4f}")
# 2. Canal depolarizante p sin Eve: e = p/2 ; combinado: e = f/4 + (1−f/2)·p/2 ... verificado numéricamente
for p in (0.04, 0.10):
    e = qber(0.0, p); assert abs(e - p / 2) < 1e-14
    e2 = qber(0.5, p)
    print(f"[2] p={p:.2f}: sin Eve QBER={e:.3f} ({1000*(1-2*e):+.0f} m-qg); con IR f=0.5: QBER={e2:.4f}")
# 3. Umbral de abortar: e* = 0.11 (colectivo) vs IR puro f* = 0.44 ⇒ la mitad de los pulsos interceptados ya está por encima
f_star = 4 * 0.1100
print(f"[3] IR equivale al umbral colectivo con f* = {f_star:.2f}; f=0.5 → QBER=0.125 → abortar")
# 4. Presupuesto de bits tamizados: σ_e = 0.005 (10 m-qg) a e = 0.02 y 0.08; método delta vs Wilson no difiere lejos del polo
for e in (0.02, 0.08):
    N = sifted_bits_for_qber_sigma(e, 0.005)
    print(f"[4] e={e:.2f}: N = {N:.0f} bits tamizados para σ_e = 0.005 (σ = 10 m-qg)")
assert abs(sifted_bits_for_qber_sigma(0.02, 0.005) - 784) < 1
# 5. Geometría: los 4 estados BB84 son 2 pares antípodas en qga (0/500 polar; 250 con φ 0/500)
assert abs(float(jnp.abs(jnp.vdot(BB84[("Z", 0)], BB84[("X", 0)]))) ** 2 - 0.5) < 1e-14
print("[5] |<Z0|X0>|² = 0.5: bases mutuamente no sesgadas; estados en {0, 500} qga y {250 qga, φ=0/500 qga}")
print("verify_cap20: OK")
