"""tools/verify_cap19.py — Demarcación QKD/QRNG (qubits) vs PQC (sin qubits). python tools/verify_cap19.py"""
import numpy as np
import jax.numpy as jnp
from qang.qdemarc import (qber_to_qz, qz_to_qber, bb84_rate_from_qz, bb84_threshold_qz, qrng_min_entropy_from_qz,
                          shots_to_resolve_bias, h2)

# 1. Umbral BB84: e* = 0.1100 ⇔ qg_Z* = 0.780 (780 m-qg)
qz_star = bb84_threshold_qz()
e_star = float(qz_to_qber(qz_star))
assert abs(e_star - 0.1100) < 2e-4 and abs(1 - 2 * float(h2(e_star))) < 1e-9
print(f"[1] BB84: r = 1 − 2h(e) = 0 en e* = {e_star:.4f}  ⇔  qg_Z* = {qz_star:.4f} ({1000*qz_star:.0f} m-qg)")
for e in (0.02, 0.05, 0.08, 0.11, 0.15):
    qz = float(qber_to_qz(e)); r = float(bb84_rate_from_qz(qz))
    print(f"    QBER={e:.2f} → {1000*qz:+.0f} m-qg → r = {r:+.4f} bit/bit {'(clave)' if r > 0 else '(abortar)'}")

# 2. QRNG: H_min desde |qg_Z| leído en m-qg; presupuesto para certificar un sesgo δ
for mqg in (0.0, 20.0, 100.0, 500.0):
    H = float(qrng_min_entropy_from_qz(mqg / 1000))
    print(f"[2] |qg_Z| = {mqg:4.0f} m-qg → p_max = {(1+mqg/1000)/2:.4f} → H_min = {H:.4f} bit")
assert abs(float(qrng_min_entropy_from_qz(0.0)) - 1.0) < 1e-15
N = shots_to_resolve_bias(0.01)   # δ = 0.01 en p0 ⇔ 20 m-qg
print(f"    certificar δ=0.01 (20 m-qg) a 3σ: N = {N:.0f} disparos")

# 3. PQC: parámetro de seguridad δ ≤ 2^−138.8 (ML-KEM); 'disparos' para resolverlo: (3/(2δ))² = 2^{279.6}
delta = 2.0 ** (-138.8)
N_pqc = shots_to_resolve_bias(delta)
print(f"[3] PQC: δ = 2^-138.8 → N = 2^{np.log2(N_pqc):.1f} 'disparos' — incontable; además no hay qubit: qang no aplica (against)")
assert np.log2(N_pqc) > 270
print("verify_cap19: OK")
