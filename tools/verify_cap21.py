"""tools/verify_cap21.py — QRNG certificado y auditoría de canal en m-qg. python tools/verify_cap21.py"""
import numpy as np
from qang.qrng import certified_min_entropy, point_min_entropy, shots_for_certified_hmin, block_drift_chi2, posterior_qz_quantiles

# 1. Punto vs certificado: con k0 = N/2 el punto da 1 bit; el certificado depende de N
for N in (1_000, 10_000, 100_000, 1_000_000):
    k0 = N // 2
    lo, hi = posterior_qz_quantiles(k0, N)
    print(f"[1] N={N:>9d}, k0=N/2: H_min punto = {point_min_entropy(k0, N):.4f} | "
          f"intervalo 99.9% qg_Z = [{1000*lo:+.1f}, {1000*hi:+.1f}] m-qg | H_min cert = {certified_min_entropy(k0, N):.4f} bit")
assert certified_min_entropy(500_000, 1_000_000) > 0.99 and certified_min_entropy(500, 1000) < 0.9
# 2. Presupuesto para certificar H_min ≥ 0.99 (|qg_Z| ≤ 13.9 m-qg al 99.9%)
N99 = shots_for_certified_hmin(0.99)
print(f"[2] H_min ≥ 0.99 certificado: |qg_Z| ≤ {1000*(2*2**-0.99-1):.1f} m-qg ⇒ N ≈ {N99:.0f} disparos")
# 3. Sesgo real de 30 m-qg: el punto sobreestima; el certificado acota
rng = np.random.default_rng(0); N = 200_000; p0 = 0.515
bits = (rng.random(N) < p0).astype(int); k0 = bits.sum()
print(f"[3] sesgo real 30 m-qg, N=2e5: H_min punto={point_min_entropy(k0, N):.4f}, cert={certified_min_entropy(k0, N):.4f}, "
      f"verdadero={-np.log2(p0):.4f}")
assert certified_min_entropy(k0, N) <= -np.log2(p0) + 1e-9
# 4. Auditoría por bloques: canal estacionario vs deriva lineal de 0 → 80 m-qg
stat = (rng.random(N) < 0.5).astype(int)
drift_p = 0.5 + np.linspace(0, 0.04, N)
drift = (rng.random(N) < drift_p).astype(int)
c1, df, m1 = block_drift_chi2(stat, 10_000); c2, _, m2 = block_drift_chi2(drift, 10_000)
print(f"[4] χ²({df} g.d.l.): estacionario {c1:.1f} | deriva {c2:.1f} (bloques m-qg: {np.round(m2[::4]).astype(int)})")
assert c2 > 3 * c1 and c1 < 2 * df + 20
print("verify_cap21: OK")
