"""tools/verify_cap25.py — Arrastre de lectura, inversión por eco (criterio §108) y filtro de sector O(e²). python tools/verify_cap25.py"""
import numpy as np
from qang.qecho import readout_affine, drag_free_point, invert_readout, mse_raw_vs_inverted, sector_readout_error_order

e01, e10 = 0.02, 0.06       # asimétrico típico (T1 durante la lectura ⇒ e10 > e01)
# 1. Mapa afín y punto sin arrastre
m_star = drag_free_point(e01, e10)
assert abs(readout_affine(m_star, e01, e10) - m_star) < 1e-9
for m in (-1000, -500, 0, 500, 1000):
    mo = readout_affine(m, e01, e10)
    print(f"[1] m={m:+5d} → m'={mo:+7.1f} m-qg (sesgo {mo-m:+6.1f}); inversión exacta sin ruido: {invert_readout(mo, e01, e10):+.1f}")
print(f"    punto sin arrastre m* = {m_star:+.1f} m-qg (qg_Z = {m_star/1000:+.3f}); compuerta X: 1−e01−e10 = {1-e01-e10:.2f} ⇒ "
      f"|1000| lee {readout_affine(-1000,e01,e10):+.0f} (cf. §168: coste X = 1.64 m-qg con e≈1e-3)")

# 2. Criterio §108: invertir sólo si sesgo² > varianza añadida (N disparos, N_cal calibración)
print("[2]   m    N     N_cal   sesgo   σ_m    MSE_raw   MSE_inv  decisión")
rows = [(600, 200, 2000), (600, 20000, 2000), (600, 20000, 200), (m_star, 20000, 2000), (-900, 1000, 1000)]
res = []
for m, N, Nc in rows:
    r, i, b, s = mse_raw_vs_inverted(m, e01, e10, N, Nc)
    dec = "INVERTIR" if i < r else "NO invertir"
    res.append(dec)
    print(f"    {m:+6.0f} {N:6d} {Nc:6d} {b:+7.1f} {s:6.1f} {r:9.0f} {i:9.0f}  {dec}")
assert res[0] == "NO invertir" and res[1] == "INVERTIR" and res[3] == "NO invertir"
# Monte Carlo del caso (600, 200, 2000)
rng = np.random.default_rng(0); m, N, Nc = 600, 200, 2000
mo_true = readout_affine(m, e01, e10); q = mo_true / 1000
mc_raw, mc_inv = [], []
for _ in range(4000):
    k0 = rng.binomial(N, (1 + q) / 2); mo = 1000 * (2 * k0 / N - 1)
    e01h = rng.binomial(Nc, e01) / Nc; e10h = rng.binomial(Nc, e10) / Nc
    mc_raw.append((mo - m) ** 2); mc_inv.append((invert_readout(mo, e01h, e10h) - m) ** 2)
r, i, _, _ = mse_raw_vs_inverted(m, e01, e10, N, Nc)
print(f"    MC (m=600,N=200,N_cal=2000): MSE_raw={np.mean(mc_raw):.0f} (teoría {r:.0f}), MSE_inv={np.mean(mc_inv):.0f} (teoría {i:.0f})")
assert abs(np.mean(mc_raw) / r - 1) < 0.15 and abs(np.mean(mc_inv) / i - 1) < 0.15

# 3. Filtro de sector frente a errores de lectura: O(e) → O(e01·e10)
p = np.array([0.5, 0.3, 0.15, 0.05])
for (a, b) in ((0.02, 0.06), (0.01, 0.01), (0.05, 0.10)):
    p_f, bias, bound = sector_readout_error_order(p, a, b)
    # sin filtro: sesgo de primer orden en p(e_k): −(e10 + (n−1)e01)·p_k + ...
    first = (b + 3 * a) * p.max()
    print(f"[3] e01={a}, e10={b}: sesgo máx. filtrado={bias:.5f} (≈ e01·e10 = {bound:.5f}) vs primer orden sin filtro ≈ {first:.4f} → ×{first/bias:.0f}")
    assert bias < 3 * bound
print("verify_cap25: OK")
