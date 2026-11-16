"""tools/verify_cap33.py — Cap. 33: qang frente a una CFT de 1+1 dimensiones (bosón libre, c=1)."""
import numpy as np
from qang.qcft import *

# [1] identidades: fermiones libres = estado completo de la cadena XX (n=8)
n = 8
C = correlation_matrix(n)
psi = exact_state_xx(n)
for (i, j) in ((1, 2), (2, 5), (0, 7)):
    a, b = connected_qgzz(C, i, j), spin_zz_connected(psi, n, i, j)
    assert abs(a - b) < 1e-12, (i, j, a, b)
for l in (2, 4):
    assert abs(block_entropy_bits(C, l) - spin_block_entropy_bits(psi, n, l)) < 1e-10
print(f"[1] n=8: ⟨ZZ⟩_c = −4C_ij² y S(l) por correlaciones = estado completo a 1e-12/1e-10; "
      f"qg_ZZ conexa (1,2) = {1000*connected_qgzz(C,1,2):+.2f} m-qg, S(4) = {block_entropy_bits(C,4):.4f} bits")

# [2] carga central
res = {}
for n in (32, 64, 128, 256, 512):
    c, s0 = fit_central_charge(n)
    res[n] = c
    print(f"[2] n={n:3d}: c ajustada = {c:.4f}  (CFT: 1)")
err = [abs(res[k] - 1) for k in sorted(res)]
assert all(a > b for a, b in zip(err, err[1:])), err                 # converge
assert abs(res[512] - 1) < 0.01 and abs(err[-1] / err[-2] - 0.5) < 0.1  # error ∝ 1/n
print(f"    error |c−1| = {[round(e, 4) for e in err]}: se divide por ~2 al duplicar n (∝ 1/n)")

# [3] ley de potencias de la qg_ZZ conexa
ps = {nn: zz_power_law_exponent(nn, 3, 31) for nn in (400, 800, 1600)}
p = ps[1600]
print("[3] pendiente log-log de |qg_ZZ conexa| frente a r impar (3..31): " + ", ".join(f"n={k}: {v:.3f}" for k, v in ps.items()) + "  (CFT de dimensión 1: −2)")
assert abs(p + 2) < 0.03 and abs(ps[400] + 2) > abs(ps[800] + 2) > abs(p + 2)
Cb = correlation_matrix(1600)
v5 = connected_qgzz(Cb, 784, 789)
print(f"    centro, r=5: qg_ZZ conexa = {1000*v5:+.2f} m-qg frente a −4000/(π·5)² = {-4000/(np.pi*5)**2:+.2f} m-qg")
assert abs(v5 + 4 / (np.pi * 5) ** 2) < 3e-4

# [4] lo que las lecturas locales NO ven
n = 128
C = correlation_matrix(n)
qz = local_qgz(C)
bulk = slice(32, 96)
print(f"[4] n=128: |qg_Z| local en el bulk ≤ {1000*np.max(np.abs(qz[bulk])):.1e} m-qg; qg_S local = "
      f"{min(local_qgs_bits(C,i) for i in range(32,96)):.5f}…{max(local_qgs_bits(C,i) for i in range(32,96)):.5f} bits")
assert np.max(np.abs(qz[bulk])) < 0.02 and min(local_qgs_bits(C, i) for i in range(32, 96)) > 0.9998
Sl = [block_entropy_bits(C, l) for l in (2, 4, 8, 16, 32, 64)]
print("    entropía de bloque S(l), l=2..64 (bits):", np.round(Sl, 3), "— crece como (c/6) log2 l; las lecturas locales son planas")
assert 0.7 < Sl[-1] - Sl[0] < 0.9   # (1/6) log2 de la razón de cuerdas ≈ 0.73–0.8
print("verify_cap33: OK")
