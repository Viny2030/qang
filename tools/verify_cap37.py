"""tools/verify_cap37.py — Cap. 37: nanocuántica leída con qang (qang.qnano)."""
import numpy as np
from qang.qnano import *

# [1] túnel
rows = [(a, barrier_T(a)) for a in (0.2, 0.5, 1.0, 1.5)]
for a, T in rows:
    assert abs(np.cos(theta_from_p0(T) / 2) - np.sqrt(T)) < 1e-15
kap = kappa_per_nm()
sT, sA = tunnel_slopes(np.linspace(0.8, 2.0, 13))
assert abs(sT + 2 * kap) < 0.01 and abs(sA + kap) < 0.01
print(f"[1] κ = {kap:.3f} nm⁻¹; " + "; ".join(f"{a} nm: T={T:.3g}, {mqg_from_p0(T):+.1f} m-qg, θ={np.degrees(theta_from_p0(T)):.1f}°" for a, T in rows))
print(f"    cos(θ_T/2) = √T = |t| a 1e-15; pendiente de ln T: {sT:.3f} (−2κ = {-2*kap:.3f}); de ln(π−θ_T): {sA:.3f} (−κ = {-kap:.3f}) nm⁻¹")

# [2] Landauer
err = max(abs(conductance_from_qg(qpc_T(v)) - qpc_T(v).sum()) for v in np.linspace(0, 6, 300))
assert err < 1e-14
pl = [conductance_from_qg(qpc_T(v)) for v in (2, 3, 4)]
assert np.allclose(pl, [1, 2, 3], atol=1e-6)
T2 = (0.4031, 0.9716)
G, S = sum(T2), sum(t * (1 - t) for t in T2)
r = recover_two_channels(G, S)
assert abs(r[0] - T2[1]) < 1e-12 and abs(r[1] - T2[0]) < 1e-12
print(f"[2] G/G0 = N/2 + Σqg/2 a {err:.1e}; mesetas en V=2,3,4: {[round(x, 6) for x in pl]}; "
      f"dos canales (G={G:.4f}, S={S:.4f}) → T = {r[0]:.4f}, {r[1]:.4f}")

# [3] ruido de partición (binomial exacto, N=1000)
Ts = (0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98)
th = [theta_std_sqrtN(t, 1000) for t in Ts]
pp = [p_std_sqrtN(t) for t in Ts]
assert all(1.0 <= x < 1.011 for x in th) and max(pp) - min(pp) > 0.3
print(f"[3] √N·σ(θ̂) = {[round(x, 4) for x in th]} para T={list(Ts)}; √N·σ(T̂) = {[round(x, 3) for x in pp]}")

# [4] ocupación de un nivel
x = np.linspace(-8, 8, 161)
f = 1 / (1 + np.exp(x))
assert np.max(np.abs((2 * f - 1) - occupancy_qgz(x))) < 1e-15
print("[4] qg_Z = 2f − 1 = −tanh(x/2) a 1e-15")

# [5] termometría: 25 niveles, 200 lecturas, 400 repeticiones, k_BT=1
res = [thermometry(seed=s) for s in (34, 1, 2, 3)]
for (ml, ar) in res:
    assert abs(ml[0] - 1) < 0.005 and ar[0] < 0.99 and ar[1] > 1.2 * ml[1]
mls = [r[0][0] for r in res]; ars = [r[1][0] for r in res]
(ml, ar) = res[0]
print(f"[5] k_BT: máxima verosimilitud {ml[0]:.4f} ± {ml[1]:.4f}; recta en artanh {ar[0]:.4f} ± {ar[1]:.4f} (4 semillas: ML {min(mls):.4f}–{max(mls):.4f}, artanh {min(ars):.4f}–{max(ars):.4f})")

# [6] lectura de un espín (e0=0.02, e1=0.08)
rd = readout_affine([1000.0, -1000.0], 0.02, 0.08)
assert abs(rd[0] - 960) < 1e-9 and abs(rd[1] + 840) < 1e-9
print(f"[6] +1000 → {rd[0]:.0f}, −1000 → {rd[1]:.0f} m-qg: amplitud {(rd[0]-rd[1])/2:.0f} m-qg, desplazamiento {(rd[0]+rd[1])/2:+.0f} m-qg; ruido ×{1/(1-0.02-0.08):.2f} al corregir")

# [7] centro NV y ODMR
s_ap = 2000 / (0.3 * np.sqrt(444444))
s_ex = nv_sigma_mqg(0.3, 444444)
n10 = nv_photons_for(10, 0.3)
assert abs(s_ap - 10.0) < 1e-3 and abs(s_ex - 9.2195) < 1e-3 and abs(nv_sigma_mqg(0.3, n10) - 10.0) < 1e-9
assert abs(nv_sigma_mqg(1.0, 1e6) * 1.0 - 2000 / 1e3 * np.sqrt(0.5)) < 1e-9
crb = odmr_fisher_crb_hz(1e6, 0.2, 1.16e6 / 61)
print(f"[7] NV C=0.3, Nb=444 444: σ aproximada {s_ap:.2f} m-qg, exacta {s_ex:.2f} m-qg; para 10 m-qg exactos hacen falta {n10:,.0f} fotones "
      f"(C=0.3 pide {1/0.3**2:.1f}× los de C=1); ODMR 1 MHz, C=0.2, 61 puntos, 1.16 M fotones: CRB {crb/1e3:.2f} kHz = {crb/28e9*1e9:.0f} nT (γ=28 GHz/T)")
print("verify_cap37: OK")
