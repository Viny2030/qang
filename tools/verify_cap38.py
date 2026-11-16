"""tools/verify_cap38.py — Cap. 38: aplicaciones transversales leídas con qang (qang.qtransv)."""
import math
import numpy as np
from qang.qtransv import *
from qang.qnano import theta_std_sqrtN

# [1] Jordan–Wigner y el signo del intercambio
e = anticommutation_error(4)
assert e < 1e-15
f, s = exchange_qgx(True), exchange_qgx(False)
assert abs(f + 1) < 1e-15 and abs(s - 1) < 1e-15
print(f"[1] anticonmutación (4 modos): error {e:.1e}; test de Hadamard del intercambio: fermiónico qg_X = {1000*f:+.0f} m-qg, SWAP de qubits {1000*s:+.0f} m-qg")

# [2] eco de Hahn (σ = 2π·50 kHz, τ = 20 µs ⇒ σ·τ = 6.283)
st = 2 * math.pi * 50e3 * 20e-6
free, stat, perr, drift = free_decay_qgx(st), hahn_echo_ensemble(st, 0.0), hahn_echo_ensemble(st, 0.02), hahn_echo_qgx_drift(st, 0.1)
assert free < 1e-8 and abs(stat - 1) < 1e-12 and abs(perr - 0.99901) < 1e-5 and abs(drift - 0.8209) < 1e-4
print(f"[2] qg_X: decaimiento libre {free:.1e}; eco estático {stat:.6f}; eco con π de 2 % de más {perr:.5f} ({1000*(1-perr):.2f} m-qg); "
      f"con deriva de 10 % de σ {drift:.4f} ({1000*(1-drift):.0f} m-qg)")

# [3] Ising con campo transverso: lectura transversal en g = J
mx = {n: tfim_mx_mqg(n) for n in (2, 4, 6, 8, 10)}
ref = {2: 894.4, 4: 810.1, 6: 771.1, 8: 747.9, 10: 732.3}
assert all(abs(mx[n] - ref[n]) < 0.06 for n in mx)
vals = [mx[n] for n in sorted(mx)]
assert all(a > b for a, b in zip(vals, vals[1:])) and vals[-1] > 2000 / math.pi
print("[3] m_X(g=J): " + ", ".join(f"N={n}: {mx[n]:.1f}" for n in mx) + f" m-qg; límite 2000/π = {2000/math.pi:.1f} (N=10 aún {vals[-1]-2000/math.pi:.1f} m-qg arriba)")

# [4] Aharonov–Bohm
ph = np.linspace(0, 3, 301)
per = np.max(np.abs(ab_qgz(ph) - ab_qgz(ph + 1.0)))
assert per < 1e-12 and abs(ab_qgz(0.25)) < 1e-15 and abs(ab_qgz(0.5) + 1) < 1e-15 and abs(ab_qgz(0.2) - ab_qgz(-0.2)) < 1e-15
sig = [theta_std_sqrtN(math.cos(math.pi * p) ** 2, 2000) / (2 * math.pi) for p in (0.05, 0.15, 0.25, 0.35, 0.45)]
assert all(0.1591 < x < 0.1600 for x in sig)
print(f"[4] qg_Z = cos(2πΦ/Φ0): periodo Φ0 a {per:.0e}; Φ y −Φ leen lo mismo; √N·σ(Φ̂/Φ0) con 2000 electrones = {min(sig):.4f}–{max(sig):.4f} (1/2π = {1/(2*math.pi):.4f})")

# [5] rotación de Wigner
num, frm = wigner_rotation([0.6, 0, 0], [0, 0.8, 0])
assert abs(num - frm) < 1e-12 and abs(num - 0.3303) < 5e-5 and abs(math.degrees(num) - 18.92) < 0.01
assert abs(1000 * math.cos(num) - 945.9) < 0.05 and abs(1000 * num / (2 * math.pi) - 52.6) < 0.05
n0, _ = wigner_rotation([0.6, 0, 0], [0.8, 0, 0])
assert n0 < 1e-7
print(f"[5] Ω_W = {num:.4f} rad = {math.degrees(num):.2f}° (fórmula cerrada a {abs(num-frm):.0e}); qg_X = cos Ω = {1000*math.cos(num):+.1f} m-qg; {1000*num/(2*math.pi):.1f} qga; colineales: {n0:.0e}")

# [6] Fresnel (aire → vidrio n=1.5; vidrio → aire en reflexión total)
tb = math.atan(1.5)
rs, rp = fresnel(1.0, 1.5, tb)
Rs = abs(rs) ** 2
rn = abs(fresnel(1.0, 1.5, 0.0)[0]) ** 2
assert abs(rp) < 1e-15 and abs(math.degrees(tb) - 56.31) < 0.01 and abs(Rs - 0.148) < 5e-4 and abs(1000 * (2 * Rs - 1) + 704.1) < 0.1
assert abs(1000 * (2 * rn - 1) + 920) < 1e-9
tc = math.asin(1 / 1.5)
r60 = fresnel(1.5, 1.0, math.radians(60))
assert max(abs(abs(r60[0]) - 1), abs(abs(r60[1]) - 1)) < 1e-14
th = np.radians(np.linspace(math.degrees(tc) + 1e-3, 89.99, 96001))
d = np.abs([tir_phase_difference(1.5, x) for x in th])
k = int(d.argmax())
cross = [math.degrees(th[i]) for i in range(1, len(th)) if (d[i - 1] - math.pi / 4) * (d[i] - math.pi / 4) < 0]
d45 = abs(tir_phase_difference(1.5, math.radians(45)))
assert abs(math.degrees(d[k]) - 45.24) < 0.01 and abs(math.degrees(th[k]) - 51.67) < 0.01 and abs(math.sin(d45) - 0.600) < 1e-9
assert len(cross) == 2 and abs(cross[0] - 50.23) < 0.02 and abs(cross[1] - 53.26) < 0.02
print(f"[6] Brewster {math.degrees(tb):.2f}°: r_p = 0, R_s = {Rs:.3f} ({1000*(2*Rs-1):+.1f} m-qg); incidencia normal R = {rn:.2f} ({1000*(2*rn-1):+.0f} m-qg); "
      f"crítico {math.degrees(tc):.2f}°; |r|=1 a 1e-14; |δp−δs| máx {math.degrees(d[k]):.2f}° en {math.degrees(th[k]):.2f}°, = 45° en {cross[0]:.2f}° y {cross[1]:.2f}°; "
      f"a 45° de incidencia qg_Y = sin|δp−δs| = {1000*math.sin(d45):+.0f} m-qg")

# [7] Bogoliubov / Unruh
gd = max(abs(math.acos(math.sqrt(vacuum_weight(r))) - gudermannian(r)) for r in (0.25, 0.5, 1.0, 1.5))
ent = {r: abs(thermal_entropy_nats(r) - thermal_entropy_sum(r)) for r in (0.25, 0.5, 1.0, 1.5)}
assert gd < 1e-14 and ent[1.0] < 1e-12 and 1e-8 < ent[1.5] < 1e-5
print(f"[7] θ/2 = gd(r) = arctan(sinh r) a {gd:.0e}; entropía cerrada − suma: " + ", ".join(f"r={r}: {v:.0e}" for r, v in ent.items()) + " (corte de 80 niveles)")
print("verify_cap38: OK")
