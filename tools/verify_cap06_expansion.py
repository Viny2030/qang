"""Comprobación de las deducciones añadidas al capítulo 6 en la reescritura ampliada.

Cubre solo lo nuevo (lo anterior sigue en tools/verify_cap06.py y tests/test_polarization.py):
el retardador como rotación de la esfera, los extremos de la ley de Malus, el polarizador como
proyector, el sesgo exacto del cuadrado y la media de Maxwell del grado de polarización de luz
no polarizada. Solo NumPy + qang (core). Imprime PASA/FALLA y sale con 1 si algo falla.

    python tools/verify_cap06_expansion.py
"""
import math
import sys

import numpy as np

from qang import polarization as pz
from qang import statistics as st

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


def rotation(axis, angle):
    """Matriz de Rodrigues: rotación de ángulo `angle` alrededor de `axis` (unitario)."""
    n = np.asarray(axis, float)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


# 1. retardador (delta, phi) = rotación de ángulo delta alrededor de (qg_X, qg_Y, qg_Z) = (sin 2phi, 0, cos 2phi)
rng = np.random.default_rng(6)
worst = 0.0
for _ in range(200):
    delta, phi = rng.uniform(0, 2 * math.pi), rng.uniform(0, math.pi)
    M = pz.mueller_from_jones(pz.wave_plate(delta, phi))        # orden (S0, S1, S2, S3) = (I, Z, X, Y)
    R_zxy = M[1:, 1:]
    R_xyz = rotation([math.sin(2 * phi), 0.0, math.cos(2 * phi)], delta)
    perm = [2, 0, 1]                                            # (Z, X, Y) en términos de (X, Y, Z)
    worst = max(worst, np.abs(R_zxy - R_xyz[np.ix_(perm, perm)]).max(), abs(M[0, 0] - 1))
check("retardador = rotación de ángulo δ alrededor de (sin 2φ, 0, cos 2φ), 200 casos", worst < 1e-12,
      f"error máx {worst:.1e}")
H = np.array([1, 0])
check("media onda = wave_plate(π, φ); cuarto de onda = wave_plate(π/2, φ) (salvo fase global)",
      np.allclose(pz.mueller_from_jones(pz.half_wave_plate(0.3)), pz.mueller_from_jones(pz.wave_plate(math.pi, 0.3)))
      and np.allclose(pz.mueller_from_jones(pz.quarter_wave_plate(0.3)),
                      pz.mueller_from_jones(pz.wave_plate(math.pi / 2, 0.3))))

# 1b. rotador = rotación 2θ alrededor de qg_Y; media onda = reflexión r -> 2(n·r)n - r; polarizador de Mueller
perm = [2, 0, 1]
th = 0.37
R_rot = pz.mueller_from_jones(pz.rotator(th))[1:, 1:]
check("rotador de ángulo ϑ = rotación de 2ϑ alrededor de qg_Y",
      np.abs(R_rot - rotation([0, 1, 0], 2 * th)[np.ix_(perm, perm)]).max() < 1e-12)
n = np.array([math.sin(2 * th), 0.0, math.cos(2 * th)])
R_hwp = pz.mueller_from_jones(pz.half_wave_plate(th))[1:, 1:]
check("media onda: r' = 2(n·r)n − r", np.abs(R_hwp - (2 * np.outer(n, n) - np.eye(3))[np.ix_(perm, perm)]).max() < 1e-12)
a = n[perm]                                                     # vector del analizador en orden (Z, X, Y)
M_pol = pz.mueller_from_jones(pz.linear_polarizer(th))
M_ref = 0.5 * np.block([[np.ones((1, 1)), a[None, :]], [a[:, None], np.outer(a, a)]])
check("polarizador: M = ½ [[1, aᵀ], [a, a aᵀ]]", np.abs(M_pol - M_ref).max() < 1e-12)
S = pz.depolarizer(0.7) @ pz.qg_to_stokes({"Z": 0.6, "X": 0.3, "Y": 0.0})
out = pz.stokes_to_qg(S)
check("despolarizador d = 0,7 en código: P 0,671 → 0,470, qg_Z 0,60 → 0,42",
      round(pz.degree_of_polarization(S), 3) == 0.470 and round(out["Z"], 2) == 0.42 and round(out["X"], 2) == 0.21)

# 2. extremos de Malus: (1 ± sqrt(qg_Z² + qg_X²))/2 en psi y psi + 90°
luz = {"Z": 0.6, "X": 0.3, "Y": 0.0}
psi = 0.5 * math.atan2(luz["X"], luz["Z"])
p_lin = math.hypot(luz["Z"], luz["X"])
tmax, tmin = pz.malus_intensity(luz, psi), pz.malus_intensity(luz, psi + math.pi / 2)
check("máximo 0,835 en 13,3° y mínimo 0,165 en 103,3°; diferencia 0,671",
      round(math.degrees(psi), 1) == 13.3 and round(tmax, 3) == 0.835 and round(tmin, 3) == 0.165
      and abs(tmax - (1 + p_lin) / 2) < 1e-12 and abs(tmin - (1 - p_lin) / 2) < 1e-12
      and round(tmax - tmin, 3) == 0.671)
eliptica = {"Z": 0.6, "X": 0.3, "Y": 0.4}
P = math.sqrt(0.36 + 0.09 + 0.16)
vis = pz.malus_intensity(eliptica, psi) - pz.malus_intensity(eliptica, psi + math.pi / 2)
chi = 0.5 * math.asin(eliptica["Y"] / P)
check("con qg_Y = 0,4: contraste lineal 0,671 < P = 0,781; elipticidad χ = 15,4°",
      round(vis, 3) == 0.671 and round(P, 3) == 0.781 and round(math.degrees(chi), 1) == 15.4)

# 3. polarizador = proyector: transmisión (1 + r·a)/2 e idempotencia
worst = 0.0
for _ in range(200):
    th = rng.uniform(0, math.pi)
    E = rng.normal(size=2) + 1j * rng.normal(size=2)
    E /= np.linalg.norm(E)
    Pi = pz.linear_polarizer(th)
    qg = pz.jones_to_qg(E)
    worst = max(worst, abs(np.linalg.norm(Pi @ E) ** 2 - pz.malus_intensity(qg, th)), np.abs(Pi @ Pi - Pi).max())
check("polarizador: Π² = Π y ‖ΠE‖² = (1 + r·a)/2, 200 casos", worst < 1e-12, f"error máx {worst:.1e}")

# 4. sesgo exacto del cuadrado: E[q̂²] = q² + (1 - q²)/N; el estimador insesgado lo anula
q, N = 0.6, 25
p0 = (1 + q) / 2
ks = np.arange(N + 1)
pmf = np.array([math.comb(N, int(k)) * p0**k * (1 - p0) ** (N - k) for k in ks])
naive = float(np.sum(pmf * (2 * ks / N - 1) ** 2))
unb = float(np.sum(pmf * np.array([st.qg2_unbiased(int(k), N) for k in ks])))
check("E[q̂²] = q² + (1 − q²)/N (suma binomial exacta) y qg2_unbiased es insesgado",
      abs(naive - (q * q + (1 - q * q) / N)) < 1e-12 and abs(unb - q * q) < 1e-12,
      f"directo {naive:.4f}, insesgado {unb:.4f}, q² = {q*q:.2f}")

# 5. luz no polarizada: E[P̂] ≈ sqrt(8/(π N)) (media de Maxwell); 0,50 / 0,16 / 0,05
vals = []
for n in (10, 100, 1000):
    k = rng.binomial(n, 0.5, size=(40000, 3))
    phat = np.sqrt((((2 * k / n) - 1) ** 2).sum(axis=1))
    vals.append((n, float(phat.mean()), math.sqrt(8 / (math.pi * n))))
check("no polarizada: P̂ medio 0,50 / 0,16 / 0,05 con 10 / 100 / 1000 fotones, igual a sqrt(8/(πN))",
      all(abs(v[1] - ref) < 0.01 for v, ref in zip(vals, (0.50, 0.16, 0.05)))
      and all(abs(v[1] - v[2]) < 0.005 for v in vals),
      ", ".join(f"N={n}: {m:.3f} (Maxwell {t:.3f})" for n, m, t in vals))

# 6. resolución en mili-qang del ejemplo
check("σ_m = 1000 sin θ / √N = 25 m-qg con qg_Z = 0,6 y N = 1000",
      round(1000 * math.sqrt(1 - 0.36) / math.sqrt(1000)) == 25)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
