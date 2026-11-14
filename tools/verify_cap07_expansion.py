"""Comprobación de las deducciones añadidas al capítulo 7 en la reescritura ampliada.

Cubre solo lo nuevo (lo anterior sigue en tools/verify_cap07.py y en las pruebas de qang.geometric):
la conexión y la curvatura en coordenadas, la invariancia de gauge del producto de solapamientos, el
invariante de Bargmann del octante, la fórmula de Van Oosterom y Strackee, la separación de fase total
y dinámica en la precesión, la prueba de Hadamard y la varianza de la fase con dos cuadraturas.
Solo NumPy + qang (core). Imprime PASA/FALLA y sale con 1 si algo falla.

    python tools/verify_cap07_expansion.py
"""
import cmath
import math
import sys

import numpy as np

from qang import geometric, qga

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


def state(theta, phi):
    return np.array([math.cos(theta / 2), cmath.exp(1j * phi) * math.sin(theta / 2)])


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# 1. conexión y curvatura en el gauge cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>
h = 1e-5
worst_a = worst_f = 0.0


def a_phi(t, p):
    return (1j * np.vdot(state(t, p), (state(t, p + h) - state(t, p - h)) / (2 * h))).real


def a_theta(t, p):
    return (1j * np.vdot(state(t, p), (state(t + h, p) - state(t - h, p)) / (2 * h))).real


for t in np.linspace(0.2, 2.9, 12):
    p = 0.7
    worst_a = max(worst_a, abs(a_phi(t, p) + math.sin(t / 2) ** 2), abs(a_theta(t, p)))
    f = (a_phi(t + 1e-3, p) - a_phi(t - 1e-3, p)) / 2e-3 - (a_theta(t, p + 1e-3) - a_theta(t, p - 1e-3)) / 2e-3
    worst_f = max(worst_f, abs(f + 0.5 * math.sin(t)))
check("A_φ = −sin²(θ/2) = −p₁, A_θ = 0", worst_a < 1e-8, f"error máx {worst_a:.1e}")
check("curvatura F_θφ = −½ sin θ", worst_f < 1e-6, f"error máx {worst_f:.1e}")

# 2. invariancia de gauge del producto de solapamientos
rng = np.random.default_rng(7)
loop = [state(1.1, 2 * math.pi * k / 300) for k in range(300)]
regauged = [cmath.exp(1j * rng.uniform(0, 2 * math.pi)) * s for s in loop]
g0, g1 = geometric.pancharatnam_phase(loop), geometric.pancharatnam_phase(regauged)
check("el producto de solapamientos no cambia con fases arbitrarias en cada estado", abs(wrap(g0 - g1)) < 1e-10)
check("bucle a θ = 1,1: γ = −2π p₁ (discretizado en 300 puntos)",
      abs(wrap(g0 + 2 * math.pi * math.sin(0.55) ** 2)) < 1e-3)

# 3. el octante: invariante de Bargmann (1 + i)/4
Hs, Ds, Rs = np.array([1, 0]), np.array([1, 1]) / math.sqrt(2), np.array([1, 1j]) / math.sqrt(2)
barg = np.vdot(Hs, Ds) * np.vdot(Ds, Rs) * np.vdot(Rs, Hs)
check("⟨H|D⟩⟨D|R⟩⟨R|H⟩ = (1 + i)/4, γ = −π/4, 875 qga",
      abs(barg - (1 + 1j) / 4) < 1e-12 and abs(-cmath.phase(barg) + math.pi / 4) < 1e-12
      and abs((qga.qga_from_radians(-cmath.phase(barg)) % 1000) - 875) < 1e-9)

# 4. Van Oosterom–Strackee: tan(Ω/2) = a·(b×c) / (1 + a·b + b·c + c·a), y γ = −Ω/2 en triángulos al azar
worst = 0.0
for _ in range(300):
    v = rng.normal(size=(3, 3))
    a, b, c = (x / np.linalg.norm(x) for x in v)
    omega = 2 * math.atan2(float(a @ np.cross(b, c)), 1 + float(a @ b) + float(b @ c) + float(c @ a))
    sa, sb, sc = (geometric.state_from_bloch(x) for x in (a, b, c))
    gam = -cmath.phase(np.vdot(sa, sb) * np.vdot(sb, sc) * np.vdot(sc, sa))
    worst = max(worst, abs(wrap(gam + omega / 2)), abs(wrap(geometric.solid_angle([a, b, c]) - omega)))
check("triángulos geodésicos al azar: γ = −Ω/2 con Ω de Van Oosterom–Strackee (300 casos)", worst < 1e-9,
      f"error máx {worst:.1e}")
check("octante: Ω = π/2",
      abs(2 * math.atan2(1.0, 1.0) - math.pi / 2) < 1e-12)

# 5. precesión alrededor de Z: fase total − fase dinámica = −π (1 − cos θ)
worst = 0.0
for t in np.linspace(0.2, 2.9, 10):
    psi0 = state(t, 0.0)
    U = np.diag([cmath.exp(-1j * math.pi), cmath.exp(1j * math.pi)])      # exp(−i φ σ_z / 2) con φ = 2π
    total = cmath.phase(np.vdot(psi0, U @ psi0))                            # −π ≡ π
    dynamic = -math.pi * math.cos(t)                                        # −∫⟨H⟩dt, H = (ω/2) σ_z, ωT = 2π
    worst = max(worst, abs(wrap(total - dynamic + math.pi * (1 - math.cos(t)))))
check("precesión: fase total − dinámica = −Ω/2 (10 alturas)", worst < 1e-12, f"error máx {worst:.1e}")

# 6. prueba de Hadamard: p0 = (1 + cos γ)/2; con la referencia corrida un cuarto de vuelta, (1 + sin γ)/2
worst = 0.0
Had = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
for gam in np.linspace(-3, 3, 25):
    for shift, want in ((0.0, (1 + math.cos(gam)) / 2), (-math.pi / 2, (1 + math.sin(gam)) / 2)):
        ctrl = Had @ np.array([1, 0])                    # |+>
        ctrl = np.array([ctrl[0], cmath.exp(1j * (gam + shift)) * ctrl[1]])   # fase del bucle sobre la rama |1>
        worst = max(worst, abs(abs((Had @ ctrl)[0]) ** 2 - want))
check("prueba de Hadamard: p₀ = cos²(γ/2) y, con la segunda cuadratura, (1 + sin γ)/2", worst < 1e-12)

# 7. una cuadratura: σ = 1/√N; dos cuadraturas: Var = (1 − ½ sin² 2γ)/N con N disparos cada una
N, reps = 1000, 40000
check("1/√N con N = 1000 es 31,6 mrad = 1,8°",
      round(1000 / math.sqrt(N), 1) == 31.6 and round(math.degrees(1 / math.sqrt(N)), 1) == 1.8)
ok, detail = True, []
for gam in (0.3, math.pi / 4, 1.0, 2.0, -2.5):
    kc = rng.binomial(N, (1 + math.cos(gam)) / 2, size=reps)
    ks = rng.binomial(N, (1 + math.sin(gam)) / 2, size=reps)
    est = np.arctan2(2 * ks / N - 1, 2 * kc / N - 1)
    var = float(np.mean(wrap(est - gam) ** 2))
    pred = (1 - 0.5 * math.sin(2 * gam) ** 2) / N
    ok &= abs(var / pred - 1) < 0.04
    detail.append(f"γ={gam:.2f}: {var * N:.3f} vs {pred * N:.3f}")
    ok &= bool(np.mean(np.sign(est) == np.sign(gam)) > 0.999)
check("dos cuadraturas: N·Var(γ̂) = 1 − ½ sin² 2γ y el signo queda fijado", ok, "; ".join(detail))

# 8. valores del capítulo en la biblioteca
check("cone_phase_turns: 0,75 a +500 m-qg; 0,5 en el ecuador; cone_phase_qga: 750 y 500",
      geometric.cone_phase_turns(0.5) == 0.75 and geometric.cone_phase_turns(0.0) == 0.5
      and abs(qga.cone_phase_qga(0.5) - 750) < 1e-9 and abs(qga.cone_phase_qga(0.0) - 500) < 1e-9)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
