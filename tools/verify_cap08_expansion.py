"""Comprobación de las deducciones añadidas al capítulo 8 en la reescritura ampliada.

Cubre solo lo nuevo (lo anterior sigue en tools/verify_cap08.py y en los ejemplos §25, §36, §58, §126, §159):
el cambio de coordenada de la información de Fisher, la información cuántica de un estado mixto, la diagonal
de la métrica, la regla de desplazamiento, la elipse de los dos relojes, la información de Ramsey, la forma
cerrada de la ganancia GHZ con su tamaño óptimo y la termodinámica del qubit.
Solo NumPy + qang (core). Imprime PASA/FALLA y sale con 1 si algo falla.

    python tools/verify_cap08_expansion.py
"""
import math
import sys

import numpy as np

from qang import clock_comparison as cc
from qang import fisher, probability, qga
from qang import statistics as st

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


# 1. información de un disparo y cambio de coordenada
worst = 0.0
for th in np.linspace(0.2, 2.9, 15):
    q = math.cos(th)
    p0 = math.cos(th / 2) ** 2
    f_theta = (math.sin(th) / 2) ** 2 / (p0 * (1 - p0))
    worst = max(worst, abs(f_theta - 1), abs(fisher.fisher_theta(th) - 1),
                abs(fisher.fisher_qg(q) - f_theta / math.sin(th) ** 2),
                abs(fisher.cramer_rao_qg(q, 1000) - (1 - q * q) / 1000))
check("F_θ = 1; F_qg = F_θ (dθ/dqg)² = 1/(1 − qg²); cota (1 − qg²)/N", worst < 1e-12)
check("F_qga = 4π²·10⁻⁶ qga⁻² y σ = 159,15/√N qga",
      abs(qga.FISHER_PER_QGA2 - 4 * math.pi ** 2 * 1e-6) < 1e-18
      and round(qga.angle_sigma_qga(1) , 2) == 159.15 and round(qga.angle_sigma_qga(1000), 2) == 5.03)
check("10 m-qg en el ecuador: 10⁴ disparos; 1 m-qg: 10⁶; 5 m-qg a 60°: 30 000; 1 m-qg del polo: 2,56°",
      probability.shots_for_resolution(10, math.pi / 2) == 10_000
      and probability.shots_for_resolution(1, math.pi / 2) == 1_000_000
      and probability.shots_for_resolution(5, math.pi / 3) == 30_000
      and round(math.degrees(probability.pole_angle(1.0)), 2) == 2.56)

# 2. información cuántica: estado puro en un meridiano, F_Q = 4 Var(σ_y/2) = 1; estado mixto de radio r, F_Q = r²
Y = np.array([[0, -1j], [1j, 0]])
worst = 0.0
for th in np.linspace(0.1, 3.0, 9):
    psi = np.array([math.cos(th / 2), math.sin(th / 2)])
    G = Y / 2
    var = (psi.conj() @ G @ G @ psi - (psi.conj() @ G @ psi) ** 2).real
    worst = max(worst, abs(4 * var - 1))
check("estado puro: F_Q = 4 Var(σ_y/2) = 1, igual a la información de la medición en Z", worst < 1e-12)
worst = 0.0
for r in (0.3, 0.6, 0.9):
    for th in (0.4, 1.2, 2.0):
        # medición a lo largo de la dirección del vector: p = (1 + r cos(θ − θ_m))/2, óptima en cuadratura
        f_best = max(((r * math.sin(th - tm)) ** 2) / (1 - (r * math.cos(th - tm)) ** 2)
                     for tm in np.linspace(0, math.pi, 20001))
        worst = max(worst, abs(f_best - r * r), abs(st.theta_qcrb_variance(r, 100) - 1 / (100 * r * r)))
check("estado mixto: la mejor medición proyectiva da F = r² (cota 1/(N r²))", worst < 1e-6, f"error máx {worst:.1e}")

# 3. métrica de un ansatz real RY + CZ: diagonal 1/4; regla de desplazamiento exacta
rng = np.random.default_rng(8)
worst = 0.0
for _ in range(10):
    th = rng.uniform(0, 2 * math.pi, 8)
    worst = max(worst, np.abs(np.diag(fisher.full_metric(th, 4, 2)) - 0.25).max())
check("ansatz real RY + CZ (4 qubits, 2 capas): toda entrada diagonal de la métrica vale 1/4", worst < 1e-10)
Hm = fisher.tfim_hamiltonian(4)
th = rng.uniform(0, 2 * math.pi, 8)
g_shift = fisher.gradient(th, Hm, 4, 2)
eps = 1e-6
g_num = np.array([(fisher.energy(th + eps * np.eye(8)[k], Hm, 4, 2) - fisher.energy(th - eps * np.eye(8)[k], Hm, 4, 2))
                  / (2 * eps) for k in range(8)])
check("regla de desplazamiento: gradiente exacto con 2 evaluaciones por parámetro (8 parámetros)",
      np.abs(g_shift - g_num).max() < 1e-7, f"error máx {np.abs(g_shift - g_num).max():.1e}")
f1, f2 = fisher.indistinguishable_pair(1.1, 0.4)[:2]
check("una sola evaluación desplazada no determina la derivada", abs(f1(1.5) - f2(1.5)) < 1e-12)

# 4. dos relojes: la elipse, los momentos y el corrimiento al rojo del estroncio
worst = 0.0
for delta in (0.05, 0.15, 0.4, 1.0):
    phi = rng.uniform(-math.pi, math.pi, 2000)
    x, y = np.cos(phi + delta), np.cos(phi - delta)
    c = math.cos(2 * delta)
    worst = max(worst, np.abs(x * x + y * y - 2 * x * y * c - (1 - c * c)).max())
check("x² + y² − 2xy c = 1 − c² con c = cos 2δ (4 valores de δ, 2000 fases)", worst < 1e-12)
phi = np.linspace(0, 2 * math.pi, 200000, endpoint=False)
x, y = np.cos(phi + 0.15), np.cos(phi - 0.15)
check("fase común uniforme: ⟨x²⟩ = ⟨y²⟩ = ½ y ⟨xy⟩ = ½ cos 2δ",
      abs(np.mean(x * x) - 0.5) < 1e-9 and abs(np.mean(x * y) - 0.5 * math.cos(0.3)) < 1e-9)
two_delta = cc.redshift_phase(429.228e12, 1.0, 1.0)
check("estroncio, 1 m, 1 s: 2δ = 0,294 rad, δ = 0,1471", round(two_delta, 3) == 0.294 and round(two_delta / 2, 4) == 0.1471)
pairs = cc.simulate_pairs(0.1471, 200, 1000, np.random.default_rng(126))
est = cc.differential_phase_from_c(cc.ellipse_estimate(pairs[0], pairs[1], n_shots=1000)["c"])
check("ajuste de elipse con cuadrados insesgados recupera δ = 0,147 (200 corridas de 1000 disparos)",
      abs(est - 0.1471) < 0.01, f"δ estimado {est:.4f}")

# 5. Ramsey: F = (V² − qg²)/(1 − qg²); máximo V² en qg = 0; cero en qg = ±V
worst = 0.0
for V in (1.0, 0.9, 0.6):
    for ph in np.linspace(0.05, 3.0, 30):
        q = V * math.cos(ph)
        p0 = (1 + q) / 2
        f = (V * math.sin(ph) / 2) ** 2 / (p0 * (1 - p0)) if abs(q) < 1 else 1.0
        worst = max(worst, abs(f - (V * V - q * q) / (1 - q * q)))
check("Ramsey: F = (V² − qg²)/(1 − qg²), plana con V = 1, máxima V² en el centro de la franja", worst < 1e-10)

# 6. GHZ: ganancia cerrada N^s (1 − 2e)^(2N − 2), s = 0 (markoviano) y ½ (gaussiano); tabla del capítulo; N óptimo
def rate_opt(fun):
    t = np.linspace(1e-4, 3.0, 300001)
    return float(np.max(fun(t)))


def gain(N, e, gaussian):
    if gaussian:
        ind = rate_opt(lambda t: N * t * np.exp(-2 * t**2) * (1 - 2 * e) ** 2)
        ghz = rate_opt(lambda t: N**2 * t * np.exp(-2 * N * t**2) * (1 - 2 * e) ** (2 * N))
    else:
        ind = rate_opt(lambda t: N * t * np.exp(-2 * t) * (1 - 2 * e) ** 2)
        ghz = rate_opt(lambda t: N**2 * t * np.exp(-2 * N * t) * (1 - 2 * e) ** (2 * N))
    return ghz / ind


table = {(2, 0, 0): 1.000, (10, .002, 0): 0.930, (30, .01, 0): 0.310, (100, .01, 0): 0.018, (100, .002, 0): 0.452,
         (2, 0, 1): 1.414, (10, .002, 1): 2.942, (30, .002, 1): 4.341, (100, .002, 1): 4.522, (30, .01, 1): 1.697,
         (100, .01, 1): 0.183, (100, 0, 1): 10.00}
worst_t = worst_c = 0.0
for (N, e, gs), val in table.items():
    num = gain(N, e, bool(gs))
    closed = N ** (0.5 * gs) * (1 - 2 * e) ** (2 * N - 2)
    worst_t = max(worst_t, abs(num - val))
    worst_c = max(worst_c, abs(num / closed - 1))
check("GHZ: la optimización numérica reproduce la tabla del capítulo (12 celdas)", worst_t < 6e-4,
      f"desvío máx {worst_t:.1e}")
check("GHZ: ganancia = N^s (1 − 2e)^(2N − 2), s = 0 markoviano, ½ gaussiano", worst_c < 1e-3)
n_opt = {e: max(range(1, 400), key=lambda N: math.sqrt(N) * (1 - 2 * e) ** (2 * N - 2)) for e in (0.002, 0.01)}
pred = {e: -1 / (4 * math.log(1 - 2 * e)) for e in (0.002, 0.01)}
g_opt = {e: math.sqrt(n) * (1 - 2 * e) ** (2 * n - 2) for e, n in n_opt.items()}
check("GHZ gaussiano: tamaño óptimo N* = −1/(4 ln(1 − 2e)) ≈ 1/(8e): 62 y 12; ganancia máxima 4,83 y 2,22",
      n_opt[0.002] == 62 and n_opt[0.01] == 12 and abs(pred[0.002] - 62.4) < 0.1 and abs(pred[0.01] - 12.4) < 0.1
      and round(g_opt[0.002], 2) == 4.83 and round(g_opt[0.01], 2) == 2.22,
      f"N* = {n_opt}, ganancia {g_opt[0.002]:.3f} y {g_opt[0.01]:.3f}")

# 7. qubit térmico: tabla en qg contra exp(−βH), información sobre T y pico de Schottky
worst = 0.0
h = 0.7
for T in (0.2, 0.5, 1.0, 3.0):
    b = 1 / T
    w = np.array([math.exp(b * h), math.exp(-b * h)])          # H = −h Z: energías −h, +h
    Zp = w.sum()
    p = w / Zp
    q = p[0] - p[1]
    U = -h * q
    S_bits = -sum(pi * math.log2(pi) for pi in p)
    Fe = -T * math.log(Zp)
    varH = h * h * (1 - q * q)
    C = varH / T**2
    dT = 1e-6
    dp = (np.exp(np.array([h, -h]) / (T + dT)) / np.exp(np.array([h, -h]) / (T + dT)).sum()
          - np.exp(np.array([h, -h]) / (T - dT)) / np.exp(np.array([h, -h]) / (T - dT)).sum()) / (2 * dT)
    FT = float(np.sum(dp**2 / p))
    Hbin = -((1 + q) / 2) * math.log2((1 + q) / 2) - ((1 - q) / 2) * math.log2((1 - q) / 2)
    worst = max(worst, abs(q - math.tanh(b * h)), abs(S_bits - Hbin),
                abs(Fe + T * math.log(2 / math.sqrt(1 - q * q))), abs(C - math.atanh(q) ** 2 * (1 - q * q)),
                abs(FT - C / T**2) / (C / T**2), abs(math.acos(q) - (math.pi / 2 - 2 * math.atan(math.tanh(b * h / 2)))))
check("térmico: qg = tanh βh, S, F, C y ángulo de Gudermann; información sobre T igual a C/T²", worst < 1e-6,
      f"error máx {worst:.1e}")
xs = np.linspace(0.5, 2.0, 1500001)
x_star = float(xs[np.argmax(xs**2 / np.cosh(xs) ** 2)])
q_star = math.tanh(x_star)
check("Schottky: βh = 1,1997, qg* = 0,8336, qg·artanh(qg) = 1, C máx = 0,439",
      round(x_star, 4) == 1.1997 and round(q_star, 4) == 0.8336 and abs(q_star * math.atanh(q_star) - 1) < 1e-5
      and round(x_star**2 * (1 - q_star**2), 3) == 0.439)

# 8. el bloque de código del capítulo
r = probability.analyze(750, 1000)
check("analyze(750, 1000): 500 m-qg [444,5; 551,7], θ = 60,0° [56,5°; 63,6°]",
      round(r.milliqang[0]) == 500 and round(r.milliqang[1], 1) == 444.5 and round(r.milliqang[2], 1) == 551.7
      and round(math.degrees(r.theta[1]), 1) == 56.5 and round(math.degrees(r.theta[2]), 1) == 63.6)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
