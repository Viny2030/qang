"""tools/verify_cap44.py — Cap. 44: agentes híbridos clásico–cuánticos leídos con qang (identidades exactas, flujos cerrados y simulaciones).
Solo NumPy + qang. Uso: python tools/verify_cap44.py   (si algo cambia, el script se detiene)."""
import numpy as np

from qang import probability as pr
from qang.qagent import *

rng = np.random.default_rng(44)

# [1] política de Born: gradiente exacto y información de Fisher 1
fisher_ok = 0
for t in (0.3, 0.8, 1.5, 2.6):
    p0, p1 = born_policy(t)
    assert abs(p0 + p1 - 1) < 1e-15 and abs(p0 - (1 + np.cos(t)) / 2) < 1e-15
    assert abs(p0 * score(0, t) ** 2 + p1 * score(1, t) ** 2 - 1.0) < 1e-12          # Fisher = 1 para todo θ
    r0, r1 = 0.9, 0.3
    a = (rng.random(400000) > p0).astype(int)
    g = np.where(a == 0, r0 * score(0, t), r1 * score(1, t))
    assert abs(g.mean() - expected_gradient(t, r0, r1)) < 4 * np.sqrt(gradient_variance(t, r0, r1) / 400000)
    assert abs(g.var() / gradient_variance(t, r0, r1) - 1) < 0.03
    fisher_ok += 1
print(f"[1] π(0)=cos²(θ/2); ∂θJ = −Δ sinθ/2 y Var(g) = r0² sin²(θ/2)+r1² cos²(θ/2) − (∂θJ)², con 4×10⁵ muestras (4 ángulos, ±3 %); "
      f"Fisher = 1 para todo θ ({fisher_ok} ángulos)")

# [2] flujo del gradiente: θ es el gradiente natural del logit
eta, delta, dt, T = 0.05, 0.5, 0.01, 40.0
th = np.pi / 2
for _ in range(int(T / dt)):
    th += dt * eta * expected_gradient(th, delta, 0.0)
assert abs(th - flow_theta(np.pi / 2, eta, delta, T)) < 1e-4
for t in (0.5, 1.2, 2.0):
    assert abs(logit_of(flow_theta(t, eta, delta, T)) - (logit_of(t) + eta * delta * T)) < 1e-9      # φ_t = φ0 + ηΔt
t_born = time_to_wrong_probability(np.pi / 2, 1.0, 1.0, 0.01)
t_van = logit_vanilla_time(0.0, 1.0, 1.0, 0.01)
phi, dt2 = 0.0, 1e-3                       # integra el flujo plano en el logit hasta p_wrong = 0.01 y compara
tt = 0.0
while 1 / (1 + np.exp(phi)) > 0.01:
    p0 = 1 / (1 + np.exp(-phi))
    phi += dt2 * p0 * (1 - p0)
    tt += dt2
assert abs(tt / t_van - 1) < 1e-3
assert abs(t_born - np.log(99)) < 1e-12
t_b3, t_v3 = time_to_wrong_probability(np.pi / 2, 1.0, 1.0, 1e-3), logit_vanilla_time(0.0, 1.0, 1.0, 1e-3)
mt = np.mean([np.sin(run_bandit(np.pi / 2, (0.7, 0.3), 0.02, 100, np.random.default_rng(s))[-1] / 2) ** 2 for s in range(400)])
pred = np.sin(flow_theta(np.pi / 2, 0.02, 0.4, 100) / 2) ** 2
assert abs(mt / pred - 1) < 0.08, (mt, pred)
print(f"[2] tan(θ_t/2)=tan(θ0/2)e^(−ηΔt/2) ⇔ φ_t=φ0+ηΔt; tiempo a π(peor)≤1 %: θ {t_born:.3f}/(ηΔ) frente a logit plano {t_van:.1f}/(ηΔ) "
      f"({t_van / t_born:.1f}×); a 0,1 %: {t_b3:.2f} frente a {t_v3:.0f} ({t_v3 / t_b3:.0f}×); REINFORCE de Bernoulli (400 corridas, η=0,02, 100 pasos): "
      f"π(peor)={mt:.3f} frente a {pred:.3f} predicho")

# [3] el polo: gradiente que se anula, exploración
n_eq, n_pole = episodes_for_gradient_sign(np.pi / 2, 0.9, 0.3), episodes_for_gradient_sign(0.05, 0.9, 0.3)
assert n_pole / n_eq > 50 and episodes_for_gradient_sign(0.0, 0.9, 0.3) == float("inf")
stuck = run_bandit(np.pi - 1e-9, (0.7, 0.3), 0.05, 2000, np.random.default_rng(0))
assert abs(stuck[-1] - stuck[0]) == 0.0
lo = exploration_floor(0.01)
assert abs(np.sin(lo / 2) ** 2 - 0.01) < 1e-15
esc = []
for s in range(200):
    p = run_bandit(np.pi - lo, (0.7, 0.3), 0.05, 3000, np.random.default_rng(s), p_floor=0.01)
    pw = np.sin(p / 2) ** 2
    esc.append(int(np.argmax(pw < 0.5)) if (pw < 0.5).any() else -1)
esc = np.array(esc)
assert (esc >= 0).mean() >= 0.99 and 100 < np.median(esc[esc >= 0]) < 500
print(f"[3] episodios para el signo del gradiente: {n_eq:.0f} en el ecuador frente a {n_pole:.0f} a θ=0,05 ({n_pole / n_eq:.0f}×) e ∞ en el polo; "
      f"sin piso, un agente en θ=π−10⁻⁹ no se mueve en 2000 pasos; con piso p_min=1 % (θ_min={lo:.3f} rad) sale del polo malo en "
      f"{(esc >= 0).mean() * 100:.0f} % de 200 corridas (mediana {int(np.median(esc[esc >= 0]))} pasos)")

# [4] lectura de decisión
d = decision_report(0.4, z=3.0)
assert abs(d["qg_Z"] - np.cos(0.4)) < 1e-15 and abs(d["qg_SNR"] - pr.quantum_snr(np.cos(0.4))) < 1e-12
assert d["shots_for_sign"] == pr.shots_for_sign(np.cos(0.4), 3.0) and d["greedy_action"] == 0.0
assert decision_report(np.pi / 2)["shots_for_sign"] == float("inf") and abs(decision_report(np.pi / 2)["qg_S"] - 1.0) < 1e-12
assert decision_report(2.5)["greedy_action"] == 1.0
print(f"[4] θ=0,4: qg_Z={d['qg_Z']:.3f} ({d['m_Z']:.0f} m-qg), qg_S={d['qg_S']:.3f} bits, qg_SNR={d['qg_SNR']:.2f}, "
      f"{d['shots_for_sign']:.0f} disparos para el signo a 3σ; en el ecuador qg_S=1 bit y el signo no se decide con ningún número de disparos")

# [5] reparto de disparos entre K lecturas
thetas = [0.2, 1.0, np.pi / 2, 2.9]
B = 4000
nopt, nuni = neyman_allocation(thetas, B), np.full(4, B / 4)
assert abs(nopt.sum() - B) < 1e-9
v_opt, v_uni = allocation_variance(thetas, nopt), allocation_variance(thetas, nuni)
assert abs(v_uni / v_opt - uniform_to_neyman_ratio(thetas)) < 1e-12 and v_opt <= v_uni
assert abs(allocation_variance([1.0, 1.0], [10, 10]) / allocation_variance([1.0, 1.0], [10, 10]) - 1) < 1e-15
assert abs(uniform_to_neyman_ratio([1.0, 1.0, 1.0]) - 1.0) < 1e-12


def sse(n):
    errs = []
    for _ in range(4000):
        e = 0.0
        for t, ni in zip(thetas, n):
            k = rng.binomial(int(round(ni)), np.cos(t / 2) ** 2)
            e += (2 * k / int(round(ni)) - 1 - np.cos(t)) ** 2
        errs.append(e)
    return np.mean(errs)


m_opt, m_uni = sse(nopt), sse(nuni)
assert abs(m_opt / v_opt - 1) < 0.06 and abs(m_uni / v_uni - 1) < 0.06
print(f"[5] K=4 lecturas con B=4000: n_i ∝ sinθ_i = {np.round(nopt).astype(int).tolist()}; suma de varianzas {v_opt:.2e} (óptimo) frente a {v_uni:.2e} "
      f"(uniforme), cociente {v_uni / v_opt:.3f} (simulado {m_uni / m_opt:.3f})")
print("OK — verify_cap44")
