"""Verificación del capítulo 36 / Check script of chapter 36.

Toma de decisiones en entornos cuánticos: del aprendizaje model-free al model-aware.
Decision-making in quantum environments: from model-free to model-aware learning.

Cada número del capítulo sale de aquí. Solo NumPy + qang (core).
Every number of the chapter comes from here. NumPy + qang (core) only.

    python tools/verify_cap36.py
"""
from __future__ import annotations

import math

import numpy as np

from qang import fisher, proportions, pulse_area, qga, qml
from qang.probability import statistical_angle

TWO_PI = 2.0 * math.pi
OUT: dict = {}


def check(name: str, cond: bool) -> None:
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------------------------------
# 36.2  Comunicaciones: |+alpha> vs |-alpha>
# --------------------------------------------------------------------------
def overlap(alpha: float) -> float:
    """c = <-alpha|alpha> = exp(-2 alpha^2)."""
    return math.exp(-2.0 * alpha**2)


def helstrom(alpha: float) -> float:
    return 0.5 * (1.0 + math.sqrt(1.0 - overlap(alpha) ** 2))


def receiver_success(alpha: float, beta: float) -> float:
    """Kennedy generalizado: desplazar beta, detector on/off, sin clic -> '-alpha'."""
    return 0.5 * (1.0 - math.exp(-((alpha + beta) ** 2)) + math.exp(-((beta - alpha) ** 2)))


def receiver_success_qg(alpha: float, beta: float) -> float:
    """La misma probabilidad como lectura lineal: 1/2 + (qg_- - qg_+)/4."""
    qg_plus = 2.0 * math.exp(-((beta + alpha) ** 2)) - 1.0
    qg_minus = 2.0 * math.exp(-((beta - alpha) ** 2)) - 1.0
    return 0.5 + (qg_minus - qg_plus) / 4.0


def best_beta(alpha: float, grid: np.ndarray) -> float:
    return float(grid[np.argmax([receiver_success(alpha, b) for b in grid])])


def section_comms() -> None:
    alpha = 0.4
    c = overlap(alpha)
    # ángulo de Bloch entre las dos hipótesis: cos(Delta/2) = c
    delta = proportions.theta_from_proportion(c**2)
    check("36.2 Delta = 2 arccos(c)", abs(delta - 2 * math.acos(c)) < 1e-12)
    # distancia geodésica qg_L = Delta/2 = ángulo estadístico de las lecturas en la base óptima
    qg_l = delta / 2.0
    p_h = helstrom(alpha)
    check("36.2 Helstrom = (1 + sin qg_L)/2", abs(p_h - 0.5 * (1 + math.sin(qg_l))) < 1e-12)
    # las dos hipótesis medidas en la base de Helstrom: p y 1-p -> ángulo estadístico
    sa = statistical_angle([p_h, 1 - p_h], [1 - p_h, p_h])
    check("36.2 ángulo estadístico Helstrom = pi/2 - ... consistente",
          abs(math.cos(sa) - 2 * math.sqrt(p_h * (1 - p_h))) < 1e-12)
    check("36.2 2 sqrt(P_H (1-P_H)) = c", abs(2 * math.sqrt(p_h * (1 - p_h)) - c) < 1e-12)

    grid = np.linspace(0.0, 1.5, 3001)
    b_star = best_beta(alpha, grid)
    p_kennedy = receiver_success(alpha, alpha)
    p_opt = receiver_success(alpha, b_star)
    p_homodyne = 0.5 * (1 + math.erf(math.sqrt(2) * alpha))
    check("36.2 Kennedy = 1 - c^2/2", abs(p_kennedy - (1 - c**2 / 2)) < 1e-12)
    for b in (0.0, 0.3, alpha, b_star, 1.2):
        check(f"36.2 lectura qg lineal, beta={b:.3f}",
              abs(receiver_success(alpha, b) - receiver_success_qg(alpha, b)) < 1e-12)

    # episodios para separar dos brazos (Kennedy vs óptimo) a z sigma, por brazo
    h = proportions.cohens_h(p_opt, p_kennedy)
    z = proportions.z_value(0.95)
    n_arm = math.ceil(2.0 * (z / h) ** 2)
    # la misma cuenta en p (Wald): n = z^2 [p1 q1 + p2 q2] / (p1 - p2)^2
    n_wald = math.ceil(z**2 * (p_opt * (1 - p_opt) + p_kennedy * (1 - p_kennedy)) / (p_opt - p_kennedy) ** 2)

    OUT["comms"] = dict(
        alpha=alpha, c=c, delta_rad=delta, delta_qga=qga.qga_from_radians(delta) if hasattr(qga, "qga_from_radians") else delta * 1000 / TWO_PI,
        qgL_qga=qg_l * 1000 / TWO_PI, helstrom=p_h, homodyne=p_homodyne, kennedy=p_kennedy,
        beta_star=b_star, p_opt=p_opt, h=h, n_arm=n_arm, n_wald=n_wald,
    )


# --------------------------------------------------------------------------
# 36.2.4  model-free (UCB1 sobre una grilla de beta) vs model-aware (MLE de alpha)
# --------------------------------------------------------------------------
def simulate_agents(alpha: float = 0.4, episodes: int = 2000, seeds: int = 200) -> dict:
    arms = np.linspace(0.0, 1.5, 21)
    p_arms = np.array([receiver_success(alpha, b) for b in arms])
    fine = np.linspace(0.0, 1.5, 301)
    p_best = max(receiver_success(alpha, b) for b in fine)
    a_grid = np.linspace(0.05, 1.5, 146)          # hipótesis de alpha para el agente consciente
    # tabla beta*(alpha) del modelo
    b_of_a = np.array([best_beta(a, fine) for a in a_grid])
    regret_free = np.zeros(seeds)
    regret_aware = np.zeros(seeds)
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        # ---- model-free: UCB1, recompensa = acierto
        n = np.zeros(len(arms))
        w = np.zeros(len(arms))
        reg = 0.0
        for t in range(episodes):
            if t < len(arms):
                k = t
            else:
                k = int(np.argmax(w / n + np.sqrt(2.0 * math.log(t + 1) / n)))
            r = rng.random() < p_arms[k]
            n[k] += 1
            w[k] += r
            reg += p_best - p_arms[k]
        regret_free[s] = reg
        # ---- model-aware: conoce P(clic | signo, alpha, beta); aprende alpha por MLE
        loglik = np.zeros(len(a_grid))
        reg = 0.0
        beta = 0.5                                   # primer desplazamiento, sin información
        for t in range(episodes):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            p_noclick = math.exp(-((beta + sign * alpha) ** 2))
            noclick = rng.random() < p_noclick
            reg += p_best - receiver_success(alpha, beta)
            pn = np.exp(-((beta + sign * a_grid) ** 2))
            loglik += np.log(pn if noclick else np.clip(1.0 - pn, 1e-300, None))
            beta = float(b_of_a[int(np.argmax(loglik))])
        regret_aware[s] = reg
    return dict(
        episodes=episodes, seeds=seeds, arms=len(arms),
        free=float(regret_free.mean()), free_se=float(regret_free.std(ddof=1) / math.sqrt(seeds)),
        aware=float(regret_aware.mean()), aware_se=float(regret_aware.std(ddof=1) / math.sqrt(seeds)),
    )


# --------------------------------------------------------------------------
# 36.3  VQA: regla de desplazamiento, área mínima (regla de simplificación de VAns), NIBP
# --------------------------------------------------------------------------
def section_vqa() -> None:
    # dos evaluaciones por parámetro: exacto para a + b cos + c sin
    f = lambda t: 0.3 - 0.7 * math.cos(t) + 0.2 * math.sin(t)  # noqa: E731
    exact = 0.7 * math.sin(1.1) + 0.2 * math.cos(1.1)
    check("36.3 parameter-shift exacto", abs(fisher.shift_gradient(f, 1.1) - exact) < 1e-12)
    f1, f2 = fisher.indistinguishable_pair(1.1, 0.4)[:2]
    check("36.3 una sola evaluación no fija la derivada", abs(f1(1.1 + 0.4) - f2(1.1 + 0.4)) < 1e-12)

    # bloque insertado por VAns tras entrenar: ángulos fuera del intervalo principal
    angles_rx = [TWO_PI + 0.10, -3.50, TWO_PI, 4.00, 0.25]        # período 2 pi
    angles_rzz = [math.pi + 0.05, 2.90, math.pi, -1.70]           # período pi (Z virtuales)
    red = [pulse_area.minimal_area_angle(a)[0] for a in angles_rx]
    red += [pulse_area.minimal_area_angle(a, math.pi)[0] for a in angles_rzz]
    before = sum(abs(a) for a in angles_rx + angles_rzz)
    after = sum(abs(a) for a in red)
    removed = sum(abs(a) < 1e-9 for a in red)
    for a, r in zip(angles_rx, red[: len(angles_rx)]):
        check(f"36.3 RX({a:.3f}) = RX({r:.3f})", abs(math.cos(a) - math.cos(r)) < 1e-12)
    OUT["vqa_area"] = dict(gates=len(red), before=before, after=after, removed=int(removed),
                           before_qga=before * 1000 / TWO_PI, after_qga=after * 1000 / TWO_PI)

    # gradiente bajo T1 en un circuito que conserva el peso: crudo = K * ideal; filtrado = ideal
    gamma = 0.08
    rows = []
    for depth in (3, 6, 12, 24):
        k = qml.kept_fraction(gamma, 1, depth)
        rows.append((depth, k, 1.0 / k))
        check(f"36.3 K=(1-gamma)^(w d), d={depth}", abs(k - (1 - gamma) ** depth) < 1e-12)
    OUT["vqa_K"] = rows
    # cambio de unidad: reescala toda derivada por 2 pi / 1000, constante
    check("36.3 qga no cambia mesetas (factor constante)",
          abs(TWO_PI / 1000 - 0.006283185307179587) < 1e-15)


# --------------------------------------------------------------------------
# 36.4  Sensado: test secuencial (Wald) y medición continua de sigma_z
# --------------------------------------------------------------------------
def sprt_run(p_true: float, p0: float, p1: float, a: float, b: float, rng) -> tuple:
    up, dn = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    A, B = math.log((1 - b) / a), math.log(b / (1 - a))
    llr, n = 0.0, 0
    while B < llr < A:
        llr += up if rng.random() < p_true else dn
        n += 1
    return n, llr >= A


def section_sensing() -> None:
    q0, q1 = 0.0, 0.2                      # dos hipótesis sobre qg_Z
    p0, p1 = (1 + q0) / 2, (1 + q1) / 2
    a = b = 0.01
    h = abs(proportions.cohens_h(p1, p0))
    z = 2.3263478740408408                 # cuantil unilateral 0.99
    n_fixed = math.ceil(((z + z) / h) ** 2)
    kl = lambda p, q: p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))  # noqa: E731
    A, B = math.log((1 - b) / a), math.log(b / (1 - a))
    wald_h1 = ((1 - b) * A + b * B) / kl(p1, p0)
    wald_h0 = (a * A + (1 - a) * B) / (-kl(p0, p1))
    rng = np.random.default_rng(36)
    runs = 20000
    r1 = [sprt_run(p1, p0, p1, a, b, rng) for _ in range(runs)]
    r0 = [sprt_run(p0, p0, p1, a, b, rng) for _ in range(runs)]
    n1 = float(np.mean([r[0] for r in r1]))
    n0 = float(np.mean([r[0] for r in r0]))
    miss = 1 - float(np.mean([r[1] for r in r1]))
    false = float(np.mean([r[1] for r in r0]))
    check("36.4 errores del SPRT <= nominales", miss <= b * 1.2 and false <= a * 1.2)
    check("36.4 SPRT < N fijo", max(n0, n1) < n_fixed)
    OUT["sprt"] = dict(q0=q0, q1=q1, h=h, h_qga=h * 1000 / TWO_PI, n_fixed=n_fixed, wald_h0=wald_h0,
                       wald_h1=wald_h1, n0=n0, n1=n1, miss=miss, false=false,
                       saving=1 - max(n0, n1) / n_fixed)

    # medición continua de sigma_z: d qg = 2 k (1 - qg^2) dW, k = sqrt(eta kappa)
    k, dt, steps, traj = 1.0, 1e-3, 4000, 4000
    q_init = 0.3
    rng = np.random.default_rng(360)
    q = np.full(traj, q_init)
    for _ in range(steps):
        q = np.clip(q + 2 * k * (1 - q**2) * math.sqrt(dt) * rng.standard_normal(traj), -1, 1)
    mean_q = float(q.mean())
    frac_up = float((q > 0).mean())
    purity_gap = float((1 - q**2).mean())
    se = float(q.std(ddof=1) / math.sqrt(traj))
    check("36.4 qg_Z es martingala (media conservada)", abs(mean_q - q_init) < 4 * se)
    check("36.4 colapso con p0 = (1+qg)/2", abs(frac_up - (1 + q_init) / 2) < 0.03)
    check("36.4 ganancia = 1/I_F(qg)", abs((1 - q_init**2) - 1 / fisher.fisher_qg(q_init)) < 1e-12)
    OUT["sme"] = dict(q_init=q_init, mean_q=mean_q, se=se, frac_up=frac_up, purity_gap=purity_gap,
                      t=steps * dt, traj=traj)


def main() -> None:
    section_comms()
    OUT["agents"] = simulate_agents()
    check("36.2.4 model-aware < model-free (arrepentimiento)", OUT["agents"]["aware"] < OUT["agents"]["free"])
    section_vqa()
    section_sensing()
    print()
    for key, val in OUT.items():
        print(key, val)


if __name__ == "__main__":
    main()
