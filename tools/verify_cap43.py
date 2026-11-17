"""tools/verify_cap43.py — Cap. 43: robótica cuántica con qang (identidades exactas, presupuestos de recursos y simulaciones de disparos).
Solo NumPy + qang. Uso: python tools/verify_cap43.py   (si algo cambia, el script se detiene)."""
import numpy as np

from qang import probability as pr, qga, statistics as st
from qang.qrobot import *

rng = np.random.default_rng(43)
links = [1.0, 0.8, 0.5]


def shots_z(theta, n, rng):
    return rng.binomial(n, np.cos(theta / 2) ** 2)


# [1] cinemática de referencia
for _ in range(200):
    t = rng.uniform(-2.5, 2.5, 3)
    J = planar_jacobian(t, links)
    Jn = np.stack([(planar_fk(t + e, links) - planar_fk(t - e, links)) / 2e-6 for e in 1e-6 * np.eye(3)], axis=1)
    assert np.allclose(J, Jn, atol=1e-6)
ok = 0
for _ in range(200):
    t1, t2 = rng.uniform(-2.5, 2.5), rng.uniform(0.1, 2.5)
    p = planar_fk([t1, t2], links[:2])
    for el in (1, -1):
        s = two_link_ik(p, links[0], links[1], el)
        assert s is not None and np.allclose(planar_fk(s, links[:2]), p, atol=1e-9)
        ok += 1
assert two_link_ik([5.0, 0.0], 1.0, 0.8) is None
print(f"[1] FK/jacobiana (diferencias finitas) y IK cerrada de 2 eslabones: {ok} soluciones comprobadas; blanco fuera de alcance → None")

# [2] la lectura de una articulación
r = joint_readings(0.7)
assert abs(r["qg_Z"] - np.cos(0.7)) < 1e-15 and abs(r["m_Z"] - pr.milliqang_from_theta(0.7)) < 1e-12
assert abs(joint_from_readings(np.cos(-0.7), np.sin(-0.7)) + 0.7) < 1e-15
assert abs(np.arccos(np.cos(-0.7)) - 0.7) < 1e-15            # trampa del signo: con qg_Z solo, θ y −θ coinciden
N = 2000
for th in (0.6, 1.0, 1.5708, 2.2, 2.6):
    assert abs(st.propagated_theta_std(th, N) - np.sqrt(joint_variance_z(N))) < 1e-12
    tr = 5000
    kz = rng.binomial(N, np.cos(th / 2) ** 2, tr)
    kx = rng.binomial(N, (1 + np.sin(th)) / 2, tr)
    zz, xx = 2 * kz / N - 1, 2 * kx / N - 1
    v_z = np.var(np.arccos(np.clip(zz, -1, 1)))
    v_a = np.var(np.arctan2(xx, zz))
    assert abs(v_z / joint_variance_z(N) - 1) < 0.10, (th, v_z * N)
    assert abs(v_a / joint_variance_atan2(th, N) - 1) < 0.10, (th, v_a * N)
    assert joint_variance_atan2(th, N) >= joint_crb_two_bases(N) - 1e-15
assert abs(joint_variance_atan2(np.pi / 4, N) - joint_crb_two_bases(N)) < 1e-15
assert abs(joint_sigma_qga(400) - qga.angle_sigma_qga(400)) < 1e-12 and abs(joint_sigma_qga(1) - 159.1549) < 1e-3
print(f"[2] σ_θ(Z) = 1/√N exacta a primer orden (N={N}, 5 ángulos, simulada ±10 %); atan2: (cos⁴+sin⁴)/N ≥ CRB 1/(2N), igualdad solo en 45°; "
      f"σ = {joint_sigma_qga(400):.2f} qga con 400 disparos; arccos(qg_Z) no distingue −0,7 de +0,7")

# [3] de la articulación al efector
th = np.array([0.6, 1.1, 0.9])
for N in (500, 4000):
    pos0 = planar_fk(th, links)
    errs = []
    for _ in range(4000):
        est = np.array([np.arccos(np.clip(2 * shots_z(t, N, rng) / N - 1, -1, 1)) for t in th])
        errs.append(np.sum((planar_fk(est, links) - pos0) ** 2))
    emp = np.sqrt(np.mean(errs))
    assert abs(emp / end_effector_rms(th, links, N) - 1) < 0.06, (N, emp, end_effector_rms(th, links, N))
eps = 0.01
Nn = shots_for_end_effector(th, links, eps)
assert end_effector_rms(th, links, Nn) <= eps + 1e-12 and end_effector_rms(th, links, Nn - 1) > eps
print(f"[3] RMS del efector = √(tr(JJᵀ)/N) (simulado, N=500 y 4000, ±6 %); para ε=0,01 hacen falta N={Nn} disparos por articulación "
      f"(= {Nn * len(th)} en total)")

# [4] tiempo real (parámetros ilustrativos, no mediciones)
assert shots_per_cycle(100, 1e-3) == 10 and abs(realtime_joint_sigma(100, 1e-3) - 10 ** -0.5) < 1e-15
assert shots_per_cycle(100, 1e-3, 0.002) == 8 and realtime_joint_sigma(1000, 1e-3, 0.002) == float("inf")
assert abs(max_control_rate(0.01, 1e-3) - 0.1) < 1e-12
th_pf = two_link_ik((1.2, 0.9), 1.0, 0.8)
pf = preflight(th_pf, (1.0, 0.8), 0.02, 100, 1e-3)
assert pf["shots_required"] == 7225 and pf["shots_available"] == 10 and pf["feasible"] == 0.0
assert preflight(th_pf, (1.0, 0.8), 0.02, 0.1, 1e-3)["feasible"] == 1.0
s100 = realtime_joint_sigma(100, 1e-3)
print(f"[4] ciclo de 100 Hz con t_shot = 1 ms: N={shots_per_cycle(100, 1e-3)}, σ_θ = {s100:.3f} rad = {np.degrees(s100):.1f}° = "
      f"{joint_sigma_qga(10):.1f} qga; para σ_θ=0,01 rad el lazo no pasa de {max_control_rate(0.01, 1e-3):.1f} Hz; "
      f"blanco (1,2; 0,9) con ε=0,02: requiere {pf['shots_required']:.0f} disparos por articulación y a 100 Hz hay {pf['shots_available']:.0f}")

# [5] articulación que deriva
q, N = 4e-4, 400
r_ = joint_variance_z(N)
prior, post = steady_state_variance(q, r_)
assert abs(prior - post - q) < 1e-15
T, reps, errs = 400, 200, []
for _ in range(reps):
    x = np.pi / 2
    xs_true, ys = [], []
    for _t in range(T):
        x += rng.normal(0, np.sqrt(q))
        xs_true.append(x)
        ys.append(np.arccos(np.clip(2 * shots_z(x, N, rng) / N - 1, -1, 1)))
    xh, _ = kalman_scalar_track(ys, r_, q, x0=np.pi / 2, p0=post)
    errs.append(np.array(xh[100:]) - np.array(xs_true[100:]))
emp = np.mean(np.array(errs) ** 2)
assert abs(emp / post - 1) < 0.20, (emp, post)
raw = np.mean((np.array(ys) - np.array(xs_true)) ** 2)
print(f"[5] Kalman escalar: varianza a posteriori estacionaria {post:.2e} rad² (simulada {emp:.2e}); sin filtro, 1/N = {r_:.2e}; "
      f"σ = {np.sqrt(post):.4f} rad frente a {np.sqrt(r_):.4f} rad")

# [6] Grover para elegir trayectorias
for Nc, M in ((16, 1), (64, 3), (1024, 1), (1000, 7)):
    k = grover_optimal_iterations(Nc, M)
    ps = grover_success(Nc, M, k)
    assert abs(ps - grover_simulate(Nc, list(range(M)), k)) < 1e-12
assert grover_optimal_iterations(1024, 1) == 25 and abs(grover_success(1024, 1, 25) - 0.99946) < 1e-5
Nc, M = 1000, 5
cl = []
for _ in range(20000):
    perm = rng.permutation(Nc)
    cl.append(int(np.argmax(perm < M)) + 1)
assert abs(np.mean(cl) / classical_expected_queries(Nc, M) - 1) < 0.03
big = grover_query_ratio(2 ** 20, 1)
assert abs(big / (2 / np.pi * np.sqrt(2 ** 20)) - 1) < 0.01
print(f"[6] Grover: sin²((2k+1)θ0) = simulación explícita (4 casos); N=1024, M=1: k=25, P={grover_success(1024, 1, 25):.5f}; "
      f"clásico esperado (N+1)/(M+1) simulado; N=2²⁰: cociente de consultas {big:.0f} ≈ (2/π)√N; "
      f"empata si la consulta cuántica cuesta ≤ {grover_break_even_overhead(2 ** 20, 1):.0f} evaluaciones clásicas")
# [7] regla del desplazamiento de parámetro: se optimiza en θ (250 qga = π/2 rad), no en qg_Z
for th in (-1.3, 0.2, 2.0):
    assert abs(joint_parameter_shift(th) + np.sin(th)) < 1e-12
assert abs(qga.qga_from_radians(np.pi / 2) - 250.0) < 1e-12
print("[7] ∂qg_Z/∂θ = −sinθ por desplazamiento de ±250 qga (π/2 rad); el gradiente en θ no diverge en los polos, el de qg_Z sí")
print("OK — verify_cap43")
