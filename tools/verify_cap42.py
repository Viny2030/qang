"""tools/verify_cap42.py — Cap. 42: lo que de la teoría de cuerdas se puede leer con qang (identidades exactas + un análogo de red)."""
import numpy as np
from qang.qstring import *

rng = np.random.default_rng(42)

# [1] cuerda bosónica de luz-cono: degeneraciones, dimensión crítica, nivel sin masa cerrado
deg = colored_partitions(5, 24)
assert deg == [1, 24, 324, 3200, 25650, 176256]
assert critical_dimension() == 26 and [D for D in range(3, 60) if zero_point_energy(D) == 1] == [26]
assert open_mass2_alpha(0) == -1 and open_mass2_alpha(1) == 0 and open_mass2_alpha(1, D=25) != 0
dec = closed_massless_decomposition(24)
assert dec == {"total": 576, "graviton": 299, "kalb_ramond": 276, "dilaton": 1} and 299 + 276 + 1 == 576 == deg[1] ** 2
qb = [int(np.ceil(np.log2(d))) for d in deg[1:]]
assert qb == [5, 9, 12, 15, 18]
print(f"[1] d_N (24 osciladores), N=0..5: {deg}; D crítica = {critical_dimension()} (única en 3..59); α'M²(N=0,1) = {open_mass2_alpha(0)}, {open_mass2_alpha(1)}; "
      f"cerrada N=Ñ=1: {dec['total']} = {dec['graviton']} (gravitón) + {dec['kalb_ramond']} (Kalb–Ramond) + {dec['dilaton']} (dilatón); qubits para rotular el nivel N=1..5: {qb}")

# [2] el nivel es una lectura Z lineal
n = 10
dp = distinct_partitions(n)
sizes = [level_sector_size(N, n) for N in range(n + 1)]
assert sizes == dp
n6 = 6
p = rng.dirichlet(np.ones(2 ** n6))
m = mode_readings_mqg(p, n6)
assert abs(level_from_readings(m) - level_from_z(p, n6)) < 1e-12
pu = np.full(2 ** n6, 1 / 2 ** n6)
assert abs(level_variance(pu, n6) - sum(k * k for k in range(1, n6 + 1)) / 4) < 1e-12
ns = level_shots(pu, n6, 0.1)
print(f"[2] tamaños del sector de nivel N=0..{n} (n={n} qubits): {sizes} = particiones en partes distintas; "
      f"|⟨N⟩ por modos (m-qg) − ⟨N⟩ por histograma| = {abs(level_from_readings(m) - level_from_z(p, n6)):.1e}; "
      f"uniforme n=6: Var N = {level_variance(pu, n6):.2f}, disparos para σ_N=0.1 (95 %): {ns}")

# [3] filtro de nivel bajo T1 y sus dos puntos ciegos
x = 0b101101  # modos 1, 3, 4, 6 ocupados (n=6)
for g in (0.1, 0.3):
    r = level_filter_damped(x, 6, g)
    assert abs(r["kept"] - r["kept_closed"]) < 1e-12 and r["level_target"] == 14
    assert abs(r["mean_level_raw"] - (1 - g) * 14) < 1e-12
rs = [level_filter_damped(x, 6, g) for g in (0.1, 0.3)]
b = level_vs_weight_blindness(4)
assert b["level_preserving"] == (3, 3, 1, 2) and b["weight_preserving"] == (1, 2, 1, 1)
print(f"[3] estado nivel 14 (modos 1,3,4,6), T1: γ=0.1 retención {rs[0]['kept']:.4f} = (1−γ)⁴; γ=0.3 retención {rs[1]['kept']:.4f}; "
      f"nivel medio crudo {rs[1]['mean_level_raw']:.2f} frente a 14 (todo decaimiento baja el nivel)")
print(f"    puntos ciegos: {{3}}↔{{1,2}} conserva el nivel {b['level_preserving'][0]} y cambia el peso {b['level_preserving'][2]}→{b['level_preserving'][3]} (filtro de nivel no lo ve); "
      f"{{1}}→{{2}} conserva el peso y cambia el nivel {b['weight_preserving'][0]}→{b['weight_preserving'][1]} (filtro de peso no lo ve)")

# [4] T-dualidad y estados sin masa
Rs = (0.5, 0.7, 1.0, 1.3, 2.0, 3.0)
for R in Rs:
    assert t_dual_equal(R) and massless_states(R) == massless_states(1.0 / R)
cnt = {R: massless_states(R) for R in Rs}
assert cnt[0.7] == cnt[1.3] == 576 and cnt[1.0] == 676 and cnt[1.0] - 576 == 4 * 24 + 4
bd = massless_breakdown(1.0)
assert sum(o[4] for o in bd) == 100 and sorted(sum(o[4] for o in bd if o[2] + o[3] == 1) for _ in [0]) == [96] and sum(o[4] for o in bd if o[2] + o[3] == 0) == 4
print("[4] espectro (h, h̄) invariante bajo R → α'/R en R = " + ", ".join(str(R) for R in Rs) + "; estados sin masa: "
      + ", ".join(f"R={R}: {cnt[R]}" for R in Rs) + "; en el radio autodual 676 = 576 + 4·24 (vectores, n=±1,w=±1) + 4 (escalares, (±2,0),(0,±2))")

# [5] radio leído con una sola base Z: varianza del peso de un bloque en una cadena XXZ (finito, SciPy)
rows = []
est = {}
for d, tol in ((0.0, 0.03), (-0.5, 0.03), (0.5, 0.15), (0.7071, 0.15)):
    st, pz = xxz_ground_distribution(20, d)
    k = luttinger_k_from_variance(st, pz, 20)
    ke = luttinger_k_exact(d)
    est[d] = (k, ke)
    assert abs(k - ke) / ke < tol, (d, k, ke)
    rows.append(f"Δ={d:+.4f}: K̂={k:.3f} (exacto {ke:.3f}, error {100 * abs(k - ke) / ke:.1f} %)")
prod = est[-0.5][0] * est[0.7071][0]
assert abs(prod - 1.0) < 0.17
e1, e2 = dual_exponents(1.5)
assert abs(e1 * e2 - 1.0) < 1e-12
print("[5] anillo XXZ de 20 sitios, ℓ pares ≥ 4, un histograma Z: " + "; ".join(rows))
print(f"    par T-dual K=3/2 (Δ=−0.5) y K=2/3 (Δ=cos π/4): producto leído {prod:.3f} frente a 1 (exacto en el continuo; exponentes 2K y 1/(2K) → producto 1)")
print("verify_cap42: OK")
