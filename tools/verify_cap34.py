"""tools/verify_cap34.py — Cap. 34: límites estructurales y resultados negativos, con código exacto."""
import numpy as np
from qang.qlimits import *
from qang.qledger import load_ledger, summarize, primary_status

rng = np.random.default_rng(34)

# [1] estado de grafo (anillo de 6) = producto en Z; los estabilizadores sí los separan (base X)
n = 6
g, pl = ring_graph_state(n), plus_state(n)
d = np.max(np.abs(z_distribution(g) - z_distribution(pl)))
assert d == 0.0
pg = z_distribution(g)
assert abs(mean_qgz(pg, n)) < 1e-15 and abs(qgs_per_qubit(pg, n) - 1) < 1e-12
sg = [stabilizer_expectation(g, i, n) for i in range(n)]
sp = [stabilizer_expectation(pl, i, n) for i in range(n)]
assert np.allclose(sg, 1.0) and np.allclose(sp, 0.0)
print(f"[1] anillo n=6: |P_Z(grafo) − P_Z(producto)| = {d:.1f}; qg_Z medio 0 m-qg, qg_S 1.00000; "
      f"⟨X_i Z_(i−1) Z_(i+1)⟩ = {sg[0]:+.1f} (grafo) frente a {sp[0]:+.1f} (producto)")

# [2] qg_S no monótona bajo T1; qg_Z medio sí (exacto); ruido simétrico: qg_S monótona
n, K = 4, 24
gam = np.linspace(0.0, 0.5, 51)
nonmono_s, mono_z = 0, 0
ex = None
for k in range(K):
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    # peso de Hamming sesgado para que el estado inicial sea de entropía alta (mezcla de pesos)
    p = z_distribution(psi)
    s, z = t1_noise_scan(p, n, gam)
    up_down = (np.max(s) - s[0] > 1e-3) and (np.max(s) - s[-1] > 1e-3)
    nonmono_s += int(up_down)
    mono_z += int(is_monotone(z, True))
    if up_down and ex is None:
        i_max = int(np.argmax(s))
        ex = (k, s[0], gam[i_max], s[i_max], gam[-1], s[-1], z[i_max], z[-1])
print(f"[2] {K} estados de 4 qubits, γ∈[0,0.5]: qg_S sube y baja en {nonmono_s}/{K}; "
      f"qg_Z medio monótono creciente en {mono_z}/{K} (exacto: 1−2(1−γ)P1)")
assert mono_z == K
k, s0, gm, sm, gl, sl, zm, zl = ex
print(f"    ejemplo (estado {k}): qg_S {s0:.3f} → {sm:.3f} (γ={gm:.2f}) → {sl:.3f} (γ={gl:.2f}); "
      f"qg_Z medio {1000*zm:+.0f} → {1000*zl:+.0f} m-qg")
assert nonmono_s >= 1 and sm > s0 and sm > sl

# [3] trampa del rango de arccos
ts = -0.5
e_clip, th_clip = best_cost_clipped(ts)
assert abs(cost_cos(ts, ts) + 1.0) < 1e-15 and abs(e_clip + np.cos(0.5)) < 1e-9 and th_clip == 0.0
print(f"[3] E(θ)=−cos(θ−θ*), θ*=−0.5: mínimo exacto {cost_cos(ts, ts):+.4f} (θ<0); con θ=arccos(qg_Z)∈[0,π] "
      f"el mejor valor es {e_clip:+.4f} en θ=0 → pérdida {e_clip + 1:.4f}")

# [4] conservación de peso, compuerta por compuerta
n = 4
rows4 = []
for name, gates in (("QFT 4q", qft_gates(n)), ("GHZ 4q", ghz_gates(n)), ("RBS escalera 4q", unary_rbs_gates(n, [0.3, 0.7, 1.1]))):
    U = circuit_unitary(n, gates)
    assert np.allclose(U.conj().T @ U, np.eye(2 ** n), atol=1e-10)
    rows4.append((name, preserves_weight(U, n), len(weight_breakers(n, gates))))
d4 = {r[0]: r for r in rows4}
assert d4["QFT 4q"][1:] == (False, 4) and d4["GHZ 4q"][1:] == (False, 4) and d4["RBS escalera 4q"][1:] == (True, 0)
print("[4] " + "; ".join(f"{a}: conserva={b}, compuertas que lo rompen={c}" for a, b, c in rows4))

# [5] red de peso 1 = forma cuadrática
n = 5
err = 0.0
for _ in range(20):
    th = rng.uniform(-np.pi, np.pi, n - 1)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    O = weight1_orthogonal(n, th)
    assert np.allclose(O @ O.T, np.eye(n), atol=1e-12)
    err = max(err, np.max(np.abs(weight1_qgz(O, v) - simulate_qgz_weight1(n, th, v))))
assert err < 1e-13
print(f"[5] cargador unario + red RBS, n=5, 20 redes al azar: |forma cuadrática − simulación completa| ≤ {err:.1e} "
      f"(coste O(n²) frente a 2^n)")

# [6] error dentro del sector: el filtro lo deja pasar
r = in_sector_error(0.3)
assert abs(r["error_retained"] - 1.0) < 1e-12 and abs(r["ideal_retained"] - 1.0) < 1e-12 and abs(r["shift_mqg"]) > 100
print(f"[6] RBS(π/2+0.3): retención del filtro {r['error_retained']:.1f}; qg_Z del qubit 0: "
      f"{r['ideal_qgz0_mqg']:+.1f} → {r['error_qgz0_mqg']:+.1f} m-qg (desvío {r['shift_mqg']:+.1f} m-qg, invisible al filtro)")

# [7] inventario de negativos en el libro de cuentas (Cap. 7–33)
rows = load_ledger("book", 7, 33)
inv = negative_inventory(rows)
S = summarize(rows)
assert S.get("measured gain", 0) == 0
print(f"[7] libro de cuentas Cap. 7–33: {len(rows)} submedidas; "
      + ", ".join(f"{k}: {len(v)}" for k, v in inv.items()) + f"; measured gain: {S.get('measured gain', 0)}")
for k, v in inv.items():
    for cap, i in v:
        print(f"      {k:11s} Cap.{cap:02d} {i}")

# [8] una sola base de lectura: ninguna función de P_Z certifica entrelazamiento (teorema del gemelo diagonal)
rowsS = []
for n_, name, psi in ((2, "arista G2", path_graph_state(2)), (3, "camino G3", path_graph_state(3))):
    pz = z_distribution(psi); pp = z_distribution(plus_state(n_))
    assert np.allclose(pz, pp) and abs(total_correlation_z(pz, n_)) < 1e-12 and abs(shannon_bits(pz) - n_) < 1e-12
    lp = local_purities(np.outer(psi, psi.conj()), n_)
    rowsS.append(f"{name}: H_Z={shannon_bits(pz):.0f} bit, C_Z={total_correlation_z(pz, n_):.0f}, P={purity(np.outer(psi, psi.conj())):.0f}, P_i={lp[0]:.2f}")
    assert np.allclose(lp, 0.5)
g2 = path_graph_state(2); ptm = partial_transpose_min_eig(np.outer(g2, g2.conj()), 2)
assert abs(ptm + 0.5) < 1e-12
print("[8] misma P_Z uniforme que |+>^n y I/2^n: " + "; ".join(rowsS) + f"; |+>^n: P_i=1; I/2^n: P=1/2^n; PT mín. G2={ptm:+.3f}")
p0 = np.array([0.6, 0.15, 0.2, 0.05]); lo, hi = purity_range_from_z(p0)
assert abs(purity(diagonal_twin(p0)) - lo) < 1e-12 and abs(purity(np.outer(pure_state_from_z(p0), pure_state_from_z(p0).conj())) - hi) < 1e-12
print(f"    pureza compatible con p=(.6,.15,.2,.05): [{lo:.4f}, {hi:.1f}], extremos alcanzados por σ_p y |ψ_p⟩")
a_ = np.array([np.sqrt(.75), np.sqrt(.25)]); b_ = np.array([np.sqrt(.8), np.sqrt(.2)])
pr = np.kron(a_, b_); cp = cz_matrix(2, 0, 1) @ pr
assert np.allclose(z_distribution(pr), z_distribution(cp)) and abs(total_correlation_z(z_distribution(pr), 2)) < 1e-12
lpc = local_purities(np.outer(cp, cp.conj()), 2); ptc = partial_transpose_min_eig(np.outer(cp, cp.conj()), 2)
assert abs(lpc[0] - 0.76) < 1e-12 and abs(ptc + 0.346410162) < 1e-8
print(f"    sesgado (z=(0.5,0.6)): producto y CZ·producto con la misma P_Z, H_Z={shannon_bits(z_distribution(pr)):.6f}, C_Z=0; P_i=1 frente a {lpc[0]:.2f}; PT mín. {ptc:+.6f}")

# [9] C_Z no es entrelazamiento: falsos negativos (grafo) y falsos positivos (mezclas separables)
gh = ghz_state_n(3); pgh = z_distribution(gh)
sep = np.zeros((8, 8)); sep[0, 0] = sep[7, 7] = 0.5
assert np.allclose(np.real(np.diag(sep)), pgh)
ev = np.zeros(8); ev[[0, 3, 5, 6]] = 0.25
mut = []
for (qa, qb) in ((0, 1), (0, 2), (1, 2)):
    pab = np.zeros(4)
    for x in range(8):
        pab[(int(bit(np.array([x]), qa, 3)[0]) << 1) | int(bit(np.array([x]), qb, 3)[0])] += ev[x]
    mut.append(total_correlation_z(pab, 2))
assert np.allclose(mut, 0.0, atol=1e-12) and abs(total_correlation_z(ev, 3) - 1) < 1e-12 and abs(shannon_bits(ev) - 2) < 1e-12
print(f"[9] GHZ₃ y mezcla 000/111: misma P_Z, C_Z={total_correlation_z(pgh, 3):.0f} bits (máximo), la mezcla es separable (P=1/2); "
      f"paridad par: H_Z=2, C_Z=1, información mutua por pares {max(mut):.1f}")
bq = apply_local(path_graph_state(2), 2, {1: HADAMARD}); pbq = z_distribution(bq)
assert abs(shannon_bits(pbq) - 1) < 1e-12 and abs(total_correlation_z(pbq, 2) - 1) < 1e-12 and partial_transpose_min_eig(np.outer(bq, bq.conj()), 2) < -0.49
print(f"    (I⊗H)|G₂⟩ = Φ⁺: H_Z 2 → {shannon_bits(pbq):.0f}, C_Z 0 → {total_correlation_z(pbq, 2):.0f}, entrelazamiento y pureza intactos")
cr = [relative_entropy_coherence_z(np.outer(g2, g2.conj())), relative_entropy_coherence_z(np.eye(4) / 4)]
assert abs(cr[0] - 2) < 1e-12 and abs(cr[1]) < 1e-12
print(f"    H_Z = coherencia C_rel,Z sólo si es puro: G₂ (puro) {cr[0]:.0f} = H_Z=2; I/4 (mixto) {cr[1]:.0f} frente a H_Z=2")

# [10] testigo de dos ajustes para el camino de 3 qubits
W3 = witness_path(3); g3 = path_graph_state(3)
w_g, w_p, w_m = (witness_value(W3, np.outer(g3, g3.conj())), witness_value(W3, np.outer(plus_state(3), plus_state(3).conj())), witness_value(W3, np.eye(8) / 8))
assert abs(w_g + 1) < 1e-12 and abs(w_p - 2) < 1e-12 and abs(w_m - 2) < 1e-12
rad = witness_radius(3, 1000, 0.05); assert -1 + rad < 0 and abs(rad - 0.293556) < 1e-6
ts = [witness_value(W3, dephased_path3(t)) + 0.0 for t in (0, 1 / 3, 1)]
assert np.allclose(ts, [-1, 0, 2])
print(f"[10] W′₃ = 2I − XZI − ZXZ − IZX: grafo {w_g:+.0f}, producto {w_p:+.0f}, I/8 {w_m:+.0f}; XZX y ZXZ (2 ajustes), 1000 disparos c/u: radio 95 % {rad:.6f} → certifica ({-1 + rad:+.3f} < 0)")
print(f"     desfase final ρ_t: P_Z fija, W′₃ = {ts[0]:+.0f}, {abs(ts[1]):.0f}, {ts[2]:+.0f} en t = 0, 1/3, 1")

# [11] relajación de un producto: entropía y pureza no monótonas, siempre producto
amp = [product_amp_damped(x) for x in (0.0, 0.5, 1.0)]
assert np.allclose([a["H_Z"] for a in amp], [0, 2, 0]) and np.allclose([a["P"] for a in amp], [1, 0.25, 1]) and all(abs(a["C_Z"]) < 1e-12 for a in amp)
hs_, ps_ = ", ".join(f"{abs(x['H_Z']):.0f}" for x in amp), ", ".join(f"{x['P']:.2f}" for x in amp)
print(f"[11] producto bajo amortiguación γ=0, ½, 1: H_Z = {hs_} bits; P = {ps_}; C_Z=0 siempre")

# [12] presupuesto de disparos y sesgo de muestra finita (cotas suficientes, no necesarias)
Ns = (sufficient_shots_entropy(2, .05, .05, "H"), sufficient_shots_entropy(2, .05, .05, "C"), sufficient_shots_entropy(3, .05, .05, "H"), sufficient_shots_entropy(3, .05, .05, "C"))
assert hoeffding_shots(.05, .05) == 2952 and Ns == (133966, 1346071, 253199, 3973513)
print(f"[12] ε=0.05 bit, δ=0.05: una media en [−1,1] N={hoeffding_shots(.05, .05)}; H_Z n=2/3 N={Ns[0]}/{Ns[2]}; C_Z n=2/3 N={Ns[1]}/{Ns[3]} (cotas suficientes, holgadas)")
rows12 = []
for N_ in (100, 10000):
    mh, mc = plugin_study_uniform(3, N_, 1000, 34)
    pred = plugin_cz_bias_uniform(3, N_)
    assert mc > 0 and abs(mc - pred) < 0.35 * pred + 1e-4
    rows12.append(f"N={N_}: Ĥ_Z={mh:.4f}, Ĉ_Z={mc:.4f} (teoría {pred:.4f})")
print("     n=3, uniforme independiente, 1000 repeticiones, semilla 34 — exacto H_Z=3, C_Z=0: " + "; ".join(rows12))
print("verify_cap34: OK")
