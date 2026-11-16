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
print("verify_cap34: OK")
