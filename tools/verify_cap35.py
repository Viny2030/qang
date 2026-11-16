"""tools/verify_cap35.py — Cap. 35: primeras corridas en hardware real (IQM Garnet). Recalcula todo de los archivos guardados."""
import os
import sys
import numpy as np
from qang.qgarnet import *

D = os.environ.get("QANG_DATA", "examples/data")
REPO = os.environ.get("QANG_REPO", ".")

# [1] §165: neutrinos de tres sabores, trabajo 01a15b04, 8 circuitos × 300 disparos
run = load(f"{D}/iqm_neutrino_qg_run.json")
S = neutrino_summary(run)
assert abs(S["raw_error"] - run["raw_error"]) < 1e-12 and abs(S["filtered_error"] - run["filtered_error"]) < 1e-12
assert S["factor"] > 1.5 and S["emin_filtered"] == 2.6 and S["emin_raw"] == 0.8 and S["kept"] >= 0.5          # I1..I4
assert abs(S["emin_exact"] - 2.6) < 1e-12 and abs(S["kept_shots"] - 2113) < 1e-6 and run["shots"] == 300
ex, raw, fil = (np.asarray(run[k]) for k in ("exact", "raw", "filtered"))
i26 = list(run["energies"]).index(2.6)
print(f"[1] error crudo {S['raw_error']:.4f} → filtrado {S['filtered_error']:.4f} (×{S['factor']:.2f}); conservado {S['kept']:.3f} "
      f"({S['kept_shots']:.0f} de 2400 disparos)")
print(f"    mínimo de supervivencia: crudo {S['emin_raw']} GeV, filtrado {S['emin_filtered']} GeV, exacto {S['emin_exact']:.1f} GeV")
print(f"    por sabor (crudo → filtrado): " + ", ".join(f"{a:.3f}→{b:.3f}" for a, b in zip(S['per_flavour_raw'], S['per_flavour_filtered'])))
print(f"    2.6 GeV, supervivencia: exacta {ex[i26,1]:.3f}, cruda {raw[i26,1]:.3f}, filtrada {fil[i26,1]:.3f}")
assert abs(ex[i26, 1] - 0.013) < 5e-4 and abs(raw[i26, 1] - 0.173) < 5e-4 and abs(fil[i26, 1] - 0.101) < 5e-4

# [2] §168: milli-qang en un qubit de Garnet (4000 disparos por circuito)
R = load(f"{D}/iqm_milliqang_qg_run.json")["counts"]
A = load(f"{D}/iqm_milliqang_qg_analysis.json")
e0, e1 = readout_from_poles(R["pole0"], R["pole1"])
c = contrast(e0, e1)
assert abs(e0 - A["e0"]) < 1e-12 and abs(e1 - A["e1"]) < 1e-12 and abs(c - A["contrast"]) < 1e-12
print(f"[2] lectura desde los polos: e0={e0:.4f}, e1={e1:.4f}, contraste {c:.4f}")
eq = [R[f"equator{i}"] for i in range(10)]
pool = {"1": sum(x.get("1", 0) for x in eq), "0": sum(x.get("0", 0) for x in eq)}
zs, devs1 = [], []
for r in A["M1"]:
    meas = raw_mqg(pool) if r["target"] == 0 else raw_mqg(R[f"sweep{int(r['target']):+d}"])
    pred = predicted_raw_mqg(r["target"], e0, e1)
    assert abs(meas - r["measured"]) < 1e-9 and abs(pred - r["predicted"]) < 1e-9
    zs.append((meas - pred) / r["sigma"])
    assert abs(zs[-1] - r["z"]) < 1e-9
    devs1.append((r["target"], meas - pred))
ratio = equator_dispersion_ratio(eq)
assert abs(ratio - A["M2_ratio"]) < 1e-3
print("    M1 (lectura cruda − predicha, m-qg): " + ", ".join(f"{t:+.0f}: {d:+.1f}" for t, d in devs1))
print(f"    z = {[round(z, 2) for z in zs]} → dentro de 3σ: {sum(abs(z) < 3 for z in zs)}/5; M2: dispersión / σ binomial = {ratio:.3f} (se pedía 0.55–1.45)")
assert sum(abs(z) < 3 for z in zs) == 4 and max(range(5), key=lambda i: abs(zs[i])) == 2          # el punto de −500 queda a 3.9σ
ks, ss, sg = [], [], []
for k in sorted(int(x) for x in A["chains"]):
    s, sig = chain_shortfall(R[f"chain{k}"], R["pole1"], c)
    assert abs(s - A["chains"][str(k)]["s"]) < 1e-9 and abs(sig - A["chains"][str(k)]["sigma"]) < 0.05
    ks.append(k); ss.append(s); sg.append(sig)
a, chi = fit_slope_origin([k - 1 for k in ks], ss, sg)
print(f"    cadenas de X (k=3..41): déficit {ss[0]:+.1f} … {ss[-1]:+.1f} ± {sg[-1]:.1f} m-qg (M4: {ss[-1] / sg[-1]:.1f}σ); pendiente por X = {a:.2f} m-qg, χ²/gl = {chi:.2f}")
assert abs(a - 1.64) < 0.01 and abs(chi - 0.81) < 0.01 and ss[-1] / sg[-1] > 2
assert A["verdict"] == {"M1": True, "M2": False, "M3": True, "M4": True}
print(f"    veredicto guardado: {A['verdict']} (M2 falló: {ratio:.2f} está fuera de 0.55–1.45)")
assert not (0.55 <= ratio <= 1.45)

# [3] §170: E=mc² como lectura (un qubit, 2000 disparos por circuito)
Z = load(f"{D}/iqm_zitterbewegung_qg_analysis.json")
RZ = load(f"{D}/iqm_zitterbewegung_qg_run.json")["counts"]
z0, z1 = readout_from_poles(RZ["pole0"], RZ["pole1"])
assert abs(z0 - Z["e0"]) < 1e-12 and abs(z1 - Z["e1"]) < 1e-12
devs = []
for i, r in enumerate(Z["zb"]):
    m = correct_mqg(raw_mqg(RZ[f"zb{i}"]), z0, z1)
    assert abs(m - r["m"]) < 1e-9, (i, m, r["m"])
    devs.append(m - r["exact"])
print(f"[3] e0={z0:.4f}, e1={z1:.4f}; 16 puntos: lectura corregida − exacta entre {min(devs):+.1f} y {max(devs):+.1f} m-qg; veredicto guardado {Z['verdict']}")
assert all(Z["verdict"].values()) and len(devs) == 16

# [4] pre-registros frente a la historia de git (si está disponible)
for name, commit, path in (("neutrinos §165", "a74c07c", "examples/iqm_neutrino_qg.py"), ("milli-qang §168", "32f3645", "examples/iqm_milliqang_qg.py"),
                           ("E=mc² §170", "ef41991", "examples/iqm_zitterbewegung_qg.py")):
    pre = prereg_hash(f"{D}/{os.path.basename(path).replace('.py', '')}.prereg")
    got = blob_hash_at_commit(REPO, commit, path)
    if got:
        assert got == pre, (name, got[:8], pre[:8])
        print(f"[4] {name}: sha256 del script en {commit} = {got[:8]}… = pre-registro {pre[:8]}…")
    else:
        print(f"[4] {name}: historia de git no disponible; pre-registro {pre[:8]}… sin comprobar")

# [5] presupuesto de créditos (cifras de las notas §168; aritmética recalculada)
steps = [("prueba nu", 0.5), ("corrida nu", 0.5), ("prueba milli", 0.5), ("corrida milli", 16.0)]
B = budget_summary(steps, 30.0, ("prueba nu", "prueba milli"))
assert abs(B["spent"] - 17.5) < 1e-12 and abs(B["left"] - 12.5) < 1e-12 and abs(B["tests_frac"] - 1.0 / 30) < 1e-12 and abs(B["runs_frac"] - 0.55) < 1e-12
print(f"[5] gastado {B['spent']:.2f} de 30 (pruebas {100*B['tests_frac']:.1f} %, corridas {100*B['runs_frac']:.1f} %), saldo {B['left']:.2f}; con §170 (≈7.7, estimado): ≈{B['left']-7.7:.1f}")
print("verify_cap35: OK")
