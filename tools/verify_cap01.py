"""Independent check of chapter 1 of the book (the qang probability equation and the milli-qang).

The chapter's numbers come from qang.probability (§167). This script recomputes the central claims without
qang, on states prepared with R_y in four SDKs (Qiskit, Cirq, PennyLane, QuTiP), samples shots in three of them,
and redoes the analyze(750, 1000) example with scipy's Wilson interval. qang is imported only at the end, to
compare. It prints PASA/FALLA per claim and exits with 1 if any fails.

    .venv-int/Scripts/python tools/verify_cap01.py
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


def H(p):
    return 0.0 if p in (0.0, 1.0) else float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


# ---- states prepared with R_y(theta)|0> in each SDK: (p0, <Z>) ----

def qiskit_state(theta):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    sv = Statevector(qc)
    return float(sv.probabilities()[0]), float(np.real(sv.expectation_value(Pauli("Z"))))


def cirq_state(theta):
    import cirq
    q = cirq.LineQubit(0)
    psi = cirq.Simulator(dtype=np.complex128).simulate(cirq.Circuit(cirq.ry(theta)(q))).final_state_vector
    z = cirq.Z(q).expectation_from_state_vector(psi, {q: 0})
    return float(abs(psi[0]) ** 2), float(np.real(z))


def pennylane_state(theta):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circ():
        qml.RY(theta, wires=0)
        return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

    pr, z = circ()
    return float(pr[0]), float(z)


def qutip_state(theta):
    import qutip as qt
    psi = (-1j * theta / 2 * qt.sigmay()).expm() * qt.basis(2, 0)
    return float(abs(psi.full()[0, 0]) ** 2), float(np.real(qt.expect(qt.sigmaz(), psi)))


SDKS = {"Qiskit": qiskit_state, "Cirq": cirq_state, "PennyLane": pennylane_state, "QuTiP": qutip_state}


# ---- shot sampling of R_y(theta)|0> in each SDK: array of 0/1 outcomes ----

def qiskit_shots(theta, n, seed):
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    res = StatevectorSampler(seed=seed).run([qc], shots=n).result()[0]
    return (res.data.c.array[:, 0] & 1).astype(int)


def cirq_shots(theta, n, seed):
    import cirq
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.ry(theta)(q), cirq.measure(q, key="m"))
    return cirq.Simulator(seed=seed).run(c, repetitions=n).measurements["m"][:, 0].astype(int)


def pennylane_shots(theta, n, seed):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=1, seed=seed)

    @qml.qnode(dev)
    def circ():
        qml.RY(theta, wires=0)
        return qml.sample(wires=0)

    return np.asarray(qml.set_shots(circ, shots=n)()).reshape(-1).astype(int)


SAMPLERS = {"Qiskit": qiskit_shots, "Cirq": cirq_shots, "PennyLane": pennylane_shots}


def part_equation():
    print("\n1. Ecuación de probabilidad sobre estados R_y(θ)|0> (tolerancia 1e-12)")
    thetas = [0.0, math.pi / 4, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi, 0.123, 1.9, 2.9]
    for sdk, state in SDKS.items():
        err_born = err_theta = err_z = err_m = 0.0
        for th in thetas:
            p0, z = state(th)
            err_born = max(err_born, abs(p0 - math.cos(th / 2) ** 2))
            err_theta = max(err_theta, abs(2 * math.acos(math.sqrt(min(1.0, max(0.0, p0)))) - th))
            err_z = max(err_z, abs(z - (2 * p0 - 1)), abs(z - math.cos(th)))
            err_m = max(err_m, abs(1000 * z - (2000 * p0 - 1000)), abs(math.acos(max(-1, min(1, (2000 * p0 - 1000) / 1000))) - th))
        check(f"{sdk}: p0 = cos²(θ/2)", err_born < 1e-12, f"error máx {err_born:.1e}")
        # arccos is ill-conditioned at the poles (θ=0, π): an error 1e-16 in p0 gives ~1e-8 in θ
        check(f"{sdk}: θ = 2 arccos √p0", err_theta < 1e-7, f"error máx {err_theta:.1e} (polos incluidos)")
        check(f"{sdk}: qg_Z = <Z> = 2p0 − 1 = cos θ", err_z < 1e-12, f"error máx {err_z:.1e}")
        check(f"{sdk}: m = 1000 qg_Z = 2000 p0 − 1000, θ = arccos(m/1000)", err_m < 1e-7, f"error máx {err_m:.1e}")

    print("\n   Tabla de anclajes (tbl-anclajes), con QuTiP y Qiskit")
    table = [(0.0, "1", "+1", "+1000"), (math.pi / 4, "0.854", "+0.707", "+707"), (math.pi / 3, "0.75", "+0.5", "+500"),
             (math.pi / 2, "0.5", "0", "0"), (2 * math.pi / 3, "0.25", "-0.5", "-500"), (math.pi, "0", "-1", "-1000")]
    for th, p_txt, z_txt, m_txt in table:
        for sdk in ("QuTiP", "Qiskit"):
            p0, z = SDKS[sdk](th)
            dp = len(p_txt.split(".")[1]) if "." in p_txt else 0
            dz = len(z_txt.split(".")[1]) if "." in z_txt else 0
            ok = (abs(round(p0, dp) - float(p_txt)) < 1e-9 and abs(round(z, dz) - float(z_txt)) < 1e-9
                  and round(1000 * z) == int(m_txt))
            check(f"{sdk}: θ = {math.degrees(th):.0f}° → p0 {p_txt}, qg_Z {z_txt}, {m_txt} m-qg", ok,
                  f"p0 {p0:.4f}, qg_Z {z:+.4f}, m {1000 * z:+.1f}")

    p0, z = SDKS["PennyLane"](math.pi / 3)
    ok = abs(math.degrees(2 * math.acos(math.sqrt(p0))) - 60) < 1e-9 and abs(H(p0) - 0.811) < 5e-4 and round(1000 * z, 6) == 500
    check("PennyLane: readings_from_p0(0.75) → 60°, qg_Z 0.5, qg_S 0.811, 500 m-qg", ok,
          f"θ {math.degrees(2 * math.acos(math.sqrt(p0))):.3f}°, qg_S {H(p0):.4f}, m {1000 * z:.3f}")


def part_shots():
    print("\n2. Resolución por disparos σ_m = 1000|sin θ|/√N, por muestreo (R = 2000 lotes de N = 1000)")
    n_shots, batches = 1000, 2000
    for sdk, sampler in SAMPLERS.items():
        for k, th in enumerate([math.pi / 6, math.pi / 3, math.pi / 2, 5 * math.pi / 6]):
            # distinct seeds per SDK: Qiskit and PennyLane both seed numpy's default_rng, and the same seed
            # would give them identical shot records
            bits = sampler(th, n_shots * batches, 1000 * (1 + list(SAMPLERS).index(sdk)) + k).reshape(batches, n_shots)
            m = 1000 * (1 - 2 * bits.mean(axis=1))          # outcome 0 counts +1, outcome 1 counts −1
            pred = 1000 * abs(math.sin(th)) / math.sqrt(n_shots)
            sd = float(m.std(ddof=1))
            # the sample sd of R batches has relative sd ≈ 1/√(2R) = 1.6 %; 6 % is about 4 sigma
            ok_sd = abs(sd / pred - 1) < 0.06
            ok_mean = abs(m.mean() - 1000 * math.cos(th)) < 4 * pred / math.sqrt(batches)
            check(f"{sdk}: θ = {math.degrees(th):.0f}°", ok_sd and ok_mean,
                  f"σ_m muestreado {sd:.2f}, predicho {pred:.2f}; media {m.mean():+.1f} vs {1000 * math.cos(th):+.1f}")

    print("\n   Números citados del texto (aritmética de la fórmula)")
    s60 = 1000 * math.sin(math.pi / 3) / math.sqrt(1000)
    check("σ_m a 60° con 1000 disparos = 27,4 m-qg", round(s60, 1) == 27.4, f"{s60:.3f}")
    n5 = (1000 * math.sin(math.pi / 3) / 5) ** 2
    check("5 m-qg a 60° requieren 30 000 disparos", round(n5) == 30000, f"{n5:.1f}")
    n10, n1 = (1000 / 10) ** 2, (1000 / 1) ** 2
    check("ecuador: 10 m-qg → 10⁴ disparos, 1 m-qg → 10⁶", n10 == 1e4 and n1 == 1e6, f"{n10:.0f}, {n1:.0f}")
    pole1, pole10 = math.acos(1 - 1 / 1000), math.acos(1 - 10 / 1000)
    check("1 m-qg del polo = 0,0447 rad (2,56°); 10 m-qg = 8,1°",
          round(pole1, 4) == 0.0447 and round(math.degrees(pole1), 2) == 2.56 and round(math.degrees(pole10), 1) == 8.1,
          f"{pole1:.5f} rad = {math.degrees(pole1):.3f}°, {math.degrees(pole10):.2f}°")
    check("1 m-qg = 1 mrad en el ecuador, 2 mrad a 30°; Δp0 = 0,0005",
          abs(1 / (1000 * math.sin(math.pi / 2)) - 1e-3) < 1e-15 and abs(1 / (1000 * math.sin(math.pi / 6)) - 2e-3) < 1e-15
          and abs(1 / 2000 - 0.0005) < 1e-15)
    m502 = 1000 * 0.502
    check("qg_Z = 0,502 está a 501,8 m-qg, 1,8 m-qg del ideal 0,5", abs(m502 - 501.8) < 1e-9 and abs(m502 - 500 - 1.8) < 1e-9,
          f"m = 1000·0,502 = {m502:.1f} m-qg, diferencia {m502 - 500:.1f} m-qg")


def part_analyze():
    print("\n3. analyze(750, 1000): intervalo de Wilson con scipy, mapeado a m, θ y qg_S")
    from scipy.stats import binomtest
    ci = binomtest(750, 1000).proportion_ci(confidence_level=0.95, method="wilson")
    p_lo, p_hi = float(ci.low), float(ci.high)
    m = (500.0, 2000 * p_lo - 1000, 2000 * p_hi - 1000)
    th = (2 * math.acos(math.sqrt(0.75)), 2 * math.acos(math.sqrt(p_hi)), 2 * math.acos(math.sqrt(p_lo)))  # bounds swap
    qs = (H(0.75), H(p_hi), H(p_lo))                                     # H decreasing for p0 > 1/2
    check("milliqang = (500.0, 444.5, 551.7) m-qg", tuple(round(x, 1) for x in m) == (500.0, 444.5, 551.7),
          f"({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f})")
    check("theta = (1.047, 0.986, 1.110) rad", tuple(round(x, 3) for x in th) == (1.047, 0.986, 1.110),
          f"({th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f})")
    deg = tuple(round(math.degrees(x), 1) for x in th)
    check("theta = 60.0°, 56.5°, 63.6°", deg == (60.0, 56.5, 63.6), f"{deg}")
    check("qg_s = (0.811, 0.768, 0.852)", tuple(round(x, 3) for x in qs) == (0.811, 0.768, 0.852),
          f"({qs[0]:.4f}, {qs[1]:.4f}, {qs[2]:.4f})")
    check("deviation_milliqang = 0.0 respecto de qg_Z = 0.5", abs(m[0] - 500.0) < 1e-12)

    sys.path.insert(0, ROOT)
    from qang import probability as pb
    r = pb.analyze(750, 1000, target_qg_z=0.5)
    diff = max(abs(a - b) for a, b in zip(r.milliqang, m))
    # qang uses z = 1.96, scipy the exact 97.5 % quantile 1.959964: they differ in the 3rd decimal of m at most
    check("qang.analyze coincide con scipy (|Δm| < 0,01 m-qg)", diff < 0.01, f"|Δm| máx {diff:.2e} m-qg")


def part_other():
    print("\n4. Otras afirmaciones del capítulo")
    import qutip as qt
    # gate-error budget: average R_x(θ(1+ε)) over ε ~ N(0, σ²) with Gauss–Hermite quadrature, in QuTiP
    x, w = np.polynomial.hermite_e.hermegauss(60)
    w = w / w.sum()
    for th, want_m, want_z in [(math.pi, 12.3, -0.988), (math.pi / 2, 0.0, 0.0)]:
        ez = sum(wi * qt.expect(qt.sigmaz(), (-1j * th * (1 + 0.05 * xi) / 2 * qt.sigmax()).expm() * qt.basis(2, 0))
                 for xi, wi in zip(x, w))
        shift = 1000 * (ez - math.cos(th))
        check(f"QuTiP: R_x({th / math.pi:g}π) con σ = 0,05 → qg_Z {want_z}, corrimiento {want_m} m-qg",
              round(shift, 1) == want_m and round(ez, 3) == want_z, f"qg_Z {ez:.4f}, corrimiento {shift:.2f} m-qg")
    # Wootters / Bhattacharyya angle as arccos of the fidelity of two commuting (diagonal) states
    f = qt.fidelity(qt.Qobj(np.diag([0.5, 0.3, 0.2])), qt.Qobj(np.diag([0.4, 0.4, 0.2])))
    check("QuTiP: d(p, q) = arccos Σ√(p q) = 0,113 rad", round(math.acos(f), 3) == 0.113, f"{math.acos(f):.5f} rad")
    rho = 0.2 * qt.fock_dm(2, 0) + 0.8 * qt.fock_dm(2, 1)
    zq, s = float(qt.expect(qt.sigmaz(), rho)), float(qt.entropy_vn(rho, base=2))
    check("QuTiP: mezcla 0,2/0,8 → qg_Z −0,6, entropía de von Neumann 0,72", round(zq, 6) == -0.6 and round(s, 2) == 0.72,
          f"qg_Z {zq:.4f}, S {s:.4f}")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector, entropy, partial_trace
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    sv = Statevector(bell)
    probs = sv.probabilities()
    joint = float(-sum(p * math.log2(p) for p in probs if p > 0)) / 2
    loc = [float(entropy(partial_trace(sv, [k]), base=2)) for k in (0, 1)]
    zloc = [float(np.real(sv.expectation_value(op))) for op in (Pauli("IZ"), Pauli("ZI"))]
    check("Qiskit: Bell → qg_Z local 0, entropía local 1 bit, conjunta normalizada 0,5",
          max(map(abs, zloc)) < 1e-12 and all(abs(v - 1) < 1e-12 for v in loc) and abs(joint - 0.5) < 1e-12,
          f"qg_Z {zloc}, S local {loc}, conjunta/2 {joint}")
    # identity qg_S = H((1 + qg_Z)/2) on random mixed states (Cirq density matrices)
    import cirq
    rng = np.random.default_rng(7)
    err = 0.0
    for _ in range(50):
        a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        rho = a @ a.conj().T
        rho /= np.trace(rho)
        q = cirq.LineQubit(0)
        z = float(np.real(cirq.Z(q).expectation_from_density_matrix(rho, {q: 0})))
        err = max(err, abs(H(float(np.real(rho[0, 0]))) - H((1 + z) / 2)))
    check("Cirq: qg_S = H((1 + qg_Z)/2) en 50 estados mixtos al azar", err < 1e-12, f"error máx {err:.1e}")


def part_hardware():
    print("\n5. Primera prueba en hardware (§168), recalculada desde examples/data con numpy y scipy")
    from scipy.stats import chi2
    with open(os.path.join(ROOT, "examples", "data", "iqm_milliqang_qg_analysis.json"), encoding="utf-8") as fh:
        r = json.load(fh)
    ks = np.array(sorted(int(k) for k in r["chains"]))
    s = np.array([r["chains"][str(k)]["s"] for k in ks])
    sg = np.array([r["chains"][str(k)]["sigma"] for k in ks])
    wt = 1 / sg ** 2
    c = float((wt * (ks - 1) * s).sum() / (wt * (ks - 1) ** 2).sum())
    chi = float((((s - c * (ks - 1)) / sg) ** 2).sum() / (len(ks) - 1))
    check("pendiente única = 1,64 m-qg por compuerta X", round(c, 2) == 1.64, f"{c:.4f}")
    check("χ²/gl = 0,81", round(chi, 2) == 0.81, f"{chi:.3f}")
    s41, g41 = r["chains"]["41"]["s"], r["chains"]["41"]["sigma"]
    check("tras 41 compuertas: 63 m-qg, ocho desviaciones", round(s41) == 63 and int(s41 / g41) == 8,
          f"{s41:.2f} ± {g41:.2f} = {s41 / g41:.2f} σ")
    res = [row["measured"] - row["predicted"] for row in r["M1"] if row["target"] != 0.0]
    check("ángulos intermedios leen entre 16 y 55 m-qg por debajo", round(-max(res)) == 16 and round(-min(res)) == 55,
          f"{sorted(round(x, 1) for x in res)}")
    passed = sum(abs(row["z"]) < 3 for row in r["M1"])
    check("M1 pasa por 4 de 5", passed == 4, f"{passed} de {len(r['M1'])}")
    ratio = r["M2_ratio"]
    dof = len(r["equator_m"]) - 1
    p_low = float(chi2.cdf(dof * ratio ** 2, dof))
    check("M2: razón 0,50, probabilidad por azar 0,013", round(ratio, 2) == 0.50 and round(p_low, 3) == 0.013,
          f"razón {ratio:.3f}, P(χ²_{dof} ≤ {dof * ratio ** 2:.2f}) = {p_low:.4f}")


def part_criteria():
    print("\n6. Criterios de teoría qang (book/REVISION.md) sobre los datos de §168")
    with open(os.path.join(ROOT, "examples", "data", "iqm_milliqang_qg_analysis.json"), encoding="utf-8") as fh:
        r = json.load(fh)
    e0, e1, shots = r["e0"], r["e1"], 4000
    c = 1 - e0 - e1
    # criterion 1: the affine readout m_pred = (1 - e0 - e1) m + 1000 (e1 - e0)
    err = max(abs(c * row["target"] + 1000 * (e1 - e0) - row["predicted"]) for row in r["M1"])
    check("lectura afín m_pred = (1 − e0 − e1) m + 1000 (e1 − e0) en los 5 puntos", err < 1e-6, f"error máx {err:.1e} m-qg")

    # criterion 3: shot errors in m-qg are binomial, divided by the contrast when the reading is readout-corrected
    def sigma_corrected(m_corr):
        raw = c * m_corr + 1000 * (e1 - e0)
        p = (1000 + raw) / 2000
        return 2000 * math.sqrt(p * (1 - p) / shots) / c
    err = max(abs(sigma_corrected(-1000 + v["s"] if int(k) % 2 else 1000 - v["s"]) - v["sigma"])
              for k, v in r["chains"].items())
    check("σ de las cadenas = binomial de 4000 disparos / contraste", err < 1e-6, f"error máx {err:.1e} m-qg")
    s_eq = 1000 / math.sqrt(shots)
    check("M2: σ binomial de una copia del ecuador = 15,8 m-qg; la dispersión medida 7,9", round(s_eq, 1) == 15.8
          and round(float(np.std(r["equator_m"], ddof=1)), 1) == 7.9,
          f"{s_eq:.2f} y {np.std(r['equator_m'], ddof=1):.2f} m-qg")
    eq = next(row for row in r["M1"] if row["target"] == 0.0)
    s_cal = 1000 * math.hypot(math.sqrt(e0 * (1 - e0) / shots), math.sqrt(e1 * (1 - e1) / shots))
    s_tot = math.hypot(s_eq / math.sqrt(len(r["equator_m"])), s_cal)
    check("σ del ecuador en M1 = media de 10 copias ⊕ calibración de los polos", abs(s_tot - eq["sigma"]) < 0.01,
          f"{s_tot:.3f} frente a {eq['sigma']:.3f} m-qg")

    # criteria 2 and 6: the 16-55 m-qg residuals as angles, and their sign
    angles = []
    for row in r["M1"]:
        if row["target"] in (0.0,):
            continue
        th_t = math.acos(row["target"] / 1000)
        th_m = math.acos((row["measured"] - 1000 * (e1 - e0)) / c / 1000)
        angles.append(math.degrees(th_m - th_t))
    check("los cuatro ángulos intermedios están sobre-rotados (θ medido > θ pedido)", all(a > 0 for a in angles),
          f"Δθ = {', '.join(f'{a:+.2f}°' for a in angles)}")


def part_theory():
    print("\n7. Teoría (revisión reiniciada): identidades simbólicas, estados mixtos, sumas binomiales exactas, ruido")
    import sympy as sp
    from scipy.stats import binom, norm

    # T9: every identity of the chapter, symbolically
    t, p, q = sp.symbols("theta p q", positive=True)
    p0 = sp.cos(t / 2) ** 2
    ids = {
        "qg_Z = 2 p0 − 1 = cos θ": sp.simplify(2 * p0 - 1 - sp.cos(t)),
        "θ = 2 arccos √p0 en [0, π]": sp.simplify(2 * sp.acos(sp.sqrt(p0)) - t).subs(t, sp.Rational(7, 5)),
        "p0 = (1000 + m)/2000 con m = 1000 cos θ": sp.simplify((1000 + 1000 * sp.cos(t)) / 2000 - p0),
        "dm/dθ = −1000 sin θ": sp.simplify(sp.diff(1000 * sp.cos(t), t) + 1000 * sp.sin(t)),
        "dE/dq = −(1/sin θ) dE/dθ": sp.simplify(sp.diff(sp.acos(q), q) + 1 / sp.sqrt(1 - q ** 2)),
        "d qg_S/d qg_Z = ½ log₂((1 − q)/(1 + q))": sp.simplify(
            sp.diff(-((1 + q) / 2) * sp.log((1 + q) / 2, 2) - ((1 - q) / 2) * sp.log((1 - q) / 2, 2), q)
            - sp.log((1 - q) / (1 + q), 2) / 2),
        "Var(θ̂) = Var(qg_Z) (dθ/dqg_Z)² = 1/N, θ = arccos qg_Z": sp.simplify((1 - q ** 2) * sp.diff(sp.acos(q), q) ** 2 - 1),
        "Var(φ̂) = 1/N para φ = 2 arcsin √p": sp.simplify(p * (1 - p) * sp.diff(2 * sp.asin(sp.sqrt(p)), p) ** 2 - 1),
        "d(p, polo) = arccos √p0 = θ/2": sp.simplify(sp.acos(sp.sqrt(p0)) - t / 2).subs(t, sp.Rational(9, 7)),
    }
    for name, expr in ids.items():
        check(f"sympy: {name}", abs(float(sp.N(expr))) < 1e-12 if expr.free_symbols == set() else expr == 0, str(expr))

    # T4: qang angle arccos(qg_Z) vs Bloch polar angle arccos(qg_Z / r) for a mixed qubit (§45)
    import qutip as qt
    from qiskit.quantum_info import DensityMatrix
    n, reps, rng = 10_000, 4000, np.random.default_rng(7)
    for th, r, book in ((math.pi / 2, 0.8, 1.56), (math.pi / 8, 0.8, 4.84)):
        rho = 0.5 * (qt.qeye(2) + r * (math.sin(th) * qt.sigmax() + math.cos(th) * qt.sigmaz()))
        pz = float(np.real(rho.full()[0, 0]))
        dm = DensityMatrix(rho.full())
        dm.seed(int(rng.integers(1 << 31)))
        k0 = np.array([dm.sample_counts(n).get("0", 0) for _ in range(reps // 10)] +
                      list(rng.binomial(n, pz, reps - reps // 10)))
        qz = 2 * k0 / n - 1
        v_qang = n * np.var(np.arccos(np.clip(qz, -1, 1)), ddof=1)
        v_bloch = n * np.var(np.arccos(np.clip(qz / r, -1, 1)), ddof=1)
        formula = 1 + (1 - r ** 2) / (r ** 2 * math.sin(th) ** 2)
        check(f"Qiskit/QuTiP r = {r}, θ = {th / math.pi:.3g}π: N·Var(arccos qg_Z) = 1 (ángulo qang, cualquier estado)",
              abs(v_qang - 1) < 0.1, f"{v_qang:.3f}")
        check(f"  ídem: N·Var(arccos(qg_Z/r)) = {book} (ángulo de Bloch, §45)", abs(v_bloch - formula) / formula < 0.08,
              f"{v_bloch:.3f}, fórmula {formula:.3f}")

    # §125 S1, S2: exact binomial sums on p ∈ {0.1, …, 0.9}, Anscombe angle, interval θ ± z/√(n + 1/2)
    z = norm.ppf(0.975)
    grid = np.round(np.arange(0.1, 0.95, 0.1), 2)

    def ang(k, nn):
        return 2 * np.arccos(np.sqrt((k + 3 / 8) / (nn + 3 / 4)))

    s1 = []
    for pp in grid:
        k = np.arange(101)
        w = binom.pmf(k, 100, pp)
        a = ang(k, 100)
        s1.append(100.5 * (w @ a ** 2 - (w @ a) ** 2))
    check("§125 S1 exacto: (n + ½) Var(θ̂) con n = 100 entre 1,000 y 1,001", 0.9995 <= min(s1) and max(s1) < 1.0015,
          f"{min(s1):.4f}–{max(s1):.4f}")
    cov = []
    for nn in (30, 100, 1000):
        k = np.arange(nn + 1)
        a = ang(k, nn)
        for pp in grid:
            th = 2 * math.acos(math.sqrt(pp))
            cov.append(binom.pmf(k, nn, pp)[np.abs(a - th) <= z / math.sqrt(nn + 0.5)].sum())
    check("§125 S2 exacto: cobertura de θ̂ ± z/√(n + ½) entre 0,930 y 0,962", round(min(cov), 3) >= 0.930
          and round(max(cov), 3) <= 0.962, f"{min(cov):.3f}–{max(cov):.3f}")

    # §51: Wilson interval for qg_Z mapped through H, exact binomial coverage
    def wilson(k, nn):
        ph = k / nn
        c = (ph + z * z / (2 * nn)) / (1 + z * z / nn)
        h = z * np.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
        return c - h, c + h

    def hv(x):
        x = np.clip(x, 1e-300, 1 - 1e-16)
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    covs = []
    for qz in (0.0, 0.3, 0.9, 0.99):
        pp = (1 + qz) / 2
        hs = H(pp)
        for nn in (20, 100, 1000):
            k = np.arange(nn + 1)
            lo, hi = wilson(k, nn)
            h_lo = np.minimum(hv(lo), hv(hi))
            h_hi = np.where((lo <= 0.5) & (hi >= 0.5), 1.0, np.maximum(hv(lo), hv(hi)))
            covs.append(binom.pmf(k, nn, pp)[(h_lo <= hs + 1e-12) & (hs <= h_hi + 1e-12)].sum())
    check("§51 exacto: cobertura de Wilson mapeado a qg_S en 0,90–0,99 (12 casos)", round(min(covs), 2) >= 0.90
          and round(max(covs), 2) <= 0.99, f"{min(covs):.3f}–{max(covs):.3f}")

    # pulse budget: ε_m ∝ θ² cos θ, not θ² — zero at odd multiples of π/2, sign alternating at multiples of π
    from qang import probability as pb
    e = [pb.pulse_error_milliqang(k * math.pi, 0.05) for k in (1, 2, 3)]
    check("ε_m(kπ) ≈ k² ε_m(π) en módulo, con signo alterno (hacia el ecuador)", abs(abs(e[1] / e[0]) - 4) < 0.2
          and abs(abs(e[2] / e[0]) - 9) < 1 and e[0] > 0 > e[1], f"{e[0]:+.2f}, {e[1]:+.2f}, {e[2]:+.2f} m-qg")
    check("ε_m(3π/2) = 0 aunque θ² sea 2,25 veces mayor que en π", abs(pb.pulse_error_milliqang(1.5 * math.pi, 0.05)) < 1e-9)

    # §168 reasoning: T1 relaxation per gate gives a deficit affine in k, a coherent over-rotation one ∝ k²
    # (Cirq, QuTiP); fitted as a + b k + c k², the share of the k² term at k = 41 tells them apart
    import cirq
    ks = np.arange(3, 42, 2)
    qb = cirq.LineQubit(0)
    sim = cirq.DensityMatrixSimulator(dtype=np.complex128)
    lin = []
    for k in ks:
        c = cirq.Circuit([cirq.X(qb), cirq.amplitude_damp(0.0016)(qb)] * int(k))
        lin.append(1000 + 1000 * float(np.real(np.trace(sim.simulate(c).final_density_matrix @ np.diag([1, -1])))))
    coh = []
    for k in ks:
        psi = (-1j * k * math.pi * 1.003 / 2 * qt.sigmax()).expm() * qt.basis(2, 0)
        coh.append(1000 + 1000 * float(qt.expect(qt.sigmaz(), psi)))

    def quad_share(y):
        c2, c1, _ = np.polyfit(ks, y, 2)
        return abs(c2) * 41 ** 2 / (abs(c2) * 41 ** 2 + abs(c1) * 41)

    s_lin, s_coh = quad_share(lin), quad_share(coh)
    check("Cirq: amortiguamiento por compuerta → faltante afín en k (término k² < 5 % a k = 41)", s_lin < 0.05,
          f"{s_lin:.3f}; {lin[-1]:.1f} m-qg a k = 41")
    check("QuTiP: sobre-rotación coherente → faltante cuadrático en k (término k² > 95 %)", s_coh > 0.95,
          f"{s_coh:.3f}; {coh[-1]:.1f} m-qg a k = 41")

    # the same test on the §168 chains: a pure k² model is rejected, a single slope is not
    from scipy.stats import chi2
    with open(os.path.join(ROOT, "examples", "data", "iqm_milliqang_qg_analysis.json"), encoding="utf-8") as fh:
        chains = json.load(fh)["chains"]
    kk = np.array([int(x) for x in chains], dtype=float)
    ss = np.array([v["s"] for v in chains.values()])
    ww = 1 / np.array([v["sigma"] for v in chains.values()]) ** 2

    def chi2_of(x):
        b = (ww * x * ss).sum() / (ww * x * x).sum()
        return float((ww * (ss - b * x) ** 2).sum())

    c_lin, c_quad = chi2_of(kk), chi2_of(kk ** 2)
    dof = len(kk) - 1
    check("§168: el modelo c·k² queda rechazado (p < 0,001) y el lineal b·k no", chi2.sf(c_quad, dof) < 1e-3 < chi2.sf(c_lin, dof),
          f"χ²/gl {c_quad / dof:.2f} (p = {chi2.sf(c_quad, dof):.1e}) frente a {c_lin / dof:.2f}")

    # Bell: the 0.5 is the Shannon entropy of the joint outcomes; the von Neumann entropy of the register is 0
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, entropy, partial_trace
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    bell = Statevector(qc)
    pr = bell.probabilities()
    sh = -sum(x * math.log2(x) for x in pr if x > 0) / 2
    check("Qiskit Bell: Shannon conjunta normalizada 0,5; von Neumann conjunta 0; local 1",
          abs(sh - 0.5) < 1e-12 and abs(entropy(bell)) < 1e-12 and abs(entropy(partial_trace(bell, [1])) - 1) < 1e-12,
          f"{sh:.3f}, {entropy(bell):.1e}, {entropy(partial_trace(bell, [1])):.3f}")

    # §65: (qg_Z, sign qg_X) is one-to-one on the whole xz circle (Cirq)
    errs = []
    for th in np.linspace(-math.pi + 0.01, math.pi, 37):
        psi = cirq.Simulator(dtype=np.complex128).simulate(cirq.Circuit(cirq.ry(th)(qb))).final_state_vector
        zz = float(np.real(cirq.Z(qb).expectation_from_state_vector(psi, {qb: 0})))
        xx = float(np.real(cirq.X(qb).expectation_from_state_vector(psi, {qb: 0})))
        errs.append(abs(math.copysign(math.acos(max(-1, min(1, zz))), xx if abs(xx) > 1e-12 else 1) - th))
    check("Cirq: θ = sign(qg_X)·arccos(qg_Z) recupera θ en (−π, π]", max(errs) < 1e-6, f"error máx {max(errs):.1e}")


def main():
    part_equation()
    part_shots()
    part_analyze()
    part_other()
    part_hardware()
    part_criteria()
    part_theory()
    n_ok = sum(RESULTS)
    print(f"\n{n_ok} de {len(RESULTS)} afirmaciones PASAN")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
