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


def main():
    part_equation()
    part_shots()
    part_analyze()
    part_other()
    part_hardware()
    n_ok = sum(RESULTS)
    print(f"\n{n_ok} de {len(RESULTS)} afirmaciones PASAN")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
