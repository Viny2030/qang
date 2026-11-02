"""
E = mc^2 as a reading on real hardware: the Dirac beta qubit and Zitterbewegung (§170, IQM Resonance, Garnet)

Appendix G of the book shows that for a free Dirac particle the expectation value of the Dirac matrix beta is
<beta> = mc^2 / E: since beta has eigenvalues +-1, it is a qg_Z, read in m-qg as 1000/gamma, with the sign of the
energy branch. This experiment tests that reading, and the Zitterbewegung it implies, on one physical qubit.

The model. In one dimension at fixed momentum the Dirac Hamiltonian is a single qubit,
    H = p X + m Z          (units hbar = c = 1; beta = Z, alpha = X),
with eigenvalues +-E, E = sqrt(p^2 + m^2). Here m = 1, p = 0.75: E = 1.25, v = p/E = 0.6, gamma = 1.25, and the
relativistic angle theta_r = atan(p/m) = 36.87 deg = 102.4 qga. The positive-energy eigenstate is R_y(theta_r)|0>
and the negative one R_y(theta_r + pi)|0>; |0> (beta = +1) is a superposition of the two branches. Evolving it,
    <beta>(t) = cos^2(theta_r) + sin^2(theta_r) cos(2 E t),
the Zitterbewegung at the frequency 2E: the reading oscillates from +1000 m-qg (the pole) to 1000 cos(2 theta_r)
= +280 m-qg, around the time average 1000 cos^2(theta_r) = 1000/gamma^2 = +640 m-qg, which is 1000 times the
"P_rest" = (mc^2/E)^2 of the source document. U(t) = exp(-i H t) is compiled to native gates by the transpiler.

Circuits, all on one physical qubit (QUBIT), optimization_level 0, a barrier before the measurement:
  poles    |0> and |1> (one X): readout errors e0 = P(1|0), e1 = P(0|1)
  eplus    R_y(theta_r) then U(t_j), t_j = j T/4, j = 0..3, with T = pi/E the Zitterbewegung period
  eminus   R_y(theta_r + pi) then U(t), t = 0 and T/2
  zb       U(t_k) on |0>, t_k = k (2T)/16, k = 0..15 (two periods)
SHOTS = 2000 per circuit, 24 circuits in one job (48 000 shots).

Pre-registered predictions (for --run; m_c is the reading corrected with the poles, m_c = (m - 1000 (e1 - e0)) / (1 -
e0 - e1); sigma is the binomial sigma of the point divided by the contrast; ALLOW = 60 m-qg is a systematic
allowance fixed from §168, where R_y at intermediate angles read 16-55 m-qg below the prediction):
  Z1  E = mc^2 is a reading: eplus at t = 0 reads m_c = +800 (1000/gamma) within 3 sigma + ALLOW.
  Z2  the negative branch mirrors it: eminus at t = 0 reads -800 within 3 sigma + ALLOW, and the charge-conjugation
      reflection holds, |m_c(eplus, 0) + m_c(eminus, 0)| <= 3 sigma + ALLOW.
  Z3  an energy eigenstate does not tremble: the four eplus readings span (max - min) at most 108 m-qg, 15 % of the
      720 m-qg peak-to-peak of the Zitterbewegung.
  Z4  the Zitterbewegung frequency is 2E: fitting A + B cos(w t) + C sin(w t) to the 16 zb readings (w scanned on a
      grid, least squares), the best w lies within 5 % of 2E = 2.5.
  Z5  its mean and amplitude: the fitted A lies within ALLOW of +640 (1000/gamma^2) and sqrt(B^2 + C^2) within ALLOW
      of 360 (1000 sin^2(theta_r)).

What the predictions discriminate: Z1 separates the beta qubit (+800) from the "Pythagoras qubit" of appendix G
(+640) and from the rest pole (+1000) by more than the tolerance (about 3 x 15 + 60 = 105 m-qg); Z5 separates the
time average 1000/gamma^2 = +640 from 1000/gamma = +800 by 160 m-qg, more than ALLOW.

Free rehearsal, 2026-11-02 (IQM's fake backends, 2000 shots, seed_simulator 170; at most 6 native gates per circuit):
  IQMFakeApollo     e0 0.054, e1 0.054; eplus 807, 800, 785, 797; eminus -814, -810; reflection -7; span 22;
                    fit w 2.494, A 634, amplitude 378; Z1-Z5 pass
  IQMFakeAphrodite  e0 0.041, e1 0.051; eplus 778, 766, 744, 776; eminus -789, -793; reflection -11; span 34;
                    fit w 2.477, A 630, amplitude 346; Z1-Z5 pass

Cost (§168: about 0.15 IQM credits per 1000 shots plus about 0.50 per job): 48 000 shots, about 7.7 credits.

Modes (the token is read from the environment variable IQM_TOKEN, never from a file in the repository; run in the
environment that has iqm-client[qiskit], e.g. .venv-iqm):
  python examples/iqm_zitterbewegung_qg.py --exact          the exact predictions, no backend
  python examples/iqm_zitterbewegung_qg.py                  free rehearsal on IQM's fake backends
  python examples/iqm_zitterbewegung_qg.py --prereg         freeze this file (sha256) before --run
  python examples/iqm_zitterbewegung_qg.py --run --yes-i-accept-qpu-cost      the 24 circuits on Garnet
"""

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

SERVER = "https://resonance.iqm.tech/"
QC_NAME = "garnet"
QUBIT = 0
SHOTS = 2000
MASS, MOMENTUM = 1.0, 0.75
ENERGY = math.hypot(MASS, MOMENTUM)                    # 1.25
THETA_R = math.atan2(MOMENTUM, MASS)                   # 36.87 deg
PERIOD = math.pi / ENERGY                              # Zitterbewegung period, frequency 2E
EPLUS_T = tuple(j * PERIOD / 4 for j in range(4))
EMINUS_T = (0.0, PERIOD / 2)
ZB_T = tuple(k * 2 * PERIOD / 16 for k in range(16))
ALLOW = 60.0
DATA = os.path.join(HERE, "data")
PREREG = os.path.join(DATA, "iqm_zitterbewegung_qg.prereg")


def hamiltonian():
    return np.array([[MASS, MOMENTUM], [MOMENTUM, -MASS]], dtype=complex)


def evolution(t):
    w, V = np.linalg.eigh(hamiltonian())
    return V @ np.diag(np.exp(-1j * w * t)) @ V.conj().T


def exact_reading(prep_angle, t):
    """1000 <Z> after R_y(prep_angle)|0> and U(t)."""
    psi = np.array([math.cos(prep_angle / 2), math.sin(prep_angle / 2)], dtype=complex)
    psi = evolution(t) @ psi
    return 1000.0 * float(abs(psi[0]) ** 2 - abs(psi[1]) ** 2)


def zb_formula(t):
    return 1000.0 * (math.cos(THETA_R) ** 2 + math.sin(THETA_R) ** 2 * math.cos(2 * ENERGY * t))


def circuits():
    """(label, circuit) pairs, in submission order."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    out = []

    def make(label, prep, t):
        qc = QuantumCircuit(1, 1, name=label)
        if prep == "x":
            qc.x(0)
        elif prep is not None:
            qc.ry(prep, 0)
        if t:
            qc.barrier()
            qc.append(UnitaryGate(evolution(t), label=f"U({t:.3f})"), [0])
        qc.barrier()
        qc.measure(0, 0)
        out.append((label, qc))

    make("pole0", None, 0.0)
    make("pole1", "x", 0.0)
    for j, t in enumerate(EPLUS_T):
        make(f"eplus{j}", THETA_R, t)
    for j, t in enumerate(EMINUS_T):
        make(f"eminus{j}", THETA_R + math.pi, t)
    for k, t in enumerate(ZB_T):
        make(f"zb{k}", None, t)
    return out


def _p0(counts):
    n0 = counts.get("0", 0)
    n = sum(counts.values())
    return n0 / n, n


def fit_zb(ts, ms, sig):
    """Best w on a grid and the weighted least-squares A, B, C of A + B cos(w t) + C sin(w t)."""
    best = None
    w8 = 1.0 / np.asarray(sig)
    for w in np.linspace(1.0, 4.0, 3001):
        X = np.column_stack([np.ones_like(ts), np.cos(w * ts), np.sin(w * ts)])
        coef, *_ = np.linalg.lstsq(X * w8[:, None], ms * w8, rcond=None)
        chi2 = float(np.sum(((X @ coef - ms) * w8) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, float(w), coef)
    chi2, w, (A, B, C) = best
    return {"w": w, "A": float(A), "amplitude": float(math.hypot(B, C)), "chi2_per_dof": chi2 / (len(ts) - 4)}


def analyse(counts_by_label):
    p = {lab: _p0(c) for lab, c in counts_by_label.items()}
    (p00, _), (p01, _) = p["pole0"], p["pole1"]
    e0, e1 = 1.0 - p00, p01
    contrast = 1.0 - e0 - e1

    def corrected(label):
        pm, n = p[label]
        m = 2000.0 * pm - 1000.0
        mc = (m - 1000.0 * (e1 - e0)) / contrast
        s = 2000.0 * math.sqrt(max(pm * (1 - pm), 1.0 / n) / n) / contrast
        return mc, s

    ep = [corrected(f"eplus{j}") for j in range(len(EPLUS_T))]
    em = [corrected(f"eminus{j}") for j in range(len(EMINUS_T))]
    zb = [corrected(f"zb{k}") for k in range(len(ZB_T))]
    target = 1000.0 / (ENERGY / MASS)                   # 1000/gamma = 800
    z1 = abs(ep[0][0] - target) <= 3 * ep[0][1] + ALLOW
    reflect = ep[0][0] + em[0][0]
    z2 = (abs(em[0][0] + target) <= 3 * em[0][1] + ALLOW
          and abs(reflect) <= 3 * math.hypot(ep[0][1], em[0][1]) + ALLOW)
    span = max(m for m, _ in ep) - min(m for m, _ in ep)
    z3 = span <= 0.15 * 1000.0 * 2 * math.sin(THETA_R) ** 2
    fit = fit_zb(np.array(ZB_T), np.array([m for m, _ in zb]), [s for _, s in zb])
    z4 = abs(fit["w"] / (2 * ENERGY) - 1) <= 0.05
    z5 = (abs(fit["A"] - 1000.0 * math.cos(THETA_R) ** 2) <= ALLOW
          and abs(fit["amplitude"] - 1000.0 * math.sin(THETA_R) ** 2) <= ALLOW)
    return {"e0": e0, "e1": e1, "contrast": contrast,
            "eplus": [{"t": t, "m": m, "sigma": s} for t, (m, s) in zip(EPLUS_T, ep)],
            "eminus": [{"t": t, "m": m, "sigma": s} for t, (m, s) in zip(EMINUS_T, em)],
            "zb": [{"t": t, "m": m, "sigma": s, "exact": zb_formula(t)} for t, (m, s) in zip(ZB_T, zb)],
            "reflection": reflect, "eplus_span": span, "fit": fit,
            "verdict": {"Z1": z1, "Z2": z2, "Z3": z3, "Z4": z4, "Z5": z5}}


def _summary(r):
    return {"e0": round(r["e0"], 4), "e1": round(r["e1"], 4),
            "eplus m-qg": [round(x["m"], 1) for x in r["eplus"]], "eminus m-qg": [round(x["m"], 1) for x in r["eminus"]],
            "reflection": round(r["reflection"], 1), "eplus span": round(r["eplus_span"], 1),
            "fit w (2E = 2.5)": round(r["fit"]["w"], 3), "fit A (640)": round(r["fit"]["A"], 1),
            "fit amplitude (360)": round(r["fit"]["amplitude"], 1), "verdict": r["verdict"]}


def exact():
    """The noiseless predictions, from the matrix exponential and from the closed form."""
    rows = {"theta_r deg": round(math.degrees(THETA_R), 2), "theta_r qga": round(THETA_R * 1000 / (2 * math.pi), 1),
            "eplus": [round(exact_reading(THETA_R, t), 6) for t in EPLUS_T],
            "eminus": [round(exact_reading(THETA_R + math.pi, t), 6) for t in EMINUS_T],
            "zb": [round(exact_reading(0.0, t), 3) for t in ZB_T],
            "zb closed form max error": max(abs(exact_reading(0.0, t) - zb_formula(t)) for t in ZB_T),
            "zb mean over a period": round(float(np.mean([zb_formula(t) for t in np.linspace(0, PERIOD, 4001)[:-1]])), 3)}
    print(json.dumps(rows))
    counts = {}
    for lab, qc in circuits():
        from qiskit.quantum_info import Statevector
        sv = Statevector(qc.remove_final_measurements(inplace=False))
        p0 = float(abs(sv.data[0]) ** 2)
        counts[lab] = {"0": round(p0 * 10 ** 7), "1": 10 ** 7 - round(p0 * 10 ** 7)}
    print(json.dumps(_summary(analyse(counts))))


def _transpile(pairs, backend):
    from qiskit import transpile
    return [transpile(qc, backend, initial_layout=[QUBIT], optimization_level=0) for _, qc in pairs]


def rehearsal(shots=SHOTS):
    from iqm.qiskit_iqm import IQMFakeAphrodite, IQMFakeApollo
    pairs = circuits()
    for B in (IQMFakeApollo, IQMFakeAphrodite):
        b = B()
        ts = _transpile(pairs, b)
        res = b.run(ts, shots=shots, seed_simulator=170).result()
        r = analyse({lab: res.get_counts(i) for i, (lab, _) in enumerate(pairs)})
        out = {"backend": b.name, "circuits": len(ts), "max native gates": max(sum(t.count_ops().values()) for t in ts)}
        out.update(_summary(r))
        print(json.dumps(out), flush=True)


def real_backend():
    if not os.environ.get("IQM_TOKEN"):
        sys.exit("IQM_TOKEN is not set: set it in this terminal only, never write it to a file in the repository.")
    from iqm.qiskit_iqm import IQMProvider
    return IQMProvider(SERVER, quantum_computer=QC_NAME).get_backend()


def submit(pairs, shots, tag):
    b = real_backend()
    ts = _transpile(pairs, b)
    print(json.dumps({"backend": b.name, "circuits": len(ts), "shots": shots, "qubit": QUBIT}), flush=True)
    t0 = time.time()
    job = b.run(ts, shots=shots)
    print(json.dumps({"job_id": job.job_id()}), flush=True)
    res = job.result()
    counts = {lab: res.get_counts(i) for i, (lab, _) in enumerate(pairs)}
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f"iqm_zitterbewegung_qg_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({"job_id": job.job_id(), "backend": b.name, "qubit": QUBIT, "shots": shots,
                   "wall_seconds": round(time.time() - t0, 1), "counts": counts}, fh, indent=1)
    return counts


def main(argv):
    if "--exact" in argv:
        exact()
        return
    if "--prereg" in argv:
        with open(__file__, "rb") as fh:
            d = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()  # LF-normalized: verifiable from git
        from datetime import datetime, timezone
        os.makedirs(DATA, exist_ok=True)
        with open(PREREG, "w", encoding="utf-8") as fh:
            fh.write(f"{d}  examples/iqm_zitterbewegung_qg.py  {datetime.now(timezone.utc).isoformat()}\n")
        print(d)
        return
    if "--run" in argv:
        if "--yes-i-accept-qpu-cost" not in argv:
            sys.exit("This spends IQM credits (about 7.7). Re-run with --yes-i-accept-qpu-cost once approved.")
        if not os.path.exists(PREREG):
            sys.exit("Freeze the predictions first: python examples/iqm_zitterbewegung_qg.py --prereg")
        counts = submit(circuits(), SHOTS, "run")
        r = analyse(counts)
        with open(os.path.join(DATA, "iqm_zitterbewegung_qg_analysis.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        print(json.dumps(_summary(r)))
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in r["verdict"].items()))
    else:
        rehearsal()


if __name__ == "__main__":
    main(sys.argv[1:])
