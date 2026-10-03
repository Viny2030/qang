"""
§88 revisited: a joint fit of all delays instead of subtracting the t = 0
floor (§92)

§88 read the damping gamma and the dephasing p of an idle period from the
Leung-code syndromes, after removing the flips of encoding and extraction
measured at t = 0. On fake_sherbrooke it failed: the floor (9% ZZ flips) was
as large as the signal, the ZZ rate at 10 us came out below the t = 0 floor,
and the floor-corrected gamma was 32-47% low. Here the floor is not taken
from the t = 0 circuits alone. All delays are fitted at once by binomial
maximum likelihood with four parameters, the two floor rates (r0_ZZ,
r0_XXXX), an effective T1 and an effective T_phi:

    gamma(t) = 1 - exp(-t/T1),   p(t) = (1 - exp(-t/T_phi))/2
    idle rates (§33 closed forms):  ZZ: gamma(1 - gamma);
                                    XXXX: (1 - (1 - gamma)^2 (1 - 2p)^4)/2
    observed rate = r0 + r_idle - 2 r0 r_idle   (independent flips)

The delays go up to 160 us (6 delays), so that qubits with T1 ~ 300 us show
enough damping. The same circuits as §88 (examples/qec_syndrome_hardware_qg.py).
Every result is given for the joint fit and for the §88 floor subtraction on
the same counts.

Pre-registered predictions (committed before the run; fake IBM backends
brisbane, sherbrooke and torino, 4000 shots per delay and logical state):
  J1  with the joint fit, gamma(t) is within 30% of the calibration gamma at
      t = 40, 80 and 160 us on all three backends.
  J2  with the joint fit, p(t) is within a factor 2 of the calibration p at
      t = 40, 80 and 160 us on all three backends.
  J3  on fake_sherbrooke the joint fit is closer to the calibration gamma
      than the §88 floor subtraction at t = 40 and 80 us.

Needs qiskit, qiskit-aer, qiskit-ibm-runtime and scipy.

Findings (python examples/qec_syndrome_jointfit_qg.py):

Fake IBM backends, 4000 shots per delay and logical state. gamma and p at
t = 40 / 80 / 160 us: calibration | joint fit | §88 floor subtraction.

  brisbane    gamma 0.115/0.216/0.382 | 0.201/0.362/0.593 | 0.196/0.316/0.388
              p     0.066/0.111/0.167 | 0.133/0.230/0.355 | 0.114/0.183/0.259
  sherbrooke  gamma 0.122/0.228/0.402 | 0.113/0.213/0.380 | 0.116/0.221/0.342
              p     0.084/0.128/0.173 | 0.022/0.043/0.082 | 0.019/0.042/0.101
  torino      gamma 0.222/0.387/0.607 | 0.180/0.328/0.548 | 0.178/0.311/0.369
              p     0.070/0.123/0.194 | 0.216/0.339/0.448 | 0.233/0.403/0.388

  * J1, J2, J3 all FAIL. The joint fit is not better than the floor
    subtraction: gamma is 1.6-1.7x high on brisbane, p is 3x high on torino
    and 0.3-0.5x low on sherbrooke. On sherbrooke the §88 failure does not
    even reproduce (both methods within 3-15% for gamma), because the
    transpiler chose different qubits this time.
  * Diagnosis (after the run, not pre-registered). The fit itself is exact:
    on synthetic counts from the model it returns the floor, T1 and T_phi
    to 1e-8. What fails is the comparison. The encoder (d0 controls three
    CNOTs) and the XXXX ancilla (four CNOTs) need degree-3 and degree-4
    vertices, which heavy-hex lattices do not have, so the transpiler
    inserts SWAPs: the data sit on different physical qubits before and
    after routing, and §88-§92 read the calibration from the final layout,
    which is not where the data idled. The SWAPs also add errors that are
    not idle errors. This also weakens the §88 comparison.
  * Next step (§93): a circuit without SWAPs and without ancillas. With
    one round per shot the stabilizers can be read destructively: half
    the shots measure the data in Z (Z0Z1, Z2Z3 from the parities), half
    in X (XXXX), after a linear-chain encoder placed on a path of the
    device, so that the idle qubits are known exactly.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qec_syndrome_hardware_qg as S  # noqa: E402
from qec_syndrome_drift_tracking_qg import estimate_from_leung  # noqa: E402

DELAYS_US = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0)
CHECK_T = (40.0, 80.0, 160.0)
BACKENDS = ("fake_brisbane", "fake_sherbrooke", "fake_torino")


def run(backend, shots, delays=DELAYS_US, seed=92):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    circs, spec = [], []
    for t in delays:
        for logical in (0, 1):
            circs.append(S.build(t, logical))
            spec.append(t)
    sim = AerSimulator.from_backend(backend)
    tc = transpile(circs, backend=sim, optimization_level=1, seed_transpiler=seed, scheduling_method="alap")
    res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    per_t = {}
    for t, i in zip(spec, range(len(tc))):
        zz, xx, n = S.flips(res.get_counts(i))
        a = per_t.setdefault(t, [0, 0, 0])
        a[0] += zz
        a[1] += xx
        a[2] += n
    return per_t, tc[0].layout.final_index_layout()[:4]


def model_rates(t, r0z, r0x, T1, Tphi):
    g = 1 - math.exp(-t / T1)
    p = (1 - math.exp(-t / Tphi)) / 2
    iz = g * (1 - g)
    ix = (1 - (1 - g) ** 2 * (1 - 2 * p) ** 4) / 2
    return r0z + iz - 2 * r0z * iz, r0x + ix - 2 * r0x * ix


def joint_fit(per_t):
    from scipy.optimize import minimize

    def nll(x):
        r0z, r0x = 1 / (1 + math.exp(-x[0])), 1 / (1 + math.exp(-x[1]))
        T1, Tphi = math.exp(x[2]), math.exp(x[3])
        s = 0.0
        for t, (zz, xx, n) in per_t.items():
            pz, px = model_rates(t, r0z, r0x, T1, Tphi)
            pz, px = min(max(pz, 1e-12), 1 - 1e-12), min(max(px, 1e-12), 1 - 1e-12)
            s -= zz * math.log(pz) + (2 * n - zz) * math.log(1 - pz)
            s -= xx * math.log(px) + (n - xx) * math.log(1 - px)
        return s

    zz0, xx0, n0 = per_t[0.0]
    x0 = [math.log(max(zz0 / (2 * n0), 1e-4) / (1 - zz0 / (2 * n0))), math.log(max(xx0 / n0, 1e-4) / (1 - xx0 / n0)),
          math.log(150.0), math.log(150.0)]
    best = None
    for l1 in (math.log(50), math.log(150), math.log(400)):
        for l2 in (math.log(50), math.log(150), math.log(400)):
            r = minimize(nll, [x0[0], x0[1], l1, l2], method="Nelder-Mead",
                         options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000})
            if best is None or r.fun < best.fun:
                best = r
    x = best.x
    return {"r0_ZZ": 1 / (1 + math.exp(-x[0])), "r0_XXXX": 1 / (1 + math.exp(-x[1])),
            "T1": math.exp(x[2]), "T_phi": math.exp(x[3])}


def floor_subtraction(per_t, t):
    zz0, xx0, n0 = per_t[0.0]
    r0z, r0x = zz0 / (2 * n0), xx0 / n0
    zz, xx, n = per_t[t]
    iz = max((zz / (2 * n) - r0z) / (1 - 2 * r0z), 0.0)
    ix = max((xx / n - r0x) / (1 - 2 * r0x), 0.0)
    return estimate_from_leung(iz * 2 * n, ix * n, n)


def analyse(backend, per_t, layout):
    fit = joint_fit(per_t)
    rows = []
    for t in CHECK_T:
        g_cal, p_cal = S.calibration(backend, layout, t)
        g_fit = 1 - math.exp(-t / fit["T1"])
        p_fit = (1 - math.exp(-t / fit["T_phi"])) / 2
        g_sub, p_sub = floor_subtraction(per_t, t)
        rows.append({"t_us": t, "gamma calibration": g_cal, "gamma joint fit": g_fit, "gamma floor subtraction": g_sub,
                     "p calibration": p_cal, "p joint fit": p_fit, "p floor subtraction": p_sub})
    return fit, rows


def verdict(results):
    j1 = all(abs(r["gamma joint fit"] / r["gamma calibration"] - 1) < 0.30 for b in results.values() for r in b["rows"])
    j2 = all(0.5 <= r["p joint fit"] / max(r["p calibration"], 1e-12) <= 2.0 for b in results.values() for r in b["rows"])
    sh = {r["t_us"]: r for r in results["fake_sherbrooke"]["rows"]}
    j3 = all(abs(sh[t]["gamma joint fit"] - sh[t]["gamma calibration"]) < abs(sh[t]["gamma floor subtraction"] - sh[t]["gamma calibration"])
             for t in (40.0, 80.0))
    return {"J1": j1, "J2": j2, "J3": j3}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    results = {}
    for name in BACKENDS:
        backend = S.get_backend("fake", name)
        per_t, layout = run(backend, args.shots)
        fit, rows = analyse(backend, per_t, layout)
        results[name] = {"fit": fit, "rows": rows, "data qubits": [int(q) for q in layout]}
        print(name, {k: round(v, 4) for k, v in fit.items()})
        for r in rows:
            print("  ", {k: round(v, 4) for k, v in r.items()}, flush=True)
    v = verdict(results)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if args.out:
        json.dump({"results": results, "verdict": v}, open(args.out, "w"), indent=1)
    return results, v


if __name__ == "__main__":
    main()
