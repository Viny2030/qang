"""
Why §93 read dephasing 1.17-1.51x high: the wrong yardstick (§99)

Post-hoc diagnosis, not a pre-registered study. §93 read the damping of four
data qubits from the Leung syndromes to 1-4% of the calibration, but the
dephasing p came out 1.17-1.51x above it, more so at longer delays.

The XXXX syndrome is the coherence of |0000> + |1111>, which decays as the
product over the four qubits, prod_i sqrt(1 - gamma_i) (1 - 2 p_i). The
joint fit models it with one gamma and one p, so the p it returns is the
"product mean"

    (1 - 2 p_eff)^4 (1 - gamma)^2 = prod_i sqrt(1 - gamma_i) (1 - 2 p_i),

dominated by the worst qubit. §93 compared it with the arithmetic mean of the
four p_i. On every path one qubit has a short T2 (30 us on brisbane, 63 us
on sherbrooke, 28 us on torino, against 110-332 us for the others), and
then the two means differ.

This script checks the two steps:
  1  Aer follows the model: the measured XXXX decay (200 000 shots) equals
     prod_i exp(-t/T2_i) of the four qubits.
  2  Against the product mean the §93 fit agrees to 0.99-1.04 at every delay
     on all three backends; against the arithmetic mean it is 1.17-1.51x.

python examples/qec_syndrome_dephasing_target_qg.py  (needs qiskit,
qiskit-aer, qiskit-ibm-runtime; about two minutes)

Findings:

  backend      worst T2   t (us)   §93 fit p   arithmetic mean   product mean
  brisbane     30 us      40       0.165       0.131 (1.26x)     0.166 (1.00x)
                          80       0.276       0.193 (1.43x)     0.276 (1.00x)
                          160      0.399       0.265 (1.51x)     0.400 (1.00x)
  sherbrooke   63 us      40       0.087       0.074 (1.17x)     0.084 (1.04x)
                          80       0.159       0.125 (1.27x)     0.153 (1.04x)
                          160      0.267       0.189 (1.41x)     0.260 (1.03x)
  torino       28 us      40       0.184       0.152 (1.21x)     0.186 (0.99x)
                          80       0.301       0.227 (1.33x)     0.302 (1.00x)
                          160      0.420       0.311 (1.35x)     0.422 (1.00x)

  The measured XXXX decay at 40 us matches prod_i exp(-t/T2_i): 0.1434
  against 0.1450 (brisbane), 0.3625 against 0.3622 (sherbrooke), 0.1013
  against 0.1021 (torino).
  Verdict. The syndromes were right; the comparison was not. With a spread
  of T2 the Leung XXXX syndrome reads the product mean of the dephasing,
  which is what limits the logical coherence of the code, so it is the
  relevant number. The §93 verdict (M3 within a factor 2) stands; the bias is
  explained, and a real-device comparison should use the product mean.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FIT_93 = {  # joint-fit p and gamma of §93 at 40, 80, 160 us
    "fake_brisbane": {"p": (0.165, 0.276, 0.399), "gamma": (0.149, 0.275, 0.474)},
    "fake_sherbrooke": {"p": (0.087, 0.159, 0.267), "gamma": (0.132, 0.246, 0.432)},
    "fake_torino": {"p": (0.184, 0.301, 0.420), "gamma": (0.191, 0.346, 0.572)},
}
CHECK_T = (40.0, 80.0, 160.0)


def targets(T1, T2, t, gamma_fit):
    """Arithmetic and product means of the per-qubit dephasing p_i at delay t (us)."""
    T1, T2 = np.asarray(T1, float), np.asarray(T2, float)
    g = 1 - np.exp(-t / T1)
    inv = np.maximum(1 / T2 - 1 / (2 * T1), 0.0)
    p = (1 - np.exp(-t * inv)) / 2
    coh = np.prod(np.sqrt(1 - g) * (1 - 2 * p))
    p_prod = (1 - (coh / (1 - gamma_fit) ** 2) ** 0.25) / 2
    return float(p.mean()), float(p_prod)


def xxxx_decay(backend, path, shots=200000):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    import qec_syndrome_destructive_qg as D

    sim = AerSimulator.from_backend(backend)
    e = {}
    for t in (0.0, 40.0):
        tc = transpile(D.build(t, 0, "X"), backend=sim, initial_layout=path, optimization_level=1,
                       seed_transpiler=93, scheduling_method="alap")
        c = sim.run(tc, shots=shots, seed_simulator=1).result().get_counts()
        e[t] = sum(m * (-1) ** (k.replace(" ", "").count("1") % 2) for k, m in c.items()) / shots
    return e[40.0] / e[0.0]


def main():
    import qec_syndrome_destructive_qg as D
    import qec_syndrome_hardware_qg as S

    for name in D.BACKENDS:
        b = S.get_backend("fake", name)
        path = D.best_path(b)
        qp = [b.qubit_properties(q) for q in path]
        T1 = [x.t1 * 1e6 for x in qp]
        T2 = [x.t2 * 1e6 for x in qp]
        pred = float(np.prod(np.exp(-40.0 / np.minimum(T2, 2 * np.asarray(T1)))))
        print(f"{name}: T2 = {np.round(T2, 1)} us; XXXX decay at 40 us measured {xxxx_decay(b, path):.4f}, "
              f"predicted {pred:.4f}")
        for i, t in enumerate(CHECK_T):
            pm, pp = targets(T1, T2, t, FIT_93[name]["gamma"][i])
            f = FIT_93[name]["p"][i]
            print(f"  t = {t:5.0f} us: fit {f:.3f}, arithmetic mean {pm:.3f} ({f / pm:.2f}x), "
                  f"product mean {pp:.3f} ({f / pp:.2f}x)")


if __name__ == "__main__":
    main()
