"""
qang.iqm on IQM's fake Garnet (IQMFakeApollo): readout correction from the poles, and the chain choice (book ch. 31)

Free: nothing is sent to IQM. Run in the environment that has iqm-client[qiskit] (.venv-iqm):

    python examples/iqm_readout_chain_qg.py

The §165 neutrino circuits (8 energies, initial nu_mu, L = 1300 km) are transpiled as in §165 (optimization_level 3,
seed_transpiler 137), once with the default layout and once on ``qang.iqm.best_chain``; two pole circuits on the same
physical qubits give the readout errors, and every circuit runs 4000 shots (seed_simulator 31). Error: mean
|P_hat - P| over 3 flavours x 8 energies, for the four readings of ``qang.iqm.iqm_qg``.

Findings (2026-10-25):
  * The two poles of the stored §168 run give e0 = 0.0055, e1 = 0.029, the published values.
  * The fake backend's error profile and the poles agree to within 0.006 (e.g. 0.053 against 0.057); the poles
    also carry the X gate's error and shot noise.
  * default layout: raw 0.1240, filtered 0.0554, readout-corrected 0.0971, filtered after correction 0.0470;
    the correction §165 did not apply removes a further 15% of the filtered error.
  * kept fraction 0.689 simulated; predicted 0.640 (``kept``, a lower bound) and 0.775 (``kept_mixed_floor``):
    the prediction brackets it but is less accurate than on IBM's models (§138).
  * best_chain scored 61 candidates in 3 s; the prediction barely separates them (0.777 against 0.775 for the
    default), because the fake's qubits are nearly uniform. Its chain read 0.0416 filtered after correction, one
    seed: not a claim that the chain choice helps on Garnet.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import iqm as Q  # noqa: E402
from qang import neutrino as nu  # noqa: E402
from qang.ibm import measured_physical_qubits  # noqa: E402

L, DCP = 1300.0, -np.pi / 2
ENERGIES = (0.8, 1.2, 1.6, 2.0, 2.6, 3.2, 4.0, 5.0)
READINGS = ("raw", "filtered", "readout", "filtered_readout")


def poles_168():
    with open(os.path.join(HERE, "data", "iqm_milliqang_qg_run.json"), encoding="utf-8") as fh:
        c = json.load(fh)["counts"]
    return Q.readout_from_poles(c["pole0"], c["pole1"])[0]


def run(backend, layout=None, shots=4000, seed=31):
    from qiskit import transpile

    kw = {"initial_layout": layout} if layout else {}
    ts = [transpile(nu.three_flavour_circuit("mu", L, E, DCP), backend, optimization_level=3, seed_transpiler=137, **kw)
          for E in ENERGIES]
    phys = measured_physical_qubits(ts[0])
    poles = [transpile(p, backend, initial_layout=phys, optimization_level=0) for p in Q.pole_circuits(3)]
    res = backend.run(ts + poles, shots=shots, seed_simulator=seed).result()
    counts = [res.get_counts(i) for i in range(len(ts) + 2)]
    errors = Q.readout_from_poles(counts[-2], counts[-1])
    exact = [nu.three_flavour_probabilities(L, E, DCP)[1] for E in ENERGIES]
    err, kept = {k: [] for k in READINGS}, []
    for c, t, ex in zip(counts, ts, exact):
        r = Q.iqm_qg(c, t, errors=errors, weight=1)
        kept.append(r["kept_raw"])
        for k in READINGS:
            err[k].append(np.abs(np.array([(1 - z) / 2 for z, _ in r[k]]) - ex).mean())
    pred = Q.predict_kept_fraction(ts[0], backend, 1)
    return {"physical_qubits": phys, "poles": errors, "profile": Q.readout_errors(backend, phys),
            "error": {k: float(np.mean(v)) for k, v in err.items()}, "kept": float(np.mean(kept)),
            "predicted_kept": pred["kept"], "predicted_floor": pred["kept_mixed_floor"]}


def main():
    from iqm.qiskit_iqm import IQMFakeApollo

    print(json.dumps({"poles §168": poles_168()}))
    b = IQMFakeApollo()
    best = Q.best_chain(nu.three_flavour_circuit("mu", L, 2.6, DCP), b, 1, limit=60)
    print(json.dumps({"best layout": best["layout"], "candidates": best["candidates"],
                      "predicted floor": round(best["prediction"]["kept_mixed_floor"], 3),
                      "default floor": round(best["default_prediction"]["kept_mixed_floor"], 3)}))
    for tag, lay in (("default", None), ("best_chain", best["layout"])):
        r = run(b, lay)
        print(json.dumps({"layout": tag, "error": {k: round(v, 4) for k, v in r["error"].items()},
                          "kept": round(r["kept"], 3), "predicted": round(r["predicted_kept"], 3),
                          "predicted floor": round(r["predicted_floor"], 3),
                          "poles": [[round(x, 3) for x in e] for e in r["poles"]], "profile": r["profile"]}))


if __name__ == "__main__":
    main()
