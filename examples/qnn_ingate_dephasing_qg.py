"""
Dephasing inside the gates in qang's QNN simulator (§133)

§132 found, after scoring its predictions, that the kept-fraction residual on
torino is pure dephasing during the gates: inside the CZ/ECR decomposition of
an RBS the qubits pass through non-computational frames, so a phase error
there changes the Hamming weight, while qang's simulator applies dephasing
only between RBS gates, where it cannot. Here that error is put inside every
RBS.

Model. For qang qubit q on physical qubit Q, the pure-dephasing rate is
1/T_phi = 1/T2 - 1/(2 T1) (published values). Its gate time per RBS, t_q, is
the summed duration of the gates acting on Q in the transpiled circuit
divided by the number of RBS gates on q (4 loader + trained). A Z error with
probability z_q = (1 - exp(-t_q / T_phi)) / 2 seen in a random frame is
twirled into a single-qubit depolarizing channel of probability 4 z_q / 3,
applied after each RBS on each of its two qubits. Everything else is the §132
simulator: the two-qubit depolarizing channel (§124), the single-qubit gate
channel (§132), T1/T2 between sublayers (§119) and the asymmetric readout
error (§129).

No new device simulation is run: the Aer values (thermal channel alone, full
model) are those recorded by §132 in examples/data/qnn_torino_residual.json,
and the device values are §124's.

Pre-registered predictions (committed before the run):
  D1  on torino, qang's thermal channel alone with in-gate dephasing predicts
      Aer's thermal channel alone (0.919) within 0.02.
  D2  on brisbane and sherbrooke, the same within 0.03 (Aer 0.777, 0.888).
  D3  the full simulator with in-gate dephasing predicts the device kept
      fraction within 0.03 on all three backends.
  D4  on torino, the full prediction moves at least half of the way from
      §132's 0.761 (full + 1q) toward the device value 0.724.

python examples/qnn_ingate_dephasing_qg.py

Findings:

Kept fraction of model A (mean over the four datasets):

                          brisbane   sherbrooke   torino
  in-gate depolarizing    0.004-0.010  0.001-0.003  0.0005-0.006  (per qubit per RBS)
  thermal, qang §132        0.882      0.934       0.974
  thermal + in-gate         0.818      0.914       0.937
  thermal, Aer              0.777      0.888       0.919
  full, qang §132 (+ 1q)    0.644      0.680       0.761
  full + in-gate            0.606      0.668       0.736
  device (§124)             0.671      0.697       0.724

  D1 PASS: on torino the thermal channel is now within 0.019 of Aer's
     (0.056 before).
  D2 FAIL: within 0.026 on sherbrooke, but 0.040 on brisbane.
  D3 FAIL: within 0.03 on torino (0.012) and sherbrooke (0.029), not on
     brisbane (0.065 too low).
  D4 PASS: on torino the full prediction moves 0.025 of the 0.037 toward
     the device (68%).
  * Dephasing inside the gates explains most of the torino residual: with
    it the simulator is within 0.012 of the device there, against 0.059 in
    §129 and 0.037 in §132.
  * On brisbane, the noisiest device, every channel of qang's simulator
    loses less than Aer's, yet the full simulator loses more than Aer's
    full model: the channels do not combine the same way. Per-channel
    agreement does not carry into the total there; the cause is not
    identified.
  * For prediction, §129's simpler model remains the closest on the two
    ECR devices (within 0.006), and this one on torino (within 0.012).
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import qnn_gate_noise_qg as G  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
import qnn_readout_error_qg as RO  # noqa: E402
import qnn_torino_residual_qg as T  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

N = G.N
ONE = G.ONE
PREV = os.path.join(HERE, "data", "qnn_torino_residual.json")
OUT = os.path.join(HERE, "data", "qnn_ingate_dephasing.json")


class InGateQNN(T.GateNoise1QNN):
    """GateNoise1QNN with a per-qubit single-qubit depolarizing channel ``ingate`` (array over qang qubits)
    after each RBS on each of its two qubits, on top of the uniform ``one_qubit_error``."""

    ingate = None

    def probs(self, theta, psi, gamma=None, dephasing=0.0):
        if self.ingate is None or not np.any(self.ingate):
            return super().probs(theta, psi, gamma, dephasing)
        self._tables()
        n = self.n
        gam = None if gamma is None else np.broadcast_to(np.asarray(gamma, float), (n,))
        rho = np.einsum("si,sj->sij", psi, psi)
        k = 0
        p1 = self.one_qubit_error
        for _ in range(self.layers):
            for pairs in self.sublayers:
                for a, b in pairs:
                    U = self._rbs(a, b, theta[k])
                    k += 1
                    rho = U[None] @ rho @ U.T[None]
                    if self.gate_error > 0:
                        rho = (1 - self.gate_error) * rho + self.gate_error * self._depol_pair(rho, a, b)
                    for q in (a, b):
                        p = 1 - (1 - p1) * (1 - self.ingate[q])
                        if p:
                            rho = (1 - p) * rho + p * self._depol_one(rho, q)
                if dephasing:
                    rho = rho * ((1 - 2 * dephasing) ** self._ham)[None]
                for q in range(n):
                    g = 0.0 if gam is None else gam[q]
                    if g:
                        bq = self._b[q]
                        f = np.where(bq, math.sqrt(1 - g), 1.0)
                        nb = (~bq).astype(float)
                        jump = g * rho[:, self._perm[q]][:, :, self._perm[q]] * (nb[:, None] * nb[None, :])[None]
                        rho = rho * (f[:, None] * f[None, :])[None] + jump
        return np.einsum("sii->si", rho).copy()


def rbs_per_qubit(model):
    """RBS gates acting on each qang qubit in the device circuit: 4 loader gates (k, k+1) + the trained ones."""
    c = np.zeros(N)
    for k in range(N - 1):
        c[k] += 1
        c[k + 1] += 1
    for _ in range(model.layers):
        for pairs in model.sublayers:
            for a, b in pairs:
                c[a] += 1
                c[b] += 1
    return c


def ingate_calibration(backend, circuit, layout, model):
    """Per qang qubit: gate time per RBS, pure-dephasing time and the twirled depolarizing probability."""
    from qiskit import transpile

    tc = transpile(circuit, backend=backend, optimization_level=1, seed_transpiler=119, initial_layout=layout)
    t = backend.target
    busy = {}
    for ins in tc.data:
        name = ins.operation.name
        if name in ("barrier", "measure", "rz"):
            continue
        qs = tuple(tc.find_bit(q).index for q in ins.qubits)
        dur = t[name][qs].duration or 0.0
        for q in qs:
            busy[q] = busy.get(q, 0.0) + dur
    nr = rbs_per_qubit(model)
    out = {"t per RBS": [], "T_phi": [], "z": [], "depolarizing": []}
    for q in range(N):
        phys = layout[N - 1 - q]  # qang qubit q = Qiskit virtual qubit N-1-q
        qp = t.qubit_properties[phys]
        rate = max(1 / min(qp.t2, 2 * qp.t1) - 1 / (2 * qp.t1), 0.0)
        tq = busy.get(phys, 0.0) / nr[q]
        z = (1 - math.exp(-tq * rate)) / 2
        out["t per RBS"].append(tq)
        out["T_phi"].append(None if rate == 0 else 1 / rate)
        out["z"].append(z)
        out["depolarizing"].append(4 * z / 3)
    return out


def run_backend(name, datasets=Q.DATASETS):
    backend = H.get_backend("fake", name)
    layout = None
    kept = {"thermal": [], "thermal + in-gate": [], "full 132": [], "full + in-gate": []}
    for ds in datasets:
        Xtr, Xte, ytr, yte = P.data(ds)
        A = WeightQNN(N, 1).fit(Xtr, ytr, G.EPOCHS, seed=DS.TRAIN_SEED)
        if layout is None:
            c0 = H.build_circuit(A, Xte[0])
            layout, cal = P.calibration(backend, c0)
            p2 = G.gate_calibration(backend, c0, layout)["depolarizing"]
            p1 = T.one_qubit_calibration(backend, c0, layout)["depolarizing"]
            icl = ingate_calibration(backend, c0, layout, A)
            noise = {"gamma": cal["gamma"], "dephasing": cal["dephasing"]}
            ro = RO.readout_errors(backend, layout)
        th, psi = A.params_[: A.n_theta], A.encode(Xte)

        def sim(p2_=0.0, p1_=0.0, ingate=False, readout=False):
            m = InGateQNN(N, 1)
            m.gate_error, m.one_qubit_error = p2_, p1_
            m.ingate = np.array(icl["depolarizing"]) if ingate else None
            pr = m.probs(th, psi, **noise)
            if readout:
                pr = RO.apply_readout(pr, ro)
            return float(np.mean(pr[:, ONE].sum(axis=1)))

        kept["thermal"].append(sim())
        kept["thermal + in-gate"].append(sim(ingate=True))
        kept["full 132"].append(sim(p2, p1, False, True))
        kept["full + in-gate"].append(sim(p2, p1, True, True))
    return {"backend": name, "in-gate calibration": icl, "sim": {k: float(np.mean(v)) for k, v in kept.items()}}


def previous():
    with open(PREV, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    return {r["backend"]: {"aer thermal": r["aer"]["thermal"], "aer full": r["aer"]["full"],
                           "device": r["kept device 124"], "sim full + 1q 132": r["sim"]["full + 1q"]} for r in rows}


def verdict(rows, prev):
    by = {r["backend"]: r["sim"] for r in rows}
    d1 = abs(by["fake_torino"]["thermal + in-gate"] - prev["fake_torino"]["aer thermal"]) <= 0.02
    d2 = all(abs(by[b]["thermal + in-gate"] - prev[b]["aer thermal"]) <= 0.03 for b in ("fake_brisbane", "fake_sherbrooke"))
    d3 = all(abs(by[b]["full + in-gate"] - prev[b]["device"]) <= 0.03 for b in by)
    t = prev["fake_torino"]
    d4 = (t["sim full + 1q 132"] - by["fake_torino"]["full + in-gate"]) >= 0.5 * (t["sim full + 1q 132"] - t["device"])
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4}


def main():
    prev = previous()
    rows = []
    for name in P.BACKENDS:
        r = run_backend(name)
        p = prev[name]
        print(json.dumps({"backend": name, "device": round(p["device"], 4), "aer thermal": round(p["aer thermal"], 4),
                          "sim": {k: round(v, 4) for k, v in r["sim"].items()},
                          "in-gate depolarizing": [round(x, 4) for x in r["in-gate calibration"]["depolarizing"]],
                          "t per RBS (ns)": [round(1e9 * x) for x in r["in-gate calibration"]["t per RBS"]]}),
              flush=True)
        rows.append(r)
    v = verdict(rows, prev)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
