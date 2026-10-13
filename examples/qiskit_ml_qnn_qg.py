"""
qang's QNN inside qiskit-machine-learning: the SamplerQNN on a noiseless and
on device-noise samplers (§140)

qang.qiskit_ml.qang_sampler_qnn builds a qiskit-machine-learning SamplerQNN
from qang's weight-1 QNN (loader angles as inputs, RBS angles as weights);
QangQNNClassifier runs a trained qang.qml.WeightQNN through it and reads the
decision from the raw or the qg-filtered qg_Z. Any qiskit-machine-learning
sampler works; here the qiskit StatevectorSampler and IBM Runtime's Sampler in
local mode on FakeBrisbane, FakeSherbrooke and FakeTorino (Aer with each
device's noise model), with a preset pass manager for the device
(optimization_level 1, seed 137).

Plumbing note (before any noisy result was seen): the first attempt used Aer's
SamplerV2 directly; on a 127/133-qubit device circuit it returned no shots,
and qiskit-machine-learning 0.9.1 reads measured bits at the physical-qubit
positions of the layout, which returns 0 for circuits measured before
transpilation. qang_sampler_qnn now transpiles itself and hands over the
circuit without its layout, and the Runtime local Sampler is used.

Set-up (fixed before the run): model A of §124 (iris, cancer, wine, digits;
189 test inputs), 1000 shots per input everywhere. Noiseless model: 185.

Pre-registered predictions (committed before the run):
  S1  with the noiseless StatevectorSampler (1000 shots) the filtered
      decision gets at least 183 of 189.
  S2  on every noisy backend the filtered decision is at least as accurate
      as the raw one.
  S3  pooled over the three noisy backends the filtered decision is within
      3 inputs of §139's default-layout result with the filter (550).

python examples/qiskit_ml_qnn_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import qnn_gate_noise_qg as G  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
from qang import qiskit_ml as QM  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino")
SHOTS = 1000
OUT = os.path.join(HERE, "data", "qiskit_ml_qnn_qg.json")


def samplers():
    from qiskit.primitives import StatevectorSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit_ibm_runtime.executor_sampler import Sampler

    yield "noiseless", StatevectorSampler(default_shots=SHOTS, seed=140), None
    for name in BACKENDS:
        b = getattr(fp, name)()
        s = Sampler(mode=b, options={"default_shots": SHOTS})
        yield name, s, generate_preset_pass_manager(backend=b, optimization_level=1, seed_transpiler=137)


def run():
    models = {}
    for ds in Q.DATASETS:
        Xtr, Xte, ytr, yte = P.data(ds)
        models[ds] = (WeightQNN(G.N, 1).fit(Xtr, ytr, G.EPOCHS, seed=DS.TRAIN_SEED), Xte, yte)
    rows = {}
    for name, sampler, pm in samplers():
        acc = {"raw": 0, "qang": 0, "inputs": 0}
        for ds, (A, Xte, yte) in models.items():
            c = QM.QangQNNClassifier(A, sampler=sampler, pass_manager=pm)
            probs = c.probabilities(Xte)
            for k, f in (("raw", False), ("qang", True)):
                acc[k] += int(np.sum((c.decision(Xte, f, probs=probs) > 0).astype(int) == yte))
            acc["inputs"] += len(yte)
        rows[name] = acc
        print(json.dumps({name: acc}), flush=True)
    return rows


def verdict(rows):
    return {"S1": rows["noiseless"]["qang"] >= 183,
            "S2": all(rows[b]["qang"] >= rows[b]["raw"] for b in BACKENDS),
            "S3": abs(sum(rows[b]["qang"] for b in BACKENDS) - 550) <= 3}


def main():
    rows = run()
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
