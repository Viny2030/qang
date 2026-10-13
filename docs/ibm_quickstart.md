# qang on IBM Quantum — quickstart

`qang.ibm` reads qg quantities from IBM devices through IBM Runtime. Everything
below runs locally on the fake backends (`qiskit_ibm_runtime.fake_provider`,
which simulate each device's noise) and unchanged on a real device: only the
backend changes.

```bash
pip install "qang[hardware]"     # qiskit, qiskit-aer, qiskit-ibm-runtime
```

## 1. Is the qg filter applicable? (transpiler pass)

The qg filter keeps the shots of one Hamming weight. It applies when every gate
conserves the weight (excitation number).

```python
from qiskit.transpiler import PassManager
from qang import ibm

pm = PassManager([ibm.WeightPreservationCheck()])
pm.run(qc)
pm.property_set["qg_weight_preserving"]   # True / False / None (undecided)
pm.property_set["qg_initial_weight"]      # the sector to filter on, e.g. 1
pm.property_set["qg_non_preserving_gates"]
```

Run it on the logical circuit: after transpilation the native gates (√X, CZ,
ECR) are not weight-preserving one by one, even when the whole RBS is.

## 2. Sampler: filter + calibrated readout correction

```python
from qiskit import transpile
from qiskit_ibm_runtime.executor_sampler import Sampler
from qiskit_ibm_runtime.fake_provider import FakeTorino

backend = FakeTorino()
t = transpile(qc_measured, backend, optimization_level=1)
res = Sampler(mode=backend).run([t], shots=4000).result()[0]
out = ibm.ibm_qg(res, t, backend, weight=1)
out["raw"], out["readout"], out["filtered"], out["filtered_readout"]   # [(qg_Z, se)] per qubit
```

Readout correction uses the published asymmetric readout errors of the
physical qubits the circuit measures. On three fake backends (§137) the qg_Z
error falls from 0.065–0.104 raw to 0.013–0.022 with filter + readout
correction.

## 3. Estimator

```python
from qiskit_ibm_runtime import EstimatorV2

t = transpile(qc_unmeasured, backend, optimization_level=1)
ibm.estimator_qg(EstimatorV2(mode=backend), t, weight=1)
# {"qg": [(value, std)], "kept": ..., "filtered": [...]}
```

The qg observables (Z_i, the weight projector P_w and P_w Z_i) are mapped
through the circuit's layout, so Runtime's own mitigation (TREX, ZNE) applies
to them on a real device.

## 4. Choose the qubits before spending device time

```python
pred = ibm.predict_kept_fraction(t, backend, weight=1)
pred["kept_mixed_floor"]                  # predicted share of shots the filter keeps
layout, pred, t = ibm.best_chain(qc_measured, backend, weight=1)
```

`best_chain` searches chains of the coupling map and keeps the one with the
highest predicted kept fraction; on three fake backends (§138) it kept 6–17
points more shots than the transpiler's default layout for chain circuits.
For circuits that are not chains (rings, like qang's QNN) use
`ibm.best_layout(circuit, backend, weight)`: it scores the circuit as it will
run, routing included, and includes the transpiler's default among the
candidates, so it is never worse than the default by the prediction (§141).

## 4a. How many shots, how many device seconds?

```python
ibm.plan_shots(t, backend, weight=1, target_se=0.02)
# {"shots": ~3000-3400, "kept": ..., "qpu_seconds": ~0.8-0.9, ...}
```

The plan accounts for the shots the filter will discard; on three device
noise models it met a 0.02 target (§143). Leave `qg=None` (worst case) unless
you know the expected values.

## 4b. Zero-noise extrapolation with the filter

```python
r = ibm.zne_qg(Sampler(mode=backend), t, backend, weight=1, scales=(1, 3))
r["zne"]["filtered"]                      # qg_Z: filter + ZNE (folded two-qubit gates)
```

On IBM devices Runtime's own ZNE (`resilience_level`) can be used with the
Estimator path; in local testing mode it has no effect, so `zne_qg` folds the
gates itself. Filter + ZNE gave qg_Z errors of 0.011–0.015 on three device
noise models (§142).

## 5. The weight-conserving QNN in one job

```python
from qang.qml import WeightQNN

model = WeightQNN(5, 1).fit(X_train, y_train)          # trained classically
r = ibm.run_qnn(model, X_test, backend, layout="best", shots=1000)
r["decision"]["qang_readout"] > 0                       # predictions
```

`qnn_template(model)` is the model as one parameterized circuit of native
XX+YY gates; `run_qnn` transpiles it once and sends the whole test set as a
single Sampler pub.

## 6. A PCA spectrum in the weight-1 sector

```python
O, lam = ibm.pca_rotation(X)                  # X: (m, d) data; O sends component k to qubit k
qcs = []
for x in X - X.mean(axis=0):
    qc = ibm.unary_loader(x)                   # sum_k x_k |e_k>, d - 1 RBS
    qc.compose(ibm.unary_orthogonal(O), inplace=True)   # d(d-1)/2 nearest-neighbour RBS
    qc.measure_all()
    qcs.append(qc)
# run them, ibm_qg(..., weight=1) on each, then
w = ((X - X.mean(axis=0)) ** 2).sum(axis=1)
ibm.unary_spectrum(outs, w)                    # estimate of lam / lam.sum()
```

The weighted mixture of loaded vectors is the density matrix that quantum PCA
diagonalizes; the filter removes the upward bias that out-of-sector shots
give every eigenvalue. On 21 device noise models (§146, d = 5) filter +
readout cut the spectrum error 4.6× (median). No quantum speed-up is claimed:
d features use d qubits.

## 7. Hamiltonian simulation (Trotter) and quantum walks

```python
qc = ibm.xy_trotter(n, J, h, t, steps, initial=(0,))    # XY chain, first-order Trotter, measured
ibm.xy_exact_occupations(n, J, h, t, initial=0)        # exact single-excitation dynamics
out = ibm.ibm_qg(res, transpile(qc, backend), backend, weight=1)
ibm.occupations(out)                                    # P(qubit k excited), filter + readout
```

Every gate conserves the excitation number, so the filter applies. On 21
device noise models (§147) it halved the best simulation error and let the
simulation use more Trotter steps (median optimum 12 against 8); with h = 0
and one excitation the circuit is a quantum walk, whose ballistic spreading
is visible only with the filter.

## On a real device

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()                        # saved account
backend = service.least_busy(operational=True, simulator=False)
# then exactly as above
```

Mind the cost: device time counts against your plan. `predict_kept_fraction`
and `best_chain` cost nothing and tell you in advance how many shots the filter
will keep.
