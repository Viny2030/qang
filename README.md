# quang — The Qang (qg) unit

Reference implementation for:

> V. H. Monteverde, *"The Qang (qg): A Unified Angular-Probability Unit and
> Metric for Parametric Quantum Circuit Design."*
> ORCID: [0000-0001-8884-4811](https://orcid.org/0000-0001-8884-4811) ·
> DOI: [10.5281/zenodo.22832150](https://doi.org/10.5281/zenodo.22832150)

The paper defines two operational units on the Bloch-sphere polar angle
theta: the **Polar Bias Qang** `qg_Z(theta) = cos(theta) = <sigma_z>`, and
the **Measurement-Outcome Entropy Qang** `qg_S(theta) = H(cos^2(theta/2))`
(the classical Shannon entropy of the Z-basis outcome distribution — *not*
the von Neumann entropy of the state, which is identically zero for any
pure state). This package implements both exactly as defined, reproduces
every anchor value in the paper's Tables 1 and 2, and then extends the
implementation along all four concrete lines the paper's Section 6
("Future Research Directions") leaves open.

## Install

```bash
pip install -e .            # core package (numpy only)
pip install -e ".[qiskit]"  # + Qiskit gate integration
pip install -e ".[cirq]"    # + Cirq gate integration
pip install -e ".[dev]"     # + pytest, matplotlib (tests, benchmark plot)
```

## Package layout

| Module | What it covers | Paper section |
|---|---|---|
| `quang.core` | `Qang` class: `qg_Z` / `qg_S`, full (theta, phi) Bloch-sphere conversions, milliqang subunit, and the explicit half-domain inversion `theta_from_entropic()` | Sections 2.1, 2.2, 2.3 |
| `quang.qiskit_gate` | `RQangGate`, `FullRQangGate`: qg as a native single-qubit Qiskit gate | Future Research Direction #1 |
| `quang.cirq_gate` | `rqang_gate`, `full_rqang_gate`: qg as a native single-qubit Cirq gate | Future Research Direction #1 |
| `quang.gradients` | The Section 4.1 gradient singularity, two regularized alternatives, and a toy-VQE convergence benchmark | Section 4.1 + Future Research Direction #3 |
| `quang.mixed` | Generalization to mixed states (`qg_z_density`, `von_neumann_entropy`) and POVMs (`qg_s_povm`) | Future Research Direction #4 (part 1) |
| `quang.multiqubit` | Generalization to multi-qubit tensor-product projection profiles (per-qubit and joint qang, entanglement witness) | Future Research Direction #4 (part 2) |
| `quang.statistics` | Shot-noise error propagation between probability-space and theta/qg-space (closed-form + empirical Qiskit validation) | Future Research Direction #2 |

## Quickstart

```python
from quang import Qang

# Table 1 / Table 2 anchor points
Qang.from_angles(0.0, mode="polar").value          # +1.0  (|0>)
Qang.from_angles(3.14159265, mode="polar").value    # -1.0  (|1>)
Qang.from_angles(1.5707963, mode="entropic").value  #  1.0  (max uncertainty)

# full Bloch-sphere round trip
q = Qang.from_angles(theta=0.9, phi=1.1, mode="polar")
q.to_bloch_vector()      # (x, y, z)
q.to_statevector()       # (alpha, beta)

# milliqang
Qang(0.5).milliqang      # 500.0
```

```python
# Qiskit: prepare a state directly from a qg_Z value, no manual arccos
from quang import Qang
from quang.qiskit_gate import FullRQangGate
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.append(FullRQangGate(Qang(0.5, phi=0.3)), [0])
qc.measure(0, 0)
```

```python
# mixed states / POVMs
import numpy as np
from quang.mixed import qg_z_density, von_neumann_entropy, mix, standard_z_povm, qg_s_povm

rho0, rho1 = np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])
rho = mix(rho0, rho1, p=0.5)          # classical 50/50 mixture (NOT |+>)
qg_z_density(rho)                     # 0.0  -- same qg_Z as the |+> superposition ...
von_neumann_entropy(rho)              # 1.0  -- ... but qg_Z alone can't tell them apart; S(rho) can
```

```python
# multi-qubit: entanglement witness
from quang.multiqubit import bell_state, per_qubit_qg_z, marginal_von_neumann_entropy, joint_qg_s

psi = bell_state("phi_plus")
per_qubit_qg_z(psi, n_qubits=2)              # [0.0, 0.0]  -- each qubit looks maximally mixed alone
marginal_von_neumann_entropy(psi, n_qubits=2) # [1.0, 1.0] -- ... yet each qubit carries 1 bit of entropy ...
joint_qg_s(psi, n_qubits=2, normalize=False)  # 1.0        -- ... while the WHOLE 2-qubit state is exactly pure
```

```python
# Cirq: the same gate integration as Qiskit, native to Cirq
from quang import Qang
from quang.cirq_gate import full_rqang_gate
import cirq

q = cirq.LineQubit(0)
circuit = cirq.Circuit([full_rqang_gate(Qang(0.5, phi=0.3)).on(q), cirq.measure(q, key="m")])
```

```python
# shot-noise error propagation: how many shots to pin down theta to +/- 0.01 rad?
from quang.statistics import propagated_theta_std, confidence_interval_theta

propagated_theta_std(theta=1.0, n_shots=10000)        # ~= 0.01 rad, ALMOST INDEPENDENT of theta
confidence_interval_theta(theta_hat=1.0, n_shots=10000, confidence=0.95)
```

## Running the tests

```bash
pytest -q
```

116 tests: Table 1/2 anchor reproduction, the paper's stated symmetry and
invertibility properties, round-trip conversions, the gradient-singularity
limitation from Section 4.1 (and that the regularized versions are bounded
everywhere), the mixed-state/POVM generalization, the multi-qubit
generalization (including the Bell-state entanglement witness), the
shot-noise error-propagation formulas (closed-form and empirical, validated
against Qiskit's AerSimulator), and the Qiskit/Cirq gate integrations
(each skipped automatically if that SDK isn't installed).

## Empirical benchmark (Future Research Direction #3)

```bash
python examples/benchmark_qg_vs_theta.py
```

Starts an optimizer of the toy loss `E(theta) = -sin(theta)` (minimum at
`theta = pi/2`) directly **at** a pole (`theta0 = 0.01` rad), and compares
theta-space gradient descent against raw qg-space (as defined in the paper)
and the two regularized qg-space variants this package adds. See
`RESEARCH_NOTES.md` for the results and what they show.

## What's new relative to the paper's stated "Code Availability"

The paper's Code Availability section describes a reference implementation
of `qg_Z`, `qg_S`, their inversion relations, and the milliqang subunit,
with a test suite reproducing Tables 1 and 2 and the Section 4.1 gradient
singularity. All of that is here (`quang.core`, `tests/test_core.py`,
`tests/test_gradients.py`). Everything else in this package —
`quang.gradients`'s regularizations and benchmark, `quang.mixed`,
`quang.multiqubit`, `quang.statistics`, and `quang.qiskit_gate` /
`quang.cirq_gate` — is new work carried out against the paper's own
Section 6 roadmap, not something the paper already claimed. All four
Future Research Directions listed in Section 6 now have a concrete
implementation in this package.

## Citation

```
Monteverde, V.H. (2026). The Qang (qg): A Unified Angular-Probability Unit
and Metric for Parametric Quantum Circuit Design. Zenodo.
https://doi.org/10.5281/zenodo.22832150
```
