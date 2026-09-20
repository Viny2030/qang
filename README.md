# qang — The Qang (qg) Python Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22832150.svg)](https://doi.org/10.5281/zenodo.22832150)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Reference implementation and computational toolkit for:

> V. H. Monteverde, *"The Qang (qg): A Unified Angular-Probability Unit and Metric for Parametric Quantum Circuit Design."*
> ORCID: [0000-0001-8884-4811](https://orcid.org/0000-0001-8884-4811) · DOI: [10.5281/zenodo.22832150](https://doi.org/10.5281/zenodo.22832150)

`qang` bridges the gap between continuous Bloch-sphere rotation angles ($\theta \in [0, \pi]$) and projective measurement spaces in parameterized quantum circuits (PQAs). It provides native representations for:
- **Polar Bias Qang ($qg_Z$):** $qg_Z(\theta) = \cos\theta = \langle\sigma_z\rangle \in [-1, 1]$.
- **Measurement-Outcome Entropy Qang ($qg_S$):** $qg_S(\theta) = H(\cos^2(\theta/2)) \in [0, 1]$ (Shannon entropy of computational-basis outcomes).
- Native SDK gate classes for **Qiskit** and **Cirq**, regularized gradient optimizers, shot-noise error propagation, and density matrix/POVM metrics.

---

## Installation

Install the minimal core library (NumPy only):
```bash
pip install .
```

```bash
pip install ".[qiskit]"   # Native Qiskit gate integration
pip install ".[cirq]"     # Native Cirq gate integration
pip install ".[all]"      # Everything (Qiskit, Cirq, Pytest, Matplotlib)
```

## Quickstart

### 1. Basic Unit Conversions & Inversion
```python
from qang import Qang

# Anchor values
q_zero = Qang.from_angles(0.0, mode="polar")
print(q_zero.value)  # +1.0 (|0>)

q_max_ent = Qang.from_angles(1.5707963, mode="entropic")
print(q_max_ent.value)  # 1.0 (Maximum measurement uncertainty)

# Full Bloch sphere and statevector round-trip
q = Qang.from_angles(theta=0.9, phi=1.1, mode="polar")
bx, by, bz = q.to_bloch_vector()
alpha, beta = q.to_statevector()

# Engineering subunit (milliqang)
print(Qang(0.5).milliqang)  # 500.0 m-qg
```

### 2. Qiskit Native Gate Integration
```python
from qang import Qang
from qang.qiskit_gate import FullRQangGate
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
# Append gate directly parameterized in qg_Z without manual arccos conversion
qc.append(FullRQangGate(Qang(0.5, phi=0.3)), [0])
qc.measure(0, 0)
```

### 3. Cirq Native Gate Integration
```python
import cirq
from qang import Qang
from qang.cirq_gate import full_rqang_gate

q = cirq.LineQubit(0)
circuit = cirq.Circuit(
    full_rqang_gate(Qang(0.5, phi=0.3)).on(q),
    cirq.measure(q, key="m")
)
```

### 4. Error Propagation & Shot Budgeting
```python
from qang.statistics import propagated_theta_std, confidence_interval_theta

# First-order delta-method uncertainty: invariant across the Bloch sphere (~1/sqrt(N))
std_theta = propagated_theta_std(theta=1.0, n_shots=10000)
ci_lower, ci_upper = confidence_interval_theta(theta_hat=1.0, n_shots=10000, confidence=0.95)
print(f"Theta 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] rad")
```

## Testing
The package includes an extensive test suite (116 tests) verifying analytical anchors, numerical stability, gradient regularizations, and backend fidelity:

```bash
pytest -v
```

## Contributing & Community
We welcome contributions, bug reports, and suggestions!

Issues: Please use the [GitHub Issue Tracker](https://github.com/Viny2030/qang/issues) to report bugs or request features.

Contributions: See [CONTRIBUTING.md](CONTRIBUTING.md) for local development and pull request guidelines.

## Citation
If you use qang in your research, please cite:

```
@article{monteverde2026qang,
  title={The Qang (qg): A Unified Angular-Probability Unit and Metric for Parametric Quantum Circuit Design},
  author={Monteverde, Vicente Humberto},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22832150}
}
```
