# qang — The Qang (qg) Python Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_full_reference.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22832150.svg)](https://doi.org/10.5281/zenodo.22832150)
[![PyPI](https://img.shields.io/pypi/v/qang.svg)](https://pypi.org/project/qang/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/Viny2030/qang/actions/workflows/tests.yml/badge.svg)](https://github.com/Viny2030/qang/actions/workflows/tests.yml)

Reference implementation and computational toolkit for:

> V. H. Monteverde, *"The Qang (qg): A Unified Angular-Probability Unit and Metric for Parametric Quantum Circuit Design."*
> ORCID: [0000-0001-8884-4811](https://orcid.org/0000-0001-8884-4811) · DOI: [10.5281/zenodo.22832150](https://doi.org/10.5281/zenodo.22832150)

`qang` bridges the gap between continuous Bloch-sphere rotation angles ($\theta \in [0, \pi]$) and projective measurement spaces in parameterized quantum circuits (PQAs). It provides native representations for:
- **Polar Bias Qang ($qg_Z$):** $qg_Z(\theta) = \cos\theta = \langle\sigma_z\rangle \in [-1, 1]$.
- **Measurement-Outcome Entropy Qang ($qg_S$):** $qg_S(\theta) = H(\cos^2(\theta/2)) \in [0, 1]$ (Shannon entropy of computational-basis outcomes).
- Native SDK gate classes for **Qiskit**, **Cirq** and **PennyLane** (differentiable in qg_Z), regularized gradient optimizers, shot-noise error propagation, and density matrix/POVM metrics.

---

## Installation

Install the minimal core library (NumPy only):
```bash
pip install qang          # from PyPI
pip install "qang @ git+https://github.com/Viny2030/qang.git@v0.4.0"   # or from GitHub
pip install .             # or from a local clone
```

```bash
pip install "qang[qiskit]" # Native Qiskit gate integration (from a clone: pip install ".[qiskit]")
pip install ".[cirq]"     # Native Cirq gate integration
pip install ".[pennylane]" # Native PennyLane operations (autodiff in qg_Z)
pip install ".[hardware]" # IBM Quantum hardware / calibrated fake backends
pip install ".[ionq]"     # IonQ simulator / hardware scripts (qiskit-ionq)
pip install ".[qec]"      # Surface-code decoding examples (pymatching)
pip install ".[chemistry]" # PySCF/OpenFermion molecules (Linux/macOS or WSL)
pip install ".[all]"      # Everything except chemistry (Qiskit, IBM Runtime, IonQ, pymatching, Cirq, PennyLane, SciPy, Pytest, Matplotlib)
pip install -r requirements.txt  # same as .[all], without installing qang itself
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

### 3b. PennyLane: optimize directly in qg_Z
```python
import pennylane as qml
from pennylane import numpy as pnp
from qang.pennylane_gate import rqang

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def z_expval(qg):
    rqang(qg, wires=0)          # RY(arccos(qg)), differentiable in qg
    return qml.expval(qml.PauliZ(0))

qg = pnp.array(0.3, requires_grad=True)
print(qml.grad(z_expval)(qg))   # 1.0: <Z> = qg_Z exactly
```

### 4. Error Propagation & Shot Budgeting
```python
from qang.statistics import propagated_theta_std, confidence_interval_theta

# First-order delta-method uncertainty: invariant across the Bloch sphere (~1/sqrt(N))
std_theta = propagated_theta_std(theta=1.0, n_shots=10000)
ci_lower, ci_upper = confidence_interval_theta(theta_hat=1.0, n_shots=10000, confidence=0.95)
print(f"Theta 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] rad")
```

## Start here: gates and algorithms in qg

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_start_here.ipynb) English &nbsp;·&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_inicio_formulacion.ipynb) Español

`notebooks/qang_start_here.ipynb` (English; Spanish: `qang_inicio_formulacion.ipynb`) installs `qang` from PyPI
and walks through the 15 main gates and 14 main algorithms in qg language with `qang.formulation`: a state is
described by its qg values (qg_P = <P> for every Pauli string) and a gate acts as qg'_P = qg_{U^dag P U}.
Article: `manuscript/qang_formulation.pdf` (Spanish draft: `manuscript/qang_teoria_es.pdf`).

```python
from qang import formulation as F
bell = F.apply_gate(F.apply_gate(F.qg_values([1, 0, 0, 0]), F.kron(F.H, F.I2)), F.CX)
# {'II': 1.0, 'XX': 1.0, 'YY': -1.0, 'ZZ': 1.0}   (qang >= 0.5.1)
```

## Quantum neural networks with and without qang

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_qml.ipynb) English &nbsp;·&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_qml_es.ipynb) Español

`qang.qml` (qang >= 0.6.0) trains weight-conserving QNN classifiers and reads every result with qang (the qg
filter keeps only the shots that stayed in the input's Hamming-weight sector) and without it. Over 60 runs
(RESEARCH_NOTES §80): trained on a simulator and run under T1, qang adds +2.5 points at weight 1 and +16.5 at
weight 2 and recovers the noiseless accuracy exactly; trained under the calibrated noise, the model without qang
catches up; dephasing is not corrected. These models are classically simulable: a robustness tool, not a quantum
advantage. Note: `manuscript/qang_qml.pdf` (Spanish: `manuscript/qang_qml_es.pdf`).

```python
from qang.qml import WeightQNN
m = WeightQNN(n_qubits=5, weight=2).fit(X_train, y_train)          # 4 features in [-1, 1]
m.score(X_test, y_test, gamma=0.08, qang=True)                      # = noiseless accuracy (equal T1)
m.score(X_test, y_test, gamma=0.08, qang=False)                     # raw readout, biased by T1
```

## Full Reference Notebook

`notebooks/qang_full_reference.ipynb` is the single canonical, self-contained walkthrough
of the whole project — the `Qang` class, the Section 4.1 gradient singularity (regularized
and benchmarked), mixed states/POVMs, multi-qubit profiles, the native Qiskit and Cirq
gates, shot-noise error propagation, Quantum Natural Gradient, a live round-trip against
IonQ's cloud simulator, and a closing section that reformulates every result above into
concrete, measured speed/cost comparisons (including one negative result, kept in on
purpose). Open it directly in Colab with the badge above.

Three further notebooks (in Spanish) cover the research notes:

* `notebooks/qang_verificado.ipynb` — §1–§19: exact identities, blind spots, noise
  diagnostics, optimization, knitting, few-shot estimation, QML, control quantization.
* `notebooks/qang_avances_colab.ipynb` — §20–§74: the qg symmetry witness and filter
  (chemistry, Hubbard, constrained QAOA), filter vs ZNE, qubit characterization, error
  correction (Leung code, syndrome tracking), coherence, few-shot estimation, and the
  recorded IonQ noisy-simulator runs, each with an honest "for / against" reading.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_avances_colab.ipynb)
* `notebooks/qang_criptografia_colab.ipynb` — cryptography line, kept separate: why qg does
  not apply to post-quantum cryptography, BB84 drift-vs-eavesdropper monitoring (§40) and
  certified quantum random numbers (§41); companion PDF `manuscript/qang_criptografia.pdf`.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_criptografia_colab.ipynb)

## Testing
The package includes an extensive test suite (1217 tests, run in CI on Python 3.9–3.12) verifying analytical anchors, numerical stability, gradient regularizations, backend fidelity, and every numerical finding quoted in `RESEARCH_NOTES.md`:

```bash
pytest -v
```

## Contributing & Community
We welcome contributions, bug reports, and suggestions!

Issues: Please use the [GitHub Issue Tracker](https://github.com/Viny2030/qang/issues) to report bugs or request features.

Contributions: See [CONTRIBUTING.md](CONTRIBUTING.md) for local development and pull request guidelines.

## Citation
If you use qang in your research, please cite the paper that defines the unit:

```bibtex
@article{monteverde2026qang,
  title     = {The Qang (qg): A Unified Angular-Probability Unit and Metric for Parametric Quantum Circuit Design},
  author    = {Monteverde, Vicente Humberto},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22832150}
}
```

and, for the software itself, the repository: <https://github.com/Viny2030/qang>.
