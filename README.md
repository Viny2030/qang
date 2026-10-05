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

`qang` bridges the gap between continuous Bloch-sphere rotation angles (θ ∈ [0, π]) and projective measurement spaces in parameterized quantum circuits (PQCs). It provides native representations for:
- **Polar Bias Qang (qg_Z):** qg_Z(θ) = cos θ = ⟨σ_z⟩ ∈ [−1, 1].
- **Measurement-Outcome Entropy Qang (qg_S):** qg_S(θ) = H(cos²(θ/2)) ∈ [0, 1] (Shannon entropy of computational-basis outcomes).
- Native SDK gate classes for **Qiskit**, **Cirq** and **PennyLane** (differentiable in qg_Z), regularized gradient optimizers, shot-noise error propagation, and density matrix/POVM metrics.

### What is in the library

| module | what it does |
|---|---|
| `qang.core` | the qg_Z / qg_S unit, Bloch-sphere round trips |
| `qang.formulation` | the 15 main gates and 14 main algorithms in qg units (qg'_P = qg_{U†PU}), and the local radius r² = qg_X² + qg_Y² + qg_Z² |
| `qang.sectors` | the qg symmetry filter and Hamming-weight tools; echo calibration of in-sector errors |
| `qang.statistics` | qg_Z intervals from shot counts (Bayesian, Wilson, delta method); unbiased qg² and r² |
| `qang.qml` | weight-conserving QNN classifiers (binary and multiclass) read with and without the qg filter |
| `qang.polarization` | polarized light: Stokes parameters are qg values, Mueller matrices are the qg gate rule |
| `qang.geometric` | geometric (Berry/Pancharatnam) phase in qg units, Stokes' theorem on the Bloch sphere |
| `qang.qiskit_gate`, `qang.cirq_gate`, `qang.pennylane_gate` | native gates for the three SDKs (optional) |

---

## Installation

Install the minimal core library (NumPy only):
```bash
pip install qang          # from PyPI
pip install "qang @ git+https://github.com/Viny2030/qang.git"   # or from GitHub (latest main)
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
Sections 5–7 of the notebooks add the local radius, the angle in radians separated from the radius, and the echo
calibration of the errors the filter keeps (with the check for when not to use it).
Article: `manuscript/qang_formulation.pdf` (Spanish draft: `manuscript/qang_teoria_es.pdf`).

```python
from qang import formulation as F
bell = F.apply_gate(F.apply_gate(F.qg_values([1, 0, 0, 0]), F.kron(F.H, F.I2)), F.CX)
# {'II': 1.0, 'XX': 1.0, 'YY': -1.0, 'ZZ': 1.0}   (qang >= 0.5.1)
F.radius_profile(bell)        # [0.0, 0.0]: both qubits at the centre of the sphere  (qang >= 0.6.4)
```

**The local radius (§97).** Each qubit lives on a sphere of squared radius r² = qg_X² + qg_Y² + qg_Z² = 2 Tr ρ² − 1
(area 4π r², surface deficit 4π(1 − r²)). One-qubit gates rotate the sphere and keep r; SWAP exchanges radii; CNOT,
CZ, iSWAP, Toffoli and Fredkin can move a qubit to the centre. For a pure global state 1 − r² is the one-tangle
(known physics: Coffman–Kundu–Wootters, Meyer–Wallach); for a noisy state it mixes entanglement and noise. In a
definite-weight state r = |qg_Z|, so after the qg filter (equal T1) the deficit is an entanglement measure read
from Z-basis shots alone (§98). On five device noise models (IBM, IonQ) the filter cuts the deficit error 3.2–4.0×,
but in-sector gate errors still make product states look entangled (§100). `F.algorithm_radii()` gives the radius of
the readout qubits of the 14 algorithms. The angle in radians separates from the radius,
θ = atan2(√(qg_X² + qg_Y²), qg_Z) = arccos(qg_Z / r) (`F.direction_from_qg`): it is exact under depolarizing noise,
where arccos(qg_Z) drifts towards π/2 (§102, §103). Short note: `manuscript/qang_radius.pdf` (Spanish: `manuscript/qang_radio_es.pdf`).

## Quantum neural networks with and without qang

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_qml.ipynb) English &nbsp;·&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_qml_es.ipynb) Español

`qang.qml` (qang >= 0.6.4; `MultiClassQNN` and `block_unitaries` from 0.6.17) trains weight-conserving QNN classifiers and reads every result with qang (the qg
filter keeps only the shots that stayed in the input's Hamming-weight sector) and without it. These models are
classically simulable: a robustness tool, not a quantum advantage. Every prediction was committed before its run
(152 judged so far in this line, §75–§118: 109 passed, 43 failed, all reported; 4 more wait for the IonQ hardware run, §116). Note (with §104–§118): `manuscript/qang_qml.pdf` (Spanish:
`manuscript/qang_qml_es.pdf`).

**Results so far (simulation; RESEARCH_NOTES §75–§98):**

| question | result |
|---|---|
| trained on a simulator, run under T1 (60 runs, §80) | with qang the noiseless accuracy is recovered exactly: +2.5 points at weight 1, +16.5 at weight 2 |
| trained under the calibrated noise (§80, §81, §101) | without qang the model catches up; the one case where it was ahead (§81, pair-product encoding, rich readout, +1.2 points) shrank to +0.5, not significant, over 60 new runs (§101), and with the simple readout the filter is 0.6 points better |
| two channels, filtered + raw features (§95, 60 runs, `qang="both"`) | with the dual encoding the 1.2-point effect does not replicate: trained under T1 the filtered model equals the noiseless one (0.953), raw 0.950, both channels 0.950; the two channels are safe but add nothing |
| T1 spread across qubits (§82) | the filter alone loses < 1 point up to a ±80% spread of 1/T1 |
| weight 2 (§87, §90) | the pair-product encoding limited it; with the `"dual"` encoding and `readout="zz"` weight 2 matches weight 1 (0.947 = 0.947) |
| device noise models (§85, §91) | IBM fake backends and the IonQ simulator (Aria-1, Forte-1): qg_Z error 2.7–3.6× smaller with qang; 189 inputs on Forte-1: 0.968 with vs 0.958 without (noiseless 0.979) |
| scaling (§94, 108 configurations, 4–8 qubits, weight 1–4, depth 3–24) | the filter has the lower readout error in 106 of 108 (up to 1000×); a closed-form rule, MSE ≈ (1 − z²)/(K·S) with K = (1 − γ)^(weight·depth), predicts the winner in 99% |
| low kept fraction, unequal T1, dephasing (§96, 405 configurations, K down to 2.6·10⁻⁷) | the filter wins in 399 of 405 and in every configuration with at least 20 kept shots (MSE up to 1871× lower with equal T1, 94× with unequal T1, 58× with dephasing); the rule picks the winner in 342 of 342 with K·S ≥ 5 |
| the local radius as an entanglement measure (§98) | under equal T1 the filtered deficit 1 − qg_Z² is the noiseless tangle to 3·10⁻¹⁵; on product states it is exactly 0, while the raw readout shows false entanglement of 0.60–0.95 |
| distorting against faithful encoding (§101, 60 runs, weight 2) | no readout trained under T1 beats the filter significantly; the best weight-2 model is the dual encoding with the filter (0.955) |
| the radius on device noise models (§100, §104 in IonQ's native gate set, 3 IBM fake backends, IonQ aria-1 and forte-1) | deficit error of trained circuits 0.05–0.08 with qang against 0.20–0.27 without; on product states the filter removes only 15–33% of the false entanglement (in-sector errors pass it) |
| the angle in radians separated from the radius (§102, §103) | exact under depolarizing noise (arccos(qg_Z) errs 0.12–0.29 rad); 5–9× more accurate on the IBM noise models and, once the noise gates are sent in IonQ's native gate set, on the IonQ ones (8.7–9.0×); loses under pure dephasing and without noise (three bases cost shots) |
| echo calibration of in-sector errors (§105, five device noise models) | five echo circuits measure where the filter's kept shots land; inverting that 5×5 matrix removes 38–47% of the remaining deficit error; filter + echo is 6–7× more accurate than the raw readout (`qang.sectors.echo_transfer_matrix`, `unmix_sector`) |
| echo calibration and the classifier (§106, 189 inputs, five device noise models) | decisions that differ from the noiseless model: 15 without qang, 7 with the filter, 3 with filter + echo; on the IBM noise models filter + echo reaches the noiseless accuracy (0.979) |
| echo calibration at weight 2 (§107, 10 × 10 transfer matrix) | removes 37–43% of the qg_Z error and 39–48% of the qg_ZZ error left by the filter on five device noise models; filter + echo 3.4–4.0× below raw |
| echo calibration on narrow-margin inputs (§108, 600 readings) | the filter cuts flipped decisions from 47 to 17; the echo adds nothing there (19), because the filtered error is close to shot noise: use the echo only when the filtered error is well above its shot-noise level |
| training from shots (§109, SPSA, 100 or 1000 shots per evaluation, T1) | trained with qang 0.939 / 0.930, without 0.929 / 0.915; the filter's shot cost did not show; noiseless training + filter at run time 0.945 |
| multiclass, each qubit a class (§110, `MultiClassQNN`, 30 runs) | with qang exactly the noiseless accuracy; without qang −5.3 points (qubit readout) and −9.1 (linear head), −15 to −18 on 5-class digits |
| 5, 6 and 8 qubits (§111, weight 1 and 2, 54 runs) | with qang exactly the noiseless accuracy at every size; without qang −3 points at weight 1 (flat in n at fixed depth) and −9 to −26 at weight 2 (8 qubits: 0.965 against 0.705) |
| trainability under T1 (§112, 4-8 qubits, 2400 gradient draws) | with qang the gradient is exactly the noiseless one (without: K times smaller at weight 1); fewer shots to resolve it in 10 of 12 cells, more when the kept fraction drops below about 0.15 |
| multiclass on device noise models with echo (§113, 5-class digits, 500 readings per readout) | flipped decisions: 46 raw / 19 filter / 15 filter + echo (qubit readout), 51 / 24 / 15 (head); head accuracy 0.858 / 0.890 / 0.900, noiseless 0.91 |
| multiclass at 8 qubits (§114, 3, 5 and 8 classes of digits, 30 runs) | with qang exactly the noiseless accuracy; without qang the head readout loses 0.7 / 15.9 / 11.1 points (3 / 5 / 8 classes), the qubit readout 5.1 / 4.4 / 3.4; trained under the noise without qang, 2–3 points behind with 5 and 8 classes |
| echo calibration at 8 qubits, weight 2 (§115, 28 × 28 transfer matrix, five device noise models) | removes 27–38% of the qg_Z error left by the filter (32% on average, against 40% at 5 qubits); filter + echo 3.4–4.1× below raw |
| unequal T1 corrected (§117, spread ±100%, 40 runs) | training with qang under the calibrated decay rates: 0.951 / 0.940 against noiseless 0.950 / 0.945 (filter alone 0.931 / 0.918), unaffected by a 10% calibration error; the filtered readout depends only on the ratios of the rates, so at weight 2 it survives a 1.5× drift of all T1 (−0.7 points) that costs the raw noise-aware model 10 |
| dephasing: filter + zero-noise extrapolation (§118, T1 0.08 + dephasing 0.03/0.06, 40 runs) | the filter turns T1 + dephasing into dephasing alone (exact), so ZNE of the filtered features cuts their error to 50–58% (linear) or 27–35% (Richardson) of the filter's and beats ZNE of the raw readout by 3–10 points; with 1000 shots per scale the gain is lost at weight 2; training with the filter under the calibrated noise recovers the noiseless accuracy (0.951–0.956) |
| not corrected exactly | dephasing; unequal T1 (the filter still has the lower error, but is biased); errors that move an excitation inside the sector |

Hardware runs are prepared, with explicit cost confirmation, and pending:
`examples/qnn_hardware_qg.py --mode ionq_qpu` (IonQ Forte-1) or `--mode ibm`, and
`examples/qec_syndrome_destructive_qg.py --mode ibm` (T1 and dephasing from Leung-code syndromes; within 1–4% of the
calibration on IBM noise models, §93; the dephasing reads the product mean over the four qubits, §99).

```python
from qang.qml import WeightQNN
m = WeightQNN(n_qubits=5, weight=2, encoding="dual", readout="zz").fit(X_train, y_train)  # 4 features in [-1, 1]
m.score(X_test, y_test, gamma=0.08, qang=True)                      # = noiseless accuracy (equal T1)
m.score(X_test, y_test, gamma=0.08, qang=False)                     # raw readout, biased by T1
```

## Polarized light and geometric phase in qg units

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_optics.ipynb) English &nbsp;·&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_optics_es.ipynb) Español

`qang.polarization` and `qang.geometric` (qang >= 0.6.0): the normalized Stokes parameters are qg values (the
Poincare sphere is the Bloch sphere), a lossless Mueller matrix is the qg gate rule, Malus's law reads
I = I0 (1 + qg_Z cos 2theta + qg_X sin 2theta)/2, and the geometric phase of a loop at constant qg_Z is
(qg_Z - 1)/2 of a turn (Stokes' theorem on the sphere). Note: `manuscript/qang_optics.pdf` (Spanish:
`manuscript/qang_optics_es.pdf`).

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
* `notebooks/qang_avances_colab.ipynb` — §20–§93: the qg symmetry witness and filter
  (chemistry, Hubbard, constrained QAOA), filter vs ZNE, qubit characterization, error
  correction (Leung code, syndrome tracking), coherence, few-shot estimation, and the
  recorded IonQ noisy-simulator runs, each with an honest "for / against" reading.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_avances_colab.ipynb)
* `notebooks/qang_criptografia_colab.ipynb` — cryptography line, kept separate: why qg does
  not apply to post-quantum cryptography, BB84 drift-vs-eavesdropper monitoring (§40) and
  certified quantum random numbers (§41); companion PDF `manuscript/qang_criptografia.pdf`.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Viny2030/qang/blob/main/notebooks/qang_criptografia_colab.ipynb)

## Testing
The package includes an extensive test suite (1272 tests, run in CI on Python 3.9–3.12) verifying analytical anchors, numerical stability, gradient regularizations, backend fidelity, and every numerical finding quoted in `RESEARCH_NOTES.md`:

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
