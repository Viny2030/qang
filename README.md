# quang — The Qang (qg) Python Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22832150.svg)](https://doi.org/10.5281/zenodo.22832150)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Reference implementation and computational toolkit for:

> V. H. Monteverde, *"The Qang (qg): A Unified Angular-Probability Unit and Metric for Parametric Quantum Circuit Design."*  
> ORCID: [0000-0001-8884-4811](https://orcid.org/0000-0001-8884-4811) · DOI: [10.5281/zenodo.22832150](https://doi.org/10.5281/zenodo.22832150)

`quang` bridges the gap between continuous Bloch-sphere rotation angles ($\theta \in [0, \pi]$) and projective measurement spaces in parameterized quantum circuits (PQAs). It provides native representations for:
- **Polar Bias Qang ($qg_Z$):** $qg_Z(\theta) = \cos\theta = \langle\sigma_z\rangle \in [-1, 1]$.
- **Measurement-Outcome Entropy Qang ($qg_S$):** $qg_S(\theta) = H(\cos^2(\theta/2)) \in [0, 1]$ (Shannon entropy of computational-basis outcomes).
- Native SDK gate classes for **Qiskit** and **Cirq**, regularized gradient optimizers, shot-noise error propagation, and density matrix/POVM metrics.

---

## Installation

Install the minimal core library (NumPy only):
```bash
pip install .
