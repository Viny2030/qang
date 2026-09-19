---
title: 'quang: A Python library for angular-probability parameterization in quantum circuits'
tags:
  - Python
  - quantum computing
  - quantum circuits
  - Qiskit
  - Cirq
authors:
  - name: Vicente Humberto Monteverde
    orcid: 0000-0001-8884-4811
    affiliation: 1
affiliations:
  - name: UMSA, Argentina
    index: 1
date: 19 September 2026
bibliography: paper.bib
---

# Summary

In quantum state engineering and parameterized quantum algorithms, single-qubit rotations are traditionally parameterized by continuous angles (radians) on the Bloch sphere, while measurement outcomes correspond to statistical projection probabilities via Born's rule. Bridging these domains in software traditionally requires repeated explicit trigonometric conversions and ad hoc error propagation.

`quang` is an open-source Python library providing a native, bidirectional framework based on the **qang ($qg$) unit**. It defines two primary operational representations: the **Polar Bias Qang** ($qg_Z = \langle\sigma_z\rangle = \cos\theta$), bridging state rotation directly to expectation values, and the **Measurement-Outcome Entropy Qang** ($qg_S$), mapping polar deviations to the binary Shannon entropy of computational-basis measurements.

# Statement of Need

Modern quantum software development kits (SDKs) such as Qiskit and Cirq parameterize rotation gates primarily through raw angles. When implementing variational algorithms (e.g., VQE, QAOA) or calibrating microwave pulse errors against fidelity thresholds, researchers frequently convert between rotation angles, expectation biases, and shot-noise uncertainties. 

`quang` addresses this gap by offering:
- **First-class unit representations:** Seamless conversions between angles, Bloch vectors, statevectors, and the milliqang ($m\text{-}qg$) engineering subunit.
- **Native SDK gate integrations:** `RQangGate` for Qiskit and `rqang_gate` for Cirq, allowing users to define parameterized circuits natively in $qg$-space without manual trigonometry.
- **Gradient regularization & optimization:** Implementations and benchmarks addressing the coordinate singularity of $qg$-space gradients near the poles via regularized updates.
- **Statistical and multi-qubit tooling:** Exact shot-noise error propagation, generalizations to density matrices/POVMs, and multi-qubit entanglement diagnostics.

`quang` has no mandatory dependencies beyond `NumPy`, keeping it lightweight while providing optional integration extras for `qiskit`, `cirq`, and development toolchains.

# References
