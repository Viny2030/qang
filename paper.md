---
title: 'quang: A Python library for angular-probability parameterization and metric analysis in quantum circuits'
tags:
  - Python
  - quantum computing
  - quantum circuits
  - Qiskit
  - Cirq
  - variational quantum algorithms
authors:
  - name: Vicente Humberto Monteverde
    orcid: 0000-0001-8884-4811
    affiliation: 1
affiliations:
  - name: Universidad del Museo Social Argentino (UMSA), Argentina
    index: 1
date: 20 September 2026
bibliography: paper.bib
---

# Summary

In quantum state engineering, quantum optimal control, and parameterized quantum algorithms (PQAs), single-qubit rotations are conventionally parameterized by continuous polar and azimuthal angles (radians) on the Bloch sphere [@nielsen2010quantum]. In contrast, experimental measurements and target observables correspond to statistical projection probabilities determined by Born's rule [@mcclean2016theory]. Interfacing between angular control parameters and observational probability spaces typically requires repeated, ad hoc trigonometric transformations ($\cos\theta$, $\arccos$), empirical conversion heuristics, and manual shot-noise error propagation across software workflows.

`quang` is an open-source Python library providing a mathematically grounded, bidirectional bridge between angular parameterization and probability-space metrics through the **Qang ($qg$)** framework [@monteverde2026qang]. The package operationalizes two primary metrics defined on the polar angle $\theta \in [0, \pi]$:
1. The **Polar Bias Qang** ($qg_Z(\theta) = \cos\theta = \langle\sigma_z\rangle \in [-1, 1]$), which directly maps polar rotations to the $Z$-observable expectation value and computational basis state deviations.
2. The **Measurement-Outcome Entropy Qang** ($qg_S(\theta) = H(\cos^2(\theta/2)) \in [0, 1]$), representing the binary Shannon entropy of computational-basis projective measurements.

In addition to core transformations and milliqang ($m\text{-}qg$) engineering units, `quang` provides SDK-native gate interfaces for Qiskit and Cirq, gradient regularizations for variational optimization, analytical shot-noise error propagation models, and extensions to mixed states (POVMs) and multi-qubit tensor networks.

# Statement of Need

Modern quantum software development kits (SDKs), such as Qiskit and Cirq, parameterize single-qubit gates almost exclusively in radians ($\theta, \phi, \lambda$). However, in practical quantum engineering contexts—such as variational quantum eigensolvers (VQE), quantum approximate optimization algorithms (QAOA), calibration of microwave pulse errors, and quantum sensor tuning—the objective functions and stopping criteria are defined in terms of expectation values, statistical fidelities, and shot budgets.

Translating raw angles into observable spaces presents non-trivial software and numerical challenges:
- **Trigonometric Coordinate Singularities in Optimization:** Updating variational parameters directly in $qg_Z$-space involves an inverse Jacobian $(d\theta/dqg_Z = -1/\sin\theta)$ that diverges at the computational poles $\theta \in \{0, \pi\}$. Without proper numerical regularization, gradient-based optimizers overshoot catastrophically or oscillate indefinitely across the boundary.
- **Error Propagation Under Finite Sampling:** Real hardware estimates expectation values via finite measurement shots $N$, introducing binomial shot noise. Understanding how variance in $\widehat{qg}_Z$ propagates into the angular estimate $\hat{\theta}$ is vital for setting realistic error tolerances in hardware benchmarks.
- **Mixed State and Entanglement Characterization:** Pure-state geometric parameters are insufficient when dealing with open quantum systems or entangled registers. A unified software metric must scale consistently to density matrices $\rho$, positive operator-valued measures (POVMs), and partial-trace subsystems.

`quang` addresses these needs by supplying a lightweight, modular library that automates these conversions, resolves numerical singularities, and embeds directly into existing quantum circuit compilation pipelines.

# State of the Field

Existing quantum computing frameworks handle parameterizations and state diagnostics across distinct, disjoint modules:
- **Circuit Construction (Qiskit, Cirq, Pennylane):** Standard libraries focus on unitary execution and statevector manipulation using raw angles. Users must manually compose parameterized gates with custom cost functions, without native types representing bias or entropy metrics directly inside the circuit DAG.
- **Quantum Information Toolkits (QETLAB, QuTiP):** Software packages such as QuTiP provide extensive density-matrix and entropy routines. However, they are heavy scientific frameworks geared toward master-equation simulation rather than lightweight, parameter-level circuit abstraction or drop-in gate definitions for execution on physical backends.

`quang` fills this distinct niche by introducing a dedicated unit abstraction layer. It does not replace quantum execution backends; rather, it augments Qiskit and Cirq circuits with native gate classes (`RQangGate`, `full_rqang_gate`) that accept $qg$ instances directly, eliminating boilerplates and providing built-in domain validation, branch inversion, and noise propagation.

# Software Design and Architecture

`quang` is architected as a lightweight package with minimal overhead, requiring only `NumPy` as a core runtime dependency, with optional integration extras for `qiskit`, `cirq`, and development toolchains. The codebase is organized into seven specialized modules:
