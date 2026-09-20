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
quang/
├── init.py           # Public API exposure
├── core.py               # Qang class, conversions, milliqang, branch inversion
├── gradients.py          # Jacobian inverses, clipped and Tikhonov regularizations
├── statistics.py         # Binomial shot-noise propagation, confidence intervals
├── mixed.py              # Density matrix qg_Z, von Neumann entropy, POVM metrics
├── multiqubit.py         # Marginal profiles, joint qg_S, entanglement witness
├── qiskit_gate.py        # Qiskit native RQangGate and FullRQangGate
└── cirq_gate.py          # Cirq native rqang_gate and full_rqang_gate

### Core Representations and Branch Inversion (`quang.core`)
The `Qang` class encapsulates both $qg_Z$ and $qg_S$ operational modes. Because $qg_S(\theta) = qg_S(\pi - \theta)$, inversion from entropy to angle is non-injective over $[0, \pi]$. `quang.core` formalizes the half-domain invertibility: it enforces monotonic branches via bisection over $H(p)$ for $\theta \in [0, \pi/2]$ (`branch='lower'`) and $\theta \in [\pi/2, \pi]$ (`branch='upper'`), providing bidirectional consistency within $10^{-4}$ rad.

### Gradient Regularization (`quang.gradients`)
To handle the divergence of $dE/dqg = -(1/\sin\theta) \cdot dE/d\theta$ at the poles, `quang.gradients` provides two bounded alternatives:
- **Clipped Inverse Jacobian:** Clamps the denominator magnitude to $\max(\vert{}\sin\theta\vert{}, \epsilon)$, bounding the update step to $1/\epsilon$.
- **Tikhonov Regularization:** Computes $-\sin\theta / (\sin^2\theta + \epsilon^2)$. This formulation smoothly suppresses updates directly at the poles ($0$ push at $\theta \in \{0, \pi\}$), preventing boundary overshooting during variational optimization.

### Shot-Noise Error Propagation (`quang.statistics`)
`quang.statistics` implements analytical error propagation using the first-order delta method. Given $N$ measurement shots, the variance of the estimator $\widehat{qg}_Z$ is $\operatorname{Var}(\widehat{qg}_Z) = \sin^2\theta / N$. Propagating through the inverse Jacobian yields:
$$\operatorname{Var}(\hat{\theta}) \approx \operatorname{Var}(\widehat{qg}_Z) \left(\frac{d\theta}{dqg_Z}\right)^2 = \left[\frac{\sin^2\theta}{N}\right] \left[\frac{1}{\sin^2\theta}\right] = \frac{1}{N}$$
The angular variance is invariant to $\theta$ to first order ($\operatorname{std}(\hat{\theta}) \approx 1/\sqrt{N}$). The module provides analytical confidence intervals and validates this behavior against empirical bootstrap trials executed via Qiskit's `AerSimulator`.

### Open Quantum Systems and Multi-Qubit Registers (`quang.mixed`, `quang.multiqubit`)
The framework generalizably evaluates mixed states via $\operatorname{Tr}(\rho \sigma_z)$ and POVM measurements normalized by $\log_2(n_{\text{outcomes}})$. For multi-qubit systems, it computes both single-qubit marginal projection profiles and joint registration entropy ($qg_S^{\text{joint}}$). For maximally entangled states (e.g., Bell pairs), `quang.multiqubit` acts as an operational entanglement witness: marginal states exhibit maximal mixing ($\operatorname{Tr}(\rho_i \sigma_z) = 0, S(\rho_i) = 1$), while the global state remains strictly pure ($S(\rho_{\text{global}}) = 0$).

# Research Impact Statement

`quang` provides a reproducible, standardized foundation for quantum software engineering and education:
1. **Algorithm Development:** In variational optimization (VQE/QAOA), using regularized $qg$-gradients allows researchers to optimize directly in expectation-value space while avoiding numerical instabilities at computational basis states.
2. **Error Budgeting and Calibration:** Experimentalists can directly estimate necessary shot budgets $N$ required to achieve angular fidelity benchmarks without running bespoke Monte Carlo noise simulations.
3. **Cross-Platform Reproducibility:** By providing unified single-qubit gate implementations matching unitary conventions across Qiskit's `UGate` and Cirq's `MatrixGate`, the software guarantees statevector equivalence across backends.

The library includes an automated test suite comprising 116 unit and regression tests reproducing all analytical tables and validation benchmarks.

# AI Usage Disclosure

Generative AI assistance (Claude 3.5 Sonnet / OpenAI GPT-4o) was utilized during code refactoring, test-suite expansion, and documentation drafting. All mathematical derivations, numerical algorithms, architectural implementations, and scientific validations were reviewed, verified, and confirmed by the author.

# References
