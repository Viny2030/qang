---
title: 'qang: A Python library for angular-probability parameterization and metric analysis in quantum circuits'
tags:
  - Python
  - quantum computing
  - quantum circuits
  - Qiskit
  - Cirq
  - variational quantum algorithms
  - quantum benchmarking
authors:
  - name: Vicente Humberto Monteverde
    orcid: 0000-0001-8884-4811
    affiliation: 1
affiliations:
  - name: Universidad del Museo Social Argentino (UMSA), Argentina
    index: 1
date: 23 September 2026
bibliography: paper.bib
---

# Summary

In quantum state engineering, quantum optimal control, and parameterized quantum algorithms (PQAs), single-qubit rotations are conventionally parameterized by continuous polar and azimuthal angles (radians) on the Bloch sphere [@nielsen2010quantum]. In contrast, experimental measurements and target observables correspond to statistical projection probabilities determined by Born's rule [@mcclean2016theory]. Interfacing between angular control parameters and observational probability spaces typically requires repeated, ad hoc trigonometric transformations ($\cos\theta$, $\arccos$), empirical conversion heuristics, and manual shot-noise error propagation across software workflows.

`qang` is an open-source Python library providing a mathematically grounded, bidirectional bridge between angular parameterization and probability-space metrics through the **Qang ($qg$)** framework [@monteverde2026qang]. The package operationalizes two primary metrics defined on the polar angle $\theta \in [0, \pi]$:

1. The **Polar Bias Qang** ($qg_Z(\theta) = \cos\theta = \langle\sigma_z\rangle \in [-1, 1]$), which directly maps polar rotations to the $Z$-observable expectation value and computational basis state deviations.
2. The **Measurement-Outcome Entropy Qang** ($qg_S(\theta) = H(\cos^2(\theta/2)) \in [0, 1]$), representing the binary Shannon entropy of computational-basis projective measurements.

The two are linked by an exact, branch-free identity, $qg_S = H\big((1 + qg_Z)/2\big)$, valid for any single-qubit state, pure or mixed. A third unit, $qg_\Phi(\phi) = e^{i 2\pi\phi}$, covers the relative phase. `qang` also provides:

- SDK-native gate interfaces for Qiskit and Cirq.
- Regularized gradients and a pole-damped optimizer for variational algorithms.
- Analytical shot-noise error propagation.
- Extensions to mixed states, POVMs, and multi-qubit registers.
- Worked, test-pinned connections to standard benchmarking and error-mitigation protocols.

# Statement of Need

Modern quantum software development kits (SDKs), such as Qiskit and Cirq, parameterize single-qubit gates almost exclusively in radians ($\theta, \phi, \lambda$). However, in practical quantum engineering contexts—such as variational quantum eigensolvers (VQE), quantum approximate optimization algorithms (QAOA), calibration of microwave pulse errors, and quantum sensor tuning—the objective functions and stopping criteria are defined in terms of expectation values, statistical fidelities, and shot budgets.

Translating raw angles into observable spaces presents non-trivial software and numerical challenges:

- **Trigonometric Coordinate Singularities in Optimization:** Updating variational parameters directly in $qg_Z$-space involves an inverse Jacobian $(d\theta/dqg_Z = -1/\sin\theta)$ that diverges at the computational poles $\theta \in \{0, \pi\}$. The inverse map $\arccos$ also returns only $[0, \pi]$, so qg-space updates cannot reach optima with $\theta < 0$. Without care, gradient-based optimizers overshoot, oscillate, or stall at the Hartree-Fock point.
- **Error Propagation Under Finite Sampling:** Real hardware estimates expectation values via finite measurement shots $N$, introducing binomial shot noise. Understanding how variance in $\widehat{qg}_Z$ propagates into the angular estimate $\hat{\theta}$ is vital for setting realistic error tolerances in hardware benchmarks.
- **Mixed State and Entanglement Characterization:** Pure-state geometric parameters are insufficient when dealing with open quantum systems or entangled registers. A unified software metric must scale consistently to density matrices $\rho$, positive operator-valued measures (POVMs), and partial-trace subsystems, and must state plainly what a Z-basis metric cannot detect.

`qang` addresses these needs by supplying a lightweight, modular library that automates these conversions, resolves numerical singularities, and embeds directly into existing quantum circuit compilation pipelines.

# State of the field

Existing quantum computing frameworks handle parameterizations and state diagnostics across distinct, disjoint modules:

- **Circuit Construction (Qiskit, Cirq, PennyLane):** Standard libraries focus on unitary execution and statevector manipulation using raw angles. Users must manually compose parameterized gates with custom cost functions, without native types representing bias or entropy metrics directly inside the circuit DAG.
- **Quantum Information Toolkits (QETLAB, QuTiP):** Software packages such as QuTiP [@johansson2012qutip] provide extensive density-matrix and entropy routines. However, they are heavy scientific frameworks geared toward master-equation simulation rather than lightweight, parameter-level circuit abstraction or drop-in gate definitions for execution on physical backends.

`qang` fills this distinct niche by introducing a dedicated unit abstraction layer. It does not replace quantum execution backends; rather, it augments Qiskit and Cirq circuits with native gate classes (`RQangGate`, `full_rqang_gate`) that accept $qg$ instances directly, eliminating boilerplate and providing built-in domain validation, branch inversion, and noise propagation.

# Software Design

`qang` is architected as a lightweight package with minimal overhead, requiring only `NumPy` as a core runtime dependency, with optional integration extras for `qiskit`, `cirq`, and development toolchains. Modules that need an SDK raise a clear error only when imported without it; `import qang` never does.

```
qang/
├── __init__.py         # Public API exposure
├── core.py             # Qang class, conversions, milliqang, branch inversion, qg_Z <-> qg_S identity
├── phase.py            # qg_Phi phase unit (full-domain invertible)
├── gradients.py        # Jacobian inverses, clipped/Tikhonov regularization, pole-damped descent
├── statistics.py       # Binomial shot-noise propagation, confidence intervals
├── mixed.py            # Density-matrix qg_Z, von Neumann entropy, POVM metrics
├── multiqubit.py       # Marginal profiles, joint qg_S, Miller-Madow estimator, qg_correlation
├── transformations.py  # Closed-form transition probabilities between qang states
├── qiskit_gate.py      # Qiskit native RQangGate and FullRQangGate
├── cirq_gate.py        # Cirq native rqang_gate and full_rqang_gate
├── circuits.py         # Bell/GHZ/W/Dicke/graph-state builders and circuit -> qg profile
├── ansatze.py          # Excitation and hardware-efficient ansatze, incl. qg-native layers
├── qec.py              # 3-qubit bit-flip code with qg_Z syndrome readout
└── algorithms.py       # Teleportation, superdense coding, Grover
```

### Core Representations and Branch Inversion (`qang.core`)
The `Qang` class encapsulates both $qg_Z$ and $qg_S$ operational modes. Because $qg_S(\theta) = qg_S(\pi - \theta)$, inversion from entropy to angle is non-injective over $[0, \pi]$. `qang.core` formalizes the half-domain invertibility: it enforces monotonic branches via bisection over $H(p)$ for $\theta \in [0, \pi/2]$ (`branch='lower'`) and $\theta \in [\pi/2, \pi]$ (`branch='upper'`), providing bidirectional consistency within $10^{-4}$ rad. The forward direction needs no branch: `qg_s_from_qg_z` implements $qg_S = H((1+qg_Z)/2)$ on the whole domain.

### Gradient Regularization and Pole Damping (`qang.gradients`)
To handle the divergence of $dE/dqg = -(1/\sin\theta) \cdot dE/d\theta$ at the poles, `qang.gradients` provides two bounded alternatives:

- **Clipped Inverse Jacobian:** Clamps the denominator magnitude to $\max(\vert{}\sin\theta\vert{}, \epsilon)$, bounding the update step to $1/\epsilon$.
- **Tikhonov Regularization:** Computes $-\sin\theta / (\sin^2\theta + \epsilon^2)$. This formulation smoothly suppresses updates directly at the poles ($0$ push at $\theta \in \{0, \pi\}$), preventing boundary overshooting during variational optimization.

Neither fixes the $\arccos$ range limit. The module therefore also provides `theta_pole_damped`, which stays in $\theta$-space and scales each parameter's step by $\max(\vert\sin\theta_i\vert, \epsilon)$. This is Levenberg–Marquardt-style damping [@levenberg1944; @marquardt1963] with an untuned, geometry-derived schedule. Because it depends only on populations, it applies unchanged to $R_x$- and $R_y$-parameterized qubits.

### Shot-Noise Error Propagation (`qang.statistics`)
`qang.statistics` implements analytical error propagation using the first-order delta method. Given $N$ measurement shots, the variance of the estimator $\widehat{qg}_Z$ is $\operatorname{Var}(\widehat{qg}_Z) = \sin^2\theta / N$. Propagating through the inverse Jacobian yields:
$$\operatorname{Var}(\hat{\theta}) \approx \operatorname{Var}(\widehat{qg}_Z) \left(\frac{d\theta}{dqg_Z}\right)^2 = \left[\frac{\sin^2\theta}{N}\right] \left[\frac{1}{\sin^2\theta}\right] = \frac{1}{N}$$
The angular variance is invariant to $\theta$ to first order ($\operatorname{std}(\hat{\theta}) \approx 1/\sqrt{N}$). The module provides analytical confidence intervals and validates this behavior against empirical bootstrap trials executed via Qiskit's `AerSimulator`, including the regime near the poles where the first-order result breaks down.

### Open Quantum Systems and Multi-Qubit Registers (`qang.mixed`, `qang.multiqubit`)
The framework evaluates mixed states via $\operatorname{Tr}(\rho \sigma_z)$ and POVM measurements normalized by $\log_2(n_{\text{outcomes}})$. For multi-qubit systems, it computes single-qubit marginal projection profiles and the joint measurement entropy ($qg_S^{\text{joint}}$). For finite-shot data, `joint_qg_s_from_counts` applies the Miller–Madow bias correction [@miller1955], and `mean_qg_z_from_counts` gives the register's mean $qg_Z$ (its relaxation bias) as an unbiased sample mean. `qg_correlation` $= \sum_i qg_S^{(i)} - qg_S^{\text{joint}}$ is the total correlation [@watanabe1960] of the Z-basis outcomes: it is zero for product states, 1 bit for Bell states, and $n-1$ bits for GHZ$_n$. The documentation states its limits plainly. It is a classical correlation measure, not an entanglement measure: it is positive for classical mixtures and zero for graph states, whose entanglement lives entirely in phases that no Z-diagonal statistic can see [@hein2004graph].

# Validation and research applications

Every example script in `examples/` has a companion regression test that pins its numerical findings. `RESEARCH_NOTES.md` collects the derivations and results. The main applications are:

- **Molecular VQE.** H$_2$ (2 qubits) and LiH (4 qubits, 52 Pauli terms, mixed $R_y$/$R_x$ ansatz; cf. [@omalley2016; @kandala2017]), with Hamiltonians derived with PySCF [@sun2018pyscf] and hard-coded, so the tests need neither PySCF nor qiskit-nature. qg-space updates stall at the Hartree–Fock energy because of the $\arccos$ range. Pole damping recovers the exact ground state at learning rates where plain gradient descent diverges. The cost is 3–20× more iterations at well-tuned rates.
- **Benchmarking.** On random Quantum Volume circuits [@cross2019], $qg_S$ tracks Heavy Output Probability (Pearson $r = -0.93$). Compared with linear XEB [@arute2019], $qg_S$ needs no circuit-specific calibration, but its estimator is biased at finite shots while XEB's is not. Under gate-level noise, readout-only dephasing is exactly invisible to $qg_S$, and mid-circuit amplitude damping makes it non-monotonic (it returns to 0 at full damping). $qg_S$ alone therefore cannot certify a T1-dominated device. Pairing it with the mean $qg_Z$ separates the two regimes. This holds with idealized channels and also on a noise model built from a real IBM device's calibration data, where qg_S at 200 µs of idle relaxation falls below its noiseless value. The script runs unchanged on real IBM hardware.
- **Randomized benchmarking and error mitigation.** Under depolarizing noise, the RB signal is exactly $qg_Z(m) = (1-p)^{m+1}$, independent of the Clifford sequence [@magesan2011]. For zero-noise extrapolation [@temme2017], extrapolating in $qg_Z$-space is exact for Bloch-vector shrinkage, and extrapolating in $\theta$-space is exact for coherent angle drift.
- **Trainability.** In barren-plateau settings [@mcclean2018; @cerezo2021], regularization rescales the exponential gradient-variance decay without removing it, and raw qg gradients still diverge at the poles.

# Research impact statement

`qang` provides a reproducible, standardized foundation for quantum software engineering and education:

1. **Algorithm Development:** In variational optimization (VQE/QAOA), regularized $qg$-gradients and pole-damped updates let researchers work with expectation-value geometry while avoiding numerical instabilities and range traps at computational basis states.
2. **Error Budgeting and Calibration:** Experimentalists can directly estimate necessary shot budgets $N$ required to achieve angular fidelity benchmarks without running bespoke Monte Carlo noise simulations, and can see which noise channels an entropy-based benchmark can and cannot detect.
3. **Cross-Platform Reproducibility:** By providing unified single-qubit gate implementations matching unitary conventions across Qiskit's `UGate` [@javadiabhari2024qiskit] and Cirq's `MatrixGate`, the software guarantees statevector equivalence across backends.

The library includes an automated test suite of 633 unit and regression tests, run in continuous integration on Python 3.9–3.12. It reproduces all analytical tables and every numerical finding cited above.

# AI usage disclosure

Generative AI tools were used: Anthropic Claude 3.5 Sonnet and Claude Opus 5.5, and OpenAI GPT-4o. They assisted with code generation and refactoring (library modules and example scripts), test-suite expansion, numerical experiments, and drafting of documentation and this paper. The author framed the research questions, made the core design decisions, and reviewed, edited, and validated all AI-assisted outputs, including every derivation and numerical result.

# References
