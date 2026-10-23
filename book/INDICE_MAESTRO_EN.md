# Quantum Computing, Information and Physics in qang Space (qg)

**A comprehensive treatise — consolidated master index (English edition).**
Merges the two October 2026 indices (*Treatise… Consolidated Master Index* and *Master Index… Experimental*)
without dropping any proposal, and adds what the repository already contains that neither index covered.
Basis: qang 0.12.0, `RESEARCH_NOTES.md` §1–§164 and Appendix A. Spanish edition: `book/INDICE_MAESTRO_ES.md`
(same numbering, item by item).

## How to read the marks

| mark | meaning |
|---|---|
| **§N** | backed: a section of `RESEARCH_NOTES.md` with code, tests and, where relevant, pre-registered predictions |
| ⚠️ | in the repository, but the original index stated it differently: the wording here is already corrected |
| 🔬 | proposal from the original index with no backing in the repository yet: prove it (or drop it) before writing the chapter |
| 🆕 | new proposal to develop within the book |
| 🧪 | candidate for a measurement on real hardware (IQM Resonance) |

General rule of the book: anything said to happen "on a device" refers to **simulated noise models** of IBM or IonQ
devices until a real run exists (Part X).

---

## PART I. Foundations: the angular–probabilistic metric (the qang unit)

### Chapter 1. The qang unit: definition, milli-qang and the probability equation ✅ written
1.1 Definition of the qang: Born's rule p₀ = cos²(θ/2), polar bias qg_Z = ⟨σ_z⟩ = 2p₀ − 1 = cos θ, qang angle and qg_S; the qang as an engineering unit (decibel analogy) — §7; figure: the qang circle with an m-qg scale
1.2 Theory of the milli-qang: definition and scale (−1000 to +1000 m-qg); linear in probability (1 m-qg = Δp₀ = 0.0005), not linear in angle (1 mrad on the equator, 2.56° next to the pole); resolution from shots σ_m = 1000 sin θ/√N and N = (1000 sin θ/δ)²; gate error budget in m-qg — §160, §167; figures: milli-qang ruler, resolution from shots, pulse budget
1.3 The probability equation of the qang and the milli-qang: θ, qg_Z, qg_S and m as exact functions of p₀; anchor table; generalization to several outcomes through Wootters' statistical distance (Bhattacharyya angle) — §167; figure: the readings of one probability
1.4 Why measure in qg: the geometry–probability gap (gradient singularity and state-dependent variance) — §1, §5
1.5 Inversion θ = arccos(qg_Z) and the branch sign — §65
1.6 The outcome entropy qg_S, its invertibility by halves and its error bars — §2, §7, §51
1.7 The qang angle as the arcsine transformation of a proportion (Bartlett, Anscombe, Freeman–Tukey) — §125
1.8 Mixed states and multi-qubit registers — §3, §4, §45
1.9 In code: `qang.probability` as the analysis tool (`analyze`) — §167
1.10 Scope and limits of the base: not new physics but a working scale; first hardware test (§168, IQM Garnet): 1.64 m-qg per X gate, angle errors of 16–55 m-qg, M2 failed; hardware figure — §168

### Chapter 2. Native gates, closed algebra and the universal operator
2.1 The elementary gate RQangGate(qg_Z) and its native decomposition into Ry(θ) — §6
2.2 🔬 The universal operator U3_qang(qg_Z, φ, λ): a closed algebraic matrix with decoupled phases
2.3 🔬 Verification of U3_qang against standard libraries to machine precision (I, H, X, Z)
2.4 🔬 Generalization to qudits (d ≥ 3) through the Cartan subalgebra of su(d) and the Gell-Mann matrices (starting point: the qutrit test, §70–§72)
2.5 The minimal pulse-area principle (`qang.pulse_area`) and the damping cos θ·e^(−θ²σ²/2) — §160
2.6 The signed qg gate and the noise-free screen for how much T1 the filter lets through — §68
2.7 One qubit order for every SDK (Qiskit, Cirq, PennyLane, Braket, pytket) — §130
2.8 On the base of Chapter 1: preparing from p₀ or m-qg; the gate as a probability equation (one m-qg number fixes its action, capacity 1 − qg_S); pulse area in m-qg — §160, §167; figure: m-qg error of R_x, requested angle against the minimal-area representative

### Chapter 3. Reflections, Householder and Grover dynamics
3.1 🔬 Unistochastic and stochastic closed-rotation matrices without transcendental functions
3.2 🔬 The Householder reflector in qang space: H(qg_Z) = −qg_Z·σ_z − √(1−qg_Z²)·σ_x
3.3 🔬 The Grover diffuser as a pure qg reflection: D(qg_Z) = −H(qg_Z)
3.4 🔬 Transition probability between two qg states: P(q1→q2) = [1 + q1q2 + √((1−q1²)(1−q2²))]/2
3.5 The Grover trajectory as a Chebyshev polynomial: qg_Z(θ_k) = cos((2k+1)θ_0) — §37
3.6 Analytic limits, coarse angles and overshoot when M/N is not negligible; Grover with noise: the qg reading loses to the histogram — §37, §63
3.7 Grover amplitude estimation on IonQ's noisy simulator — §39
3.8 On the base of Chapter 1: the transition probability has qang angle γ (the angle between the states) and reading 1000·r₁·r₂ m-qg; the Grover trajectory and depth resolution in m-qg — §37, §166, §167; figure: Grover trajectory in m-qg (a = 0.1 and 0.3)

### Chapter 4. Sampling statistics, Bayesian inference and the Haar distribution
4.1 Shot-noise propagation by the delta method: Var(θ̂) ≈ 1/N — §5, §125
4.2 The Haar distribution in qg space: Archimedes' hat-box theorem, qg_Z ~ U(−1, 1) — §15
4.3 The Beta-Binomial Bayesian estimator: exact intervals near the poles where the delta method collapses — §5, §125
4.4 Second-order unbiased estimators for qg² and the local radius r² (`qg2_unbiased`) — §97, §126
4.5 ⚠️ Adaptive shot budgeting — §143 (planning shots for a target qg precision). 🔬 The figure "33.9% of shots saved" is not in the repository: measure it or drop it
4.6 Few-shot single-qubit tomography; classical shadows against direct measurement — §35, §62
4.7 Functional-analysis foundation: Riesz–Fréchet, why qg_Z is the canonical linear readout and qg_S is not — Appendix A
4.8 On the base of Chapter 1: Haar is uniform in m-qg and in p₀ (½ sin θ in angle); intervals in m-qg (the delta method collapses or runs off the scale); planning shots in m-qg; the milli-qang as a linear functional (Riesz–Fréchet) — §15, §143, §167; figures: Haar in m-qg and in angle, interval widths with 50 shots

---

## PART II. Local geometry, quantum optics and geometric phases

### Chapter 5. The local Bloch radius as a measure of purity and entanglement ✅ written
5.0 The probability equation with a radius: p₀ = (1 + r cos θ)/2 and r² = (m_X² + m_Y² + m_Z²)/10⁶; at definite weight, 1 − r² = 4p₀(1 − p₀) — §167; figures: deficit in m-qg, direction separated from the radius, duality P² + V² = r²
5.1 r² = qg_X² + qg_Y² + qg_Z² = 2·Tr(ρ²) − 1 and the surface deficit D = 1 − r² — §97
5.2 Correspondence with the Coffman–Kundu–Wootters one-tangle and the Meyer–Wallach measure — §98
5.3 The radius under elementary gates (rotations, CNOT, CZ, iSWAP, Fredkin) — §97
5.4 Separating polar direction from radius: θ = atan2(√(qg_X²+qg_Y²), qg_Z) = arccos(qg_Z/r) under depolarizing noise — §102, §103
5.5 The collapse r = |qg_Z| in fixed-Hamming-weight states and reading entanglement in the Z basis — §98, §100, §104
5.6 Wave–particle duality as the Bloch radius, and entropic uncertainty with qg_S — §127

### Chapter 6. Optical polarization and Stokes parameters in qg units
6.0 Malus's law as a probability equation: the transmission of a polarizer is p_0=cos ²Δ, so the qang angle is 2Δ and each normalized Stokes parameter reads in m-qg (S_1/S_0=qg_Z) — §167
6.0b ✅ written: Malus as p₀ = cos²Δ (θ = 2Δ, m = 1000·r·a), Stokes as qg values, degree of polarization = radius, Mueller = gate rule, photon-counting polarimetry with the bias of P̂ — §83, §167; figures: angle doubling, Malus in m-qg, bias of the degree of polarization
6.1 The Poincaré–Bloch equivalence: qg_Z = S1/S0, qg_X = S2/S0, qg_Y = S3/S0 — §83
6.2 Degree of polarization as the norm of the qg vector, and the purity of mixed states — §83
6.3 Lossless Mueller matrices and the rule qg_P′ = qg{U†PU} — §83
6.4 Malus's law unified: I(θ) = (I0/2)(1 + qg_Z cos 2θ + qg_X sin 2θ) — §83
6.5 Single-photon polarimetry from counts, with credible intervals — §83

### Chapter 7. Geometric phases and Stokes' theorem on the sphere
7.0 The geometric phase in the probability equation: a loop at constant qg_Z picks up (qg_Z-1)/2=-p_1 turns, that is, (m-1000)/2000 turns in milli-qang — §167
7.0b ✅ written: the phase of a loop at constant height is p₀ turns (mod 1); the interferometer turns the phase into cos²(γ/2) and its qang angle is the phase, with 1/√N for any value; Pancharatnam's octant — §84, §167; figures: phase against m-qg with three computations, interferometric measurement
7.1 The Berry connection A = i⟨ψ|∇ψ⟩ and its curvature as a monopole field — §84
7.2 Stokes on the sphere: γ(C) = −Ω(C)/2 as a curvature flux — §84
7.3 Loops at constant qg_Z: phase φ = (qg_Z − 1)/2 turns (mod 1) — §84
7.4 Three independent computations (gauge-invariant overlap product, spherical fan, numerical flux) — §84
7.5 The Pancharatnam phase and the canonical octant H → D → R → H — §84

### Chapter 8. Fisher information, clocks, gravity and metrology 🆕 (not in either index)
8.0 From the base: the qang angle is the Fisher-flat coordinate (information 1 per shot) and qg_Z the curved one (1/(1-qg_Z²)); Chapter 1's m-qg resolution is its Cramér–Rao bound — §167
8.0b Information geometry in qg: Fisher information, the Fubini–Study metric (g_ii=1/4 in the real R_Y+CZ ansatz) and the block of qg_Y correlators; ⚠️ natural gradient in qg coordinates equals Euclidean descent in θ: the 1.3× gain comes from the block metric (4.6× with the full one), not from the coordinate — §159
8.1 Two clocks and gravity: the differential phase read from the ellipse (gravitational redshift over 1 m) — §126
8.2 Ramsey sensing: where to operate — §36
8.3 GHZ metrology: when entanglement beats N independent qubits — §58
8.4 Thermal states: qg_Z = tanh(βh) — §25

---

## PART III. Neutrino physics, oscillations and quantum simulation (`qang.neutrino`)

### Chapter 9. Quantum kinematics of flavour oscillation in vacuum
9.0 The oscillation probability P(ν_α→ν_β) as a qang probability: its angle 2 arcsin √P, its reading m = 1000(1 − 2P) m-qg and its error bar on the scale of Chapter 1 — §167
9.1 ν_e ↔ ν_μ ↔ ν_τ oscillation as a unitary geometric rotation on the qg sphere — §151
9.2 Vacuum probability: qg_Z = cos²2θ + sin²2θ·cos Δ (the spherical law of cosines) and P(ν_α → ν_β) — §151
9.3 At the oscillation maximum the mixing angle is a quarter of the qang angle: θ_mix = θ_qg/4; for θ > π/4 it returns π/2 − θ (octant degeneracy) — §151
9.4 Error bars independent of θ: σ = 1/(4√(N+½)) — §151
9.5 Three flavours in the weight-1 sector (|100⟩, |010⟩, |001⟩) on 3 qubits; PMNS as three RBS gates and two phases — §151
9.6 ⚠️ Noise suppression with the qg filter: the error falls ~3× (3.0–3.8×) **on simulated noise models** — §151; on real hardware (IQM Garnet) 1.6× and the oscillation maximum returns to 2.6 GeV — §165; figures: precession cone, three-flavour spectrum, Garnet run
9.7 The CP asymmetry in vacuum, and why it survives even without the filter — §151

### Chapter 10. Matter effects, solar neutrinos and collective oscillations
10.0 Flavour probabilities in matter, in the Sun and in collective oscillations read in m-qg: matter effects as shifts measured on the same scale as shot noise — §167
10.1 The MSW effect in the Earth as a weight-preserving Trotter circuit; matter mimics CP violation — §152
10.2 Solar neutrinos: the adiabatic MSW conversion and why the whole Sun is one rotation (`solar_circuit`) — §156
10.3 Collective oscillations: the conserved number is the mass number, not the flavour number (`reweight_sectors`) — §154
10.4 Mass ordering (normal against inverted) read through matter (NUFIT_IO) — §163
10.5 🔬 Varying densities and comparison with the Standard Solar Model (a realistic density profile; §156 uses a simplified one today)

### Chapter 11. Majorana phases, neutrinoless double beta decay and sterile neutrinos
11.0 Majorana phases and the sterile neutrino seen from the probability equation: what changes no probability changes no qang reading — §167
11.1 ⚠️ Majorana phases are invisible to oscillations: they are diagonal phases that the transpiler deletes (`majorana_invisible`); not a deliberate "analytic elimination" — §163
11.2 Majorana phases in the effective mass m_ββ of neutrinoless double beta decay — §163
11.3 The sterile neutrino ν_s as a fourth qubit in the weight-1 sector (`sterile_circuit`) — §163
11.4 ⚠️ A 5.5× filter gain with 4 flavours (3+1) **on simulated noise models** — §163 🧪
11.5 🔬 Phenomenological limits and comparison of 3+1 models with short-baseline experimental data
11.6 The four quantum numbers in qg terms — §155

---

## PART IV. Quantum teleportation in optical networks and entanglement distribution (`qang.qpon`)

### Chapter 12. Optical teleportation and the limit of the Bell pair
12.1 Foundations of Bennett et al. (1993) and the role of Bell states — §157
12.2 Asymmetric lossy fibre channels: why the Bell pair stops being the optimal resource — §157
12.3 ⚠️ The polar design rule of a quantum passive optical network (QPON): **tan α* = √η / κ** (the original index said √(η/κ)) — §157
12.4 Optimal fidelity F* = (2 + η/κ)/3; with assistance F* = 1 at tan α* = √N > 1: a "more than maximally entangled" resource — §157
12.5 The optimal resource state read in its polar polarization: qg_Z* = cos 2α* = (κ² − η)/(κ² + η) — §157
12.6 Bell pairs in a quantum network, read in qg — §57

### Chapter 13. Dual rail, sector filtering and long-distance fidelity
13.1 Teleportation fidelity as a linear readout of three qg correlators: F = (3 + qg_XX − qg_YY + qg_ZZ)/6 — §158
13.2 Loss heralding by the network users as a sector-filter projector — §158
13.3 Dual-rail encoding: detecting and discarding every photon loss — §158
13.4 ⚠️ At 20 km, dual rail with the filter beats the noiseless single-rail optimum on Torino and Fez (2 of 3 simulated models; a post hoc finding, not pre-registered) — §158
13.5 🔬 Passive quantum routing and switching in metropolitan multi-carrier topologies
13.6 🆕 Quantum repeaters and chained entanglement swapping, read with qg_ZZ and the dual-rail filter: in the book, exact algebra (the correlators of Bell-diagonal pairs multiply; F = (1 + w^n)/2); the dual-rail chain on devices is still open
13.7 🆕 Gate teleportation and its readout in qg correlators

---

## PART V. Relativity, time loops and black-hole evaporation (`qang.postselection`)

> These are **information-theoretic models**, not statements about spacetime (§164). Both are teleportation
> with postselection on a Bell outcome, that is, a filter on the Bell sector.

### Chapter 14. Postselected closed timelike curves (P-CTC) and the resolution of paradoxes
14.1 Relativistic quantum modelling: Deutsch's D-CTC formalism against the P-CTC of Bennett–Svetlichny / Lloyd et al. (2011) — §164 (P-CTC); 🆕 D-CTC to be developed
14.2 Teleportation into the past as a Bell projection with strict postselection — §164
14.3 The success probability as a pure trigonometric law: P_success = cos²(φ/2), whose qang angle arccos(2P − 1) = φ — §164
14.4 The grandfather paradox: P(paradox) = 0 at φ = π; suppressed to the noise floor on Brisbane and Fez — §164
14.5 A rotation controlled by a system: Tr_C(U) = diag(1, cos(φ/2)) and qg_Z(S) = (1 − c²)/(1 + c²) — §164
14.6 Non-linear effects of P-CTCs and limits on discriminating non-orthogonal states — 🔬 (pending: circuit and prediction)
14.7 🆕 Deutsch's D-CTC as a fixed point: a numerical comparison with P-CTC on the same circuit, and what qg sees in each: in the book, exact algebra (P-CTC moves m_Z of S, D-CTC does not and lowers m_X; the local radius separates Deutsch's fixed points); the device run is still open
14.8 🆕 The cost of postselection as a rate (cos²(φ/2)): how many shots "time travel" costs on a device: in the book, 1/P shots per useful shot (2 at π/2, 4 at 2π/3, 400 at 0.1 rad from the paradox; never above 2 when controlled) 🧪

### Chapter 15. Black-hole thermodynamics, quantum chaos and the Yoshida–Kitaev decoder
15.1 Hawking's information-loss paradox and black-hole complementarity — §164 (context)
15.2 Quantum chaos and information scrambling: a 3-qubit micro black hole with a Clifford that spreads every Pauli — §164
15.3 The Hayden–Preskill protocol with the Yoshida–Kitaev decoder: recovering the message thrown into the horizon from the Hawking radiation — §164
15.4 Formulation in qg units and postselection assisted by the symmetry filter: success 1/4 (m = −500), fidelity 1 without noise; decoding is crossing the equator (F > 1/2 ↔ m_F > 0); the exact law P·F = 1/4 as a hyperbola (1000 + m_P)(1000 + m_F) = 10⁶, and 4PF as a diagnostic (full noise also gives success 1/4) — §164; in the book, checked on Cliffords and Haar
15.5 ⚠️ Recovered fidelity 0.55–0.74 (> 1/2) **on IBM noise models** (Brisbane, Torino, Fez); without scrambling, 0.25. The original index cited IonQ: there is no IonQ run of this experiment — §164 🧪
15.6 🆕 Larger black holes: 5–7-qubit scramblers, the gate cost and how far the filter reaches: in the book, exact (Haar with one radiated qubit gives F ≈ 4/7; with 2 and 3, 0.84 and 0.95) and on the FakeFez model (Clifford: 35, 126 and 252 gates; F 0.80–0.89 but the success falls to 0.105 and 4PF to 0.37); the §164 scrambler as a Clifford drops from 90 to 35 gates; no device run
15.7 🆕 A toy Page curve: entropy of the radiation against evaporation time, read with qg_S: in the book, ⚠️ qg_S does not see the curve (the Z reading keeps rising); the Rényi-2 entropy from the sum of qg_P² does, at the cost of 4^k − 1 correlators
15.8 🆕 Traversable wormholes as teleportation (an ER = EPR-type protocol) with the fidelity read in qg correlators: in the book, F = (1 + qg_XX − qg_YY + qg_ZZ)/4 exact, (1, −1, 1) with the scrambler and (0, 0, 0) without; a Jafferis-2022-type experiment stays 🔬

---

## PART VI. Quantum cybersecurity, random-number generation and post-quantum frontiers

### Chapter 16. Demarcation: quantum cryptography against post-quantum cryptography (PQC)
16.1 The nature of PQC: classical lattice algorithms (ML-KEM/Kyber, ML-DSA/Dilithium), codes and hashes — §13, §14 (context)
16.2 Why qg does not apply to PQC: no Hermitian operators, qubits or Bloch sphere in those schemes — §13, §14
16.3 The quantum attack vector (Shor/QPE): the eigenstate register projects onto qg_Z = −1 without resolving the phase — §13
16.4 Conclusion: qang is a physical- and quantum-layer tool, not classical algebraic cryptography — §14
16.5 ⚠️ There is no `qang.crypto` module: the cryptography lives in `examples/` (bb84_*, certified_qrng_*) and in `manuscript/crypto*`

### Chapter 17. Quantum key distribution (QKD) and BB84 in qg space
17.1 QBER in qg observables: QBER_Z = (1 − |qg_Z|)/2, QBER_X = (1 − |qg_X|)/2 — §40
17.2 The asymmetry witness A_Z = qg(0) + qg(1): instrumental T1 relaxation or an eavesdropper (Eve) — §40
17.3 Fewer false alarms under hardware degradation while keeping intrusion detection — §40, §43, §44. 🔬 The figures "48–96% → 0.6%" are not in the notes (the 0.6% in §40 says something else: the qg monitor *misses* Eve in that attack); use the real figures
17.4 Finite blocks (the Tomamichel–Lim–Gisin–Renner bound): early drift detection without shortening the key — §43
17.5 Combined attacks (T1 drift with active interception), partial cloning, one-basis attacks and T2 drift — §49
17.6 BB84 on IonQ's noisy simulator — §44

### Chapter 18. Certified quantum random numbers (QRNG) and channel auditing
18.1 Why output-bias analysis fails: 50/50 hides the loss of private randomness to decoherence — §41
18.2 A closed min-entropy bound from transverse coherence: H_min = −log₂[(1 + √(1 − qg_X² − qg_Y²))/2] — §41
18.3 Security in 100% of the noisy runs against 0% for the usual estimators — §41
18.4 Asymmetric readout calibration (the §31 protocol): thermal population against detector error — §31, §41
18.5 QRNG on IonQ's calibrated simulator (Aria-1/Forte-1) with leakage and attenuation — §42

### Chapter 19. Quantum batteries and networks 🆕 (not in either index)
19.1 Qubit batteries in qg: storage, certification and locked charge (ergotropy) — §56

---

## PART VII. Quantum machine learning (QML) and the symmetry filter

### Chapter 20. Weight-conserving quantum neural networks (WeightQNN)
20.1 RBS (Reconfigurable Beam Splitter) gates as weight-conserving Givens rotations — §76
20.2 Architectures in weight 1 (model E) and weight 2 (model W) on 5, 6 and 8 qubits — §75, §79, §90, §111, §114
20.3 Data encoding: unary normalization, pair products, ring and dual encoding — §16, §87, §101
20.4 Readouts: direct qubit, linear head and ⟨Z_iZ_j⟩ correlators; a two-channel readout — §81, §95
20.5 Classical simulability and the limits of quantum advantage — §75, §80
20.6 The multiclass QNN: each qubit a class; 3, 5 and 8 classes on 8 qubits — §110, §114
20.7 QNN inside qiskit-machine-learning; a differentiable filter in PennyLane — §131, §140
20.8 Weighted-QSP activations in a one-layer KAN; quantum signal processing in qg — §134, §135, §136

### Chapter 21. The qg symmetry filter and exactness under T1 relaxation
21.1 Kraus operators of amplitude damping and the no-jump decay (1 − γ)^(k/2) — §77
21.2 Theorem: postselecting the weight-k sector exactly restores the noiseless state under homogeneous T1 — §77
21.3 Fraction of shots kept: K = (1 − γ)^(k·d); how it scales with qubits, weight, depth and shots — §94
21.4 Identical gradient: training on filtered noise equals noiseless training; training from shots — §77, §109, §112
21.5 Property F5: invariance under homogeneous drift; dependence only on the ratios (1 − γ_q)/(1 − γ_r); unequal T1 corrected — §78, §82, §117
21.6 Analytic ceiling and decision rule: MSE ≈ (1 − z²)/(K·S) against the raw readout; low kept fractions — §96
21.7 Dephasing: the filter plus ZNE, at 5 and 8 qubits; dephasing inside the gates — §118, §122, §132, §133
21.8 Gate and readout errors in qang's simulator — §124, §129

### Chapter 22. Echo calibration and the noise bias (pull effect)
22.1 In-sector leakage: errors that keep the weight but permute the excitation — §105
22.2 Echo transfer matrices M (5×5, 10×10, 28×28) and unmixing with M^(−1/2) — §105, §107, §115
22.3 Fewer classification errors on simulated IBM and IonQ backends — §106, §113. 🔬 The figure "43–56% of the decision value" is not in the notes: verify
22.4 The noise bias (pull effect) in multiclass classifiers: the term K·L + (1 − K)·v_c — §121
22.5 Pull-free readouts: robust even without the filter — §123
22.6 Calibrating with the device itself; the whole pipeline on noise models — §119, §120

### Chapter 23. Information geometry, the natural gradient (QNG) and kernels
23.1 The singularity dE/dqg = −(1/sin θ)·dE/dθ and its regularizations (Tikhonov, clipping) — §1
23.2 Quantum Fisher information in qg: F_Q = 1/(1 − qg_Z²); the qang angle is the Fisher-flat coordinate — §159
23.3 ⚠️ QNG in qg coordinates equals Euclidean descent in θ: the 1.3× gain comes from the qg_YY block metric (4.6× with the full metric), not from the coordinate — §159
23.4 Pole-damped descent: robustness to aggressive learning rates in VQE — §9
23.5 Filtered quantum kernels: Gram-matrix repair 2.4–6× — §161
23.6 Barren plateaus in qg space; global cost against the local qg cost — §12, §23

---

## PART VIII. Many-body dynamics, open channels, matter and chemistry

### Chapter 24. Hamiltonian simulation, quantum walks and VarQTE
24.1 Trotterized XY chains in the weight-1 sector — §147
24.2 Continuous-time quantum walks: ballistic spreading recovered (exponent 1.84 against 1.94) against noise-driven diffusion (0.87) — §147
24.3 XXZ chains with several excitations: the filter at weights k = 1, 2, 3 — §60, §61, §148
24.4 Quantum channels as affine maps of (qg_X, qg_Y, qg_Z) — §162
24.5 Variational time evolution (VarQTE) at fixed depth by McLachlan's principle — §162
24.6 Differential equations — §22
24.7 1D Hubbard dynamics with N and spin filters; Gauss-law witnesses in a lattice gauge theory — §29, §66
24.8 Transverse-field Ising: is the entropy gap a coherence witness? — §34
24.9 RBM states: the I–η plane swept by the qg marginals — §47

### Chapter 25. Quantum spectroscopy, PCA and molecular chemistry
25.1 Quantum PCA of data in the unary weight-1 sector — §146
25.2 Adjacent RBS rotations to isolate eigenvalues on physical qubits — §146
25.3 The molecular symmetry witness and occupation number: N_e = Σ(1 − qg_Z,i)/2 — §21, §52
25.4 H2 and LiH: accuracy recovered under T1 and the limit of deep circuits; why the filter helps H2 and not LiH — §21, §50, §53, §54
25.5 Spin-resolved filters (N↑, N↓), total S² and Λ (the molecular m_l) as a third filter — §28, §55, §69, §153
25.6 Overlaps: the filtered unary inversion test against the ancilla cost of the swap test — §149
25.7 Constrained optimization: QAOA with "exactly K" — §30, §38

---

## PART IX. Error correction, continuous syndromes and characterization

### Chapter 26. Error correction and continuous syndromes
26.1 Leung's 4-qubit code for T1-dominated channels and the T2 > T1 criterion — §27, §32
26.2 The syndrome as a continuous qg witness: tracking drift and switching codes — §33, §88, §92, §93, §99
26.3 A T1-aware decoder under circuit-level noise; the d = 3 surface code in qg — §59, §64
26.4 Qutrits and erasure qubits: the |2⟩ flag, the (1 − h) rule — §70, §70b, §71, §72, §73, §74
26.5 Circuit knitting: the shot multiplier γ = 1 + 2√(1 − qg²) for RZZ — §26
26.6 Randomized benchmarking and ZNE in qg units; qg_S as a benchmarking signal — §10, §11

### Chapter 27. Hardware characterization in qg
27.1 T1, T2, effective temperature and readout in a single sweep — §31, §86
27.2 ⚠️ Which noise is it? Mean qg_Z against XEB/HOP — §18. 🔬 The figure "88–91% with 100 shots" is not in the repository: verify it in §18 or drop it 🧪
27.3 Coherent conditional-phase errors and always-on ZZ crosstalk — §145, §150
27.4 Control quantization: the even qg grid lowers the error in 43 of 45 distributions (median 1.56×) — §17

---

## PART X. Experimental evidence, real hardware and the falsification audit

### Chapter 28. Methodology: the pre-registered falsification protocol
28.1 The discipline of pre-registration: hypotheses and predictions committed (commit or sha256) before running — §46, §75–§164
28.2 ⚠️ Passes and failures: "123 passed / 53 failed" covers **only §75–§123** (the QML line). 🔬 Recount over §1–§164
28.3 Test beds: calibrated noise models of IBM (Brisbane, Sherbrooke, Torino, Fez, a fleet of 21) and IonQ (Aria-1, Forte-1) — §20, §137, §144
28.4 🔬 Runs on real hardware: QuTech/Quantum Inspire (Tuna-5, Tuna-9, Tuna-17) does not appear in the repository. IonQ Forte-1 (§128) was refused by the provider before the job was accepted. **There is no real-hardware data today**
28.5 🔬 The falsification criterion for QPUs: variance of the inversion echo Var_θ(F_echo) over continuous trajectories (not in the repository)
28.6 🆕 🧪 First run on real hardware: IQM Resonance (Garnet/Emerald) — see Chapter 31

### Chapter 29. The confirmed empirical advantages (where qg brings a measurable benefit)
29.1 Exact restoration under T1: ideal accuracy recovered in paired runs — §80. 🔬 The figures "+2.5 points at weight 1, +16.5 at weight 2, 60 runs" are not in the notes: verify
29.2 Lower error of polar observables on simulated noisy IBM and IonQ backends — §85, §144. 🔬 The range "2.7–3.6×" is not in the notes: take the figures from §85 and §144
29.3 In-sector leakage mitigated by echo: wrong decisions from 15 to 3 in 945 readings — §106–§108 (945 readings confirmed; verify "15 → 3")
29.4 Control quantization (see 27.4) — §17
29.5 T1 diagnosis without classical simulation (see 27.2) — §18
29.6 ⚠️ Mitigation without extra circuits: 3× for 3-flavour neutrinos and 5.5× with sterile neutrinos (simulated) — §151, §163
29.7 Operational cybersecurity: fewer BB84 false alarms under thermal drift — §40, §43

### Chapter 30. Structural limits, equivalences and negative results
30.1 Blindness to relative phases and multipartite entanglement in the Z basis: graph states — §8
30.2 The non-monotonicity of qg_S under pure amplitude damping, and the fix with the pair (qg_S, mean qg_Z) — §10, §14
30.3 The QNG equivalence: no intrinsic speed-up from a change of coordinate — §159
30.4 The filter fails in deep circuits: LiH with 60 entangling gates and Hubbard at L ≥ 4 — §29, §50
30.5 WeightQNN is not quantum supremacy against classical quadratic models — §75
30.6 Inapplicability to pure phase estimation (QPE, Shor, HHL) and to PQC — §13
30.7 The P-CTC paradox on Torino, the decoder fidelity and the other pre-registered failures of §157–§164 — §157–§164
30.8 The consolidated list of known limitations — §14

### Chapter 31. 🆕 🧪 First experiment on real hardware: IQM Resonance
31.1 Why IQM: native PRX and CZ gates, a square lattice; how RBS and XX+YY gates compile
31.2 The `qang/iqm.py` module (analogous to `qang/ibm.py`): readout errors from calibration, choice of qubit chain, the qg filter
31.3 A free rehearsal on IQM's simulated backends before spending credits
31.4 Budget: a minimal probe job (≈10%), the main experiment (≈60%), a reserve (≈30%); the token only in an environment variable
31.5 The pre-registered main experiment: 3-flavour neutrinos (§151) — 3 qubits, ~18 two-qubit gates
31.6 Next candidates: single-qubit P-CTC (§164), the Yoshida–Kitaev decoder (§164), H2 (§20/§128)
31.7 Result, passes and failures, and comparison with the simulated models

### Chapter 32. Software architecture, the multi-SDK ecosystem and physical execution
32.1 The unified `qang.interop` order (Qiskit, Cirq, PennyLane, Braket, pytket) — §130
32.2 Qiskit V2 primitives (SamplerV2, EstimatorV2) with filtered observables — §131
32.3 Sector-aware compilation on IBM Runtime and chain selection (`best_chain`); predicting the kept fraction from calibration — §137, §138, §141
32.4 ⚠️ Static ZZ: `ibm.add_static_zz` **simulates** the crosstalk to measure how much the filter sees; it does not mitigate it — §150
32.5 ZNE with the filter, and shot planning — §142, §143
32.6 Guidelines for superconducting (IBM, IQM) and trapped-ion (IonQ Cloud / AWS Braket) hardware — §128, §137, Ch. 31

## PART XI. Nanoquantum 🆕

### Chapter 33. Nanoquantum: transmission, partition noise and qubits at the nanoscale
⚠️ No module or research-notes section yet: each item starts from the probability equation of chapter 1 and needs its own development and verification.
33.1 The tunnelling transmission probability T as a qang probability: θ_T = 2 arccos √T and its reading in m-qg
33.2 The Landauer formula G = (2e²/h) Σ T_n: each channel of a quantum point contact is a qang reading
33.3 Partition noise S = 2eI(1 − T) and the Fano factor F = 1 − T: the binomial variance T(1 − T), stabilized by the qang angle
33.4 Coulomb blockade and the single-electron transistor: the occupation of a level as p₀ and qg_Z = −tanh(x/2)
33.5 Quantum dots as charge and spin qubits: Rabi and charge-sensor read-out in qg; read-out fidelity in m-qg
33.6 NV centres in nanodiamond: the ODMR contrast as qg_Z and nanoscale magnetometry in m-qg

---

## Appendices
A. Catalogue of the 15 fundamental gates in qg representation
B. Transition formulas, Gell-Mann matrices 🔬 and Chebyshev polynomials in parametric circuits — §37
C. The echo-calibration protocol and matrix inversion in Hamming sectors — §105–§108
D. Quick reference to the qang 0.12.0 API: core, formulation, sectors, qml, ibm, interop, pulse_area, neutrino, postselection, qpon, fisher, kernels, varqte, channels, angular, qsp, knitting, qec, statistics, polarization, geometric, clock_comparison, complementarity, proportions (⚠️ no `crypto`; 🆕 `iqm`)
E. Falsification statement and the register of pre-registered predictions (full count §1–§164)
F. Functional-analysis foundation: Riesz–Fréchet — Appendix A of the notes

General bibliography and Zenodo DOI record (10.5281/zenodo.22832150 and associated preprints).

---

## Open items before writing

**To prove or drop (🔬):** the decibel analogy (1.6); U3_qang and its verification (2.2–2.3); qudits with Gell-Mann
(2.4); Householder, Grover as a reflection and the transition formula (3.1–3.4); the 33.9% figure (4.5); the Standard
Solar Model (10.5); 3+1 phenomenology (11.5); metropolitan routing (13.5); non-linear P-CTCs (14.6); the figures for
BB84 (17.3), classification (22.3), T1 (27.2), 29.1 and 29.2; Quantum Inspire (28.4); Var_θ(F_echo) (28.5); the full count (28.2).

**New proposals (🆕):** clocks and metrology (Ch. 8); repeaters and gate teleportation (13.6–13.7);
D-CTC (14.7) and the cost of postselection (14.8); larger black holes, the Page curve and traversable wormholes
(15.6–15.8); batteries (Ch. 19); the IQM experiment (Ch. 31).
