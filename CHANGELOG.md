# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `examples/lih_parity_verification_qg.py`: electron-parity and N mod 4 checks on ancillas reach every
  LiH measurement group; error cut 6-8x, below Hartree-Fock at 3 layers (RESEARCH_NOTES §52).
- `qang.statistics.qg_s_estimate`, `qg_s_from_qg_z_interval`: qg_S point estimate (Miller-Madow)
  and interval (Wilson interval for qg_Z mapped through qg_S = H((1+qg_Z)/2));
  `examples/qg_s_error_bars.py`: coverage of qg_S intervals (RESEARCH_NOTES §51).
- `examples/filter_scaling_lih_qg.py`: why the qg filter helps H2 and not LiH -- the LiH error sits
  in the X/Y measurement groups a Z-basis filter cannot reach; witness-based rescaling of those
  groups overcorrects (RESEARCH_NOTES §50).
- `examples/bb84_attacks_beyond_ir_qg.py`: BB84 attacks as Pauli channels (phase-covariant
  cloner, one-basis intercepts) and T2 drift; a dephasing family removes the §43 false alarm and
  exposes a Z-only-intercept blind spot (RESEARCH_NOTES §49).
- `examples/coherent_drift_filter_zne.py`: coherent MS over-rotation and drift on the native
  circuits of the hardware plan; native folding is blind to coherent angle errors, the qg
  filter removes 94 % of them (RESEARCH_NOTES §48).
- `examples/rbm_mutual_information_qg.py`: the I-eta bounds of RBM neural quantum states
  (Singh et al., Academia Quantum 2026) written in qg marginals; lower-bound saturation traced
  to Z2 symmetry and broken by a longitudinal field or a symmetry-broken learner (RESEARCH_NOTES §47).
- Pre-registered Forte-1 predictions for the IonQ hardware plan (RESEARCH_NOTES §46):
  `examples/ionq_sim_h2_stretched.py` (H2 at 1.5 and 2.5 A on the forte-1 noise model),
  `examples/ionq_hardware_plan.py`, and `tests/test_ionq_hardware_predictions.py` pinning them.
- `qang.statistics`: `propagated_theta_variance_mixed`, `theta_qcrb_variance`, `qg_covariance`,
  `delta_method_variance`, `register_witness_variance`; `examples/multiqubit_error_propagation_qg.py`
  (RESEARCH_NOTES §45: §5 error propagation for mixed states and correlated registers).
- `examples/ionq_sim_bb84.py`: BB84 with an ancilla-built T1 memory and a measure-and-resend
  Eve as native circuits on IonQ's noisy simulator; the §43 flags with a fitted device-noise
  model, observed vs predicted flag rates (RESEARCH_NOTES §44).
- `examples/bb84_finite_key_qg.py`: BB84 finite-key rate (Tomamichel et al. 2012) on the §40
  channel and a qg diagnosis of every block from post-error-correction counts (joint T1 +
  intercept-resend likelihood ratios; catches an attack hidden under drift) (RESEARCH_NOTES §43).
- `examples/ionq_sim_qrng.py`: certified QRNG (§41) on IonQ's noisy simulator with an
  environment ion that learns the output through a partial MS gate; naive vs qg estimators,
  native-gate circuits, recorded results (RESEARCH_NOTES §42).
- Cryptography line kept separate: `notebooks/qang_criptografia_colab.ipynb` and the note
  `manuscript/crypto/main.tex` (PDF `manuscript/qang_criptografia.pdf`) on §40-§41 and why qg
  does not apply to post-quantum cryptography.
- `examples/qrng_qg_certified.py`: certified min-entropy of a qubit QRNG,
  H_min(Z|E) = -log2[(1 + sqrt(1 - qg_X^2 - qg_Y^2))/2], naive vs qg estimators with the
  §31 readout calibration (RESEARCH_NOTES §41).
- `examples/bb84_qg_eve_vs_noise.py`: BB84 on a noisy qubit channel; qg asymmetry monitor vs
  QBER monitor for intercept-resend, T1 drift and a T1-mimicking attack (RESEARCH_NOTES §40).
- `notebooks/qang_avances_colab.ipynb`: Colab notebook (Spanish) reproducing fast versions of
  RESEARCH_NOTES §20-§39 with a for/against reading per section; README badge.
- `examples/ionq_sim_zne_grover.py`: native-gate (MS) folding for ZNE on IonQ (CX folding
  is undone by the service), ZNE vs the qg filter on H2 and 3-qubit Grover amplitude
  estimation on IonQ noise models (RESEARCH_NOTES §39).
- `examples/ionq_sim_hubbard_qaoa.py`: Hubbard dynamics and constrained QAOA on
  IonQ's noisy cloud simulator (aria-1, forte-1), resumable job submission, recorded
  results in `examples/data/ionq_sim_results.json` (RESEARCH_NOTES §38).
- `examples/few_shot_tomography_qg.py`, `examples/ramsey_qg_operating_point.py`,
  `examples/amplitude_estimation_chebyshev_qg.py`: few-shot tomography (qg-Haar
  vs Jeffreys, LI, MLE, Bayes), Ramsey Fisher information F = (V^2 - qg^2)/(1 - qg^2),
  and Grover amplitude estimation as T_{2k+1}(qg) with F = m^2/(1 - qg^2), MLAE vs
  Monte Carlo with noise (RESEARCH_NOTES §35-§37).
- `examples/ising_coherence_witness_qg.py`: the §25 entropy gap as the relative
  entropy of coherence, a superadditive single-qubit qg lower bound (92-100%
  tight on thermal transverse-field Ising states), the failing basis-entropy
  bound and the graph-state counterexample (RESEARCH_NOTES §34).
- `examples/qec_syndrome_drift_tracking_qg.py`: syndrome ancillas as a continuous
  qg witness (closed forms <XXXX> = qg_X^4, <X1X2> = qg_X^2), drift tracking and
  adaptive switching between no code, phase code and Leung code (RESEARCH_NOTES §33).
- `examples/qec_leung_code_t1_qg.py`: the 4-qubit Leung code for amplitude
  damping vs no code and repetition codes, the p < gamma/4 (T2 > T1) rule, and
  the qg witness policy with the Leung option (RESEARCH_NOTES §32).
- `manuscript/witness/`: companion preprint on the register-mean qg_Z as a
  symmetry witness (chemistry, noise-type decision rule vs ZNE, Hubbard,
  constrained QAOA, qubit characterization).

### Changed
- Preprints: `manuscript/main.tex` summary table extended to §32-§39; companion witness paper
  gains the trapped-ion (IonQ noise models) and syndrome-witness sections; PDFs regenerated
  (`manuscript/preprint_qang_revisado.pdf`, `manuscript/preprint_testigo_simetria.pdf`).
- `manuscript/main.tex`: revised with a summary table of the follow-up studies
  (RESEARCH_NOTES §15–§31), updated limitations and test count.
- `examples/hardware_characterization_qg.py`: heralded qg sweep that separates
  thermal population (effective temperature) from asymmetric readout error and
  fits T1 and T2, vs the standard calibration suite (RESEARCH_NOTES §31).
- `examples/qaoa_k_constraint_qg.py`: QAOA for maximum K-vertex cover with the
  qg constraint filter, qg budget and warm starts, vs penalty QAOA, XY-mixer
  QAOA, greedy and brute force (RESEARCH_NOTES §30).
- `examples/hubbard_trotter_qg_filters.py`: 1D Fermi-Hubbard Trotter dynamics
  (8 qubits) with N and spin-resolved qg filters vs ZNE, on an all-to-all
  depolarizing device and on fake_brisbane (RESEARCH_NOTES §29).
- `examples/chemistry_spin_resolved_qg_filter.py`: H2 with separate
  spin-up / spin-down qg filters vs the total-N filter, with a spin-leak
  witness, on fake_brisbane and controlled noise (RESEARCH_NOTES §28).
- `examples/qec_repetition_code_choice_qg.py`: bit-flip vs phase-flip
  repetition code vs no code under T1 + dephasing, with a qg_Z / qg_X
  witness that picks the option, cross-checked with a Qiskit
  density-matrix circuit (RESEARCH_NOTES §27).
- `examples/circuit_knitting_qg_cut_selection.py`: choosing which gates to cut
  with the closed-form qg cost, vs counting gates and cutting the weakest
  bonds, checked against qiskit-addon-cutting's find_cuts and end to end
  with proportional vs uniform shot allocation (RESEARCH_NOTES §26).
- `examples/thermal_states_qg_tanh.py`: thermal states with qg_Z = tanh(beta h):
  single-qubit thermodynamics in qg, the optimal-thermometer condition
  qg * artanh(qg) = 1, few-shot thermometry (plug-in vs Haar vs Jeffreys
  posteriors), a purification circuit, qubit effective temperature, and a
  6-qubit Ising ring where the qg entropy decomposition is exact without
  and breaks with a transverse field (RESEARCH_NOTES §25).
- `examples/error_mitigation_qg_vs_zne.py`: H2 under T1, dephasing,
  depolarizing, readout and fake_brisbane noise; qg symmetry filter vs
  zero-noise extrapolation vs both, with the qg witness as decision rule
  (RESEARCH_NOTES §24).
- `examples/barren_plateau_qg_local_cost.py`: barren plateaus with the global
  cost vs the qg local cost (1 - mean qg_Z)/2, finite-shot training, and a
  classical light-cone control that trains 100 qubits (RESEARCH_NOTES §23).
- `examples/ode_qg_vs_angle.py`: differentiable-quantum-circuit ODE solver
  with qg vs angle encoding and a classical Chebyshev spectral control
  (RESEARCH_NOTES §22).
- `examples/qml_multiqubit_and_shots.py`: qg vs angle encoding on two qubits,
  with two input features and with finite-shot parameter-shift training,
  plus a classical Chebyshev-regression control (RESEARCH_NOTES §19).
- `examples/ionq_validation.py`: QV noise benchmark and the qg of an RZZ
  coupling on IonQ (local Aer, IonQ noisy cloud simulator, or hardware only
  with `--yes-i-accept-qpu-cost`); key read from `IONQ_API_KEY` or a
  git-ignored `.ionq_key`. Results on the aria-1 and forte-1 noise models
  in RESEARCH_NOTES §20, including H2 chemistry with the qg witness and
  filter on trapped-ion noise (`--only h2`).
- `examples/chemistry_qg_symmetry_witness.py`: H2 (Jordan-Wigner) classical
  Hartree-Fock / FCI vs noisy quantum energies, raw, with readout
  mitigation, and with the qg electron-number witness and filter
  (RESEARCH_NOTES §21), including the H2 dissociation curve
  (`examples/data/h2_dissociation_jw.json`).
- `examples/chemistry_lih_deep_circuit.py`: 6-qubit LiH at 3.0 Angstrom,
  where the qg witness diagnoses unital noise and the qg filter does not
  help (data in `examples/data/lih_3p0_jw.json`).

## [0.3.0] - 2026-09-24

### Added
- Install straight from GitHub:
  `pip install "qang @ git+https://github.com/Viny2030/qang.git@v0.3.0"`.
- `qang.pennylane_gate`: `rqang`, `full_rqang`, `append_qang` for PennyLane,
  the third SDK after Qiskit and Cirq. Accepts a `Qang` or a raw, trainable
  qg_Z (autograd / torch / jax / tf), so gradients flow directly to qg_Z.
  Same U(theta, phi, 0) convention as Qiskit's UGate. New `[pennylane]`
  extra, also in `[all]`; pinned in `tests/test_pennylane_gate.py`.
  Requires Python >= 3.10 and PennyLane >= 0.42; on Python 3.9 the extra
  installs nothing and the PennyLane tests are skipped.
- `qang.core.qg_s_from_qg_z`: exact, branch-free identity qg_S = H((1 + qg_Z)/2).
- `qang.multiqubit.qg_correlation` (Z-basis total correlation) and the
  Miller-Madow finite-shot estimator `joint_qg_s_from_counts`.
- `theta_pole_damped` optimization space and `multi_param_gradient_descent`
  in `qang.gradients`.
- New modules: `qang.phase` (qg_Phi), `qang.circuits`, `qang.ansatze`,
  `qang.qec`, `qang.algorithms`.
- Examples and pinned tests: H2 and LiH (mixed Ry/Rx) VQE, Quantum Volume
  vs HOP / linear XEB / finite shots / T1-T2 gate-level noise, randomized
  benchmarking, ZNE, barren plateaus, QPE.

- `qang.multiqubit.mean_qg_z` / `mean_qg_z_from_counts`: the register's
  relaxation bias, a T1-aware companion to qg_S (Finding D in
  `examples/quantum_volume_qg_s_realistic_noise.py`).
- `examples/nisq_hardware_validation.py`: qg_S / mean qg_Z / HOP / XEB under
  an idle-delay T1 sweep and raw LiH energies, via Qiskit Runtime on a
  calibration-based fake backend (default) or a real IBM device
  (`--mode ibm`). New `[hardware]` extra (`qiskit-ibm-runtime`), also in
  `[all]`.
- `manuscript/`: arXiv-style preprint draft (`main.tex`, figure script).
- `qang.knitting`: sampling cost of circuit cutting in qg units,
  gamma = 1 + 2*sqrt(1 - qg^2) for RXX/RYY/RZZ/RZX and
  1 + 2*sqrt((1 - qg)/2) for controlled rotations, pinned against
  `qiskit-addon-cutting`. New `[knitting]` extra, also in `[all]`.
- `qang.gradients.qfi_qg` and `natural_gradient_step_qg`: the quantum
  natural gradient in qg coordinates, shown to equal plain descent in theta.
- `qang.statistics.bayes_qg_estimate`, `wilson_qg_estimate`,
  `delta_qg_estimate` (and SciPy-free `beta_cdf` / `beta_ppf`): few-shot
  qg_Z intervals under the Haar (uniform-in-qg_Z) prior.
- `notebooks/qang_verificado.ipynb`: executed notebook that checks every
  claim against an independent reference (RESEARCH_NOTES §15).

- `examples/qml_encoding_qg_vs_angle.py`: qg (arccos) data encoding vs angle
  encoding in a single-qubit re-uploading regressor (RESEARCH_NOTES §16);
  also added as §7 of `notebooks/qang_verificado.ipynb`.
- `examples/control_quantization_qg_vs_theta.py`: b-bit angle grids uniform
  in theta vs uniform in qg, including Grover-Rudolph distribution loading
  (RESEARCH_NOTES §17).
- `examples/noise_type_detection_qg_vs_xeb.py`: T1-vs-unital detection from
  counts, mean qg_Z vs qg_S / XEB / HOP on held-out circuits (§18).

### Changed
- `examples/lih_vqe_ry_rx_ansatz.lih_energy` builds the statevector with
  NumPy instead of simulating a circuit per call (30x faster, identical to
  1e-15; cross-checked by a new test). Full test suite: ~3 min -> ~1 min.
- CI caches pip downloads and reports the slowest tests.

### Fixed
- Leftover `quang` names in module docstrings, user-facing ImportError
  messages and the reference notebook (including its repository link).
- `README.md`: the Citation section was cut off; test count was stale (116).
- `qang/__init__.py` docstring listed only 5 of the 14 modules.
- Docstring numbers that no longer matched the scripts' output
  (pole-trapping rate, finite-shot bias growth, H2 behaviour at lr=5).
- Unused imports in `qang.algorithms`, `qang.mixed` and three examples.
- `examples/quantum_volume_qg_s.py` now warns that qg_S is not monotonic
  under amplitude damping.

### Documentation
- `paper.md` updated to the current module set, test count (633) and
  findings; package name corrected from `quang` to `qang`; references added.
- `RESEARCH_NOTES.md` Part II (§7–§14) and Appendix A (Riesz–Fréchet).
- Fixed the active-space description in `examples/lih_vqe_ry_rx_ansatz.py`
  (2 electrons in 3 spatial orbitals, not 2).

## [0.2.2] - 2026-09-20

### Added
- Downloadable `.whl` and `.tar.gz` installer files, attached automatically to
  GitHub Releases via `.github/workflows/tests.yml`-style CI, as an
  installation path independent of PyPI.

### Fixed
- Synced `qang/__init__.py`'s `__version__` to match `pyproject.toml`.

## [0.2.1] - 2026-09-20

### Known issue
- This release was published before the asset-building workflow existed, and
  GitHub's immutable-release policy blocks attaching files to it after the
  fact. It contains no installable `.whl`/`.tar.gz`. Superseded by 0.2.2.

## [0.2.0] - 2026-09-20

### Added
- Initial public release: `qg_Z` / `qg_S` core metrics, native Qiskit and Cirq
  gate integrations (`RQangGate`, `FullRQangGate`), regularized gradient
  optimization (clipped and Tikhonov), analytical shot-noise error
  propagation, and generalization to mixed states, POVMs, and multi-qubit
  registers.

### Known issue
- Same as 0.2.1: no installable assets attached to this release.
