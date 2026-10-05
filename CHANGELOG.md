# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `examples/qnn_full_pipeline_qg.py` (§119): calibrated training with qang (from published T1/T2) plus filter and echo
  on the IBM fake backends; K1-K4 fail: gate errors dominate, the filter alone gives 0.958 -> 0.972 and calibrated
  training adds nothing there.

## [0.6.18] - 2026-10-05

Release of §114–§118: the two noises the filter cannot remove alone are now corrected. F5: the filtered readout
depends only on the ratios of the decay rates, so training with qang under the calibrated T1s corrects unequal T1
(within 0.5 points of noiseless, insensitive to a 10% calibration error, robust to a common T1 drift at weight 2).
F6: with equal T1 the filtered readout equals the dephasing-only readout, so zero-noise extrapolation of the filtered
features removes most of the dephasing bias (with enough shots) and noise-aware training with the filter recovers the
noiseless accuracy. Also: eight classes on eight qubits, the echo calibration at 8 qubits, and a guarded IonQ QPU
package for the multiclass model (not submitted). QML Colab notebooks (EN/ES) sections 12–14 and the QML note
(EN/ES, PDF and Word) cover all of it.

### Added
- QML notebooks (EN/ES): section 14, dephasing with the filter plus zero-noise extrapolation and with noise-aware
  training with the filter (exact and 1000 shots, with and without qang); QML note (EN/ES, PDF and Word) with §118.
- `examples/qnn_dephasing_zne_qg.py` (§118): dephasing with the filter plus zero-noise extrapolation (F6: with equal T1
  the filtered readout equals the dephasing-only readout, tested); Z2, Z5 pass; Z1, Z3, Z4 fail; noise-aware training
  with the filter recovers the noiseless accuracy under dephasing.
- QML note (EN/ES, PDF and Word) updated with §114-§117: F5 (the filtered readout depends only on the ratios of
  the decay rates), unequal T1 corrected by calibrated training with the filter, eight classes on eight qubits, the
  echo at 8 qubits and the IonQ package (138 predictions, 99 pass, 39 fail).
- QML notebooks (EN/ES): section 13, unequal T1 corrected by training with qang under the calibrated rates (mean of
  three initializations, with and without qang, including a 1.5x T1 drift).
- `examples/qnn_unequal_t1_qg.py` (§117): unequal T1 corrected by training with qang under the calibrated decay rates
  (within 0.5 points of noiseless, insensitive to a 10% calibration error; the filtered readout depends only on the
  ratios of the rates, tested); U2, U3, U5 pass; U1, U4 fail.

## [0.6.17] - 2026-10-05

Release of the QNN studies §109–§116: `qang.qml.MultiClassQNN` (one class per qubit, "qubit" and "head"
readouts) and `WeightQNN.block_unitaries` (RBS layers built directly in each weight sector, 8-qubit training 13x
faster) are now on PyPI; the QML Colab notebooks (EN/ES) install `qang>=0.6.17` and show every result with and without
qang (sections 8–12); the QML note (EN/ES, PDF and Word) covers §107–§113.

### Added
- `examples/qnn_multiclass_qpu_qg.py` (§116): guarded IonQ QPU package for the multiclass QNN with filter and echo
  (plan / rehearsal / qpu with `--yes-i-accept-qpu-cost`); predictions H1-H4 pre-registered; simulator rehearsal on the
  forte-1 noise model: 8 / 4 / 1 flipped decisions raw / qang / qang + echo. Nothing submitted to hardware.
- `examples/qg_echo_8q_qg.py` (§115): echo calibration at 8 qubits, weight 2 (28 x 28 transfer matrix) on five device
  noise models; removes 32% of the filtered qg_Z error on average, against 40% at 5 qubits (X1-X5 pass).
- QML notebooks (EN/ES): section 12, eight classes on eight qubits (head readout) with the with/without-qang table.
- `examples/qnn_multiclass_8q_qg.py` (§114): multiclass QNN at 8 qubits with 3, 5 and 8 classes of digits, with and
  without qang (head readout loses 0.7 / 15.9 / 11.1 points without the filter; Q1, Q6 pass; Q2-Q5 fail).
- QML note (EN/ES, PDF and Word) updated with §107-§113: echo calibration at weight 2 and on narrow margins,
  training from shots, multiclass, 5-8 qubits, gradients and the multiclass echo (122 predictions, 89 pass, 33 fail).
- QML notebooks (EN/ES): section 11, class confusions inside the sector with the filter and the echo calibration
  (synthetic device with accuracy and flipped decisions without qang / qang / qang + echo, plus the §113 table).
- `examples/qnn_multiclass_echo_qg.py` (§113): the multiclass QNN (5-class digits) on five device noise models with
  the filter and the echo calibration of class confusions. Flipped decisions 46 / 19 / 15 (qubit readout) and
  51 / 24 / 15 (head) for raw / filter / filter + echo over 500 readings (E1, E3 pass; E2, E4 fail).
- QML notebooks (EN/ES): section 10, gradients under T1 with and without qang (variance ratio 1 with qang, K^2 at
  weight 1 without; shots to resolve the gradient).
- `examples/qnn_trainability_qg.py` (§112): gradients of weight-conserving circuits under T1 with and without qang.
  With qang the gradient is exactly the noiseless one; at weight 1 the raw one is K times smaller; the filter
  resolves gradients with fewer shots in 10 of 12 cells but not when K < 0.15 (T1, T2, T4 pass; T3, T5 fail).
- `qang.qml.WeightQNN.block_unitaries`: the RBS layers built directly in each weight sector; `probs` uses it
  (8-qubit training 13x faster, results unchanged).
- QML notebooks (EN/ES): section 9, eight qubits at weight 1 and 2 with the with/without-qang table (in the saved
  run qang adds 15-26 points at weight 1 and 46 at weight 2); the install cell needs `block_unitaries`.
- `examples/qnn_scaling_qg.py` (§111): QNNs at 5, 6 and 8 qubits, weight 1 and 2, with and without qang. With qang
  exactly noiseless at every size; without qang -3 points at weight 1 (flat in n at fixed depth) and up to -26 at
  weight 2 with 8 qubits (SC1, SC3-SC5 pass; SC2 fails for weight 1).
- `qang.qml.MultiClassQNN` (weight 1): C classes with readout "qubit" (class c is qubit c) or "head" (linear
  softmax on the qg_Z values); `logits`, `predict`, `score` with and without qang, under T1, dephasing and shots.
- `examples/qnn_multiclass_qg.py` (§110): multiclass with and without qang on iris, wine and digits 0-4. With qang
  exactly the noiseless accuracy; without qang -5.3 (qubit) and -9.1 (head) points, -15 to -18 on 5 classes
  (MC1, MC3-MC5 pass; MC2 fails narrowly).
- QML notebooks (EN/ES): section 8, multiclass on 5-class digits with the with/without-qang table; the install cell
  falls back to GitHub until the next PyPI release.
- `examples/qnn_shot_training_qg.py` (§109): the QNN trained from shots (SPSA, 100 or 1000 shots per evaluation)
  under T1, with and without qang. With qang 0.939 / 0.930 against 0.929 / 0.915 without; the filter's shot cost did
  not show in training (S3, S4 pass; S1, S2 fail). Noiseless training plus the filter at run time stays the best
  route (0.945).

## [0.6.16] - 2026-10-05

### Changed
- The start-here notebooks (`qang_start_here`, `qang_inicio_formulacion`) gain section 6, the angle in radians
  separated from the radius (`direction_from_qg`, `direction_estimate`), and section 7, the echo calibration of the
  errors the filter keeps (`echo_transfer_matrix`, `unmix_sector`) with the shot-noise check of §108 for when not to
  use it; built by `tools/add_radius_section.py`, executed, install `qang>=0.6.15`.

## [0.6.15] - 2026-10-05

### Added
- `examples/qnn_echo_narrow_margin_qg.py` (§108): the echo calibration on 120 narrow-margin inputs on five device
  noise models. The filter cuts flipped decisions from 47 to 17 (600 readings, +6.4 accuracy points); the echo adds
  nothing (19), because the filtered decision error is close to its shot-noise level (M1-M4 fail). Rule: use the echo
  only when the filtered error is well above shot noise.

## [0.6.14] - 2026-10-05

### Added
- `examples/qg_echo_weight2_qg.py` (§107): the echo calibration in the weight-2 sector (10 x 10 transfer matrix from
  10 echo circuits) on three IBM fake backends and the IonQ simulator (native); it removes 37-43% of the qg_Z error and
  39-48% of the qg_ZZ error left by the filter (W1-W4 pass).

## [0.6.13] - 2026-10-05

### Changed
- The QML note (`manuscript/qang_qml.pdf`, Spanish `manuscript/qang_qml_es.pdf`, Word versions) gains a section on
  the echo calibration of the errors the filter keeps (§104-§106): on five device noise models, decisions that differ
  from the noiseless model fall from 15 (raw) and 7 (filter) to 3 (filter + echo) over 945 readings (91 predictions:
  70 pass, 21 fail).

## [0.6.12] - 2026-10-04

### Added
- `examples/qnn_echo_calibration_qg.py` (§106): the echo calibration on the §91 classifier (189 inputs) on three
  IBM fake backends and the IonQ simulator (native). It halves the decision-value error left by the filter; decisions
  that differ from the noiseless model drop from 15 (raw) and 7 (filter) to 3 (filter + echo) over 945 readings; on
  the IBM noise models filter + echo reaches the noiseless accuracy (L1-L4 pass).

## [0.6.11] - 2026-10-04

### Added
- `qang.sectors.echo_transfer_matrix`, `unmix_sector`, `sector_states` and `matrix_power_stochastic`: echo
  calibration of the errors that move an excitation inside a weight sector (§105), for any weight; tests check
  that they reproduce the §105 example.

### Changed
- The radius note (EN/ES, PDF and Word) gains a section on the echo calibration (27 predictions, 3 failed).

## [0.6.10] - 2026-10-04

### Added
- `examples/qg_sector_echo_mitigation_qg.py` (§105): echo calibration of the errors that move an excitation inside
  the weight sector. Five echo circuits give a 5x5 transfer matrix; inverting its square root on the filtered
  distribution removes 38-47% of the remaining deficit error on three IBM fake backends and the IonQ simulator
  (native); filter plus echo is 6-7x more accurate than the raw readout (C1-C4 pass).

## [0.6.9] - 2026-10-04

### Added
- `examples/qg_radius_ionq_native_qg.py` (§104): the §100 IonQ part rerun in the native gate set. The conclusions
  hold (filter 3.2-3.3x more accurate on trained circuits; removes 17% of the false deficit of product states); the
  QIS echo rows had understated the noise on aria-1 (0.62 against 0.74) (Q1-Q4 pass).

### Changed
- The radius note (EN/ES, PDF and Word) replaces the IonQ caveat of §100 with the §104 result (23 predictions, 3
  failed).

## [0.6.8] - 2026-10-04

### Added
- `examples/qg_direction_ionq_native_qg.py` (§103): the §102 IonQ failure explained. In the QIS gate set IonQ's
  compiler removed the identity CX pairs; sent as 16 native MS gates the noise acts, and the radius-separated angle
  is 8.7-9.0x more accurate (N1-N3 pass). Circuits with identities must go to IonQ in the native gate set.

### Changed
- The radius note (`manuscript/qang_radius.pdf`, Spanish `manuscript/qang_radio_es.pdf`, Word versions) gains a
  section on the angle in radians separated from the radius (§102, §103) and the IonQ caveat for the §100 echo
  circuits (19 predictions, 3 failed).

## [0.6.7] - 2026-10-04

### Added
- `qang.formulation.direction_from_qg` (polar angle and azimuth in radians, separated from the radius) and
  `qang.statistics.direction_estimate` (the same from counts in three bases, with unbiased squares).
- `examples/qg_direction_radians_qg.py` (§102): the angle in radians from the three qg values against
  arccos(qg_Z). Exact under depolarizing noise (arccos(qg_Z) errs 0.12-0.29 rad), 29-52% less error under amplitude
  damping, worse under pure dephasing and without noise; 5-9x more accurate on the IBM noise models, worse on the IonQ
  ones where the vector barely shrank (A1-A5 pass; A6 fails).

## [0.6.6] - 2026-10-04

### Added
- `examples/qg_radius_hardware_qg.py` (§100): the radius deficit on three IBM fake backends and the IonQ simulator
  (aria-1, forte-1). The filter cuts the deficit error of trained circuits 3.2-4.0x, but removes only 15-33% of the
  false deficit of a product state, because errors that move an excitation inside the sector pass it (R1, R2, R4
  pass; R3 fails).
- `examples/qec_syndrome_dephasing_target_qg.py` (§99, post hoc): the §93 dephasing bias is the yardstick; the XXXX
  syndrome reads the product mean over the four qubits, and against it the fit agrees to 0.99-1.04.
- `examples/qnn_distorting_encoding_qg.py` (§101): distorting against faithful encoding over 60 new runs; the §81
  effect (raw ahead under noise-aware training) shrinks to +0.5 points, not significant, and with the qg_Z readout
  the filter is significantly better (D2, D4 pass; D1, D3, D5 fail).
- A short note on the local Bloch radius, `manuscript/qang_radius.pdf` (Spanish `manuscript/qang_radio_es.pdf`, Word
  versions, figure), and its Zenodo record.

### Changed
- The QML note (`manuscript/qang_qml[_es]`, PDF and Word) updated with §100 and §101: the one case where the filter
  lost (§81) does not hold up as a significant effect (79 predictions: 58 pass, 21 fail).

## [0.6.5] - 2026-10-03

### Added
- `examples/qnn_two_channel_qg.py` (§95): the two-channel readout `qang="both"` over 60 runs. With the dual encoding
  the §81 effect does not replicate (trained under T1, filter 0.953 = noiseless, raw 0.950, both 0.950); the two
  channels are safe but add nothing (B4 passes; B1-B3 fail).
- `docs/zenodo/` and `.zenodo.json`: Zenodo metadata for the nine preprints and the software
  (`tools/make_zenodo_pack.py`).

### Changed
- The QML note (`manuscript/qang_qml[_es]`, PDF and Word) updated with §94-§98: two-channel readout, shot cost of
  the filter over 513 configurations, entanglement read after the filter (70 predictions: 53 pass, 17 fail).
- Word versions of the formulation article with the local radius.

## [0.6.4] - 2026-10-03

### Added
- The local radius (sphere surface) metric in `qang.formulation` (§97): `radius2`, `radius_profile`, `radius_deficit`
  (1 - r^2, the one-tangle for pure states), `sphere_area` (4 pi r^2), `meyer_wallach`, `gate_radius_class` (preserves /
  permutes / entangling, with the maximal deficit), `hadamard_test_radius`, `weight_sector_radius` (r = |qg_Z| in a
  definite-weight state) and `algorithm_radii` (the 14 algorithms); `qang.statistics.qg2_unbiased` and
  `radius2_estimate` (unbiased qg^2 from counts).
- `examples/qg_radius_witness_qg.py` (§98): the radius deficit as an entanglement measure under T1, with and without
  qang; the filtered deficit equals the noiseless tangle to 3e-15 under equal T1, no false entanglement on product
  states (raw: 0.60-0.95), filter MSE lower in 30 of 30 configurations (E1, E3-E6 pass; E2 fails).
- `examples/qg_filter_scaling_lowk_qg.py` (§96): the filter at kept fractions down to 2.6e-7 and under unequal T1 and
  dephasing; it wins in 399 of 405 configurations and in every one with K S >= 20; the generalized rule picks the
  winner in 342 of 342 with K S >= 5 (H1, H3, H4 pass; H2 fails).
- `qang.qml`: two-channel readout `qang="both"` (filtered and raw features); study §95 pre-registered.

### Changed
- The formulation article (`manuscript/qang_formulation.pdf`, Spanish `manuscript/qang_teoria_es.pdf`): the 15 gates
  and 14 algorithms restated with the local radius (a radius column in the gates table, a radius line per algorithm,
  the radius of the readout qubits in the readout-classes table, and a section on the radius after the qg filter).
- The start-here notebooks (`qang_start_here`, `qang_inicio_formulacion`) gain section 5, the local radius, built by
  `tools/add_radius_section.py`; they install `qang>=0.6.4`.

## [0.6.3] - 2026-10-03

### Changed
- README (PyPI page): a results table for the QML line (§75-§94), the hardware scripts that are ready, the dual
  encoding in the example, and the avances notebook range (§20-§93).
- `qang.qml.WeightQNN` simulates any weight 1 <= k <= n - 1 (probs, qg_z, kept_fraction); data encodings remain for
  weights 1 and 2 (§94).

### Added
- `examples/qg_filter_scaling_qg.py` (§94): scaling of the qg filter with qubits (4-8), weight (1-4), depth (3-24) and
  shots; under equal T1 the filter has the lower readout MSE in 106 of 108 configurations (up to 1000x), a closed-form
  rule predicts the winner in 99%, break-even at 3-110 shots (G1-G4 pass).
- `examples/qnn_ionq_datasets_qg.py` (§91): the filtered QNN on the IonQ simulator with Forte-1 noise on four datasets
  (189 inputs): qang +1.1 points pooled, qg_Z error 2.7-2.9x smaller (N1-N3 pass). IonQ results now use the measured
  frequencies returned by the API (qiskit-ionq's get_counts resamples them); §85 IonQ numbers corrected.
- `examples/qec_syndrome_jointfit_qg.py` (§92): joint fit of the Leung syndromes over all delays; J1-J3 fail, and the
  diagnosis shows that routing SWAPs made the §88/§92 calibration comparison ill-posed.
- `examples/qec_syndrome_destructive_qg.py` (§93): Leung syndromes read destructively, without ancillas or SWAPs,
  on the best 4-qubit path; on three fake IBM backends the damping is read to 1-4% of the calibration (M1-M4 pass);
  replaces §88 as the hardware test of §33 (real-device run pending).
- `notebooks/qang_avances_colab.ipynb` extended to §20-§93 (sections 31-38: QML with and without qang, T1 spread,
  weight-2 encodings, device noise models, destructive syndromes, optics, geometric phase), built by
  `tools/extend_avances_notebook.py`.
- The QML note (`manuscript/qang_qml[_es]`) updated with §90 and §91 (52 predictions: 40 pass, 12 fail).

## [0.6.2] - 2026-10-03

### Changed
- The notebook builder scripts moved from `notebooks/` to `tools/` (`tools/make_qml_notebooks.py`,
  `tools/make_optics_notebooks.py`), so `notebooks/` holds only the Colab notebooks (.ipynb).
- README (the PyPI project page): formulas in plain text so they render on PyPI, a table of the library's modules,
  the optics section and Colab badges, install from GitHub without a fixed old tag. No code changes.

## [0.6.1] - 2026-10-03

### Added
- `qang.qml`: `WeightQNN(encoding="ring")` loads the weight-1 data on n weight-2 states (§87).
- `qang.qml`: `WeightQNN(encoding="dual")` fills all 10 weight-2 states of 5 qubits (v on the ring, (x^2, 1)/norm on the
  chords) (§90).
  (Version 0.6.0 on PyPI was built before these two options; the study scripts of §87 and §90 need 0.6.1.)
- `notebooks/qang_optics.ipynb` (English) and `notebooks/qang_optics_es.ipynb` (Spanish): polarized light and geometric
  phase in qg units with `qang.polarization` and `qang.geometric`, each result compared with standard optics; built by
  `notebooks/make_optics_notebooks.py`.

## [0.6.0] - 2026-10-02

### Added
- `qang.qml` (NumPy only): `WeightQNN`, a weight-conserving QNN classifier (weight 1 or 2, RBS layers, readout
  over all qg_Z) whose every reading can be taken with qang (the qg filter from `qang.sectors`) or without it;
  exact block simulator for per-qubit T1 and dephasing; noise-aware training (`fit(..., gamma, dephasing, qang)`),
  finite shots, `kept_fraction` ((1 - gamma)^(weight x depth)) and `compare_qang` (accuracy with and without qang
  and their difference). Reproduces the §77-§80 models exactly (tests/test_qml.py, checked against a full density
  matrix).
- `notebooks/qang_qml.ipynb` (English) and `notebooks/qang_qml_es.ipynb` (Spanish): QNNs with and without qang,
  installed from PyPI; built by `notebooks/make_qml_notebooks.py`.
- `examples/qnn_weight2_full_sector_qg.py` (§90, 60 runs): with the dual encoding and the qg_ZZ readout the weight-2
  model reaches the weight-1 accuracy (0.947 = 0.947) and qang adds 14-20 points under T1 (Y1, Y3-Y5 pass; Y2 fails:
  the full sector alone does not help with the qg_Z readout).
- `examples/qnn_weight2_encoding_qg.py` (§87, 60 runs): the weight-2 deficit comes from the pair-product encoding
  (its quadratic-form ceiling is 2.4 points lower; X1 fails); loaded like weight 1, weight 2 is within 1 point of
  weight 1 and qang adds +23.9 points under T1 (X2-X4 pass).
- `examples/qec_syndrome_hardware_qg.py` (§88): the §33 Leung-code syndromes as a T1/dephasing witness on IBM
  circuits; on fake backends D1-D3 pass on brisbane and torino, D1 and D3 fail on sherbrooke (extraction floor
  comparable to the signal); real-device run pending.
- `manuscript/optics/`, `manuscript/optics_es/` (PDF and Word `manuscript/qang_optics[_es]`): note on polarized light
  and geometric phase in qg units (§83-§84); five optics references in `refs.bib`.
- The QML note (`manuscript/qang_qml[_es]`) updated with §81, §82, §85 and §87 (44 predictions: 33 pass, 11 fail).
- `qang.qml`: `WeightQNN(readout="zz")` adds the two-qubit qg_ZZ correlations to the readout (§81).
- `qang.polarization` (NumPy only): Stokes parameters as qg values (the Poincare sphere is the Bloch sphere),
  degree of polarization and purity, Jones and Mueller matrices (a lossless Mueller matrix is the qg gate rule),
  wave plates, polarizers, rotators, depolarizer, Malus's law in qg units, qg values with intervals from analyser
  counts (RESEARCH_NOTES §83).
- `qang.geometric` (NumPy only): geometric (Berry/Pancharatnam) phase in qg units, gamma = -Omega/2 checked three ways
  (overlaps, solid angle, curvature flux: Stokes' theorem on the Bloch sphere); a cone loop's phase is
  (qg_Z - 1)/2 turns; results as `qang.phase.QangPhi` (RESEARCH_NOTES §84).
- `examples/qnn_zz_readout_qg.py` (§81, 60 runs): the qg_ZZ readout does not close the weight-2 gap (U2, U3 fail);
  trained under T1 with a rich readout the model without qang is 1.2 points better (U5 fails); qang adds +14.9
  points when training without noise.
- `examples/qnn_t1_spread_qg.py` (§82, 60 runs): the filter alone loses < 1 point up to a +-80% spread of 1/T1
  across qubits (V1-V3 pass; V4, V5 fail).
- `examples/qnn_hardware_qg.py` (§85): the filtered QNN compiled to Qiskit for IBM (fake or real) and IonQ
  (simulator or QPU, with explicit cost flags); on three fake IBM backends the filter cuts the qg_Z error 2.9-3.6x
  (K1-K3 pass), and on the IonQ simulator with aria-1 and forte-1 noise 3.2x and 2.9x; resumable IonQ job lists (--jobs); QPU default qpu.forte-1; real-device runs pending.
- `examples/hardware_characterization_ibm.py` (§86): the §31 heralded characterization as IBM circuits with a
  mid-circuit herald; C1-C3 pass on three fake backends; real-device run pending.
- `manuscript/qml/` and `manuscript/qml_es/`: note "Training quantum neural networks for T1-limited hardware: what
  the qg symmetry filter does and does not do" (English and Spanish; PDF and Word `manuscript/qang_qml[_es]`),
  §75-§80 with every result with and without qang; figure script `manuscript/qml/make_figure.py`; LaTeX-to-Word
  script `manuscript/to_docx.py`; two references added to `refs.bib`.

### Added (research notes)
- `examples/qnn_seeds_qg.py`: pre-registered replication of the QNN results over 5 seeds (60 runs per model), with
  and without qang and 95% CIs across seeds. All five predictions pass: trained without noise, qang adds +2.5 points
  [0.8, 4.1] at weight 1 and +16.5 [11.8, 21.2] at weight 2; with noise-aware training the difference is 0.1 points;
  the unequal-T1 loss is 0.5-0.7 points; dephasing needs noise-aware training; weight 2 costs 3.0 points
  (RESEARCH_NOTES §80). Needs scipy.
- `examples/qnn_weight2_qg.py`: pre-registered weight-2 QNN, every result with and without qang (filtered vs raw
  readout) and their difference. Trained without noise, qang adds +2 to +12 points for the weight-1 model and +13 to
  +21 for the weight-2 model (exact under equal T1 on all splits); trained under the noise, the model without qang
  catches up (differences within +-0.7 points). Weight 2 costs 3.7 points of accuracy (P3 fails) and P4 fails
  narrowly; P1, P2, P5 pass. Exact block simulator for any Hamming weight with per-qubit T1 and dephasing
  (RESEARCH_NOTES §79).
- `examples/qnn_realistic_noise_qg.py`: pre-registered test of the qg-filtered QNN under unequal T1 across qubits
  and under dephasing. The filter is exact in every fixed-weight sector under equal T1 (F4, checked to 3e-16), but
  a +-50% T1 spread costs it 1.5 points (prediction S2 fails) and dephasing 3.1 points; training under the
  calibrated noise with the filter recovers the noiseless accuracy (0.964/0.963 vs 0.965) and stays at or above
  the noise-aware standard QNN (S1, S3-S5 pass). Exact simulators with per-qubit T1 and dephasing
  (RESEARCH_NOTES §78).
- `examples/qnn_noise_aware_qg.py`: pre-registered comparison of noise-aware training and the qg filter under T1
  (gamma = 0.08). All four predictions pass: training under T1 with the filter reproduces noiseless training
  exactly (F3, parameters within 5e-9); the standard QNN collapses under T1 (0.771) unless trained under it (0.944);
  E with the filter reaches 0.949 with no noisy training, ahead of both by less than one test sample. The kept
  fraction is exactly (1 - gamma)^depth. Includes an exact block simulator for T1 on weight-1 states, about 100x
  faster than the full density matrix (RESEARCH_NOTES §77).
- `examples/qnn_unary_norm_qg.py`: weight-conserving QNN that keeps the norm (5 qubits, unary encoding with a
  constant component, trained readout over all qg_Z); mean accuracy 0.960 (standard QNN 0.952), exactly immune to
  T1 with the qg filter (proved and checked), classically simulable as a quadratic classifier (proved and checked);
  uses `qang.sectors` for the filter (RESEARCH_NOTES §76).
- `examples/qnn_classifier_qg.py`: pre-registered QNN study on four real datasets (iris, breast cancer, wine,
  digits): standard QNN, arccos encoding, register-mean qg readout and a weight-conserving QNN with the qg filter,
  against logistic regression, RBF-SVM, MLP and Chebyshev features. No quantum advantage (as predicted); arccos
  encoding gives no accuracy gain (prediction failed); the qg filter removes the T1 loss of the weight-conserving
  QNN completely (RESEARCH_NOTES §75). Needs scikit-learn (now in the `examples` and `all` extras).

### Added
- `CITATION.cff` (GitHub "Cite this repository"; validated against CFF 1.2.0) and
  `notebooks/qang_start_here.ipynb`, the starter notebook in English; README links both Colab notebooks.
- `manuscript/formulation/` (PDF and Word `manuscript/qang_formulation`): article in English, "Quantum gates and
  algorithms in qg units: Hamming-weight conservation and readout classes", with the original references of the
  14 algorithms in `refs.bib`; the Spanish draft now states that the HHL readout is checked for a 2x2 matrix.

## [0.5.1] - 2026-10-01

### Changed
- `qang.formulation`: `qg_values`, `apply_gate` and `conjugate` round floating-point residue (new helper
  `clean`), so a Bell state reads exactly `{'II': 1.0, 'XX': 1.0, 'YY': -1.0, 'ZZ': 1.0}` instead of
  0.9999999999999998.

## [0.5.0] - 2026-10-01

### Added
- `qang.formulation`: the qg formulation of the 15 main gates and 14 main algorithms as library code (NumPy only):
  `qg_values`, `state_from_qg`, `apply_gate`, `gate_table`, `conjugate`, `is_clifford`, `conserves_weight`, the gate
  matrices, and one readout function per algorithm (Deutsch-Jozsa, Bernstein-Vazirani, Simon, order finding, Grover,
  kickback/QPE, HHL, QFT, VQE energy, MaxCut, counting, quantum walk, product kernel). `tests/test_formulation.py`
  (25 checks) replaces `tests/test_qg_formulation_checks.py`; the example script now imports the module.
- `notebooks/qang_inicio_formulacion.ipynb`: starter Colab (Spanish), installs qang from PyPI and runs the 15 gates
  and 14 algorithms in qg language.
- `manuscript/teoria_es/` (PDF and Word `manuscript/qang_teoria_es`): theoretical draft, in Spanish, of the 15 main
  gates and 14 main algorithms in qg language (qg'_P = qg_{U^dag P U}); which gates conserve Hamming weight, which
  algorithms read out from local qg values and which need the full histogram.
- `examples/qg_formulation_checks.py` and `tests/test_qg_formulation_checks.py`: 21 numerical checks pinning every
  identity of that draft (gate tables; Deutsch-Jozsa, Bernstein-Vazirani, Simon, Grover, QFT, kickback, counting,
  order finding, purity, single-walker quantum walk).

### Changed
- qang 0.4.0 is on PyPI (`pip install qang`); README install section and PyPI badge updated.
- Symmetry-witness preprint now cited with its DOI (10.5281/zenodo.23036963) in the Z-basis preprint, the
  leakage note (EN/ES), the RFC v2 and the cryptography note (EN/ES); all regenerated as PDF and Word
  (Spanish cryptography note now also in Word), revised 30 September 2026.
- Z-basis preprint (`manuscript/main.tex`, PDF and Word `preprint_qang_revisado`): summary table extended with
  §68-§74 (fifty-two follow-up studies), leakage note cited, 1108 tests; Word table now keeps its section numbers.
- `notebooks/qang_avances_colab.ipynb`: sections 25-30 for §68-§74 (sector exposure and the analysis pass, S^2,
  the qutrit tests, the Bayesian flag weight, the transmon channel, erasure qubits), executed; summary updated.

## [0.4.0] - 2026-09-29

### Added
- RFC v2 items, now implemented: `qang.cirq_gate.signed_rqang_gate` (Cirq counterpart of `SignedRQangGate`),
  `qang.statistics.qg_estimate(k0, n_shots, method, confidence)` (one entry point for the Bayesian, Wilson and
  delta-method qg_Z intervals) and `qang.sectors.SectorExposurePass` (the sector-exposure screen as a Qiskit
  transpiler analysis pass, written to `property_set["sector_exposure"]`).
- `manuscript/rfc/`: RFC version 2 for a native qang module in Qiskit and Cirq (PDF `manuscript/qang_rfc_v2.pdf`,
  Word alongside): adds SignedRQangGate, the filter as an operation and the sector-exposure analysis pass;
  withdraws the qudit module and the Householder/Grover constructors on the evidence of §62, §63, §70-§72;
  keeps decoder rules out of the SDKs. Version 1 stays in `manuscript/old/RFC.pdf`.
- `examples/erasure_calibrated_qg.py`: pre-registered test of the (1 - h) prior rule for qg on erasure qubits;
  all four predictions pass: never worse, 1.41x (d = 3) and 1.68x (d = 5) at h = 0.5, fades at h = 0.99,
  robust to h misestimated by 0.1 (RESEARCH_NOTES §74). Needs pymatching.
- `examples/erasure_qubits_qg.py`: the qg T1 decoder on erasure qubits with heralding efficiency h; the gain falls
  from 1.59x/2.04x (h = 0) to nothing, and a naive qg + erasure combination is worse than erasure alone at high h
  (one of four pre-registered predictions fails); exploratory fix: decay priors scaled by (1 - h), never worse
  (RESEARCH_NOTES §73). Needs pymatching.
- `manuscript/leakage/` and `manuscript/leakage_es/`: short note on leakage flags and the qg T1-aware decoder
  (§70-§72), English and Spanish, PDFs `manuscript/qang_fuga.pdf` and `qang_fuga_es.pdf`, Word versions
  alongside; figure script `manuscript/leakage/make_figure.py`; seven leakage references in `refs.bib`.
- `examples/transmon_leakage_channel_qg.py`: coherent three-level transmon CZ (fidelity 0.99967); the derived
  leakage channel has a = 0 and a leaked qubit that keeps its value, so the |2> flag is useless and erasure
  hurts, while the qg T1 reweighting gains 1.81x (d = 3) and 2.02x (d = 5); all four pre-registered
  predictions pass (RESEARCH_NOTES §72). Needs scipy and pymatching.
- `examples/qutrit_bayes_weight_qg.py`: Bayesian weight for a |2> flag (flip probability a/(1+a) from the
  leak asymmetry a); never worse than standard and contains §70 as a limit; not uniformly better than
  erasure (one of four pre-registered predictions fails); the qutrit + qg gain grows from 1.48x (d = 3) to
  1.77x (d = 5) (RESEARCH_NOTES §71). Needs pymatching.
- `examples/qutrit_leakage_rounds_qg.py`: pre-registered second qutrit test (leak from |0> and |1>, random
  return, per-round flags); the |2> readout and the qg T1 reweighting combine super-additively, 1.68x over
  the best leakage-aware decoder (1.43x on a replication seed); the qutrit enters as a decoder input, qg
  stays a qubit quantity (RESEARCH_NOTES §70b). Needs pymatching.
- `examples/qutrit_leakage_qg.py`: the qutrit test on the §64 Z-memory with data-qubit leakage and a
  three-level final readout; negative result: the |2> flags make every decoder worse, and all of the qg
  gain (1.68x in the T1-dominated regime) is the qubit T1 reweighting; qang stays qubit-only
  (RESEARCH_NOTES §70). Needs pymatching.
- `examples/spin_squared_check_qg.py`: singlet (S^2 = 0) projection after the N and N_up/N_down checks on H2O and H4;
  a further 37-38 % of the error removed, at about 20x more Pauli strings to measure (RESEARCH_NOTES §69).
  Needs pyscf, openfermion, openfermionpyscf.
- `qang.sectors` (`filter_distribution`, `sector_exposure`, `hamming_weights`) and
  `qang.qiskit_gate.SignedRQangGate`; `examples/sector_exposure_qg.py`: a noise-free exposure number ranks
  compilations by how much T1 error the qg filter lets through (rank correlation 0.80) (RESEARCH_NOTES §68).
- `notebooks/qang_avances_colab.ipynb`: sections for §45-§67 (error propagation, RBM bounds, coherent errors,
  LiH checks on forte-1, batteries and Bell pairs, GHZ, surface code with T1, XXZ filter reach, shadows,
  Grover, signed qg, Gauss law, filter vs ZNE by shot budget), executed with outputs.
- `manuscript/crypto_es/`, `manuscript/qang_criptografia_es.pdf`: Spanish translation of the cryptography note.

### Changed
- Superseded PDFs moved to `manuscript/old/`.
- `examples/shot_budget_adaptive_zne_qg.py`: filter vs ZNE vs filter + ZNE across shot budgets; the filter wins
  for shallow circuits up to ~2000 shots, filter + ZNE beyond ~10^4 shots and for deep circuits; a pre-registered
  pilot-based switch fails (2.29x worst case) (RESEARCH_NOTES §67).
- `examples/lattice_gauge_gauss_qg.py`: Z2 lattice gauge theory with matter; local Gauss-law checks read as
  qg of parities; the Gauss filter removes more than the number filter (80 vs 64 % depolarizing, 58 vs 30 %
  T1), T1 leaks through CNOT-compiled gates (RESEARCH_NOTES §66).
- `qang.gradients.signed_theta`, `signed_qg_step`, `signed_natural_qg_step`: qg-space updates that keep the
  branch sign sign(qg_X) and reflect through the poles; they remove the arccos range limit (H2 reaches FCI
  from Hartree-Fock); `examples/signed_qg_range_qg.py` shows the natural signed step reduces to theta-space
  descent, so there is no optimization gain (RESEARCH_NOTES §65).
- `requirements.txt`, and optional-dependency groups `ionq` (qiskit-ionq), `qec` (pymatching),
  `chemistry` (pyscf, openfermion; not on native Windows) and `examples` (scipy, matplotlib);
  `all` now includes qiskit-ionq, pymatching and scipy, which examples and tests already used.
- `examples/surface_code_circuit_t1_qg.py`: circuit-level Z-memory (d = 3, 5; d rounds) with decay,
  depolarizing and readout noise; the T1-aware decoder of §59 gains 2.2x (d = 3) and 3.3x (d = 5)
  under pure T1, costs 15 % when depolarizing noise dominates, and a qg witness picks the right
  decoder (RESEARCH_NOTES §64). Needs pymatching.
- `examples/grover_noise_qg.py`: Grover search with gate noise; best iteration and cost per verified
  success; negative result: per-qubit qg signs and magnitudes lose to the outcome histogram
  (~1/P^2 vs ~1/P shots) (RESEARCH_NOTES §63).
- `examples/shadows_vs_direct_qg.py`: classical shadows vs direct measurement for the qg quantities;
  direct is 3-41x cheaper for Z-basis qg, the register-mean witness and the filter, ties for all
  local Paulis with a balanced (L18) design, 4x cheaper for the LiH energy (RESEARCH_NOTES §62).
- `examples/ionq_sim_xxz_filter.py`: the §60 chain in IonQ native gates on the forte-1 noise model;
  MS vs ZZ compilation give the same filter gain (ratio 0.82, predicted 0.8-1.25), no T1-like
  signature in the vendor model (RESEARCH_NOTES §61).
- `examples/xxz_trotter_filter_qg.py`: XXZ Trotter dynamics with the qg number filter; per-noise reach,
  half of the T1 error leaks through a 3-CNOT compilation and none through number-conserving gates;
  filter + ZNE at a finite shot budget; trapped-ion compilations: native MS rotations barely leak,
  a ZZ built by basis change does (RESEARCH_NOTES §60).
- `manuscript/battery_network/`: short note combining qubit batteries (§56) and Bell pairs (§57),
  PDF `manuscript/qang_baterias_redes.pdf`.
- `examples/surface_code_d3_qg.py`: the d = 3 rotated surface code read in qg -- exact code capacity,
  two-weight separation of data and ancilla-readout error, syndromes blind to T1 while mean qg_Z is
  not, T1-aware decoding and its robustness limit (RESEARCH_NOTES §59).
- `examples/ghz_metrology_qg.py`: GHZ vs N independent qubits under Markovian / Gaussian dephasing;
  readout cap N* = 3/(4|ln(1-2e)|); mean qg_Z as a T1 witness on a GHZ sensor (RESEARCH_NOTES §58).
- `examples/bell_pairs_network_qg.py`: Bell pairs read in qg -- noise signatures and a T1 witness, the
  DEJMPS slot rule from finite-shot data, closed-form memory cutoffs for BBM92 (RESEARCH_NOTES §57).
- `examples/battery_ergotropy_qg.py`: qubit battery ergotropy in qg; T2/T1 charging rule with closed-form
  crossover, certification from shots, locked charge of Dicke registers (RESEARCH_NOTES §56).
- `examples/spin_checks_qg.py`: spin-resolved (N_up, N_down) checks vs total-number checks on H4, H2O,
  H6; a further 12-29 % of the error at the same gate cost (RESEARCH_NOTES §55).
- `manuscript/rbm_comment/`: short comment on the RBM mutual-information bounds (§47), PDF
  `manuscript/qang_rbm_comment.pdf`; preprints updated with §47-§54.
- `examples/symmetry_checks_scaling_qg.py`: parity and N mod 4 checks on H4, H2O (8 qubits) and
  H6 (12 qubits); mod 4 reaches the number-projection ceiling, but the ceiling shrinks with depth
  (RESEARCH_NOTES §54). Needs pyscf, openfermion, openfermionpyscf.
- `examples/ionq_sim_lih_mod4.py`: LiH with parity + N mod 4 checks on IonQ's forte-1 noise model;
  error cut 3.8x, recorded as a pre-registered hardware prediction (RESEARCH_NOTES §53).
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
