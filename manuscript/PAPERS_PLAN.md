# Three separate papers, one per experiment line

The software paper (`paper.md`, for JOSS) describes the library. The results go
into three separate papers, each with one claim, its own pre-registered
predictions and the sections of RESEARCH_NOTES that support it.

## Paper 1 — Quantum neural networks under device noise: the qg filter and the pull-free readout

* **Claim.** Reading a weight-conserving QNN through the qg filter removes most
  of the T1 damage without retraining; a bias-free readout linear in the
  excitation probabilities is immune to T1 without any filter (§123); training
  with the device's gate errors in the simulator (§124) is or is not worth it.
* **Sections.** §75–§80 (model and exact simulator), §114–§118 (multiclass,
  echo, unequal T1, dephasing + ZNE), §119–§122 (device noise models,
  calibration, seed-to-seed pull F7), §123 (pull-free readout), §124 (gate
  errors in the simulator).
* **Data.** IBM fake backends (brisbane, sherbrooke, torino); IonQ run package
  ready (§116).
* **Existing draft.** `manuscript/qml` (EN) and `manuscript/qml_es` (ES); add
  §124 when its findings are recorded.
* **Missing before submission.** At least one real-device point (IBM or IonQ).
* **Target.** Quantum Machine Intelligence, or Quantum Science and Technology.

## Paper 2 — The qang angle as a variance-stabilized coordinate for proportions

* **Claim.** The qang angle is the classical arcsine transformation of a
  binomial proportion; the same toolkit gives error bars for a measured qubit
  and for a proportion in ordinary data, with exact coverage 93–96% for
  n ≥ 30 and a known failure at np ≈ 2.5 (S3, §125).
* **Sections.** `qang.statistics` (delta method, Wilson, Haar-prior Bayes),
  §125 (`qang.proportions`, S1–S4 by exact binomial sums).
* **Data.** Exact computations; illustration with COMPR.AR 2020 (share of
  direct contracts by agency, datos.gob.ar).
* **Scope note.** The transformation is classical (Bartlett 1947, Anscombe
  1948, Freeman & Tukey 1950, Cohen 1988); the contribution is the unified
  coordinate and the measured coverage, not a new estimator.
* **Missing before submission.** A comparison with the Clopper–Pearson and
  Jeffreys intervals over a finer grid of n and p; a second real-data example.
* **Target.** A short methods note (e.g. The American Statistician's
  teacher's corner, or a quantum-education journal for the qubit side).

## Paper 3 — First real-device characterization in qg units

* **Claim.** The heralded qg sweep separates thermal excited population from
  readout error on a real qubit (C1', C4), and the Leung-code syndromes read
  damping and dephasing as a continuous witness (D1–D3).
* **Sections.** §31 and §86 (heralded characterization), §33 and §88 (Leung
  syndromes); fake-backend results already recorded.
* **Data.** Pending: the IBM Open Plan run in `docs/IBM_RUN_PLAN.md` (about two
  QPU minutes for both experiments).
* **Missing before submission.** The real-device run itself, reported with the
  device name, date and job ids, whatever the outcome of the predictions.
* **Target.** Physical Review Applied (short format) or EPJ Quantum
  Technology.

## Order

1. Paper 1 first: the draft exists and §124 completes it.
2. Run the IBM plan; it gives Paper 3 its data and Paper 1 its real-device
   point.
3. Paper 2 can be written in parallel: it needs no hardware.
