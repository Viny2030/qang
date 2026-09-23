# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
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

### Documentation
- `paper.md` updated to the current module set, test count (610) and
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
