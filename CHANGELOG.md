# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
