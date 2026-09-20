# Contributing to qang

Thank you for your interest in contributing to `qang`! We welcome bug reports, documentation improvements, algorithmic extensions, and feature requests.

## Reporting Issues

If you encounter a bug or unexpected behavior:

1. Search the [existing issues](https://github.com/Viny2030/qang/issues) to ensure it has not already been reported.
2. If it is new, open an issue providing:
   - A clear description of the problem.
   - A minimal, reproducible code example.
   - Your Python and library versions (`qang`, `numpy`, `qiskit`, `cirq`).

## Submitting Pull Requests

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[all]"
   ```
3. Implement your changes and write accompanying tests in `tests/`.
4. Ensure all tests pass:
   ```bash
   pytest
   ```
5. Commit your changes with descriptive commit messages and submit a Pull Request against the `main` branch.
