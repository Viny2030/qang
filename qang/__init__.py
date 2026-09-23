"""
qang — A Unified Angular-Probability Unit for Parametric Quantum Circuit Design.

Reference implementation for:
    V. H. Monteverde, "The Qang (qg): A Unified Angular-Probability Unit and
    Metric for Parametric Quantum Circuit Design." ORCID: 0000-0001-8884-4811
    https://doi.org/10.5281/zenodo.22832150

This package covers, in order:
  - qang.core        the qg_Z / qg_S unit exactly as defined in the paper
                      (Section 2), plus the full Bloch-sphere (theta, phi)
                      representation consolidated from the author's
                      exploratory notebook.
  - qang.gradients    the coordinate-singularity limitation from Section 4.1,
                       a regularized alternative, and a toy VQE benchmark
                       comparing theta-space vs qg-space optimization
                       (Future Research Direction #3).
  - qang.mixed        generalization to mixed states and POVMs
                       (Future Research Direction #4, part 1).
  - qang.multiqubit   generalization to multi-qubit tensor-product
                       projection profiles (Future Research Direction #4,
                       part 2).
  - qang.qiskit_gate  qg as a native single-qubit gate for Qiskit
                       (Future Research Direction #1). Optional: only
                       importable if qiskit is installed.
  - qang.cirq_gate    the same for Cirq (optional: needs cirq).
  - qang.statistics   shot-noise error propagation theta <-> qg.
  - qang.phase        the qg_Phi phase unit.
  - qang.transformations  closed-form transition probabilities.
  - qang.circuits, qang.ansatze, qang.qec, qang.algorithms
                       Qiskit circuit builders, variational ansatze, the
                       3-qubit bit-flip code, and textbook algorithms
                       (optional: need qiskit).

See RESEARCH_NOTES.md for the derivations and results behind each module.
"""

from .core import Qang, MILLIQANG_PER_QANG

__all__ = ["Qang", "MILLIQANG_PER_QANG"]
__version__ = "0.2.2"
