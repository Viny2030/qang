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
  - qang.statistics   shot-noise error propagation theta <-> qg, and
                       few-shot qg_Z intervals (Haar-prior Bayes, Wilson).
  - qang.probability  the qang probability equation and the milli-qang:
                       every reading of one probability, with intervals (§167).
                       m-qg is the reading 1000 * qg_Z, not an angle.
  - qang.qga          the angular qang: 1000 qga per turn; conversions between
                       qga, radians, p0 and m-qg, Fisher and Berry phase in qga.
  - qang.proportions  the qang angle as the arcsine transformation of any
                       binomial proportion: intervals and Cohen's h (§125).
  - qang.clock_comparison  differential phase of two clocks from the
                       ellipse of their qg pairs (§126).
  - qang.complementarity  duality as the Bloch radius and entropic
                       uncertainty with qg_S (§127).
  - qang.ibm          IBM devices: transpiler pass, readout correction, Runtime
                       read-outs, qubit selection, the QNN in one job.
  - qang.iqm          IQM devices: readout from the poles or the error profile,
                       qg read-outs of IQMBackend counts, chain choice.
  - qang.qiskit_ml    qang's QNN as a qiskit-machine-learning SamplerQNN.
  - qang.qsp          quantum signal processing and weighted QSP in qg terms
  - qang.interop      qg read-outs (profile, filter, entropy) from Qiskit,
                       Cirq, PennyLane, Braket and pytket results, one
                       qubit order (§130).
  - qang.phase        the qg_Phi phase unit.
  - qang.transformations  closed-form transition probabilities.
  - qang.formulation  the 15 main gates and 14 main algorithms in qg language
                       (qg'_P = qg_{U^dag P U}; readout rules), NumPy only.
  - qang.qml          weight-conserving QNN classifiers with and without the
                       qg filter, exact block simulator for T1 and dephasing
                       (RESEARCH_NOTES §75-§80), NumPy only.
  - qang.polarization  polarized light in qg units: Stokes parameters are qg
                       values, Mueller matrices are the qg gate rule (§83).
  - qang.geometric    geometric (Berry/Pancharatnam) phase in qg units and
                       Stokes' theorem on the Bloch sphere (§84).
  - qang.knitting     the sampling cost of circuit cutting in qg units.
  - qang.angular      the quantum numbers n, l, m_l, m_s in qg terms: m is the
                       Hamming weight of a symmetric register (§155).
  - qang.circuits, qang.ansatze, qang.qec, qang.algorithms
                       Qiskit circuit builders, variational ansatze, the
                       3-qubit bit-flip code, and textbook algorithms
                       (optional: need qiskit).

See RESEARCH_NOTES.md for the derivations and results behind each module.
"""

from .core import MILLIQANG_PER_QANG, Qang, qg_s_from_qg_z
from .probability import (
    Reading,
    amplitudes_from_p0,
    analyze,
    angle_per_milliqang,
    dissipative_deficit,
    information_curvature,
    m_qg_from_p0,
    milliqang_from_p0,
    milliqang_from_theta,
    p0_from_milliqang,
    pole_angle,
    pulse_error_milliqang,
    qg_tau,
    quantum_snr,
    readings_from_p0,
    shot_noise_milliqang,
    shots_for_resolution,
    shots_for_sign,
    sigma_m,
    statistical_angle,
    theta_from_milliqang,
)
from .qga import (
    FISHER_PER_QGA2,
    MILLI_QGA_PER_QGA,
    QGA_PER_TURN,
    angle_sigma_qga,
    azimuth_qga,
    cone_phase_qga,
    degrees_from_qga,
    fubini_study_metric_qga,
    milliqang_from_qga,
    p0_from_qga,
    qga_from_degrees,
    qga_from_milliqang,
    qga_from_p0,
    qga_from_radians,
    radians_from_qga,
)
from .report import REPORT_FIELDS, qang_report
from .sectors import (
    SectorExposurePass,
    echo_transfer_matrix,
    filter_distribution,
    hamming_weights,
    matrix_power_stochastic,
    reweight_sectors,
    sector_exposure,
    sector_states,
    stabilizer_shot_budget,
    unmix_sector,
)

__all__ = [
    "Qang",
    "MILLIQANG_PER_QANG",
    "qg_s_from_qg_z",
    "probability",
    "qga",
    "report",
    "sectors",
    "REPORT_FIELDS",
    "qang_report",
    "Reading",
    "amplitudes_from_p0",
    "analyze",
    "angle_per_milliqang",
    "dissipative_deficit",
    "information_curvature",
    "m_qg_from_p0",
    "milliqang_from_p0",
    "milliqang_from_theta",
    "p0_from_milliqang",
    "pole_angle",
    "pulse_error_milliqang",
    "qg_tau",
    "quantum_snr",
    "readings_from_p0",
    "shot_noise_milliqang",
    "shots_for_resolution",
    "shots_for_sign",
    "sigma_m",
    "statistical_angle",
    "theta_from_milliqang",
    "FISHER_PER_QGA2",
    "MILLI_QGA_PER_QGA",
    "QGA_PER_TURN",
    "angle_sigma_qga",
    "azimuth_qga",
    "cone_phase_qga",
    "degrees_from_qga",
    "fubini_study_metric_qga",
    "milliqang_from_qga",
    "p0_from_qga",
    "qga_from_degrees",
    "qga_from_milliqang",
    "qga_from_p0",
    "qga_from_radians",
    "radians_from_qga",
    "SectorExposurePass",
    "echo_transfer_matrix",
    "filter_distribution",
    "hamming_weights",
    "matrix_power_stochastic",
    "reweight_sectors",
    "sector_exposure",
    "sector_states",
    "stabilizer_shot_budget",
    "unmix_sector",
]
__version__ = "0.17.0"
