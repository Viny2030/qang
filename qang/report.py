"""qang.report — canonical measurement report shared by every application field.

The contract is deliberately small and SDK-independent::

    report = qang_report(counts, theta_qga, n_shots, phi_qga=None) -> {
        "p0": ..., "amplitude_0": ..., "amplitude_1": ...,
        "qg_Z": ..., "m_qg": ..., "qga": ..., "theta_rad": ...,
        "phi_qga": ..., "qg_S": ..., "N": ..., "sigma_m": ..., "ci_95": ...
    }

``counts`` may be a Qiskit/Cirq/PennyLane/Braket/pytket counts dict, a result
object accepted by :func:`qang.interop.canonical_counts`, or already-canonical
``{(b_0, ..., b_{n-1}): count}`` tuples. The reading is always of qubit 0 in
that canonical order. ``theta_qga`` is the prepared polar angle in qga
(1000 qga = one turn); it fixes the pure-state shot noise
``sigma_m = 1000 sin(theta) / sqrt(N)``. ``phi_qga`` is the declared prepared
azimuth in [0, 1000) qga; it is echoed, not estimated, because Z counts cannot
see the phase. ``ci_95`` is the confidence interval
of the *estimated qga angle*, obtained by propagating the theta interval of
:func:`qang.statistics.confidence_interval_theta` to qga.

``ci_95`` is the polar-angle interval in qga; the theta-to-qga mapping uses
the Jacobian 1000/(2*pi) = 159.1549 qga/rad, so these bounds are not an
azimuthal or cone-phase interval.

No physical claim is attached to this report: it is the common reporting
format for exact, simulated and hardware readings alike.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

from .core import qg_s_from_qg_z
from .interop import canonical_counts, weight_filter
from .probability import amplitudes_from_p0, milliqang_from_p0, shot_noise_milliqang
from .qga import azimuth_qga, p0_from_qga, qga_from_p0, qga_from_radians, radians_from_qga
from .statistics import confidence_interval_theta

__all__ = ["REPORT_FIELDS", "qang_report"]

REPORT_FIELDS: Tuple[str, ...] = (
    "p0",
    "amplitude_0",
    "amplitude_1",
    "qg_Z",
    "m_qg",
    "qga",
    "theta_rad",
    "phi_qga",
    "qg_S",
    "N",
    "sigma_m",
    "ci_95",
)


def _canonical(
    counts: Any,
    sdk: Optional[str],
    n_qubits: Optional[int],
    weight: Optional[int],
) -> Dict[Tuple[int, ...], int]:
    """Normalise inputs without guessing the qubit order of multi-bit strings."""
    if isinstance(counts, Mapping) and counts:
        keys = list(counts)
        if all(isinstance(k, str) and len(k) == 1 and set(k) <= {"0", "1"} for k in keys):
            out = {(int(k),): int(v) for k, v in counts.items()}
        else:
            out = canonical_counts(counts, sdk=sdk, n_qubits=n_qubits)
    else:
        out = canonical_counts(counts, sdk=sdk, n_qubits=n_qubits)

    if any(v < 0 for v in out.values()):
        raise ValueError("counts contain a negative value.")
    if weight is not None:
        if not isinstance(weight, int) or weight < 0:
            raise ValueError("weight must be a non-negative integer.")
        out, _ = weight_filter(out, weight)
        if not out:
            raise ValueError(f"no shot has weight {weight}.")
    return out


def _declared_azimuth(phi_qga: Optional[float]) -> Optional[float]:
    """Validate a prepared azimuth and express it in the qga convention."""
    if phi_qga is None:
        return None
    if isinstance(phi_qga, bool) or not isinstance(phi_qga, (int, float)):
        raise ValueError("phi_qga must be a finite number in [0, 1000) qga.")
    phi = float(phi_qga)
    if not math.isfinite(phi) or not 0.0 <= phi < 1000.0:
        raise ValueError(f"phi_qga must lie in [0, 1000) qga, got {phi_qga}.")
    phi_rad = radians_from_qga(phi)
    canonical = azimuth_qga(math.cos(phi_rad), math.sin(phi_rad))
    if not math.isclose(canonical, phi, abs_tol=1e-9):
        raise ValueError(f"phi_qga could not be canonicalised, got {phi_qga}.")
    return phi


def qang_report(
    counts: Any,
    theta_qga: float,
    n_shots: int,
    *,
    phi_qga: Optional[float] = None,
    confidence: float = 0.95,
    sdk: Optional[str] = None,
    n_qubits: Optional[int] = None,
    weight: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the standard qang report of one Z-basis reading.

    Parameters
    ----------
    counts:
        Counts of the measured register, or an SDK result object. Strings and
        integer keys need ``sdk`` (and ``n_qubits`` for integer keys) unless a
        single bit is measured; canonical tuple keys need neither.
    theta_qga:
        Prepared polar angle in qga; one turn is 1000 qga. Values outside
        ``[0, 500]`` are folded back through ``p0_from_qga``.
    n_shots:
        Total number of shots. It must equal the sum of the retained counts.
    phi_qga:
        Declared prepared azimuth in qga, in ``[0, 1000)``; ``None`` when it is
        not declared. A Z-basis report cannot estimate the phase, so this value
        is only echoed after validation.
    confidence:
        One of the levels supported by ``confidence_interval_theta``.
    sdk, n_qubits:
        Passed to ``qang.interop.canonical_counts`` when the input needs them.
    weight:
        Optional Hamming-weight filter; when set, the report refers to the
        retained shots only.

    Returns
    -------
    dict
        Keys in the canonical order of :data:`REPORT_FIELDS`. ``theta_rad`` is
        the polar angle estimated from the counts, in radians; ``phi_qga`` is
        the declared azimuth or ``None``. ``ci_95`` is ``(low_qga, high_qga)``
        (or the corresponding level if ``confidence`` is changed).
    """
    can = _canonical(counts, sdk, n_qubits, weight)
    observed = sum(can.values())
    if observed <= 0:
        raise ValueError("counts contain no shots.")
    if not isinstance(n_shots, int) or isinstance(n_shots, bool) or n_shots <= 0:
        raise ValueError("n_shots must be a positive integer.")
    if n_shots != observed:
        raise ValueError(
            f"n_shots={n_shots} disagrees with the sum of counts={observed}."
        )

    ones = sum(c for bits, c in can.items() if bits[0] == 1)
    p0 = 1.0 - ones / n_shots
    amplitude_0, amplitude_1 = amplitudes_from_p0(p0)
    qg_z = 2.0 * p0 - 1.0
    theta_hat = 2.0 * math.acos(amplitude_0)
    theta_design = 2.0 * math.acos(math.sqrt(p0_from_qga(theta_qga)))
    lo_rad, hi_rad = confidence_interval_theta(theta_hat, n_shots, confidence)

    return {
        "p0": p0,
        "amplitude_0": amplitude_0,
        "amplitude_1": amplitude_1,
        "qg_Z": qg_z,
        "m_qg": milliqang_from_p0(p0),
        "qga": qga_from_p0(p0),
        "theta_rad": theta_hat,
        "phi_qga": _declared_azimuth(phi_qga),
        "qg_S": qg_s_from_qg_z(qg_z),
        "N": n_shots,
        "sigma_m": shot_noise_milliqang(theta_design, n_shots),
        "ci_95": (qga_from_radians(lo_rad), qga_from_radians(hi_rad)),
    }
