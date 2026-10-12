"""
qang.pennylane_gate — qg as native operations for PennyLane.

Completes Future Research Direction #1 ("implement qg as a native unit/type
in quantum-computing SDKs (Qiskit, Cirq, PennyLane wrappers) ...") for
PennyLane, alongside qang.qiskit_gate and qang.cirq_gate.

Two constructors, mirroring the Qiskit and Cirq modules:

  * ``rqang(qang, wires)``            qg_Z only -> ``qml.RY(arccos(qg_Z))``.
  * ``full_rqang(qang, wires, phi)``  full (qg_Z, phi) -> ``qml.U3(theta,
                                       phi, 0)``, the same U(theta, phi, 0)
                                       convention as Qiskit's UGate and
                                       qang.cirq_gate.full_rqang_gate, so it
                                       matches Qang.to_statevector().

What PennyLane adds over the other two SDKs is automatic differentiation.
Both functions accept either a ``Qang`` or a raw, possibly trainable,
qg_Z value (a float, or an autograd / torch / jax / tensorflow tensor).
The arccos is taken with ``qml.math``, so gradients flow straight to qg_Z
and a circuit can be optimized directly in qg coordinates.

The chain rule d<Z>/d(qg_Z) = (-sin theta) * (-1/sin theta) = 1 is exact
in the interior, but each factor is singular at the poles |qg_Z| = 1
(the §4.1 singularity), so autodiff returns nan there. Keep trainable
qg_Z strictly inside (-1, 1), for instance with the pole-damped space of
qang.gradients.

This module is optional: importing ``qang`` itself never requires
PennyLane. Only importing *this* module does, and it raises a clear,
actionable error if PennyLane is not installed.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from .core import Qang

try:
    import pennylane as qml

    _PENNYLANE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when pennylane is absent
    _PENNYLANE_AVAILABLE = False


def _require_pennylane():
    if not _PENNYLANE_AVAILABLE:
        raise ImportError(
            "qang.pennylane_gate requires PennyLane. Install it with "
            "`pip install pennylane` (or `pip install \".[pennylane]\"`)."
        )


def _qg_and_phi(qang_val: Union[float, Qang, Any], phi: Optional[Any]):
    """Return (qg_Z, phi). A Qang must be polar; its phi is used unless
    ``phi`` is given explicitly. Raw values pass through untouched so that
    autodiff tensors keep their gradient tape."""
    if isinstance(qang_val, Qang):
        if qang_val.mode != "polar":
            raise ValueError("pennylane_gate functions require a polar-mode Qang (qg_Z).")
        _, qang_phi = qang_val.to_angles(unit="rad")
        return qang_val.value, (qang_phi if phi is None else phi)
    if isinstance(qang_val, (int, float)) and not -1.0 <= float(qang_val) <= 1.0:
        raise ValueError(f"qg_Z must lie in [-1, 1], got {qang_val}.")
    return qang_val, (0.0 if phi is None else phi)


if _PENNYLANE_AVAILABLE:

    def rqang(qang_val: Union[float, Qang, Any], wires) -> "qml.operation.Operation":
        """
        Apply a qg_Z-parameterized preparation on ``wires`` (inside a QNode or
        tape): P(|0>) = (1 + qg_Z) / 2, P(|1>) = (1 - qg_Z) / 2 starting from
        |0>, with no control over the relative phase. Returns the queued
        ``qml.RY`` operation.
        """
        qg, _ = _qg_and_phi(qang_val, None)
        return qml.RY(qml.math.arccos(qg), wires=wires)

    def full_rqang(
        qang_val: Union[float, Qang, Any], wires, phi: Optional[Any] = None
    ) -> "qml.operation.Operation":
        """
        Apply the full (qg_Z, phi) Bloch-sphere preparation on ``wires``: from
        |0> it prepares exactly Qang.to_statevector(). Built on
        ``qml.U3(theta, phi, 0)``, whose matrix is identical to Qiskit's
        UGate(theta, phi, 0). ``phi`` overrides a Qang's own phase and is
        required (default 0) for a raw qg_Z value; it may be trainable too.
        """
        qg, ph = _qg_and_phi(qang_val, phi)
        return qml.U3(qml.math.arccos(qg), ph, 0.0, wires=wires)

    def append_qang(qang: Union[Qang, Any], wires, full: bool = True):
        """Convenience: apply the right operation for ``qang`` on ``wires``."""
        return full_rqang(qang, wires) if full else rqang(qang, wires)

    # ------------------------------------------------ qg read-out of qml.probs (§131)

    def _masks(n: int, weight: Optional[int]):
        import numpy as np

        idx = np.arange(2 ** n)
        # PennyLane orders probs with wire 0 as the most significant bit
        z = np.stack([1.0 - 2.0 * ((idx >> (n - 1 - i)) & 1) for i in range(n)], axis=1)
        if weight is None:
            return np.ones(2 ** n), z
        w = np.array([bin(i).count("1") for i in idx])
        return (w == weight).astype(float), z

    def _n_of(probs) -> int:
        d = qml.math.shape(probs)[-1]
        n = int(d).bit_length() - 1
        if 2 ** n != d:
            raise ValueError("probs must have 2^n entries on its last axis.")
        return n

    def qg_z_from_probs(probs):
        """Per-wire qg_Z = <Z_i> from ``qml.probs()`` (wire order of the probs), differentiable."""
        _, z = _masks(_n_of(probs), None)
        return qml.math.tensordot(probs, qml.math.convert_like(z, probs), axes=[[-1], [0]])

    def kept_fraction(probs, weight: int):
        """Probability of Hamming weight ``weight`` (the qg filter's kept fraction), differentiable."""
        m, _ = _masks(_n_of(probs), weight)
        return qml.math.tensordot(probs, qml.math.convert_like(m, probs), axes=[[-1], [0]])

    def filtered_qg_z(probs, weight: int):
        """qg_Z of each wire after the qg filter (post-selection on Hamming weight ``weight``):
        <P_w Z_i> / <P_w>, differentiable through ``probs`` in every PennyLane interface.
        Use it on ``qml.probs()`` returned by a QNode; the filter is a classical post-processing,
        so it costs no extra circuit (with shots, the ratio is the post-selected estimate)."""
        m, z = _masks(_n_of(probs), weight)
        num = qml.math.tensordot(probs, qml.math.convert_like(m[:, None] * z, probs), axes=[[-1], [0]])
        return num / qml.math.expand_dims(kept_fraction(probs, weight), -1)

else:  # pragma: no cover - exercised only when pennylane is absent

    def rqang(*args, **kwargs):
        _require_pennylane()

    def full_rqang(*args, **kwargs):
        _require_pennylane()

    def append_qang(*args, **kwargs):
        _require_pennylane()

    def qg_z_from_probs(*args, **kwargs):
        _require_pennylane()

    def kept_fraction(*args, **kwargs):
        _require_pennylane()

    def filtered_qg_z(*args, **kwargs):
        _require_pennylane()
