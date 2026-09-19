"""
quang.qiskit_gate — qg as a native single-qubit gate for Qiskit.

This addresses Future Research Direction #1: "implement qg as a native
unit/type in quantum-computing SDKs (Qiskit, Cirq, PennyLane wrappers), so
that circuit parameters ... can be constructed, converted, and displayed
natively in qg / m-qg rather than requiring manual trigonometric conversion
at each call site."

Two gates, consolidated from the author's exploratory notebook:

  * ``RQangGate``      single-parameter, qg_Z only (decomposes to RY(theta)).
  * ``FullRQangGate``  full (qg_Z, phi) Bloch-sphere gate (decomposes to the
                        native U(theta, phi, 0) gate), i.e. the qubit
                        analogue of quang.core.Qang.to_statevector().

This module is optional: importing ``quang`` itself never requires Qiskit.
Only importing *this* module does, and it raises a clear, actionable error
if Qiskit is not installed rather than failing on an obscure import line.
"""

from __future__ import annotations

from typing import Union

from .core import Qang

try:
    from qiskit.circuit import Gate, QuantumCircuit
    from qiskit.circuit.library import RYGate, UGate

    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when qiskit is absent
    _QISKIT_AVAILABLE = False


def _require_qiskit():
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "quang.qiskit_gate requires Qiskit. Install it with "
            "`pip install qiskit` (and, to run circuits, `pip install qiskit-aer`)."
        )


if _QISKIT_AVAILABLE:

    class RQangGate(Gate):
        """A single-qubit gate parameterized directly by qg_Z (polar qang).

        Prepares P(|0>) = (1 + qg_Z) / 2, P(|1>) = (1 - qg_Z) / 2 starting
        from |0>, with no control over the relative phase (equivalent to
        RY(theta) with theta = arccos(qg_Z)).
        """

        def __init__(self, qang_val: Union[float, Qang], label: str = "RQang"):
            qg = qang_val if isinstance(qang_val, Qang) else Qang(qang_val, mode="polar")
            if qg.mode != "polar":
                raise ValueError("RQangGate requires a polar-mode Qang (qg_Z).")
            self._theta = qg.to_theta()
            super().__init__(name="rqang", num_qubits=1, params=[qg.value], label=label)

        def _define(self):
            qc = QuantumCircuit(1, name=self.name)
            qc.append(RYGate(self._theta), [0])
            self.definition = qc

    class FullRQangGate(Gate):
        """
        A single-qubit gate parameterized by the full (qg_Z, phi) Bloch-sphere
        Qang, i.e. it prepares exactly the state
        quang.core.Qang.to_statevector() describes, starting from |0>.
        """

        def __init__(self, qang: Qang, label: str = "RQang3D"):
            if qang.mode != "polar":
                raise ValueError("FullRQangGate requires a polar-mode Qang (qg_Z, phi).")
            self.qang = qang
            theta, phi = qang.to_angles(unit="rad")
            self._theta = theta
            self._phi = phi
            super().__init__(name="rqang_3d", num_qubits=1, params=[qang.value, phi], label=label)

        def _define(self):
            qc = QuantumCircuit(1, name=self.name)
            # U(theta, phi, lambda=0)|0> = cos(theta/2)|0> + e^{i*phi} sin(theta/2)|1>
            qc.append(UGate(self._theta, self._phi, 0.0), [0])
            self.definition = qc

    def append_qang(qc: "QuantumCircuit", qang: Qang, qubit: int, full: bool = True) -> "QuantumCircuit":
        """Convenience: append the right gate for ``qang`` to ``qubit`` of ``qc``."""
        gate = FullRQangGate(qang) if full else RQangGate(qang)
        qc.append(gate, [qubit])
        return qc

else:  # pragma: no cover - exercised only when qiskit is absent

    class RQangGate:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_qiskit()

    class FullRQangGate:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_qiskit()

    def append_qang(*args, **kwargs):
        _require_qiskit()
