"""
Pulse area of fractional gates in qg terms: the minimal-area rule (RESEARCH_NOTES §160).

On hardware with fractional (continuously parameterized) native gates, a rotation by theta is a pulse whose area,
and often whose duration, grows with |theta| (Rabi area theorem). Two error mechanisms then grow with it:

* amplitude miscalibration: the pulse rotates by theta (1 + eps). With eps random (Gaussian, std sigma) the qg_Z
  of RX(theta)|0> is cos(theta) exp(-theta^2 sigma^2 / 2) (``rx_qg_z_random_amplitude``); with eps fixed it is
  cos(theta (1 + eps)), whose first-order error is eps theta sin(theta);
* relaxation during a duration tau = tau_pi |theta| / pi (T1, T2).

The same gate has shorter representatives. RX(theta) = -RX(theta - 2 pi): period 2 pi up to a global phase, so
|theta| never needs to exceed pi. RZZ(theta + pi) = RZZ(theta) (-i Z x Z), and Z x Z is two virtual RZ(pi): period
pi, so |theta| never needs to exceed pi/2. ``minimal_area_angle`` returns the representative and the
``MinimalAreaRotations`` pass rewrites a circuit. Qiskit's WrapAngles enforces the hardware bounds of a target;
this rule picks the representative with the smallest area, which is the one with the least error under both
mechanisms.

The qg read-out sensitivity to an angle error is d qg_Z / d theta = -sin(theta): largest at pi/2, zero to first
order at 0 and pi, where an error enters only as cos(theta) delta^2 / 2.
"""

from __future__ import annotations

import math
from typing import Tuple

PERIOD = {"rx": 2 * math.pi, "ry": 2 * math.pi, "rzz": math.pi, "rxx": math.pi, "ryy": math.pi}


def minimal_area_angle(theta: float, period: float = 2 * math.pi) -> Tuple[float, int]:
    """(theta', k) with theta' = theta - k period in (-period/2, period/2]."""
    k = math.floor((theta + period / 2) / period)
    t = theta - k * period
    if t <= -period / 2:
        t += period
        k -= 1
    return t, k


def rx_qg_z_random_amplitude(theta: float, sigma: float) -> float:
    """E[cos(theta (1 + eps))], eps ~ N(0, sigma^2): cos(theta) exp(-theta^2 sigma^2 / 2)."""
    return math.cos(theta) * math.exp(-(theta * sigma) ** 2 / 2)


def rx_qg_z_systematic_amplitude(theta: float, eps: float) -> float:
    """qg_Z of RX(theta (1 + eps))|0>."""
    return math.cos(theta * (1 + eps))


def qg_sensitivity(theta: float) -> float:
    """d qg_Z / d theta for RX(theta)|0>: -sin(theta)."""
    return -math.sin(theta)


def minimal_area_circuit(qc):
    """Copy of ``qc`` with every rx, ry, rzz, rxx, ryy angle moved to its minimal-area representative.

    rx/ry: the 2 pi shift only changes the global phase. rzz: each pi shift multiplies by -i Z x Z, added as
    RZ(pi) on both qubits (virtual on IBM devices) and a global phase; rxx and ryy likewise with X x X and Y x Y.
    """
    from qiskit import QuantumCircuit

    out = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=qc.global_phase)
    for inst in qc.data:
        op, qargs, cargs = inst.operation, inst.qubits, inst.clbits
        name = op.name
        if name in PERIOD and not op.is_parameterized():
            t, k = minimal_area_angle(float(op.params[0]), PERIOD[name])
            new = op.__class__(t)
            out.append(new, qargs, cargs)
            if name in ("rx", "ry"):
                out.global_phase += math.pi * (k % 2)
            elif k % 2:
                # exp(-i k pi P x P / 2) = (-i P x P)^k; for odd k one P x P and phase -pi/2 (mod 2 pi per k)
                pauli = {"rzz": "z", "rxx": "x", "ryy": "y"}[name]
                for q in qargs:
                    getattr(out, pauli)(q)
                out.global_phase += -math.pi / 2 * k
            else:
                out.global_phase += -math.pi / 2 * k
        else:
            out.append(op, qargs, cargs)
    return out


def area(qc) -> float:
    """Total |angle| of the fractional gates (rx, ry, rzz, rxx, ryy): the summed pulse area in radians."""
    return float(sum(abs(float(i.operation.params[0])) for i in qc.data
                     if i.operation.name in PERIOD and not i.operation.is_parameterized()))
