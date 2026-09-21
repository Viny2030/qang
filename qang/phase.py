"""
qang.phase — the deferred qg_Phi phase unit, formalized.

This addresses the "Deferred" note at the end of Section 6 of the paper
and the companion RFC addendum (GitHub issue: "Formalize qg_Phi: a
closed-form phase unit").

    qg_Phi(phi) := e^(i * 2 * pi * phi),   phi in [0, 1)

representing a fraction of a full turn. Unlike qg_Z (Section 2.1) or qg_S
(Section 2.2), this unit is invertible over its FULL domain:

    phi = arg(qg_Phi) / (2 * pi)  mod 1

with no half-domain restriction needed, because e^(i*2*pi*phi) traces out
the full unit circle bijectively as phi ranges over [0, 1) -- there is no
qg_Z-style "two theta values give the same output" ambiguity here.

This module follows the same anchor-point-first, closed-form-inversion
pattern already used for qg_Z (Table 1) and qg_S (Table 2) in qang.core.
"""

from __future__ import annotations

import cmath
import math


class QangPhi:
    """
    A single value in the qg_Phi (phase) unit.

    Parameters
    ----------
    phi : float
        Fraction of a full turn. Any real number is accepted and
        normalized to the canonical range [0, 1).
    """

    __slots__ = ("_phi",)

    def __init__(self, phi: float):
        self._phi = float(phi) % 1.0

    # ------------------------------------------------------------------ #
    # basic accessors
    # ------------------------------------------------------------------ #
    @property
    def phi(self) -> float:
        """The phase fraction, in [0, 1)."""
        return self._phi

    @property
    def value(self) -> complex:
        """qg_Phi = e^(i * 2 * pi * phi), a point on the unit circle."""
        return cmath.exp(2j * math.pi * self._phi)

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_radians(cls, angle_rad: float) -> "QangPhi":
        """Build from a raw azimuthal angle in radians: phi = angle / (2*pi)."""
        return cls(angle_rad / (2.0 * math.pi))

    @classmethod
    def from_complex(cls, z: complex, atol: float = 1e-6) -> "QangPhi":
        """
        Build from a unit-modulus complex number e^{i*theta}. This is the
        full-domain inversion: unlike Qang.from_statevector (qg_Z), no
        branch choice or half-domain restriction is needed.
        """
        modulus = abs(z)
        if abs(modulus - 1.0) > atol:
            raise ValueError(
                f"qg_Phi requires a unit-modulus complex number, got |z|={modulus:.6f}."
            )
        phi = cmath.phase(z) / (2.0 * math.pi)
        return cls(phi)

    # ------------------------------------------------------------------ #
    # inversion (exact, full domain -- Section 6 "Deferred" note contrasts
    # this directly with qg_Z / qg_S, which each need explicit domain
    # restrictions to be invertible)
    # ------------------------------------------------------------------ #
    def to_radians(self) -> float:
        """theta = 2*pi*phi, recovered exactly with no branch ambiguity."""
        return 2.0 * math.pi * self._phi

    # ------------------------------------------------------------------ #
    # dunder methods
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        v = self.value
        return f"QangPhi(phi={self._phi:.4f}, qg_Phi={v.real:+.4f}{v.imag:+.4f}j)"

    def __eq__(self, other) -> bool:
        if not isinstance(other, QangPhi):
            return NotImplemented
        return math.isclose(self._phi, other._phi, abs_tol=1e-9)


# ---------------------------------------------------------------------- #
# standard anchor points, analogous to Table 1 (qg_Z) in the paper: named
# single-qubit phase gates, each identified by the phi value that
# reproduces its phase convention.
# ---------------------------------------------------------------------- #
ANCHOR_POINTS = {
    "identity": 0.0,        # qg_Phi = 1        (I / no phase)
    "t_gate": 1.0 / 8.0,    # qg_Phi = e^{i*pi/4}   (T gate)
    "s_gate": 1.0 / 4.0,    # qg_Phi = i        (S gate)
    "z_gate": 1.0 / 2.0,    # qg_Phi = -1       (Z gate)
    "s_dagger": 3.0 / 4.0,  # qg_Phi = -i       (S-dagger gate)
}


def anchor_value(name: str) -> complex:
    """qg_Phi for one of the named standard anchor points (see ANCHOR_POINTS)."""
    if name not in ANCHOR_POINTS:
        raise ValueError(f"unknown anchor {name!r}, expected one of {list(ANCHOR_POINTS)}.")
    return QangPhi(ANCHOR_POINTS[name]).value
