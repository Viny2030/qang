"""
quang.core — The qang (qg) unit, exactly as defined in Sections 2.1 and 2.2 of
the paper, plus the full (theta, phi) Bloch-sphere representation consolidated
from the author's exploratory notebook ("quang 1.ipynb").

Two operational definitions coexist on the same class, selected by ``mode``:

  * ``mode="polar"``    -> qg_Z(theta) = cos(theta) = <sigma_z>,      range [-1, +1]
  * ``mode="entropic"`` -> qg_S(theta) = H(cos^2(theta/2)),           range [0, +1]

qg_Z is bijective with theta (arccos) and therefore supports the full set of
Bloch-sphere conversions (statevector, Cartesian vector, probabilities).
qg_S is *not* bijective with theta over the full domain [0, pi] -- the paper's
Section 2.2 "Domain and Invertibility" note -- so entropic-mode instances only
support probability-level operations, plus an explicit half-domain inversion
(:meth:`Qang.theta_from_entropic`) that formalizes exactly the restriction the
paper calls for.
"""

from __future__ import annotations

from typing import Tuple, Union
import cmath
import math

MILLIQANG_PER_QANG = 1000.0


def _binary_entropy(p: float) -> float:
    """H(p) = -p*log2(p) - (1-p)*log2(1-p), with the 0*log(0) := 0 convention."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


class Qang:
    """
    A single value in the qang (qg) unit.

    Parameters
    ----------
    value : float
        qg_Z in [-1.0, 1.0] (mode="polar") or qg_S in [0.0, 1.0] (mode="entropic").
    phi : float, optional
        Azimuthal Bloch angle in radians, only meaningful in ``mode="polar"``
        (qg_S is independent of phi -- the Z-basis outcome distribution
        cos^2(theta/2), sin^2(theta/2) does not depend on the relative phase).
    mode : {"polar", "entropic"}
    """

    __slots__ = ("_value", "_phi", "mode")

    def __init__(self, value: float, phi: float = 0.0, mode: str = "polar"):
        self.mode = mode.lower()
        if self.mode == "polar":
            if not (-1.0 <= value <= 1.0):
                raise ValueError(f"qg_Z (polar) must lie in [-1.0, 1.0], got {value}.")
        elif self.mode == "entropic":
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"qg_S (entropic) must lie in [0.0, 1.0], got {value}.")
        else:
            raise ValueError("mode must be 'polar' or 'entropic'.")
        self._value = float(value)
        self._phi = float(phi) % (2.0 * math.pi)

    # ------------------------------------------------------------------ #
    # basic accessors
    # ------------------------------------------------------------------ #
    @property
    def value(self) -> float:
        return self._value

    @property
    def phi(self) -> float:
        return self._phi

    @property
    def milliqang(self) -> float:
        """value expressed in milliqang (m-qg), Section 2.3 of the paper."""
        return self._value * MILLIQANG_PER_QANG

    @classmethod
    def from_milliqang(cls, m_value: float, phi: float = 0.0, mode: str = "polar") -> "Qang":
        return cls(m_value / MILLIQANG_PER_QANG, phi=phi, mode=mode)

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_angles(cls, theta: float, phi: float = 0.0, unit: str = "rad",
                     mode: str = "polar") -> "Qang":
        """Build a Qang from the Bloch polar angle theta (and, in polar mode, phi)."""
        is_deg = unit.lower() in ("deg", "degrees", "grados")
        theta_rad = math.radians(theta) if is_deg else float(theta)
        phi_rad = math.radians(phi) if is_deg else float(phi)

        if not (0.0 <= theta_rad <= math.pi + 1e-9):
            raise ValueError(f"theta must lie in [0, pi] rad, got {theta_rad}.")

        if mode == "polar":
            return cls(math.cos(theta_rad), phi=phi_rad, mode="polar")
        elif mode == "entropic":
            p0 = math.cos(theta_rad / 2.0) ** 2
            return cls(_binary_entropy(p0), phi=0.0, mode="entropic")
        raise ValueError("mode must be 'polar' or 'entropic'.")

    @classmethod
    def from_probability(cls, p0: float, p1: Union[float, None] = None,
                          mode: str = "polar") -> "Qang":
        """Build a Qang from the computational-basis outcome probabilities."""
        if not (0.0 <= p0 <= 1.0):
            raise ValueError(f"p0 must lie in [0, 1], got {p0}.")
        if p1 is None:
            p1 = 1.0 - p0
        elif not (0.0 <= p1 <= 1.0):
            raise ValueError(f"p1 must lie in [0, 1], got {p1}.")
        elif abs((p0 + p1) - 1.0) > 1e-4:
            raise ValueError(f"p0 + p1 must equal 1, got {p0 + p1}.")

        if mode == "polar":
            return cls(p0 - p1, mode="polar")
        elif mode == "entropic":
            return cls(_binary_entropy(p0), mode="entropic")
        raise ValueError("mode must be 'polar' or 'entropic'.")

    @classmethod
    def from_bloch_vector(cls, x: float, y: float, z: float, atol: float = 1e-3) -> "Qang":
        """Build a (polar-mode) Qang from a unit Bloch vector (x, y, z)."""
        norm = math.sqrt(x * x + y * y + z * z)
        if abs(norm - 1.0) > atol:
            raise ValueError(f"(x, y, z) must be unit-norm, got norm={norm:.6f}.")
        z_clamped = max(-1.0, min(1.0, z))
        phi = math.atan2(y, x) % (2.0 * math.pi)
        return cls(z_clamped, phi=phi, mode="polar")

    @classmethod
    def from_statevector(cls, alpha: complex, beta: complex, atol: float = 1e-3) -> "Qang":
        """
        Build a (polar-mode) Qang from amplitudes |psi> = alpha|0> + beta|1>,
        after removing the global phase (alpha is rotated to be real and >= 0).
        """
        norm = abs(alpha) ** 2 + abs(beta) ** 2
        if abs(norm - 1.0) > atol:
            raise ValueError(f"|alpha|^2 + |beta|^2 must equal 1, got {norm:.6f}.")

        ref = alpha if abs(alpha) > 1e-12 else beta
        global_phase = cmath.phase(ref)
        a = alpha * cmath.exp(-1j * global_phase)
        b = beta * cmath.exp(-1j * global_phase)

        p0 = abs(a) ** 2
        p1 = abs(b) ** 2
        phi = cmath.phase(b) % (2.0 * math.pi)
        return cls(p0 - p1, phi=phi, mode="polar")

    # ------------------------------------------------------------------ #
    # conversions back out (polar mode only, per Section 2.2's caveat)
    # ------------------------------------------------------------------ #
    def _require_polar(self, what: str):
        if self.mode != "polar":
            raise NotImplementedError(
                f"{what} requires mode='polar': qg_S is not bijective with theta "
                "over [0, pi] (Section 2.2 of the paper) and carries no phase "
                "information, so it cannot be uniquely converted back to a "
                "Bloch-sphere point. Use Qang.theta_from_entropic() for the "
                "explicit half-domain inversion instead."
            )

    def to_theta(self, unit: str = "rad") -> float:
        self._require_polar("to_theta()")
        theta = math.acos(self._value)
        return math.degrees(theta) if unit.lower() in ("deg", "degrees", "grados") else theta

    def to_angles(self, unit: str = "rad") -> Tuple[float, float]:
        self._require_polar("to_angles()")
        theta = self.to_theta(unit=unit)
        phi = math.degrees(self._phi) if unit.lower() in ("deg", "degrees", "grados") else self._phi
        return theta, phi

    def to_probabilities(self) -> Tuple[float, float]:
        if self.mode == "polar":
            p0 = (1.0 + self._value) / 2.0
            p1 = (1.0 - self._value) / 2.0
            return p0, p1
        # entropic mode: qg_S = H(p) has two solutions p, 1-p (Section 2.2);
        # report the symmetric pair rather than pretending there is one answer.
        p0 = self._invert_binary_entropy(self._value, branch="lower")
        return p0, 1.0 - p0

    def to_bloch_vector(self) -> Tuple[float, float, float]:
        self._require_polar("to_bloch_vector()")
        theta = math.acos(self._value)
        x = math.sin(theta) * math.cos(self._phi)
        y = math.sin(theta) * math.sin(self._phi)
        z = self._value
        return x, y, z

    def to_statevector(self) -> Tuple[complex, complex]:
        self._require_polar("to_statevector()")
        theta = math.acos(self._value)
        alpha = complex(math.cos(theta / 2.0))
        beta = cmath.exp(1j * self._phi) * math.sin(theta / 2.0)
        return alpha, beta

    # ------------------------------------------------------------------ #
    # explicit half-domain inversion for qg_S (Section 2.2 + Future
    # Research Direction #2: "the domain over which qg_Z and qg_S are
    # each invertible")
    # ------------------------------------------------------------------ #
    @staticmethod
    def _invert_binary_entropy(s: float, branch: str = "lower", tol: float = 1e-12) -> float:
        """Solve H(p) = s for p, on the monotonic half p in [0, 0.5] ('lower')
        or p in [0.5, 1] ('upper'). Bisection: H is monotonic and continuous
        on each half, so this always converges."""
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"qg_S must lie in [0, 1], got {s}.")
        if branch == "lower":
            lo, hi = 0.0, 0.5
            f = lambda p: _binary_entropy(p) - s  # noqa: E731  (increasing on [0, 0.5])
        elif branch == "upper":
            lo, hi = 0.5, 1.0
            f = lambda p: s - _binary_entropy(p)  # noqa: E731  (increasing on [0.5, 1])
        else:
            raise ValueError("branch must be 'lower' or 'upper'.")

        flo, fhi = f(lo), f(hi)
        if flo > 0 or fhi < 0:
            # numerical edge case at s == 0 or s == 1
            return lo if abs(s) < 1e-9 else hi
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if abs(fm) < tol:
                return mid
            if fm < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @classmethod
    def theta_from_entropic(cls, qg_s: float, branch: str = "lower", unit: str = "rad") -> float:
        """
        Explicit inversion theta = H^-1(qg_S), restricted to the half-domain
        where qg_S(theta) is monotonic and therefore invertible:

          * branch="lower" -> theta in [0, pi/2]  (qg_S increasing 0 -> 1)
          * branch="upper" -> theta in [pi/2, pi]  (qg_S decreasing 1 -> 0)

        This formalizes the restriction the paper's Section 2.2 says is
        required for a well-defined inversion.
        """
        # theta in [0, pi/2] <-> p0 in [0.5, 1]; theta in [pi/2, pi] <-> p0 in [0, 0.5]
        if branch == "lower":
            p0 = cls._invert_binary_entropy(qg_s, branch="upper")  # p in [0.5, 1]
        elif branch == "upper":
            p0 = cls._invert_binary_entropy(qg_s, branch="lower")  # p in [0, 0.5]
        else:
            raise ValueError("branch must be 'lower' or 'upper'.")
        theta = 2.0 * math.acos(math.sqrt(max(0.0, min(1.0, p0))))
        return math.degrees(theta) if unit.lower() in ("deg", "degrees", "grados") else theta

    # ------------------------------------------------------------------ #
    # dunder methods
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        symbol = "qg_Z" if self.mode == "polar" else "qg_S"
        if self.mode == "polar":
            return f"Qang({self._value:+.4f} {symbol}, φ={math.degrees(self._phi):.1f}°)"
        return f"Qang({self._value:.4f} {symbol})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Qang):
            return NotImplemented
        return (
            self.mode == other.mode
            and math.isclose(self._value, other._value, abs_tol=1e-9)
            and (self.mode != "polar" or math.isclose(self._phi, other._phi, abs_tol=1e-9))
        )
