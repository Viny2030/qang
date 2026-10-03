"""
qang.polarization -- polarized light in qg units (RESEARCH_NOTES §83).

A polarization state of light is a qubit: with |H> = |0> and |V> = |1>, the
normalized Stokes parameters are exactly the qg values (the Poincare sphere
is the Bloch sphere):

    qg_Z = S1 / S0 = (I_H - I_V) / S0
    qg_X = S2 / S0 = (I_D - I_A) / S0         D, A = (H +- V) / sqrt 2
    qg_Y = S3 / S0 = (I_R - I_L) / S0         R, L = (H +- i V) / sqrt 2

Conventions for the handedness of circular light differ between texts; here R
is defined as (|H> + i|V>)/sqrt 2, so S3 > 0 means qg_Y > 0. Use
``circular_sign=-1`` for the opposite convention.

Partially polarized light is a mixed state: its degree of polarization is the
length of the qg vector, and the purity is (1 + P^2)/2. A Mueller matrix
acting on (S0, S1, S2, S3) is the Pauli transfer matrix in the order
(I, Z, X, Y); for a lossless element (a unitary Jones matrix) it is exactly
the qg gate rule qg'_P = qg_{U^dag P U} of qang.formulation. The physics is
standard optics; the module translates it into qang's units and checks it.
NumPy only.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "stokes_to_qg", "qg_to_stokes", "degree_of_polarization", "purity_from_stokes",
    "jones_to_stokes", "jones_to_qg", "mueller_from_jones", "ptm_from_mueller",
    "mueller_from_ptm", "rotator", "linear_polarizer", "wave_plate", "half_wave_plate",
    "quarter_wave_plate", "depolarizer", "malus_intensity", "qg_from_counts",
]

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)
_SIGMA = (_I, _Z, _X, _Y)  # Stokes order: S0, S1, S2, S3
_PTM_ORDER = ("I", "Z", "X", "Y")


# --------------------------------------------------------------------- #
# Stokes vectors and qg values
# --------------------------------------------------------------------- #
def stokes_to_qg(S, circular_sign: int = 1) -> dict:
    """Stokes vector (S0, S1, S2, S3) -> {'X': qg_X, 'Y': qg_Y, 'Z': qg_Z}."""
    S0, S1, S2, S3 = (float(v) for v in S)
    if S0 <= 0:
        raise ValueError("S0 (total intensity) must be positive")
    return {"X": S2 / S0, "Y": circular_sign * S3 / S0, "Z": S1 / S0}


def qg_to_stokes(qg: dict, intensity: float = 1.0, circular_sign: int = 1) -> np.ndarray:
    """{'X','Y','Z'} qg values -> Stokes vector with total intensity S0."""
    return intensity * np.array([1.0, qg.get("Z", 0.0), qg.get("X", 0.0), circular_sign * qg.get("Y", 0.0)])


def degree_of_polarization(S) -> float:
    """sqrt(S1^2 + S2^2 + S3^2) / S0 = length of the qg vector."""
    S = np.asarray(S, float)
    return float(np.linalg.norm(S[1:]) / S[0])


def purity_from_stokes(S) -> float:
    """Tr(rho^2) = (1 + P^2) / 2 for the normalized polarization state."""
    return 0.5 * (1.0 + degree_of_polarization(S) ** 2)


def jones_to_stokes(E) -> np.ndarray:
    """Jones vector (E_H, E_V) -> Stokes vector, S_mu = <E| sigma_mu |E>."""
    E = np.asarray(E, dtype=complex)
    return np.array([float(np.real(E.conj() @ s @ E)) for s in _SIGMA])


def jones_to_qg(E, circular_sign: int = 1) -> dict:
    return stokes_to_qg(jones_to_stokes(E), circular_sign)


# --------------------------------------------------------------------- #
# Mueller matrices and the qg gate rule
# --------------------------------------------------------------------- #
def mueller_from_jones(J) -> np.ndarray:
    """Mueller matrix of a (non-depolarizing) Jones matrix:
    M_mu,nu = Tr(sigma_mu J sigma_nu J^dag) / 2."""
    J = np.asarray(J, dtype=complex)
    return np.array([[float(np.real(np.trace(a @ J @ b @ J.conj().T))) / 2 for b in _SIGMA] for a in _SIGMA])


def ptm_from_mueller(M) -> dict:
    """Mueller matrix -> Pauli transfer matrix as {(P_out, P_in): value},
    P in I, Z, X, Y (the Stokes order)."""
    M = np.asarray(M, float)
    return {(_PTM_ORDER[i], _PTM_ORDER[j]): float(M[i, j]) for i in range(4) for j in range(4)}


def mueller_from_ptm(ptm: dict) -> np.ndarray:
    return np.array([[ptm.get((a, b), 0.0) for b in _PTM_ORDER] for a in _PTM_ORDER])


def rotator(theta: float) -> np.ndarray:
    """Rotation of the polarization plane by theta (Jones matrix)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def linear_polarizer(theta: float) -> np.ndarray:
    """Ideal linear polarizer with transmission axis at theta from H."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c * c, c * s], [c * s, s * s]], dtype=complex)


def wave_plate(retardance: float, theta: float) -> np.ndarray:
    """Retarder with fast axis at theta from H (global phase dropped)."""
    R = rotator(theta)
    return R @ np.diag([1.0, np.exp(1j * retardance)]) @ R.T


def half_wave_plate(theta: float) -> np.ndarray:
    return wave_plate(math.pi, theta)


def quarter_wave_plate(theta: float) -> np.ndarray:
    return wave_plate(math.pi / 2, theta)


def depolarizer(d: float) -> np.ndarray:
    """Mueller matrix diag(1, d, d, d): the depolarizing channel; d = 1 - p."""
    return np.diag([1.0, d, d, d])


def malus_intensity(qg: dict, theta: float, intensity: float = 1.0) -> float:
    """Intensity behind a linear polarizer at theta, in qg units:
    I = I0 (1 + qg_Z cos 2theta + qg_X sin 2theta) / 2 (Malus's law for any
    polarization state; cos^2 theta for horizontal light)."""
    return intensity * 0.5 * (1 + qg.get("Z", 0.0) * math.cos(2 * theta) + qg.get("X", 0.0) * math.sin(2 * theta))


# --------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------- #
def qg_from_counts(nH, nV, nD, nA, nR, nL, method=None, confidence=0.95):
    """qg values (and degree of polarization) from photon counts in the three
    analyser bases. With ``method`` ('bayes', 'wilson' or 'delta') each value
    also gets an interval from qang.statistics.qg_estimate."""
    pairs = {"Z": (nH, nV), "X": (nD, nA), "Y": (nR, nL)}
    out = {}
    for k, (a, b) in pairs.items():
        n = a + b
        if n <= 0:
            raise ValueError(f"no counts in the {k} basis")
        out[k] = (a - b) / n
    out["P"] = math.sqrt(out["X"] ** 2 + out["Y"] ** 2 + out["Z"] ** 2)
    if method is not None:
        from .statistics import qg_estimate

        out["intervals"] = {k: qg_estimate(int(a), int(a + b), method, confidence) for k, (a, b) in pairs.items()}
    return out
