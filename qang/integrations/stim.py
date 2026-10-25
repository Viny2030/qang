"""
qang.integrations.stim — qg readings of Stim samples (book, chapters 26 and 27).

A detector that fires with probability p reads qg = 1 - 2p: a quiet detector sits at +1000 m-qg and a detector
that fires half the time sits on the equator. The same holds for measurement records and logical observables.

  * ``detector_qg(circuit, shots)``          (qg per detector, its shot-noise error);
  * ``observable_qg(circuit, shots)``        qg of each logical observable, without decoding;
  * ``measurement_qg(circuit, shots, weight=None, qubits=None)``
                                             qg_Z per measurement record, raw or after the qg filter on the
                                             Hamming weight of the chosen records (a weight-conserving circuit);
  * ``decoded_logical_qg(circuit, shots)``   qg of each observable after PyMatching decoding (needs pymatching).
"""

from __future__ import annotations

import numpy as np

from . import require

stim = require("stim", "stim")

__all__ = ["detector_qg", "observable_qg", "measurement_qg", "decoded_logical_qg"]


def _qg(bits):
    p = np.asarray(bits, dtype=float).mean(axis=0)
    q = 1.0 - 2.0 * p
    return q, np.sqrt(np.maximum(1.0 - q * q, 0.0) / len(bits))


def detector_qg(circuit, shots: int, seed=None):
    """(qg, error) of every detector over ``shots`` samples."""
    det, _ = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return _qg(det)


def observable_qg(circuit, shots: int, seed=None):
    """(qg, error) of every logical observable, raw (no decoding)."""
    _, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return _qg(obs)


def measurement_qg(circuit, shots: int, weight=None, records=None, seed=None):
    """qg_Z of the measurement records (all, or the indices in ``records``). With ``weight``, keep only the shots
    whose chosen records have that Hamming weight. Returns (qg, error, kept fraction)."""
    m = circuit.compile_sampler(seed=seed).sample(shots).astype(np.uint8)
    if records is not None:
        m = m[:, list(records)]
    kept = 1.0
    if weight is not None:
        mask = m.sum(axis=1) == weight
        kept = float(mask.mean())
        if not mask.any():
            raise ValueError(f"no shot has weight {weight}.")
        m = m[mask]
    q, e = _qg(m)
    return q, e, kept


def decoded_logical_qg(circuit, shots: int, seed=None):
    """(qg, error) of every observable after minimum-weight matching: qg = 1 - 2 (logical error rate)."""
    pm = require("pymatching", "stim")
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    matching = pm.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    pred = matching.decode_batch(det)
    return _qg(np.asarray(pred, dtype=np.uint8) ^ obs.astype(np.uint8))
