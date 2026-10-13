"""
qang.qiskit_ml — qang's weight-conserving QNN inside qiskit-machine-learning (§140).

qiskit-machine-learning builds trainable models from a ``SamplerQNN``: a
parameterized circuit, a sampler (StatevectorSampler, Aer, IBM Runtime) and an
optional pass manager for a device. This module gives it qang's pieces:

  * ``qang_sampler_qnn(model_or_n, ...)``  a SamplerQNN whose circuit is qang's
    weight-1 QNN (amplitude loader + RBS layers as native XX+YY gates, from
    qang.ibm.qnn_template); inputs are the loader angles, weights the RBS
    angles; output the full 2^n distribution (Qiskit index);
  * ``loader_inputs(X)``      features -> loader angles (the QNN's input_params);
  * ``qg_features(probs, n, weight)``  per-qubit qg_Z raw and after the qg filter
    (Hamming weight ``weight``), and the kept fraction, from SamplerQNN outputs;
  * ``QangQNNClassifier``     a trained qang.qml.WeightQNN run through a
    SamplerQNN: decisions from the raw or filtered qg_Z head, on any sampler.

Training stays classical (qang.qml.WeightQNN.fit is exact and fast); the
SamplerQNN carries the trained model to any qiskit-machine-learning sampler or
device. Its ``backward`` (parameter-shift gradients) is available for
fine-tuning on a sampler.

Requires qiskit and qiskit-machine-learning (imported on use).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["loader_inputs", "qang_sampler_qnn", "qg_features", "QangQNNClassifier"]


def loader_inputs(X) -> np.ndarray:
    """Loader angles (S, n-1) of the weight-1 amplitude encoding of features X (S, n-1): the QNN inputs."""
    from .ibm import _loader_angles

    X = np.atleast_2d(np.asarray(X, float))
    V = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return np.array([_loader_angles(v) for v in V])


def _structure(model_or_n, layers=3, sublayers=None):
    if isinstance(model_or_n, int):
        from .qml import WeightQNN

        return WeightQNN(model_or_n, 1, layers=layers, sublayers=sublayers)
    return model_or_n


def qang_sampler_qnn(model_or_n, sampler=None, pass_manager=None, layers: int = 3, sublayers=None,
                     input_gradients: bool = False):
    """SamplerQNN for qang's weight-1 QNN. Output: the 2^n outcome distribution (Qiskit integer index,
    which is also qang.qml's basis index). Returns (qnn, model_structure)."""
    from qiskit_machine_learning.neural_networks import SamplerQNN

    from .ibm import qnn_template

    m = _structure(model_or_n, layers, sublayers)
    tpl = qnn_template(m, measure=False)
    xs = [p for p in tpl.parameters if p.name.startswith("x[")]
    ws = [p for p in tpl.parameters if p.name.startswith("w[")]
    qnn = SamplerQNN(circuit=tpl, sampler=sampler, input_params=xs, weight_params=ws,
                     pass_manager=pass_manager, input_gradients=input_gradients)
    return qnn, m


def qg_features(probs, n: int, weight: int = 1):
    """From SamplerQNN outputs (S, 2^n): (raw qg_Z (S, n), filtered qg_Z (S, n), kept fraction (S,)).
    Columns follow qang qubit order (qubit i = Qiskit qubit n-1-i), like qang.qml.WeightQNN.qg_z."""
    P = np.atleast_2d(np.asarray(probs, float))
    idx = np.arange(2 ** n)
    z = np.stack([1 - 2 * ((idx >> (n - 1 - i)) & 1) for i in range(n)], axis=1)   # qang qubit i
    w = np.array([bin(i).count("1") for i in idx])
    raw = P @ z
    mask = (w == weight).astype(float)
    kept = P @ mask
    with np.errstate(invalid="ignore", divide="ignore"):
        filt = (P * mask) @ z / kept[:, None]
    return raw, filt, kept


class QangQNNClassifier:
    """A trained binary qang.qml.WeightQNN (weight 1, readout "z") executed through a SamplerQNN.

    ``decision(X, filtered=True)`` uses the model's linear head on the filtered (or raw) qg_Z read on the
    sampler; ``predict`` thresholds it at 0.
    """

    def __init__(self, model, sampler=None, pass_manager=None):
        if getattr(model, "weight", 1) != 1 or not hasattr(model, "params_"):
            raise ValueError("pass a trained weight-1 qang.qml.WeightQNN.")
        self.model = model
        self.qnn, _ = qang_sampler_qnn(model, sampler=sampler, pass_manager=pass_manager)

    def probabilities(self, X) -> np.ndarray:
        th = self.model.params_[: self.model.n_theta]
        return self.qnn.forward(loader_inputs(X), th)

    def decision(self, X, filtered: bool = True, probs: Optional[np.ndarray] = None) -> np.ndarray:
        m = self.model
        P = self.probabilities(X) if probs is None else probs
        F = m.qg_z(P, qang=bool(filtered))
        return F @ m.params_[m.n_theta:m.n_theta + m.n_head] + m.params_[-1]

    def predict(self, X, filtered: bool = True) -> np.ndarray:
        return (self.decision(X, filtered) > 0).astype(int)
