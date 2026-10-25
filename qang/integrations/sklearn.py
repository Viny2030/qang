"""
qang.integrations.sklearn — WeightQNN and MultiClassQNN as scikit-learn classifiers (book, chapter 20).

``WeightQNNClassifier`` and ``MultiClassQNNClassifier`` follow the estimator API (``fit``, ``predict``,
``decision_function``, ``score``, ``get_params`` / ``set_params``), so they work in ``Pipeline``,
``cross_val_score`` and ``GridSearchCV`` like any classifier. Labels may be any two (or C) values; they are mapped
to 0..C-1 internally and back on ``predict``.

The quantum model is qang.qml's exact simulator: ``qang=True`` reads with the qg filter (post-selection on the
input's Hamming weight), ``qang=False`` reads all shots, and ``gamma`` / ``dephasing`` set the noise (chapter 20,
§75-§80). With weight 1 the unary encoding needs ``n_features = n_qubits - 1``; ``n_qubits=None`` takes it from
the data.
"""

from __future__ import annotations

import numpy as np

from . import require

_skl_base = require("sklearn.base", "sklearn")
BaseEstimator, ClassifierMixin = _skl_base.BaseEstimator, _skl_base.ClassifierMixin

from ..qml import MultiClassQNN, WeightQNN  # noqa: E402

__all__ = ["WeightQNNClassifier", "MultiClassQNNClassifier"]


class _QNNBase(ClassifierMixin, BaseEstimator):
    def _n_qubits(self, X):
        if self.n_qubits is not None:
            return int(self.n_qubits)
        return X.shape[1] + 1

    def _noise(self):
        return {"gamma": self.gamma, "dephasing": self.dephasing, "qang": self.qang}

    def _encode_labels(self, y):
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        return y_idx


class WeightQNNClassifier(_QNNBase):
    """Binary weight-conserving QNN classifier (qang.qml.WeightQNN) with the scikit-learn API."""

    def __init__(self, n_qubits=None, weight=1, layers=3, readout="z", encoding="pairs", epochs=120, lr=0.1,
                 gamma=None, dephasing=0.0, qang=True, seed=0):
        self.n_qubits = n_qubits
        self.weight = weight
        self.layers = layers
        self.readout = readout
        self.encoding = encoding
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.dephasing = dephasing
        self.qang = qang
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_idx = self._encode_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("WeightQNNClassifier is binary; use MultiClassQNNClassifier for more classes.")
        self.model_ = WeightQNN(n_qubits=self._n_qubits(X), weight=self.weight, layers=self.layers,
                                readout=self.readout, encoding=self.encoding)
        self.model_.fit(X, y_idx, epochs=self.epochs, lr=self.lr, seed=self.seed, **self._noise())
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        return self.model_.decision(self.model_.params_, np.asarray(X, dtype=float), **self._noise())

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class MultiClassQNNClassifier(_QNNBase):
    """Multi-class weight-conserving QNN classifier (qang.qml.MultiClassQNN) with the scikit-learn API."""

    def __init__(self, n_qubits=None, layers=3, readout="qubit", encoding="pairs", epochs=120, lr=0.1,
                 gamma=None, dephasing=0.0, qang=True, seed=0):
        self.n_qubits = n_qubits
        self.layers = layers
        self.readout = readout
        self.encoding = encoding
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.dephasing = dephasing
        self.qang = qang
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_idx = self._encode_labels(y)
        self.model_ = MultiClassQNN(n_qubits=self._n_qubits(X), n_classes=len(self.classes_), layers=self.layers,
                                    readout=self.readout, encoding=self.encoding)
        self.model_.fit(X, y_idx, epochs=self.epochs, lr=self.lr, seed=self.seed, **self._noise())
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        return self.model_.logits(self.model_.params_, np.asarray(X, dtype=float), **self._noise())

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
