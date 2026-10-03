"""
qang.qml -- weight-conserving quantum neural networks with and without the
qg filter (RESEARCH_NOTES §75-§80).

A ``WeightQNN`` encodes data in one Hamming-weight sector of n qubits, applies
layers of RBS (Givens) rotations, which conserve the weight, and reads
z = sum_i c_i qg_Z^(i) + b. Every reading can be taken with qang (only the
shots that stayed in the input's weight sector, ``qang.sectors``) or without
it (all shots), so the effect of qang is always measurable:

  * F1/F4 (§76, §78): under equal T1 on every qubit the filtered readout is
    exactly the noiseless one, in any weight sector; the kept fraction is
    (1 - gamma)^(weight * depth) (``kept_fraction``).
  * F3 (§77): hence training under T1 with the filter gives exactly the
    parameters of noiseless training.
  * Trained on a simulator and run under T1, qang adds +2.5 points at weight
    1 and +16.5 at weight 2 (means over 60 runs, §80); trained under the
    calibrated noise, a model without qang catches up (difference 0.1
    points). Dephasing and unequal T1 are not corrected by the filter (§78,
    §80).

The simulator is exact and fast: T1 (amplitude damping, per qubit) and
dephasing keep the state block-diagonal in the Hamming weight, so one block
per weight is propagated instead of the full 2^n x 2^n density matrix.
These models are classically simulable (the weight-k sector has dimension
C(n, k)); the module is for studying robustness on noisy hardware, not a
claim of quantum advantage.

Qubit q is bit (n - 1 - q) of a basis index. NumPy only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .sectors import filter_distribution

__all__ = ["WeightQNN", "kept_fraction", "compare_qang"]


def kept_fraction(gamma: float, weight: int, depth: int) -> float:
    """Fraction of shots the filter keeps under equal T1 (§77, §78 F4)."""
    return float((1.0 - gamma) ** (weight * depth))


def _default_sublayers(n):
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    out = [even, odd]
    if n > 2:
        out.append([(n - 1, 0)])
    return [s for s in out if s]


class WeightQNN:
    """Weight-conserving QNN classifier (binary) on ``n_qubits`` qubits.

    Any weight 1 <= k <= n - 1 can be simulated (probs, qg_z, kept_fraction);
    data encodings exist for weight 1 and 2:

    weight 1: unary amplitude encoding of v = (x, 1)/|(x, 1)| (needs
              ``n_features = n_qubits - 1``); the constant keeps |x| (§76).
    weight 2: amplitude v_i v_j on the state with qubits i < j excited,
              normalized (§79).

    weight 2, encoding="ring": v_k on the state with qubits k and k+1 (mod n)
              excited: the weight-1 data on n of the C(n, 2) states (§87).
    weight 2, encoding="dual": v on the ring and u = (x^2, 1)/|(x^2, 1)| on
              the chords (k, k+2), each half with weight 1/sqrt 2; for n = 5
              ring and chords are all 10 weight-2 states (§90).

    readout "z":  z = sum_i c_i qg_Z^(i) + b (n features).
    readout "zz": also the two-qubit correlations qg_ZZ^(ij) = <Z_i Z_j>,
              i < j (n + C(n, 2) features; §81). In the weight-1 sector the
              correlations are linear in the qg_Z and add nothing; in the
              weight-2 sector they give the full sector distribution.
    """

    def __init__(self, n_qubits=5, weight=1, layers=3, sublayers=None, readout="z", encoding="pairs"):
        if not 1 <= int(weight) <= int(n_qubits) - 1:
            raise ValueError("weight must be between 1 and n_qubits - 1")
        if encoding not in ("pairs", "ring", "dual"):
            raise ValueError("encoding must be 'pairs', 'ring' or 'dual'")
        if encoding == "dual" and (weight != 2 or n_qubits < 5):
            raise ValueError("encoding 'dual' needs weight 2 and at least 5 qubits")
        self.encoding = encoding
        if readout not in ("z", "zz"):
            raise ValueError("readout must be 'z' or 'zz'")
        self.readout = readout
        self.n = int(n_qubits)
        self.weight = int(weight)
        self.layers = int(layers)
        self.sublayers = sublayers if sublayers is not None else _default_sublayers(self.n)
        self.n_theta = self.layers * sum(len(s) for s in self.sublayers)
        self.dim = 2**self.n
        idx = np.arange(self.dim)
        self.bits = (idx[:, None] >> np.arange(self.n)[::-1]) & 1
        self.zsign = 1.0 - 2.0 * self.bits
        self.wt = self.bits.sum(axis=1)
        self.idx = {k: np.where(self.wt == k)[0] for k in range(self.n + 1)}
        pos = {k: {s: i for i, s in enumerate(self.idx[k])} for k in self.idx}
        self._exc, self._jump, self._z = {}, {}, {}
        for k in range(1, self.n + 1):
            for q in range(self.n):
                b = self._bit(q)
                self._exc[k, q] = np.array([(s & b) != 0 for s in self.idx[k]])
                J = np.zeros((len(self.idx[k - 1]), len(self.idx[k])))
                for i, s in enumerate(self.idx[k]):
                    if s & b:
                        J[pos[k - 1][s ^ b], i] = 1.0
                self._jump[k, q] = J
                self._z[k, q] = self._exc[k, q].astype(float)
        if readout == "zz":
            pairs = list(itertools.combinations(range(self.n), 2))
            zz = np.stack([self.zsign[:, i] * self.zsign[:, j] for i, j in pairs], axis=1)
            self.features = np.concatenate([self.zsign, zz], axis=1)
        else:
            self.features = self.zsign
        self.n_head = self.features.shape[1]
        self.params_ = None

    # ------------------------------------------------------------------ #
    def _bit(self, q):
        return 1 << (self.n - 1 - q)

    def _rbs(self, a, b, t):
        U = np.eye(self.dim)
        c, s = math.cos(t), math.sin(t)
        ba, bb = self._bit(a), self._bit(b)
        for i in range(self.dim):
            if self.bits[i, a] == 1 and self.bits[i, b] == 0:
                j = i ^ ba ^ bb
                U[i, i], U[j, j], U[i, j], U[j, i] = c, c, -s, s
        return U

    def unitaries(self, theta):
        """One unitary per sublayer, in order."""
        out, k = [], 0
        for _ in range(self.layers):
            for pairs in self.sublayers:
                U = np.eye(self.dim)
                for a, b in pairs:
                    U = self._rbs(a, b, theta[k]) @ U
                    k += 1
                out.append(U)
        return out

    @property
    def depth(self):
        return self.layers * len(self.sublayers)

    def encode(self, X):
        if self.weight > 2:
            raise ValueError("data encodings exist for weight 1 and 2; for weight >= 3 pass states to probs()")
        X = np.atleast_2d(np.asarray(X, float))
        v = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        if v.shape[1] != self.n:
            raise ValueError(f"need {self.n - 1} features, got {X.shape[1]}")
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        psi = np.zeros((len(X), self.dim))
        if self.weight == 1:
            for i in range(self.n):
                psi[:, self._bit(i)] = v[:, i]
            return psi
        if self.encoding == "ring":
            for k in range(self.n):
                psi[:, self._bit(k) | self._bit((k + 1) % self.n)] = v[:, k]
            return psi
        if self.encoding == "dual":
            u = np.concatenate([X**2, np.ones((len(X), 1))], axis=1)
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            for k in range(self.n):
                psi[:, self._bit(k) | self._bit((k + 1) % self.n)] = v[:, k] / np.sqrt(2)
                psi[:, self._bit(k) | self._bit((k + 2) % self.n)] = u[:, k] / np.sqrt(2)
            return psi
        for i, j in itertools.combinations(range(self.n), 2):
            psi[:, self._bit(i) | self._bit(j)] = v[:, i] * v[:, j]
        return psi / np.linalg.norm(psi, axis=1, keepdims=True)

    def probs(self, theta, psi, gamma=None, dephasing=0.0):
        """Exact outcome probabilities (S, 2^n). ``gamma``: T1 damping per
        qubit after every sublayer (scalar or length-n array; None = noiseless);
        ``dephasing``: phase-flip probability per qubit after every sublayer."""
        k0 = self.weight
        S = len(psi)
        gam = None if gamma is None else np.broadcast_to(np.asarray(gamma, float), (self.n,))
        a = psi[:, self.idx[k0]]
        M = {k: np.zeros((S, len(self.idx[k]), len(self.idx[k]))) for k in range(k0 + 1)}
        M[k0] = np.einsum("si,sj->sij", a, a)
        for L in self.unitaries(theta):
            for k in range(1, k0 + 1):
                B = L[np.ix_(self.idx[k], self.idx[k])]
                M[k] = B[None] @ M[k] @ B.T[None]
            for q in range(self.n):
                g = 0.0 if gam is None else gam[q]
                if g:
                    for k in range(1, k0 + 1):
                        J = self._jump[k, q]
                        M[k - 1] = M[k - 1] + g * (J[None] @ M[k] @ J.T[None])
                        d = np.where(self._exc[k, q], math.sqrt(1 - g), 1.0)
                        M[k] = M[k] * d[None, :, None] * d[None, None, :]
                if dephasing:
                    for k in range(1, k0 + 1):
                        z = self._z[k, q]
                        f = np.where(z[:, None] == z[None, :], 1.0, 1 - 2 * dephasing)
                        M[k] = M[k] * f[None]
        out = np.zeros((S, self.dim))
        for k in range(k0 + 1):
            out[:, self.idx[k]] = np.einsum("sii->si", M[k])
        return out

    def qg_z(self, probs, qang=True):
        """Readout features: the local qg_Z of every qubit (and, with
        readout="zz", the qg_ZZ correlations); with ``qang`` only the shots in
        the input's weight sector are kept (the filter from qang.sectors).
        ``qang="both"`` is the two-channel readout (§95): the filtered
        features followed by the raw ones, so that a model trained under
        noise can also use the shots that decayed out of the sector."""
        probs = np.atleast_2d(probs)
        if isinstance(qang, str):
            if qang != "both":
                raise ValueError("qang must be True, False or 'both'")
            filt = np.array([filter_distribution(p, self.n, self.weight)[0] for p in probs])
            return np.hstack([filt @ self.features, probs @ self.features])
        if qang:
            probs = np.array([filter_distribution(p, self.n, self.weight)[0] for p in probs])
        return probs @ self.features

    # ------------------------------------------------------------------ #
    def decision(self, params, X, gamma=None, dephasing=0.0, qang=True, shots=None, seed=None):
        th = params[: self.n_theta]
        pr = self.probs(th, self.encode(X), gamma, dephasing)
        if shots:
            rng = np.random.default_rng(seed)
            pr = np.array([rng.multinomial(shots, p / p.sum()) / shots for p in pr])
        return self.qg_z(pr, qang) @ params[self.n_theta:-1] + params[-1]

    def fit(self, X, y, epochs=120, lr=0.1, gamma=None, dephasing=0.0, qang=True, seed=0, h=1e-4):
        """Adam on the cross-entropy, central-difference gradients on the
        angles. ``gamma``/``dephasing`` set noise-aware training."""
        y = np.asarray(y, float)
        psi = self.encode(X)
        rng = np.random.default_rng(seed)
        nt = self.n_theta
        nh = self.n_head * (2 if isinstance(qang, str) else 1)
        p = np.concatenate([rng.uniform(-np.pi, np.pi, nt), rng.normal(0, 0.5, nh), [0.0]])
        m = np.zeros_like(p)
        v = np.zeros_like(p)

        def feats(th):
            return self.qg_z(self.probs(th, psi, gamma, dephasing), qang)

        for t in range(1, epochs + 1):
            th, c, b = p[:nt], p[nt:nt + nh], p[-1]
            R = feats(th)
            dz = (1.0 / (1.0 + np.exp(-(R @ c + b))) - y) / len(y)
            g = np.zeros_like(p)
            g[nt:nt + nh] = dz @ R
            g[-1] = dz.sum()
            for k in range(nt):
                e = np.zeros(nt)
                e[k] = h
                g[k] = np.sum(dz * ((feats(th + e) - feats(th - e)) @ c)) / (2 * h)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            p = p - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        self.params_ = p
        return self

    def predict(self, X, params=None, **noise):
        params = self.params_ if params is None else params
        return (self.decision(params, X, **noise) > 0).astype(int)

    def score(self, X, y, params=None, **noise):
        return float(np.mean(self.predict(X, params, **noise) == np.asarray(y)))


def compare_qang(model, X_train, y_train, X_test, y_test, gamma=0.08, dephasing=0.0,
                 epochs=120, seed=0, noise_aware=False):
    """Accuracy with and without qang under the given noise, and the difference.

    noise_aware=False: one model trained without noise, evaluated under the
    noise with the filter and without it (the simulator-to-hardware case).
    noise_aware=True: two models trained under the noise, one with the
    filtered readout and one without.
    """
    noise = {"gamma": gamma, "dephasing": dephasing}
    if noise_aware:
        pq = model.fit(X_train, y_train, epochs, gamma=gamma, dephasing=dephasing, qang=True, seed=seed).params_
        pr = model.fit(X_train, y_train, epochs, gamma=gamma, dephasing=dephasing, qang=False, seed=seed).params_
    else:
        pq = pr = model.fit(X_train, y_train, epochs, seed=seed).params_
    exact = model.score(X_test, y_test, pq if not noise_aware else model.fit(X_train, y_train, epochs, seed=seed).params_)
    with_q = model.score(X_test, y_test, pq, qang=True, **noise)
    without = model.score(X_test, y_test, pr, qang=False, **noise)
    kept = kept_fraction(gamma, model.weight, model.depth) if np.ndim(gamma) == 0 else None
    return {"exact": exact, "with qang": with_q, "without qang": without,
            "difference": with_q - without, "kept fraction": kept}
