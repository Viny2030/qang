"""
qang.integrations.torch — a weight-conserving QNN as a PyTorch module, through PennyLane (book, chapter 20).

``WeightQNNTorch`` is a ``torch.nn.Module``: a PennyLane ``TorchLayer`` that loads the data in the weight-1 sector
(unary amplitude encoding of (x, 1)/|(x, 1)|, as qang.qml), applies layers of ``qml.SingleExcitation`` (Givens /
RBS rotations, which conserve the weight) and returns ``qml.probs``; the module then reads qg_Z with the qg filter
(``qang.pennylane_gate.filtered_qg_z``) or without it, and a linear head gives the logit. Everything is
differentiable with ``torch.autograd``, so the network trains next to classical layers.

``t1`` adds equal amplitude damping after every layer (``default.mixed``). Under equal T1 the filtered readout is
the noiseless one (chapter 21, F1); the raw readout is not.
"""

from __future__ import annotations


from . import require

torch = require("torch", "torch")
qml = require("pennylane", "torch")

from ..pennylane_gate import filtered_qg_z, qg_z_from_probs  # noqa: E402

__all__ = ["WeightQNNTorch", "unary_state"]


def unary_state(x):
    """Batch of weight-1 amplitude states for rows of ``x``: amplitude v_q on the state with only qubit q excited."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.ndim == 1:
        x = x[None, :]
    v = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
    v = v / torch.linalg.norm(v, dim=1, keepdim=True)
    n = v.shape[1]
    psi = torch.zeros(x.shape[0], 2**n, dtype=torch.complex128)
    for q in range(n):
        psi[:, 1 << (n - 1 - q)] = v[:, q].to(torch.complex128)
    return psi


class WeightQNNTorch(torch.nn.Module):
    """Weight-1 QNN with ``n_features + 1`` qubits; ``qang`` chooses the filtered (True) or raw (False) readout."""

    def __init__(self, n_features: int, layers: int = 3, qang: bool = True, t1: float = 0.0):
        super().__init__()
        self.n = n = int(n_features) + 1
        self.qang = qang
        pairs = [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]
        dev = qml.device("default.mixed" if t1 else "default.qubit", wires=n)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.StatePrep(unary_state(inputs).squeeze(0), wires=range(n))
            k = 0
            for _ in range(layers):
                for a, b in pairs:
                    qml.SingleExcitation(weights[k], wires=[a, b])
                    k += 1
                if t1:
                    for w in range(n):
                        qml.AmplitudeDamping(t1, wires=w)
            return qml.probs(wires=range(n))

        self.circuit = circuit
        self.n_angles = layers * len(pairs)
        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": self.n_angles})
        self.head = torch.nn.Linear(n, 1, dtype=torch.float64)

    def readout(self, x):
        """qg_Z of every qubit, filtered on weight 1 or raw."""
        probs = torch.stack([self.qlayer(row) for row in torch.as_tensor(x, dtype=torch.float64)])
        return filtered_qg_z(probs, 1) if self.qang else qg_z_from_probs(probs)

    def forward(self, x):
        return self.head(self.readout(x)).squeeze(-1)
