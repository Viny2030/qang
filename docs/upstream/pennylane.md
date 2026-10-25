**Title:** `qml.math.is_abstract` fails with recent JAX: `jax.core.is_concrete` no longer exists

**Labels:** bug

### Expected behavior

Differentiating a QNode with `interface="jax"` under `jax.grad` / `jax.jit` works with the current JAX release.

### Actual behavior

With PennyLane 0.45.1 and JAX 0.12.0.1 (CPU, Python 3.12, Windows), any `jax.grad` through a QNode raises

```
AttributeError: module 'jax.core' has no attribute 'is_concrete'
```

from `pennylane/math/utils.py`, in `is_abstract`:

```python
if isinstance(tensor, jax.core.Tracer):
    # If the value of the tracer is known, jax.core.is_concrete will return True.
    return not jax.core.is_concrete(tensor)
```

With JAX 0.8.3 the same code works. I did not bisect 0.9-0.11.

### Source code

```python
import jax, jax.numpy as jnp
import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, interface="jax")
def f(t):
    qml.PauliX(0)
    qml.SingleExcitation(t, wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

jax.grad(f)(jnp.array(0.4))
```

### System information

PennyLane 0.45.1, JAX 0.12.0.1 / jaxlib (CPU), NumPy 2.2.6, Python 3.12.10, Windows 11.
Workaround: `pip install "jax<0.9"`.

### Existing GitHub issues

- [ ] I have searched existing GitHub issues to make sure the issue does not already exist.
(Please check, and close this if it is already fixed in a newer PennyLane release.)
