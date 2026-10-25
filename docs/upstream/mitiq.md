**Title:** Compose symmetry post-selection with ZNE, plus a closed-form rule for when post-selection pays

**Labels:** feature-request

### Issue description

`mitiq.rem.post_select` already supports Hamming-weight post-selection (its docstring uses that example), and the
example "Composing techniques: [[4,2,2]] Error Detection and Zero Noise Extrapolation" (`combine_422_zne`) already
composes post-selection with ZNE, using stabilizer ancillas. This proposal is about two things that example does not
cover, which we found useful for number-conserving circuits:

1. **The ancilla-free variant: post-selecting on the circuit's own conserved Hamming weight, then ZNE.** It needs no
   encoding and no extra qubits, only a circuit that conserves excitation number. Under amplitude damping (T1),
   post-selecting on the circuit's Hamming-weight sector is exact when T1 is equal across qubits, and it turns T1 plus
   dephasing into dephasing alone, which folding-based ZNE extrapolates well. ZNE on the raw counts does not, because T1
   leakage out of the sector does not scale cleanly under folding.
2. **A closed-form decision rule** for whether to post-select at a given shot budget S, from quantities already measured:
   with kept fraction K, filtered expectation z_f and raw expectation z_r (both in [-1, 1]),

   ```
   MSE_postselected ≈ (1 - z_f^2) / (K S)
   MSE_raw          ≈ (z_r - z_f)^2 + (1 - z_r^2) / S
   ```

   and post-select when the first is smaller. In our simulations the predicted and simulated MSE agreed within a factor
   1.00-1.05.

### Evidence (simulation)

Mitiq 1.1.0, Cirq 1.6.1 `DensityMatrixSimulator`, a 3-qubit weight-1 circuit (X, then three Givens rotations), amplitude
damping 0.05 and dephasing 0.03 after every moment, `RichardsonFactory([1, 3, 5])` with `fold_global`. Error of the qubit-1
expectation in units of 10^-3 (exact value -0.2573):

| noise | raw | post-selected | raw + ZNE | post-selected + ZNE |
|---|---|---|---|---|
| T1 only | 233.2 | 0.0 | 49.4 | 0.0 |
| T1 + dephasing | 257.9 | 30.3 | 55.9 | 1.1 |

### Minimal example

```python
from mitiq import zne
from mitiq.rem import post_select

def executor(circuit):                      # returns a float
    result = sample(circuit)                # a MeasurementResult
    kept = post_select(result, lambda b: sum(b) == 1)
    return expectation_z(kept, qubit=1)

zne.execute_with_zne(circuit, executor,
                     factory=zne.inference.RichardsonFactory([1, 3, 5]),
                     scale_noise=zne.scaling.fold_global)
```

A tested implementation (`qg_executor`, `execute_with_filter_zne`, `filter_rule`) is in `qang.integrations.mitiq`
(https://github.com/Viny2030/qang, MIT license). If this fits Mitiq, it could be a short example next to
`combine_422_zne` (weight post-selection instead of a code), plus the decision rule as a utility; I can open a PR for
whichever form you prefer. (The earlier symmetry-verification RFC, #1901, was closed as error detection rather than
mitigation; the decision rule above is what makes post-selection a mitigation choice with a measurable cost.)
