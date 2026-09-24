# Research notes: extending the Qang beyond the paper

These notes document the extensions built on top of the paper's Section 6
("Future Research Directions") roadmap, with derivations and concrete
results, so they can be folded into a future revision of the paper or a
companion technical report. All four directions listed in Section 6 are now
covered: gradient regularization (§1), error-propagation bounds (§5),
mixed states / POVMs (§3), multi-qubit generalization (§4), and native-SDK
integration for Qiskit, Cirq and PennyLane (§6).

Part II (§7–§13) goes beyond the original roadmap: new exact identities
(§7), the structural blind spots of Z-basis metrics (§8), a pole-damped
optimizer validated on H2 and LiH (§9), qg_S as a benchmarking signal
compared against Heavy Output Probability, linear XEB and realistic T1/T2
noise (§10), qg_Z reformulations of randomized benchmarking and
zero-noise extrapolation (§11), barren plateaus (§12), and qg_Phi / QPE /
QEC (§13). Appendix A records the functional-analysis foundation
(Riesz–Fréchet) for the paper's conceptual section. §14 lists every known
limitation in one place. §15 adds three results checked against
independent references: the natural gradient in qg coordinates, the
cost of circuit cutting in qg units, and few-shot estimation under the
Haar prior. §16 compares qg data encoding with angle encoding in
quantum machine learning. §17 and §18 compare qg against standard
practice in finite-precision control and in identifying the type of
noise. §19 takes the QML result of §16 to two qubits, two input
features and finite-shot training.

Every number quoted below is produced by a script in `examples/` and is
pinned by a regression test in `tests/` (723 tests at the time of
writing); re-running the named script reproduces it.

| § | Topic | Code | Tests |
|---|---|---|---|
| 7 | qg_Z ↔ qg_S identity, `qg_correlation` | `qang.core`, `qang.multiqubit` | `test_core.py`, `test_multiqubit.py` |
| 8 | Blind spots (graph states, arccos range) | `qang.circuits`, `examples/vqe_h2_qg_vs_theta.py` | `test_circuits.py`, `test_vqe_h2.py` |
| 9 | Pole-damped gradient descent | `qang.gradients`, 5 examples | `test_gradients.py` + 5 example tests |
| 10 | qg_S vs HOP / XEB / finite shots / T1–T2, `mean_qg_z`, device-calibrated run | `examples/quantum_volume_qg_s*.py`, `examples/nisq_hardware_validation.py` | `test_quantum_volume_qg_s*.py`, `test_nisq_hardware_validation.py` |
| 11 | RB and ZNE | `examples/randomized_benchmarking_qg_z.py`, `examples/zne_qg_vs_theta_space.py` | matching tests |
| 12 | Barren plateaus | `qang.ansatze`, `examples/barren_plateaus_qg_vs_theta.py` | `test_barren_plateaus_qg_vs_theta.py` |
| 13 | qg_Phi, QPE, QEC, algorithms | `qang.phase`, `qang.qec`, `qang.algorithms` | `test_phase.py`, `test_qec*.py`, `test_algorithms.py`, `test_quantum_phase_estimation_qg_phi.py` |
| 15 | Natural gradient in qg, cutting cost, Haar-prior estimation | `qang.gradients`, `qang.knitting`, `qang.statistics`, `notebooks/qang_verificado.ipynb` | `test_gradients.py`, `test_knitting.py`, `test_statistics.py` |
| 16 | QML data encoding: qg (arccos) vs angle | `examples/qml_encoding_qg_vs_angle.py` | `test_qml_encoding_qg_vs_angle.py` |
| 17 | Finite-precision control: θ grid vs qg grid, distribution loading | `examples/control_quantization_qg_vs_theta.py` | `test_control_quantization_qg_vs_theta.py` |
| 18 | Noise-type detection: mean qg_Z vs XEB / HOP | `examples/noise_type_detection_qg_vs_xeb.py` | `test_noise_type_detection_qg_vs_xeb.py` |
| 19 | QML beyond one qubit: 2 qubits, 2D input, shot-noise training, classical control | `examples/qml_multiqubit_and_shots.py` | `test_qml_multiqubit_and_shots.py` |

## 1. Regularizing the Section 4.1 gradient singularity

The paper defines the qg-space gradient as

```
dE/d(qg) = (dE/d(theta)) * (d(theta)/d(qg)) = -(1/sin(theta)) * (dE/d(theta))
```

and states, correctly, that this diverges at theta = 0, pi. Two bounded
alternatives to the exact inverse Jacobian `-1/sin(theta)`:

* **Clipped**: `-1 / sign(sin(theta)) * max(|sin(theta)|, eps)`. Bounded by
  `1/eps` in magnitude; identical to the exact value away from the poles
  (error `-> 0` as `theta` moves away from `{0, pi}` for any fixed `eps`).
* **Tikhonov**: `-sin(theta) / (sin(theta)^2 + eps^2)`. Smooth everywhere,
  including exactly at `theta = 0, pi` (where it evaluates to 0, rather than
  saturating at `1/eps`) — it *suppresses* the update at a pole instead of
  clamping it to a large-but-finite value.

Both are implemented in `qang.gradients` alongside the exact
`inverse_jacobian_raw`, and `tests/test_gradients.py` checks that (a) the
raw version is genuinely unbounded near the poles (`>1000` at `theta =
0.001`, matching the paper's own worked example), and (b) both
regularizations stay finite there.

### Benchmark: theta-space vs raw qg-space vs regularized qg-space

`examples/benchmark_qg_vs_theta.py` optimizes the toy loss `E(theta) =
-sin(theta)` (a clean, single interior minimum at `theta* = pi/2`, chosen
specifically so the minimum is *not* at a domain boundary — see the
module's docstring) starting **at** a pole, `theta0 = 0.01` rad (`qg_Z ~
0.9999`), for 150 iterations:

| space | final theta | final E | \|grad\| at end | max \|step\| | behaviour |
|---|---|---|---|---|---|
| theta-space (lr=0.05) | 1.5698 | -1.0000 | 0.001 | 0.05 | converges cleanly to `pi/2` |
| **raw qg-space** (lr=0.05) | 3.1416 | -0.0000 | 1.000 | 3.14 | **never settles** — jumps straight past the target to the opposite pole and oscillates between `theta ~ 0` and `theta ~ pi` for the entire run |
| qg-space, clipped (lr=0.01, eps=0.05) | 1.4184 | -0.9884 | 0.152 | 0.63 | converges, more slowly (smaller lr needed for stability) |
| qg-space, Tikhonov (lr=0.01, eps=0.05) | 1.4043 | -0.9862 | 0.166 | 0.27 | converges, more slowly, smallest step sizes |

This is a sharper empirical statement than the paper's own qualitative
"should be used away from that region": started exactly at a pole with the
*same* learning rate as theta-space, raw qg-space parameterization does not
merely have "a coordinate singularity of its own" — in this run it never
recovers, oscillating indefinitely between the two poles because each
oversized step overshoots into the mirror-image high-gradient region on the
other side. Both regularizations recover, at the cost of needing a smaller
learning rate roughly proportional to `eps` (empirically, keeping
`lr <= eps` avoided the single-step-across-the-domain failure mode seen in
the raw case — a good starting rule of thumb, though not a proof of a tight
bound).

**Practical takeaway**: if a qg-space parameterization is used for VQE/QAOA
optimization near `|qg_Z| -> 1` (i.e. near a computational basis state),
either regularized inverse Jacobian removes the divergence; the clipped
version is simpler to reason about (a hard cap at `1/eps`), the Tikhonov
version degrades more gracefully (it goes to *zero* push right at the pole
rather than a large one), which is usually the safer default when the
optimizer might land exactly on a pole rather than just near one.

## 2. Formalizing qg_S's invertibility domain

Section 2.2 states `qg_S(theta) = qg_S(pi - theta)` and that inversion
requires restricting to a half-domain, without giving the restriction
explicitly. Proof sketch: `p0(theta) = cos^2(theta/2)` is strictly
monotonically *decreasing* on the full domain `theta in [0, pi]`, from 1 to
0. The binary entropy `H(p)` is strictly increasing on `p in [0, 0.5]` and
strictly decreasing on `p in [0.5, 1]`. Composing:

* On `theta in [0, pi/2]`, `p0` decreases monotonically from 1 to 0.5, so
  `H(p0(theta))` is the decreasing-`H`-on-increasing-domain composed with a
  decreasing `p0`, i.e. **strictly increasing** in `theta`, from 0 to 1 —
  invertible.
* On `theta in [pi/2, pi]`, `p0` continues decreasing from 0.5 to 0, so
  `H(p0(theta))` is now the increasing-`H` branch composed with decreasing
  `p0`, i.e. **strictly decreasing** in `theta`, from 1 to 0 — invertible.

`Qang.theta_from_entropic(qg_s, branch="lower" | "upper")` implements the
inverse on each half exactly this way (bisection on the monotonic branch of
`H`, then closed-form `theta = 2*arccos(sqrt(p0))`), and
`tests/test_core.py` round-trips both branches to `1e-4` rad.

## 3. Mixed states and POVMs (`qang.mixed`)

`qg_Z = <sigma_z>` generalizes to any state, pure or mixed, as
`Tr(rho @ sigma_z)` — `qg_z_density` reduces exactly to `qg_Z(theta)` for
`rho = |psi><psi|` (tested for every Table-1 anchor angle). The paper's own
Section 2.3 note that "for any pure state, `S(rho) = -Tr(rho log rho)` is
identically zero" is exactly the fact that makes `von_neumann_entropy`
*not* redundant with `qg_S`: two states can share the same `qg_Z` (e.g. the
pure equatorial state `|+>` and the classical 50/50 mixture of `|0>` and
`|1>` both have `qg_Z = 0`) while having very different entropy (`qg_S(|+>)
= 1` by the paper's own Table 2, vs. `von_neumann_entropy` of the classical
mixture, which is *also* 1 in this particular case — but for a mixture with
unequal weights, e.g. `mix(rho0, rho1, p=0.2)`, `qg_Z = -0.6` while
`von_neumann_entropy ~= 0.72`, a combination the pure-state theory in the
paper cannot represent at all, since it has no `theta` that produces it).

`qg_s_povm` generalizes the *measurement* side: given any POVM `{E_i}`
(not just the two-outcome Z-basis projection), it returns the Shannon
entropy of the resulting outcome distribution, normalized to `[0, 1]` by
`log2(n_outcomes)` so it stays comparable to the original `qg_S` scale
regardless of how many outcomes the POVM has. `qg_s_povm(rho,
standard_z_povm())` reduces exactly to `qg_S(theta)` (tested for every
Table-2 anchor angle); `trine_povm()` is included as a worked 3-outcome
example.

## 4. Multi-qubit tensor-product profiles (`qang.multiqubit`)

Two complementary generalizations, both reducing exactly to the
single-qubit definitions at `n_qubits = 1` (tested):

* **Per-qubit local qang** (`per_qubit_qg_z`, `marginal_qg_s`): partial-trace
  each qubit out of the full register and report its own `qg_Z` / `qg_S` —
  a "projection profile" across the register.
* **Joint qang** (`joint_qg_s`): the Shannon entropy of the full `2^n`
  computational-basis outcome distribution from measuring every qubit,
  normalized by `n_qubits` to stay in `[0, 1]`.

The Bell state is the concrete demonstration
(`tests/test_multiqubit.py::test_bell_state_is_globally_pure_but_locally_maximally_mixed`):
`per_qubit_qg_z` reports `0.0` for *both* qubits (each looks maximally
mixed on its own), `marginal_von_neumann_entropy` reports `1.0` bit for
*both* qubits individually, and yet `joint_qg_s(..., normalize=False)`
reports exactly `1.0` bit total for the *whole* 2-qubit state, which is
also exactly pure (`von_neumann_entropy` of the full `4x4` density matrix is
`0`). That combination — every part maximally mixed, the whole exactly pure
— is impossible for any unentangled (product) state, which is exactly why
it is the standard textbook entanglement witness; `marginal_von_neumann_entropy`
is the piece of this package that can detect it, while `per_qubit_qg_z`
alone cannot.

## 5. Error propagation between probability-space and theta/qg-space (`qang.statistics`)

This is the other half of Future Research Direction #2 (the half not
covered by §2 above, which handled the *domain of invertibility*; this
handles what happens to that inversion under *finite-shot measurement
noise*).

**Setup.** You never observe `P(|0>)` directly on real or simulated
hardware — you estimate it from `N` measurement shots as
`p0_hat = counts0 / N`, a `Binomial(N, p0) / N` estimator with
`Var(p0_hat) = p0*(1-p0) / N`. Since `qg_Z_hat = 2*p0_hat - 1`, that shot
noise propagates directly into `qg_Z_hat`, and then — through the *same*
singular Jacobian studied in §1 for optimization steps — into
`theta_hat = arccos(qg_Z_hat)`.

**Result (delta method, first order).**

```
Var(qg_Z_hat)  = 4*p0*(1-p0)/N = sin^2(theta)/N        (exact identity)

Var(theta_hat) ~= Var(qg_Z_hat) * (d(theta)/d(qg))^2
               = [sin^2(theta)/N] * [1/sin^2(theta)]
               = 1/N
```

The `sin^2(theta)` factors cancel **exactly**: to first order, the
propagated angular uncertainty is *constant* (`std ~= 1/sqrt(N)` rad)
across the whole Bloch sphere, independent of `theta`. The shrinking
shot-noise variance near a pole (fewer "wrong-outcome" events to measure)
exactly offsets the growing sensitivity of `arccos` there (the §1 / Section
4.1 singularity).

This is a genuinely different statement from the *optimization*-step story
in §1: there, the (non-vanishing) energy gradient `dE/d(theta)` does *not*
shrink near the poles the way `sqrt(Var(qg_Z_hat))` does here — so a
qg-space *gradient step* still blows up near a pole even though qg-space
*measurement* uncertainty does not. Regularizing the gradient (§1) and
propagating measurement error (this section) are two separate problems
with the same singular Jacobian at their core, and only the first needs a
fix; the second self-regularizes.

**Caveat.** The cancellation is only a first-order (delta-method /
Gaussian) result: it requires `N` large enough that the binomial count of
the minority outcome is itself well approximated by a Gaussian (rule of
thumb: `N * min(p0, p1) >> 1`). Very close to a pole, for fixed `N`, that
condition fails — the minority outcome becomes a rare event, `theta_hat`
is usually exactly the pole itself (zero error) with an occasional large
jump when the rare outcome does appear. `qang.statistics.empirical_theta_std`
makes this breakdown directly visible: it bootstraps `theta_hat` across many
simulated trials on Qiskit's `AerSimulator` and reports how many of them
landed exactly on a domain boundary (`n_at_pole_boundary`), rather than
just assuming the asymptotic formula holds everywhere. Away from the poles
(e.g. `theta = pi/2`), the empirical std matches the analytical `1/sqrt(N)`
prediction to within simulator/bootstrap noise; `tests/test_statistics.py`
checks both regimes.

`confidence_interval_theta(theta_hat, n_shots, confidence)` packages the
result into a practical normal-approximation CI (`theta_hat +/- z/sqrt(N)`,
clipped to `[0, pi]`) — e.g. "10,000 shots pins down theta to about
`+/- 0.02` rad (95% CI), for any state that isn't extremely close to a
computational basis state."

## 6. qg as a native gate for Qiskit, Cirq and PennyLane (`qang.qiskit_gate`, `qang.cirq_gate`, `qang.pennylane_gate`)

Future Research Direction #1 asked for qg as a native unit/type across
multiple SDKs. `qang.qiskit_gate` (`RQangGate`, `FullRQangGate`) was
already there; `qang.cirq_gate` completes the Cirq side with the same two
constructors:

* `rqang_gate(qang)` — qg_Z-only preparation, built directly on Cirq's own
  native `cirq.ry(theta)` (no custom `Gate` subclass needed — this is the
  idiomatic Cirq primitive for exactly this job).
* `full_rqang_gate(qang)` — full `(qg_Z, phi)` Bloch-sphere preparation,
  built as an explicit `U(theta, phi, lambda=0)` unitary wrapped in
  `cirq.MatrixGate`, using the *same* convention as Qiskit's `UGate`, so a
  `Qang` produces bit-for-bit the same prepared state whichever SDK backend
  is used. `tests/test_cirq_gate.py` checks this both via measured Z-basis
  probabilities and via direct statevector fidelity against
  `Qang.to_statevector()`.

`qang.pennylane_gate` adds the PennyLane side with the same pair,
`rqang` (`qml.RY(arccos qg_Z)`) and `full_rqang` (`qml.U3(theta, phi, 0)`,
whose matrix equals Qiskit's `UGate(theta, phi, 0)`). What PennyLane adds
is autodiff: both accept a raw, trainable qg_Z, and the arccos is taken
with `qml.math`, so a circuit can be optimized directly in qg coordinates.
`tests/test_pennylane_gate.py` pins that `d<Z>/d(qg_Z) = 1` exactly in the
interior (the two singular factors `-sin θ` and `-1/sin θ` cancel), the
analytic `d<X>/d(qg_Z) = -qg_Z cos φ / sqrt(1 - qg_Z²)`, and a gradient
descent run in qg_Z. At `|qg_Z| = 1` autodiff returns nan: the §4.1
singularity survives the cancellation numerically, so trainable qg_Z
values must stay strictly inside (-1, 1).

# Part II — Results beyond the original roadmap

Citation keys such as `[@cross2019]` refer to entries in `paper.bib`.

## 7. Exact identities: qg_Z ↔ qg_S, and `qg_correlation`

### 7.1 The branch-free qg_Z ↔ qg_S identity (`qang.core.qg_s_from_qg_z`)

Both metrics are functions of one quantity, the Z-basis population
`p0 = P(0) = cos^2(theta/2)`: `qg_Z = 2*p0 - 1` and `qg_S = H(p0)`.
Eliminating `p0` gives

```
qg_S = H((1 + qg_Z) / 2)            valid on all of qg_Z in [-1, 1]
```

§2 needed two branches to invert `qg_S -> theta`. This direction needs
none: it is a single-valued function of `qg_Z`. The derivation uses only
`qg_Z = Tr(rho sigma_z)` and `qg_S = H(<0|rho|0>)`, never a specific gate.
So it holds for **every single-qubit state, pure or mixed, however it was
prepared** (Ry, Rx, U, a noisy channel, or a partial trace of a larger
register). Its derivative,

```
d(qg_S)/d(qg_Z) = (1/2) * log2((1 - qg_Z) / (1 + qg_Z))
```

(checked numerically by finite differences), diverges at `qg_Z = ±1`. That
is the entropy-side counterpart of the §1 Jacobian singularity: near a
pole, a tiny change in bias produces an unboundedly large relative change
in entropy.

Consequence for multi-qubit registers: applied to each reduced
single-qubit density matrix, the identity makes `per_qubit_qg_z` and
`marginal_qg_s` two encodings of the **same** per-qubit information.
`tests/test_multiqubit.py` checks this on random Haar states, not only on
Bell/GHZ/product states.

### 7.2 `qg_correlation`: the Z-basis total correlation

Since the per-qubit profile carries no information beyond `qg_Z`, anything
new must come from the joint distribution. Define

```
qg_correlation(state) = sum_i H(X_i) - H(X_1, ..., X_n)
                      = sum_i marginal_qg_s[i] - joint_qg_s(normalize=False)
```

where `X_i` is qubit i's Z-basis outcome. This is Watanabe's *total
correlation* [@watanabe1960] of the measurement outcomes. For n = 2 it is
exactly the classical mutual information `I(X_1; X_2)`. Properties
(proved by subadditivity of Shannon entropy; checked in the test suite):

* `0 <= qg_correlation <= n - 1` bits, and it is 0 **iff** the Z-basis
  outcomes are independent.
* Product states: exactly 0.
* Bell: 1 bit. GHZ_n: exactly `n - 1` bits (attains the upper bound;
  verified for n = 3, 4, 5).
* Never negative on random Haar states (tested).

**Scope, stated plainly.** `qg_correlation` measures *classical*
correlation in the Z basis. It is not an entanglement measure in either
direction:

* It is **positive without entanglement**: the classical mixture
  `(|00><00| + |11><11|)/2` has `qg_correlation = 1` bit, the same as a
  Bell state.
* It is **zero despite entanglement**: every graph state (e.g. the 4-qubit
  linear cluster state) has `qg_correlation = 0`, because its outcome
  distribution is exactly uniform (see §8.1).

Combining it with `marginal_von_neumann_entropy` (§4) separates the cases
that matter: the Bell state has both quantities positive, the classical
mixture has correlation 1 but global von Neumann entropy 1, and the graph
state needs a measurement outside the Z basis.

## 8. Structural blind spots of Z-basis metrics

These are exact results, not numerical accidents. They belong in the paper
next to the Section 4.1 singularity, as limits of what the unit can detect.

### 8.1 Graph states are invisible (`qang.circuits`)

A graph state is `H^{⊗n}` followed by CZ gates on the graph's edges. `H^{⊗n}`
makes every `|amplitude|^2` equal to `2^-n`. CZ is diagonal, so it only
changes phases. Therefore, **for every graph**:
`qg_Z = 0` on every qubit, `joint_qg_s = 1` (maximal), and
`qg_correlation = 0`. All of the entanglement (certified independently in
`tests/test_circuits.py` through the stabilizers `X_i prod_{j∈N(i)} Z_j = +1`)
lives in phases that no Z-diagonal statistic can see. Readout-only
dephasing (§10.4, Finding A) is the same blind spot, showing up for a
class of noise instead of a class of states.

### 8.2 The arccos range trap (`examples/vqe_h2_qg_vs_theta.py`)

Every qg-space update goes through `theta = arccos(qg)`, which only
returns values in `[0, pi]`. On the H2 Hamiltonian (0.735 Å, STO-3G, 2-qubit
parity mapping, independently derived with PySCF [@sun2018pyscf]), the
optimum lies at `theta ≈ -0.22`, in the half that arccos cannot reach.
Starting from Hartree-Fock (lr = 0.3):

| space | final E (Ha) | error vs FCI | steps to chem. accuracy |
|---|---|---|---|
| theta | -1.1373060348 | 9.1e-10 | 5 |
| qg raw / clipped / Tikhonov | -1.1169989956 (= HF) | 2.0e-2 | never |
| theta_pole_damped | -1.1373060348 | 9.1e-10 | 61 |

All three qg-space variants stop exactly at the Hartree-Fock energy and
recover **zero** correlation energy. This is a hard limit of the range,
separate from the Jacobian singularity, and regularization cannot fix it.
It motivates keeping updates in theta-space (§9).

## 9. Pole-damped gradient descent (`theta_pole_damped`)

### 9.1 Definition and honest framing

```
pole_damping_factor(theta, eps) = max(|sin(theta)|, eps)      in (0, 1]
theta_new = theta - lr * pole_damping_factor(theta, eps) * dE/d(theta)
```

The update stays in theta-space, so the §8.2 trap cannot occur, and the
step shrinks near the poles. This is a position-dependent trust-region
damping in the tradition of Levenberg–Marquardt
[@levenberg1944; @marquardt1963]. **It is not a new optimizer class.**
The framework-specific part is that the damping schedule is not tuned:
it is `|d qg_Z / d theta|`, the same pole geometry as §1. Robustness is
insensitive to `eps` (tested). For vectors of parameters the factor is
applied per coordinate and depends only on that coordinate's own value
(`qang.gradients.multi_param_gradient_descent`). A parameter already
at its optimum is therefore left untouched by damping on the others
(`examples/multi_parameter_pole_damped_vqe.py`).

### 9.2 Statistical robustness (300 random one-qubit landscapes per lr)

`E = h_z cos(theta) + h_x sin(theta)`, started near a pole
(`examples/pole_damped_gradient_descent_robustness.py`):

| lr | plain success | damped success | damped trapped |
|---|---|---|---|
| 0.3 | 0.997 | 0.963 | 0.017 |
| 1.0 | 0.773 | 1.000 | 0.000 |
| 2.0 | 0.187 | 0.477 | 0.030 |
| 3.0 | 0.087 | 0.357 | 0.023 |
| 5.0 | 0.057 | 0.253 | 0.040 |
| 10.0 | 0.027 | 0.130 | 0.067 |

At a safe lr, damping costs a few percent. At lr = 1 it succeeds 100%
of the time vs 77%. From lr = 2 on, it wins by 2.5–5×. The trapped rate (the optimizer oscillating at its starting pole)
grows with lr, up to 6.7% at lr = 10. That is higher than the "2–5%"
figure in `qang.gradients`' docstring, which covers only lr ≤ 5.

### 9.3 H2 with two coupled parameters (`examples/h2_vqe_multi_parameter_ansatz.py`)

Ansatz `Ry(θ0)⊗Ry(θ1)` then `CX(1→0)`. The optimum is at `θ0 = π` exactly
(a pole) and `θ1 ≈ -0.2235`. Start: `(1.0, 1e-6)`.

* **Finding A (cost).** At safe lr, damping needs ~3× more steps to reach
  chemical accuracy (1.6 mHa): 603 vs 196 at lr = 0.05, 29 vs 9 at lr = 1.0.
  The reason is that `θ0`'s own target is a pole, and the step keeps
  shrinking as it gets there.
* **Finding B (benefit).** At lr = 2.5, 3.0 and 4.0, plain gradient descent
  never reaches chemical accuracy (final errors 0.048, 0.43 and 0.93 Ha).
  The damped optimizer reaches the FCI energy (error 9e-10) in 10, 8 and 6
  steps. At lr = 5.0, plain gradient descent touches chemical accuracy
  once (step 90) and then leaves it, ending 0.45 Ha away. Damped
  converges in 5 steps.

(A start near `θ0 = 0` was rejected on purpose: the landscape has a
spurious stationary point at `(0, -π/2)` where both partial derivatives
vanish. That is a bad local minimum, not a pole effect, and damping is
not designed to fix it.)

### 9.4 LiH with a mixed Ry/Rx ansatz (`examples/lih_vqe_ry_rx_ansatz.py`)

**Hamiltonian.** LiH at 1.5459 Å, STO-3G, active space (2 electrons, 3
spatial orbitals), ParityMapper, giving 4 qubits and 52 Pauli terms. This
is the scale of early hardware VQE work [@omalley2016; @kandala2017]. The
coefficients were derived once with PySCF + qiskit-nature and hard-coded,
so neither library is a runtime or test dependency.

**Ansatz.** `Ry(θ0), Ry(θ1)` on the two occupied spin-orbitals, `Ry(θ2),
Rx(θ3)` on the two virtual ones, then `CX(2→0), CX(3→1)`. At `(π, π, 0, 0)`
the statevector overlap with qiskit-nature's official `HartreeFock`
circuit is 1.0 to machine precision, and the energy equals
`HF = -0.0437813056 Ha` (electronic energy of the active space).

| energy (active space) | Ha |
|---|---|
| Hartree-Fock | -0.0437813056 |
| best attainable with this ansatz | -0.0440152421 |
| exact (FCI) | -0.0448309020 |

The HF–FCI gap (1.05 mHa) is the genuine correlation energy, not a bug.
It happens to be smaller than chemical accuracy, so convergence is
measured against the ansatz's own optimum instead.

**Generalization beyond Ry, settled mathematically.** From `|0>`,
`Rx(θ)|0> = cos(θ/2)|0> - i sin(θ/2)|1>` has exactly the same populations
as `Ry(θ)|0>`. They differ only by a relative phase. So `qg_Z(θ) = cos θ`
and `pole_damping_factor(θ)` are the same function for both axes. The
factor was never axis-specific, only population-specific, and needs no
generalization. The phase is still physically real: after `CX(3→1)` it
changes the joint state and the energy. The test suite checks both halves
(equal single-qubit populations, different register energies).

* **Finding A (cost, larger than for H2).** All four optimal parameters
  sit at or within ~0.04 rad of a pole (`θ1* = π` and `θ3* = 0` exactly).
  At lr = 0.3 damping needs 355 steps vs 17; at lr = 1.0, 106 vs 5 (~20×).
* **Finding B (benefit).** For lr from 2 to 10, plain gradient descent never
  converges (final errors 0.14–0.40 Ha). Damped always does, to
  errors ≤ 1.1e-7, and in **fewer** steps as lr grows: 53, 42, 26, 15 and 10
  at lr = 2, 2.5, 4, 7 and 10.

**Take-away across §9.** Damping trades iterations at a well-tuned lr for
robustness to a badly tuned one. The cost grows with the number of
parameters whose optimum sits on a pole, which is typical of
Hartree-Fock-referenced chemistry ansätze.

## 10. qg_S as a benchmarking signal

### 10.1 Versus Heavy Output Probability (`examples/quantum_volume_qg_s.py`)

Across 10 random 4-qubit Quantum Volume circuits [@cross2019] × 11 levels
of global depolarizing noise (110 points), Pearson r(HOP, qg_S) = **-0.926**.
qg_S needs no heavy set and no ideal simulation. Caveat: the noise model
is depolarizing on the output distribution, whose fixed point is uniform.
§10.4 shows what happens when it is not.

### 10.2 Finite shots: plug-in bias and Miller–Madow (`quantum_volume_qg_s_finite_shots.py`)

The plug-in entropy estimator is biased low. The Miller–Madow correction
[@miller1955], implemented as `joint_qg_s_from_counts`, removes most of that
bias:

| shots (4 qubits) | plug-in bias | Miller–Madow bias |
|---|---|---|
| 50 | -0.0535 | -0.0097 |
| 100 | -0.0292 | -0.0055 |
| 500 | -0.0055 | -0.0003 |
| 1,000 | -0.0027 | ~0 |

At a fixed 1,000 shots, the plug-in bias grows about 6× from 3 to 6 qubits
(-0.0012 → -0.0075). Miller–Madow stays at or below 5e-4 in magnitude.

### 10.3 Versus linear XEB (`quantum_volume_qg_s_vs_xeb.py`)

* **Finding A.** Linear XEB [@arute2019] has noiseless value `A - 1`, where
  `A = 2^n Σ p_ideal^2`. This equals 1 only for Porter–Thomas statistics. On
  QV circuits `A` = 1.23, 1.37, 2.33 and 2.02 for n = 3–6. So raw XEB needs
  a circuit-specific calibration. qg_S does not.
* **Finding B.** Linear XEB is **unbiased** at any shot count (tested down
  to 10 shots, always within 5 SE). It is a sample mean of a bounded
  per-shot quantity. qg_S is a strictly concave functional, so its plug-in
  estimator is biased by Jensen's inequality. Appendix A gives the
  structural reason: XEB is linear in the outcome distribution, and
  entropy is not.

Neither metric dominates. qg_S is calibration-free; XEB is unbiased.

### 10.4 Realistic gate-level noise: T1 and T2 (`quantum_volume_qg_s_realistic_noise.py`)

Qiskit Aer density-matrix simulation, 4-qubit depth-4 QV circuit (seed 0),
ideal `qg_S = 0.9193`.

* **Finding A — readout-only dephasing is exactly invisible.** `qg_S` =
  0.919303 for λ = 0, 0.3, 0.7 and 1.0. Dephasing preserves the diagonal,
  and qg_S depends only on the diagonal.
* **Finding B — mid-circuit dephasing is visible.** After every 2-qubit
  block, qg_S rises monotonically: 0.9193 → 0.9615 → 0.9952 → 0.9991 →
  0.9999 (λ = 0 → 1). Later entangling gates turn lost coherence into
  randomized populations.
* **Finding C — amplitude damping makes qg_S non-monotonic.** Mid-circuit
  T1 with γ = 0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0 gives qg_S = 0.919, 0.972,
  **0.983**, 0.974, 0.918, 0.713, **0.000**. The fixed point of amplitude
  damping is `|0…0>`, which has zero entropy, not the maximally mixed
  state.

**Consequence for the benchmark.** "More noise → higher qg_S" is a
property of noise channels whose fixed point is maximally mixed. It is
not a property of qg_S. At γ = 0.4 the register gives almost the ideal
value (0.918 vs 0.919) while being badly damaged. qg_S alone therefore
cannot certify a T1-dominated device.

* **Finding D — the fix: pair qg_S with the register's mean qg_Z.**
  `qang.multiqubit.mean_qg_z`, the average of every qubit's own qg_Z,
  measures the relaxation bias. Unital noise pulls it towards 0, and T1
  pulls it towards +1, the value at the fixed point of amplitude damping.
  At γ = 0 and γ = 0.4 the pair is (0.9193, -0.029) vs (0.9176, +0.310),
  so qg_S alone confuses the two operating points and the pair does not.
  Under mid-circuit dephasing, mean qg_Z stays within 0.03 of 0.
  Along a 21-point γ grid it rises monotonically in **23 of 24** QV
  circuits (n = 3, 4, 5; seeds 0–7). The exception (n = 3, seed 1) dips by
  about 0.015 before rising, and that counterexample is pinned in the
  tests. Monotonicity is therefore a strong empirical regularity, not a
  theorem.

  The population of `|0…0>` was also considered. It was monotone less
  often (22 of 24). Purity was rejected because it is non-monotone under
  T1 as well and cannot be estimated from Z-basis counts. mean qg_Z has two
  further advantages: it needs only counts (`mean_qg_z_from_counts`), and
  because it is linear in ρ its estimator is exactly unbiased at any shot
  count (Appendix A).

### 10.5 Device-calibrated validation (`examples/nisq_hardware_validation.py`)

The same QV circuit is run on the `fake_brisbane` backend from
`qiskit-ibm-runtime`, simulated locally with `AerSimulator.from_backend`.
That backend uses a real IBM Eagle device's published calibration data:
per-qubit T1/T2, gate and readout errors, and the device's native gates
and coupling map. An idle delay before readout serves as a controlled T1
knob. The qubits are chosen by `choose_layout`: a connected line with
valid calibration data (T2 ≤ 2·T1) whose worst T1 is maximal (on
fake_brisbane: physical qubits 112–126–125–124). 4,000 shots, seed 42:

| delay | qg_S | mean qg_Z | HOP | linear XEB |
|---|---|---|---|---|
| ideal | 0.9193 | -0.0289 | 0.7722 | +0.3662 |
| 0 µs | 0.9732 | -0.0234 | 0.6680 | +0.2266 |
| 25 µs | 0.9761 | +0.0321 | 0.6478 | +0.2096 |
| 50 µs | 0.9793 | +0.0805 | 0.6082 | +0.1605 |
| 100 µs | 0.9608 | +0.2013 | 0.5437 | +0.0769 |
| 200 µs | **0.8863** | +0.3814 | 0.4358 | -0.0436 |

HOP and XEB fall at every step, and mean qg_Z rises at every step. qg_S
rises and then falls. At 200 µs it is **below the noiseless value**, so
qg_S alone would rank the most broken operating point as the least noisy.
Findings C and D thus survive the move from one idealized channel to
full device-level noise (all channels at once, per-qubit rates, a
transpiled circuit). Seeds 1 and 7 give the same pattern (qg_S at
200 µs: 0.8883 and 0.8847).

The same script evaluates the LiH ansatz (§9.4) from Z/X/Y-basis counts
(qubit-wise commuting groups of the 52 Pauli terms), with no error
mitigation. At 8,000 shots per group, seed 42, the raw error is 47.5 mHa
at the HF point and 44.0 mHa at the ansatz optimum (seeds 1 and 7:
42–45 mHa). That is about 25–30× chemical accuracy and about 200× the
0.23 mHa the optimizer is trying to resolve. Without mitigation, a device
at this noise level cannot see the correlation energy that the noiseless
pole-damped optimizer recovers.

With `--mode ibm` the same script runs on a real device through
`QiskitRuntimeService` and `SamplerV2`. That run needs an IBM Quantum account and has not
been done yet.

## 11. Randomized benchmarking and ZNE in qg units

### 11.1 RB (`examples/randomized_benchmarking_qg_z.py`)

With per-gate depolarizing noise, the survival signal after m random
single-qubit Cliffords plus the recovery gate is exactly

```
qg_Z(m) = (1 - p)^(m + 1)
```

independent of which Cliffords were drawn: across 4 seeds, identical to 10
digits at m = 0, 5 and 10. Depolarizing noise shrinks the Bloch vector
isotropically, and Cliffords are rotations. Fitting `log qg_Z` against
`m + 1` recovers `f = 1 - p = 0.95` and `F_avg = (1 + f)/2 = 0.975` to
machine precision [@magesan2011]. The general RB result is statistical
(it comes from averaging); in this idealized case it holds exactly.

### 11.2 ZNE: which space to extrapolate in (`examples/zne_qg_vs_theta_space.py`)

Ground truth `θ0 = π/3`, `qg_Z = 0.5`:

| noise model | ZNE in qg_Z-space | ZNE in theta-space |
|---|---|---|
| Bloch shrinkage (depolarizing-like) | exact (4e-16) | biased (1.4e-3) |
| coherent angle drift (miscalibration) | biased (1.8e-3) | exact (4e-16) |

Guideline: extrapolate in whichever space makes the noise linear
[@temme2017]. The `θ ↔ qg_Z` round trip makes checking both cheap.

## 12. Barren plateaus in qg-space (`examples/barren_plateaus_qg_vs_theta.py`)

Hardware-efficient ansatz with a global `Z^{⊗n}` cost
[@mcclean2018; @cerezo2021]. In theta-space, Var(∂E/∂θ) falls from 7.2e-2
(n = 4) to 2.0e-3 (n = 10), a log-linear slope of -0.59 per qubit. After
the clipped qg conversion, the exponential decay survives: 3.7 → 0.054.
Regularization only rescales the variance by an O(1) factor. At a pole
(θ[0] = 1e-4, n = 6) the single-sample gradient is -0.072 in theta and
+721 in raw qg, versus 1.44 clipped and 0.0029 Tikhonov. **The Section 4.1
singularity and the barren plateau are independent and compound.** A
qg-space landscape can be exponentially flat almost everywhere and
divergent at isolated points.

## 13. qg_Phi, QPE, QEC and textbook algorithms

* **qg_Phi** (`qang.phase`): `qg_Phi(φ) = e^{i2πφ}`, φ ∈ [0, 1), is invertible
  on its **whole** domain, unlike qg_Z and qg_S. QPE on the T, S and Z phase
  gates (`examples/quantum_phase_estimation_qg_phi.py`) recovers φ =
  1/8, 1/4, 1/2 with probability 1 once the counting register is wide
  enough. QPE computes on a circuit what `QangPhi.from_complex` computes in
  closed form.
* **QEC** (`qang.qec`): in the 3-qubit bit-flip code, the syndrome ancillas
  always end at an exact qg_Z pole (±1). Syndrome extraction is therefore
  a qg_Z readout, and the full encode–error–correct–decode cycle has
  fidelity 1.0 for any logical amplitude.
* **Algorithms** (`qang.algorithms`): teleportation (in deferred-measurement
  form, verified by exact fidelity), superdense coding (note: with this
  gate ordering, qubit 0 decodes the Z bit and qubit 1 the X bit), and
  Grover at the optimal iteration count.

## 14. Consolidated list of known limitations

Stated together so the paper can cite them in one place:

1. **Jacobian singularity** at θ ∈ {0, π} (§1). Regularizing it costs
   exactness near the poles.
2. **arccos range trap** (§8.2). qg-space updates cannot reach θ < 0. On H2
   they lose 100% of the correlation energy.
3. **Z-basis blindness** (§8.1, §10.4 A). Graph-state entanglement and
   readout dephasing are exactly invisible to qg_Z, qg_S and
   `qg_correlation`.
4. **`qg_correlation` is not an entanglement measure** (§7.2). It is
   positive for classical mixtures and zero for graph states.
5. **qg_S is not monotonic under T1** (§10.4 C, §10.5). It cannot
   certify a device dominated by amplitude damping on its own. Pair it
   with `mean_qg_z` (§10.4 D). That pair is monotone in 23/24 circuits
   tested, not in all of them.
6. **Finite-shot bias** of qg_S (§10.2). Use Miller–Madow. XEB does not
   have this problem.
7. **Pole damping** costs 3–20× more iterations at a safe lr when optimal
   parameters sit on poles, and has a trapping rate of up to 6.7% at
   lr = 10 (§9).
8. **§5's `1/N` result is first-order only**. It breaks down when
   `N·min(p0, p1)` is O(1).
9. **No speed or energy advantage.** Closed-form qg expressions are only
   faster than trigonometric ones when the data are already stored as
   qg (1.3× on 10⁶ fidelities; 5.6× *slower* when stored as θ), and
   either way the difference is negligible next to one circuit
   simulation (~0.5 s for 20 qubits). qg does not change how many shots
   a given precision needs (§15.3).

## 15. Three results checked against independent references

All three are also worked through, with generated outputs, in
`notebooks/qang_verificado.ipynb`.

### 15.1 The natural gradient in qg coordinates is plain descent in θ

For `|ψ(θ)⟩ = Ry(θ)|0⟩` the quantum Fisher information is `F_θ = 1`. In
the coordinate `q = qg_Z = cos θ` it becomes

```
F_q = F_θ (dθ/dq)² = 1 / (1 - q²)        (qang.gradients.qfi_qg)
```

so the §1 Jacobian singularity is the Fubini–Study metric written in qg
coordinates. The natural-gradient step (Stokes et al. 2020) is

```
Δq = -η F_q⁻¹ dE/dq = η sin θ dE/dθ   ⇒   Δθ = -η dE/dθ + O(η²)
```

i.e. **the natural gradient in qg is plain gradient descent in θ**
(`natural_gradient_step_qg`; the two trajectories agree to O(η²) and
reach the same minimum, `test_gradients.py`). This is the expected
reparametrization invariance of natural gradients, and it clarifies §1
and §9:

* raw qg-space descent applies the reciprocal of the right metric,
  hence its divergence near the poles;
* the pole-damped θ step is the natural step multiplied by
  `|sin θ| = √(1 - q²)`, which for these real states is also the
  l1-norm of coherence `2|ρ₀₁|`. Pole damping slows the optimizer in
  proportion to the coherence the qubit has left. It is a
  position-dependent step size, closer to a trust region than to a
  Levenberg–Marquardt `λI` shift.

Claims that "QNG in qg converges where θ-descent does not" come from
clipping θ to `[0, π]`, which hides minima at θ < 0 (the §8.2 trap):
without the clip plain θ-descent converges exactly.

### 15.2 The cost of circuit cutting in qg units (`qang.knitting`)

Cutting a two-qubit gate multiplies the shots needed for a fixed
precision by γ² (Mitarai & Fujii 2021). With `qg = cos θ` of the gate
angle:

| Gate family | γ | in qg |
|---|---|---|
| Pauli rotations RXX, RYY, RZZ, RZX | 1 + 2 \|sin θ\| | 1 + 2√(1 − qg²) |
| Controlled rotations CRX/CRY/CRZ, CPhase | 1 + 2 \|sin(θ/2)\| | 1 + 2√((1 − qg)/2) = 1 + 2√P(1) |

Both rows are pinned against `qiskit-addon-cutting`'s own
decompositions (`QPDBasis.from_instruction(gate).overhead`) for eight
angles each, and an end-to-end cut of an RZZ(π/4) circuit reconstructs
`⟨X₀⟩`, `⟨X₁⟩`, `⟨X₀X₁⟩` within 0.03 at 20,000 shots
(`test_knitting.py`). In qg units the cost of a Pauli-rotation cut
depends only on the transverse part `√(1 - qg²)`: free at `qg = ±1`,
maximal (γ² = 9) at `qg = 0`. For RZZ(θ) acting on `|+⟩|+⟩`,
`⟨X₀⟩ = cos θ = qg` exactly, so a single-qubit X measurement reads off
the qg of an unknown ZZ coupling, and with it the cost of cutting it.

A frequently repeated shortcut, `γ² = 1 + √(1 - qg²)`, is wrong: it
underestimates the cost by up to 4.5× (2 instead of 9 at `qg = 0`).
A cut demo that checks `⟨ZZ⟩` after RZZ tests nothing, because `ZZ`
commutes with the gate.

### 15.3 Few-shot estimation: the Haar prior is uniform in qg_Z

Under the Haar measure, `qg_Z` is uniform on `[-1, 1]` (Archimedes'
hat-box theorem; checked on 200,000 Haar states). A uniform prior on
`qg_Z` is therefore the rotation-invariant one, equivalent to
`p₀ ~ Beta(1, 1)`, with posterior `Beta(k₀ + 1, N - k₀ + 1)`
(`bayes_qg_estimate`, implemented without SciPy and checked against
`scipy.stats.beta`). At `N = 50` shots:

| Truth | Delta method | Wilson | Bayes–Haar |
|---|---|---|---|
| near a pole (θ = 0.08): coverage of the 95% interval | 7.6% | 92.4% | 92.4% |
| Haar-random states: coverage | 89.8% | 95.1% | 94.9% |

On Haar-random states the posterior mean has about 4% lower mean squared
error in `qg_Z` than the raw frequency. The honest reading: the Haar
prior repairs the delta method's collapse near the poles and has a clean
geometric justification, but the standard Wilson interval achieves the
same coverage; the gain over good frequentist practice is in the point
estimate, and it is modest. qg does not reduce the number of shots a
given precision requires.

## 16. QML data encoding: qg (arccos) vs angle (`examples/qml_encoding_qg_vs_angle.py`)

A single-qubit data re-uploading regressor (Pérez-Salinas et al. 2020):
the feature `x ∈ [-1, 1]` is loaded L times with `Ry(θ(x))`, with a
trainable `Rz Ry Rz` rotation between loads, so every encoding has the
same `3(L + 1)` parameters, and the output is `⟨Z⟩`.

* **Angle encoding** `θ = αx`: the model is a truncated Fourier series in
  x with frequencies `kα` (Schuld, Sweke & Meyer 2021); α must be chosen.
* **qg encoding** `θ = arccos x`, i.e. the encoded state's `qg_Z` equals
  the feature: since `cos kθ = T_k(x)` and `sin kθ = √(1 - x²) U_{k-1}(x)`,
  the model is a Chebyshev polynomial of degree L (plus `√(1 - x²)` times
  a polynomial of degree L - 1). With identity processing it is exactly
  `T_L(x)` (checked to 1e-15). This is the single-qubit case of quantum
  signal processing (Low & Chuang 2017; Martyn et al. 2021) and of the
  Chebyshev feature maps of Kyriienko et al. (2021); the contribution
  here is the qg reading and a controlled comparison.

80 training points, 201 test points, best of 8 L-BFGS restarts, test MSE:

| Target | L | qg | best fixed α | trainable α, offset (+2L params) |
|---|---|---|---|---|
| x³ − 0.5x | 3 | **2e-15** | 1.5e-4 | 3e-5 |
| random bounded polynomial, degree d | d | ≤ 1e-10 | — | — |
| same | d − 1 | 3e-4 to 3e-3 | — | — |
| sin(πx) | 1 | 0.20 | **3e-16** (α = π) | 6e-15 |
| tanh(4x) | 4 | 7.6e-3 | 3.2e-4 | **2e-7** |
| Runge 1/(1+25x²) | 4 | 7.1e-2 | 2.2e-3 | **7e-6** |
| \|x\| | 4 | 4.9e-3 | 8.4e-4 | **1.7e-4** |

* **Finding A (the qg advantage).** L qg layers fit any bounded
  polynomial of degree L to machine precision, and L − 1 layers do not:
  the layer count is the polynomial degree. That gives a direct sizing
  rule when the target is (close to) a low-degree polynomial of a bounded
  feature — expectation values, smooth physical responses, solutions of
  differential equations — and there is no scale hyperparameter.
* **Finding B (no universal winner).** For periodic or
  non-polynomial targets the Fourier bias of angle encoding wins, by 1.5
  to 4.5 orders of magnitude with a trainable scale. The price of angle
  encoding is the scale: with the wrong one (α = 1 on cos 2πx) it fails
  completely (MSE 0.34 at L = 4).

qg encoding swaps the Fourier inductive bias for a polynomial one; it is
the better choice only when the target is polynomial-like. Open: whether
the polynomial bias survives multi-qubit, entangling re-uploading models
and shot noise, and how it compares with the Chebyshev-tower maps of
Kyriienko et al.

## 17. Finite-precision control: θ grid vs qg grid (`examples/control_quantization_qg_vs_theta.py`)

Control electronics set each angle from a grid of 2^b values. A grid
uniform in θ is the usual choice; a grid uniform in `qg = cos θ` is
uniform in the Born probability and, on the Bloch sphere, is made of
equal-area bands. Each is minimax-optimal for a different error:

| b bits | worst \|ΔP(0)\|, θ grid | worst \|ΔP(0)\|, qg grid | worst infidelity, θ grid | worst infidelity, qg grid |
|---|---|---|---|---|
| 4 | 5.2e-2 | 3.3e-2 | 2.7e-3 | 1.7e-2 |
| 8 | 3.1e-3 | 2.0e-3 | 9.5e-6 | 9.8e-4 |
| 10 | 7.7e-4 | 4.9e-4 | 5.9e-7 | 2.4e-4 |

The qg grid's worst probability error is exactly π/2 times smaller (it
saves log₂(π/2) ≈ 0.65 bits), and its mean error on Haar-random targets
is about 1.23 times smaller. Its worst infidelity is worse by a factor
that grows like 2^b, because it is coarse near the poles.

**Application: loading a distribution** with a Grover–Rudolph tree of Ry
rotations, the state-preparation step of quantum Monte Carlo. In qg
units each rotation is set directly by a conditional probability,
`qg = 2p − 1`. Over 45 cases (n = 4, 6, 8 qubits; b = 4, 6, 8 bits; five
distributions):

* the total variation distance of the loaded distribution is lower with
  the qg grid in 43 of 45 cases (median 1.56×, up to 2.9×);
* for smooth distributions (normal, log-normal, Dirichlet(1)) the
  infidelity is also lower with the qg grid in 25 of 27 cases (median
  2.0×): their conditional probabilities cluster near ½, where the qg
  grid is denser;
* for sparse or sharply peaked distributions (Dirichlet(0.1), a narrow
  normal) the θ grid gives lower infidelity in 15 of 18 cases (median
  3.3×, up to 56×): their conditional probabilities sit near 0 or 1.

**Design rule:** a qg-uniform grid when the task is to reproduce
probabilities (sampling, Monte Carlo, smooth distributions); a θ-uniform
grid when state fidelity matters and targets sit near the poles.

## 18. Which kind of noise? mean qg_Z vs XEB / HOP (`examples/noise_type_detection_qg_vs_xeb.py`)

Task: from N shots of a 4-qubit QV circuit, decide whether the noise is
dissipative (T1) or unital (dephasing, depolarizing). 24 circuits, three
channels applied after every two-qubit block, 10 strengths; every feature
is turned into a classifier by a single threshold fitted on 16 circuits
and scored on the 8 held-out ones, so the comparison is between features,
not models. XEB and HOP need the ideal distribution (a classical
simulation); mean qg_Z and qg_S need only counts. Held-out accuracy
(0.67 = always answering "unital"):

| Noise per gate | N | mean qg_Z | qg_S | XEB | HOP |
|---|---|---|---|---|---|
| strong, 2–50% | 100 | **0.88** | 0.72 | 0.70 | 0.71 |
| strong, 2–50% | 10,000 | **0.90** | 0.73 | 0.76 | 0.75 |
| weak, 0.5–5% | 100,000 | 0.67 | 0.67 | 0.67 | 0.67 |

* **Finding A.** With strong damping, mean qg_Z identifies T1 far better
  than the standard benchmarks, from only 100 shots and without the ideal
  distribution. Its accuracy rises with the strength (0.75 below 10% per
  gate, 0.98 above 25%). XEB and HOP measure how much noise there is,
  not which kind.
* **Finding B.** At realistic per-gate strengths every feature is at
  chance, even with 100,000 shots. The limit is not shot noise: the ideal
  mean qg_Z of a random 4-qubit QV circuit already varies from circuit to
  circuit (standard deviation 0.14, range −0.33 to +0.33), more than the
  T1 shift.

**Practical consequence:** mean qg_Z is a T1 detector on circuits whose
ideal value is known, such as the idle-delay probe of §10.5, not a
passive detector on arbitrary payload circuits. Subtracting the ideal
value and correcting with the XEB-estimated fidelity lifts weak-noise
accuracy only to ~0.8 in exploratory runs (not pinned).

## 19. QML beyond one qubit (`examples/qml_multiqubit_and_shots.py`)

The model of §16 extended to n qubits: in each of L layers every qubit
loads a feature with Ry(θ(x)), then a CZ ring and a trainable Rz Ry Rz on
every qubit; output ⟨Z₀⟩. Best of 6 L-BFGS restarts, test MSE.

| Setting | Target | qg | angle π/2 | angle π |
|---|---|---|---|---|
| 1D input, 2 qubits, L = 2 | x³ − 0.5x | **1e-13** | 5e-5 | 1e-2 |
| | random degree-4 polynomial | **2.7e-3** | 5.8e-3 | 8.9e-3 |
| | sin(πx) | 3e-3 | 6e-13 | **2e-14** |
| | tanh(4x) | 1.6e-2 | **1.2e-3** | 6.2e-2 |
| 2D input, 2 qubits, L = 2 | x₁x₂ | **2e-16** | 2.4e-4 | 8.3e-2 |
| | 0.5 T₂(x₁) + 0.5 x₁x₂² | **2.1e-3** | 6.4e-3 | 4.7e-2 |
| | sin(πx₁) cos(πx₂) | 9.1e-2 | 3.0e-2 | **9e-18** |
| | tanh(2(x₁ + x₂)) | 4.4e-2 | **1.9e-2** | 0.29 |

* **The polynomial bias survives more qubits and more features:** qg is
  best on every polynomial target and angle encoding on every periodic or
  saturating one.
* **n·L is a bound, not a guarantee.** The random degree-4 polynomial is
  inside the n·L = 4 budget but is not fitted exactly by this shallow
  CZ-ring architecture (the mixed 2D polynomial neither). The single-qubit
  statement of §16 — L layers reach every degree-L polynomial — does not
  carry over to every multi-qubit architecture.

**Finite shots.** x³ − 0.5x on one qubit, L = 3, trained with Adam and
parameter-shift gradients estimated from N shots per circuit (1500 steps,
lr 0.02, median of 5 seeds):

| N | qg | angle π/2 |
|---|---|---|
| exact | 1.8e-5 | 4.2e-4 |
| 10,000 | 4.5e-5 | 4.2e-4 |
| 1,000 | 6.2e-5 | 5.5e-4 |

The advantage survives shot noise but shrinks from exact (noiseless
L-BFGS) to about 10×; a gradient optimizer does not reach
machine-precision fits in 1500 steps either way.

**Classical control.** Least-squares Chebyshev regression of degree 3 on
the same 80 points reaches 6e-33 with NumPy alone. These models are
classically simulable: the result says which inductive bias to give a
quantum model — qg encoding for polynomial-like targets — not that the
quantum model beats classical regression. A claim of quantum advantage
would need models that are hard to simulate, which this study does not
test.

## Suggested next steps

* **Real-device run.** Run `examples/nisq_hardware_validation.py
  --mode ibm` and compare with §10.5. Everything except the account is in
  place.
* **Mitigated LiH on hardware.** Add readout correction and ZNE (§11.2)
  to the §10.5 energy evaluation. Then run the pole-damped optimizer
  itself with shot-based parameter-shift gradients.
* **Multi-qubit error propagation.** Extend §5 to the mixed-state and
  multi-qubit settings of §3–4, where the Jacobian is no longer the scalar
  `-1/sin θ`.

## Appendix A. Functional-analysis foundation: Riesz–Fréchet

This appendix is conceptual. It adds no new result. It records why qg_Z
is the canonical linear readout of a quantum state, and why qg_S cannot
be one.

**Setting.** For n qubits, operators on `C^{2^n}` form a finite-dimensional
Hilbert space under the Hilbert–Schmidt inner product
`<A, B> = Tr(A† B)`. Density matrices live in it.

**Riesz–Fréchet representation** [@riesz1907; @frechet1907]. Every
continuous linear functional `f` on a Hilbert space `H` has the form
`f(x) = <y_f, x>` for a unique `y_f ∈ H`. In finite dimension continuity
is automatic, and the theorem reduces to basic linear algebra. That is why
this is a foundation, not a result.

**Application.** The map `ρ ↦ qg_Z(ρ) = Tr(ρ σ_z)` is linear in ρ, so it
has a unique Hilbert–Schmidt representative, `σ_z` (on one qubit; on
qubit i of a register, `σ_z^{(i)} ⊗ I`). Every linear readout of a state
(an expectation value, a population, a per-shot average such as linear
XEB) is of this form. qg_Z is the one whose representative is the
Z observable.

**What does not have a representative.** `qg_S(ρ) = H(<0|ρ|0>)` is
strictly concave in ρ, not linear, so no operator represents it. This is
the same structural fact behind:

* §10.3 B: linear functionals have unbiased sample-mean estimators, and
  entropy does not (Jensen's inequality).
* §7.1: the identity `qg_S = H((1 + qg_Z)/2)` expresses the non-linear
  metric as a fixed scalar function of the linear one. qg_S carries no
  information about the state beyond what qg_Z's representative already
  extracts.

**On Riesz's lemma (1918), for the record.** The almost-orthogonal-vector
lemma is used to show that the closed unit ball of an infinite-dimensional
normed space is not compact. In the finite dimensions used here it is
trivial, since an exactly orthogonal vector always exists. It plays no
role in qang and should not be cited as if it did.
