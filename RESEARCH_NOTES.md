# Research notes: extending the Qang beyond the paper

These notes document the extensions built on top of the paper's Section 6
("Future Research Directions") roadmap, with derivations and concrete
results, so they can be folded into a future revision of the paper or a
companion technical report. All four directions listed in Section 6 are now
covered: gradient regularization (§1), error-propagation bounds (§5),
mixed states / POVMs (§3), multi-qubit generalization (§4), and native-SDK
integration for both Qiskit and Cirq (§6).

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

Both are implemented in `quang.gradients` alongside the exact
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

## 3. Mixed states and POVMs (`quang.mixed`)

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

## 4. Multi-qubit tensor-product profiles (`quang.multiqubit`)

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

## 5. Error propagation between probability-space and theta/qg-space (`quang.statistics`)

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
jump when the rare outcome does appear. `quang.statistics.empirical_theta_std`
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

## 6. qg as a native gate, for both Qiskit and Cirq (`quang.qiskit_gate`, `quang.cirq_gate`)

Future Research Direction #1 asked for qg as a native unit/type across
multiple SDKs. `quang.qiskit_gate` (`RQangGate`, `FullRQangGate`) was
already there; `quang.cirq_gate` completes the Cirq side with the same two
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

## Suggested next steps

* Benchmark the regularized gradients on a real multi-parameter VQE
  Ansatz (H2 or similar) rather than the single-parameter toy loss here.
* A PennyLane counterpart to `quang.qiskit_gate` / `quang.cirq_gate`.
* Extend the error-propagation analysis (§5) to the mixed-state /
  multi-qubit settings of §3–4, where the relevant Jacobian is no longer a
  simple scalar `-1/sin(theta)`.
