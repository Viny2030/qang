**Title:** Heralded readout characterization: separate thermal excited population from assignment error

**Labels:** type: feature request

### What should we add?

A characterization experiment that measures the readout assignment errors (e01, e10) **and** the qubit's residual excited
population (effective temperature) without confusing the two.

**The problem with the current approach.** Readout calibration prepares "|0>" and "|1>", but on a warm qubit those are really
the thermal state and X applied to it. With excited population p and readout `P(1|0) = e01`, `P(0|1) = e10`, the standard
assignment calibration returns approximately `e01 + p (1 - e01 - e10)` instead of `e01` (and similarly for e10), and the
qubit is reported as perfectly cold. The temperature is absorbed into the readout error.

**Proposal: a two-circuit heralded experiment.** Measure once (herald), apply I or X, measure again. With outcomes
r = ±1 and the readout written as the affine map `measured = a + b z` (`a = e10 - e01`, `b = 1 - e01 - e10`, z = ±1 the
true state), the t = 0 pairs give everything in closed form:

```
m   = E[r1]                  = a + b z_eq
V   = E[r1 r2 | I] - m^2     = b^2 (1 - z_eq^2)
C_X = E[r1 r2 | X]           = a^2 - b^2
u = (m^2 - V - C_X) / (2m),  b = sqrt(V + u^2),  a = m - u,  z_eq = u / b
```

so `e01 = (1 - b - a)/2`, `e10 = (1 - b + a)/2`, `p = (1 - z_eq)/2` and `T_eff = h f / (2 k_B artanh z_eq)`.
Adding delays after X and after a pi/2 pulse gives T1 and T2 from the same sweep, fitted jointly.

**Evidence (simulation, Aer 0.18, Qiskit 2.6.1).** Readout error e01 = 0.015, e10 = 0.04, thermal population prepared by
purification with an ancilla, 200,000 shots per circuit:

| true p | e01 standard | e01 heralded | e10 standard | e10 heralded | p heralded | T_eff (5 GHz) |
|---|---|---|---|---|---|---|
| 0 | 0.0148 | 0.0151 | 0.0401 | 0.0396 | 0.0000 | 0 mK |
| 0.01 | 0.0244 | 0.0151 | 0.0494 | 0.0397 | 0.0099 | 52.1 mK |
| 0.03 | 0.0431 | 0.0151 | 0.0683 | 0.0396 | 0.0297 | 68.8 mK |
| 0.08 | 0.0900 | 0.0150 | 0.1152 | 0.0401 | 0.0794 | 97.9 mK |

In a larger simulated study (16 circuits x 2000 shots, 100 repetitions), the joint fit resolved T_eff to about 1 mK from
52 mK up, kept T1/T2 unbiased with about 1.5x smaller spread than the standard suite, and moved by about 0.001 when the
herald was made 5 % non-QND.

**Why it matters.** Besides reporting temperature, it changes downstream decisions: for example, a randomness-certification
pipeline that subtracts "readout error" calibrated the standard way overestimates the private randomness of a warm qubit.

**Scope and limits.** Needs mid-circuit measurement (dynamic circuits) and assumes a QND herald and good single-qubit gates.
Separating thermal population by repeated measurement is known practice; the contribution is a closed-form, two-circuit
analysis that fits the framework's `BaseExperiment` / `BaseAnalysis` pattern.

**Reference implementation.** A working `BaseExperiment` subclass (`HeraldedReadout`) with its analysis and tests is in
`qang.integrations.qiskit_experiments` (https://github.com/Viny2030/qang, MIT license), tested against
qiskit-experiments 0.14.2. Happy to adapt it into a PR following your contribution guidelines if there is interest.
