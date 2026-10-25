**Title:** Support Qiskit 2.x (qiskit-dynamics 0.6.0 requires `qiskit<=1.3`)

### What is the expected behavior?

`pip install qiskit-dynamics` next to a current Qiskit (2.x) without downgrading Qiskit.

### What is the current behavior?

qiskit-dynamics 0.6.0 declares `Requires-Dist: qiskit<=1.3`. Installing it into an environment with Qiskit 2.6.1
downgrades Qiskit to 1.3.0, which in turn breaks packages that need Qiskit 2 (for example recent qiskit-ibm-runtime and
qiskit-experiments setups). The only option today is a separate environment.

### Steps to reproduce

```bash
python -m venv env && env/bin/pip install "qiskit==2.6.1" qiskit-dynamics
env/bin/pip list | grep -i qiskit        # qiskit 1.3.0, qiskit-dynamics 0.6.0
```

### Context

For our use (solving driven single-qubit Hamiltonians with `Solver`, `Signal` and the rotating-wave cutoff), the
numerical core worked as expected in the 1.3 environment: resonant pulses matched `cos(area)` to 1e-7-2e-6 for square
and Gaussian envelopes. We have not checked which parts of qiskit-dynamics actually depend on removed Qiskit 1.x APIs
(the pulse module was removed in Qiskit 2.0), so a release that drops or isolates the pulse-dependent pieces might be
enough.

If Qiskit 2 support is not planned (for example if the project is in maintenance mode), a note in the README would help
users plan environments. Please close this if it is already tracked.

qiskit-dynamics 0.6.0, Qiskit 2.6.1 / 1.3.0, Python 3.12, Windows 11.
