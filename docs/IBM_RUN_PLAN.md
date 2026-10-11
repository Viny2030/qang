# First real-device run on IBM Quantum (Open Plan)

Two experiments are already pre-registered and checked end to end on the
calibration-based fake backends; only the real-device run is missing. Both fit
comfortably in the free Open Plan (10 QPU minutes per rolling 28-day window).

| experiment | script | circuits x shots | pre-registered (real device) | estimated QPU time |
|---|---|---|---|---|
| heralded qg characterization (§31, §86) | `examples/hardware_characterization_ibm.py` | 32 one-qubit circuits x 2000 | C1' (qg T1 within 25% of the reported T1), C4 (standard suite reports a larger e01 than the qg fit) | about 1 minute |
| Leung-code syndromes as a noise witness (§33, §88) | `examples/qec_syndrome_hardware_qg.py` | 14 seven-qubit circuits x 4000 | D1 (gamma within 40%), D2 (p within a factor 2), D3 (both flip rates increase with t) | about 1 minute |

The time estimates count shots x (repetition delay + circuit duration) plus job
overhead; the real figure is shown on the IBM Quantum Platform Workloads page.
Run the characterization first: it is the smallest job and checks the account,
the backend selection and the mid-circuit measurement.

## Order of work

1. Save the account once, on your own machine (the token is never written in
   the code or committed):

   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                     token="<your API key>", instance="<your instance CRN>",
                                     set_as_default=True)
   ```

2. Re-run the fake mode to confirm the environment (no QPU time):

   ```
   python examples/hardware_characterization_ibm.py --mode fake
   python examples/qec_syndrome_hardware_qg.py --mode fake
   ```

3. Real device, one at a time, saving the raw results:

   ```
   python examples/hardware_characterization_ibm.py --mode ibm --yes-i-run-on-hardware --out ibm_char.json
   python examples/qec_syndrome_hardware_qg.py --mode ibm --yes-i-run-on-hardware --out ibm_leung.json
   ```

   Without `--backend` the least busy device is used; write down its name and
   the date, because the calibration changes daily.

4. Report every prediction as PASS or FAIL in the docstring of each script and
   in RESEARCH_NOTES (§86 and §88), including the device name, the date and the
   job ids, whether or not the predictions hold.

## What is not planned for the Open Plan

The QNN runs of §116-§124 need hundreds of circuits x 1000 shots per backend
(several QPU minutes each); they stay on the fake backends until a larger
allocation is available.
