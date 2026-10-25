# Drafts for upstream projects (book, chapter 35)

Drafts only: nothing here has been posted. Before posting each one, search the project's open and closed issues for the
same topic, and check that the problem still exists in the project's latest release (these were found with the versions
listed in each draft, on 2026-10-25).

| file | project | where | kind | strength |
|---|---|---|---|---|
| `qiskit-experiments.md` | qiskit-experiments | https://github.com/qiskit-community/qiskit-experiments/issues | feature proposal | strong |
| `mitiq.md` | Mitiq | https://github.com/unitaryfoundation/mitiq/issues | feature proposal | strong |
| `pennylane.md` | PennyLane | https://github.com/PennyLaneAI/pennylane/issues | bug report | a real incompatibility |
| `qiskit-dynamics.md` | Qiskit Dynamics | https://github.com/qiskit-community/qiskit-dynamics/issues | compatibility request | a real incompatibility |
| `amazon-braket-sdk.md` | Amazon Braket SDK | https://github.com/amazon-braket/amazon-braket-sdk-python/issues | dependency report | minor |
| `pymatching.md` | PyMatching | https://github.com/oscarhiggott/PyMatching/discussions | discussion | moderate |

## Pre-send check, 2026-10-25

Latest releases on PyPI are the versions the drafts were written with (amazon-braket-sdk 1.127.5, mitiq 1.1.0, pennylane
0.45.1, qiskit-dynamics 0.6.0, qiskit-experiments 0.14.2, PyMatching 2.4.0; jax 0.12.0.1, qiskit 2.6.1). Issues, PRs and
discussions searched in each repository, open and closed; main branches read where the problem is in the code.

| draft | verdict | why |
|---|---|---|
| `pennylane.md` | **do not post** | Duplicate of PennyLaneAI/pennylane#9930, closed 2026-08-03 by the maintainers: they support only jax 0.7.1 and will not review a fix for now. `is_concrete` is still on master. Pin jax in qang instead. |
| `qiskit-dynamics.md` | **do not post** | Already tracked: #390 (closed), #392 and #395 (open since 2025). `qiskit<=1.3` is still on main and the last commit is from 2025-10-07, so the project looks inactive. At most a reaction on #395. |
| `amazon-braket-sdk.md` | **do not post** | PR #1350 "fix: Unpin cloudpickle" is open (2026-09-02, changes requested); the pin is still on main. At most a comment on the PR saying the joblib conflict is a reason to merge it. |
| `mitiq.md` | **post, revised** | No duplicate. Revised: Mitiq's `combine_422_zne` example already composes post-selection with ZNE (with ancillas), so the draft now offers the ancilla-free weight variant and the decision rule, and answers why RFC #1901 was closed. |
| `qiskit-experiments.md` | **post as is** | No issue on thermal population, heralded readout or excited population. Repository active (last commit 2026-09-09). |
| `pymatching.md` | **post, revised** | No discussion on the topic (discussions are enabled). Revised to cite #172 (per-shot edge reweighting, open) as the API this would use. |

Not drafted: CUDA-Q (its GPU target failed because the CUDA driver visible in WSL is older than CUDA-Q's runtime, which
is a local driver issue, not a CUDA-Q bug), and the weaker items of chapter 35 (Qiskit transpiler, qiskit-machine-learning).
