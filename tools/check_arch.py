"""tools/check_arch.py — architectural audit of the qang package, as a CI gate.

Runs the audit that lives in `qang.qarch` and turns it into something a CI job
can gate on:

  1. the core is NumPy-only, the `iqm` extra stays isolated and `jax` is pinned
     below 0.9;
  2. no test imports (at module level) a package that the `all` extra does not
     install without being protected by `pytest.importorskip`, so pytest never
     dies at collection time with exit code 2.

The second check is a hard gate: it is the invariant that keeps the main CI job
collecting. It currently reports 0 problems.

A third, ADVISORY section repeats the check against the `dev` extra (the `core`
job on Python 3.9/3.10 installs no SDK at all). It is reported but never fails
the build, because `qang.qarch` counts two things as unguarded that do not break
collection in practice:

  * an import inside `try/except ImportError` — e.g. `qang/sectors.py` imports
    Qiskit for one analysis pass only and degrades gracefully without it;
  * a `pytest.importorskip("qiskit_aer")` that precedes a later `from qiskit
    import ...` — the module is skipped, not collected with an error.

So the advisory section over-reports by design. It is kept because it points at
the tests that would need attention if the `core` job ever stopped skipping
them. `examples/` is passed as a local directory; without it the scripts in
`examples/` look like third-party packages and the report shows dozens of false
positives.

Standard library only.

    python tools/check_arch.py          # human-readable report, exit 1 on failure
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qang import qarch  # noqa: E402

QANG_DIR = os.path.join(ROOT, "qang")
TESTS_DIR = os.path.join(ROOT, "tests")
# `examples` matters: its scripts are local modules, not third-party packages.
LOCAL_DIRS = [os.path.join(ROOT, d) for d in ("qang", "tests", "tools", "examples")]


def main() -> int:
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        extras = qarch.parse_pyproject(fh.read())

    groups = qarch.group_modules(qarch.requires(QANG_DIR))

    print("Modules of qang/ by dependency:")
    for name in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        print(f"  {name:10} {len(groups[name]):3d}  {', '.join(sorted(groups[name]))}")
    print()

    failures = []

    print("Extra invariants:")
    for name, ok in sorted(qarch.check_extras(extras).items()):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(name)
    print()

    print("Collection-time imports the CI cannot install (gate: the 'all' extra):")
    available = qarch.installed_by(extras, "all")
    bad = qarch.unguarded_tests(TESTS_DIR, QANG_DIR, available, LOCAL_DIRS)
    print(f"  {'OK  ' if not bad else 'FAIL'} tests unguarded against the 'all' extra: {len(bad)}")
    for name, missing in bad:
        print(f"        {name}: {missing}")
    if bad:
        failures.append("unguarded tests against the 'all' extra")
    print()

    print("Advisory (not a gate): the same check against the 'dev' extra, i.e. the")
    print("'core' CI job on Python 3.9/3.10, which installs no SDK. It over-reports:")
    print("guarded try/except imports and module-level importorskip are counted as")
    print("unguarded, although neither breaks collection.")
    available_dev = qarch.installed_by(extras, "dev")
    advisory = qarch.unguarded_tests(TESTS_DIR, QANG_DIR, available_dev, LOCAL_DIRS)
    print(f"  INFO tests flagged against the 'dev' extra: {len(advisory)}")
    for name, missing in advisory:
        print(f"        {name}: {missing}")
    print()

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("All architecture checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
