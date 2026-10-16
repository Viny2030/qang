"""
Post-hoc analysis of §154 (NOT pre-registered; written after the run of examples/neutrino_collective_qg.py).

The pre-registered errors are against the exact evolution, and at dt = 0.5 the noiseless Trotter error
(0.09) is larger than the whole noisy error, so they cannot show what the filter does to the noise. Here the
same circuits are scored against the noiseless Trotter circuit instead (as §153 and the chemistry sections
do), on new samples (the sampler is not seeded, so the numbers differ slightly from the main run).

python examples/neutrino_collective_qg_posthoc.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import neutrino_collective_qg as M  # noqa: E402

if __name__ == "__main__":
    ref = {p: M.noiseless(p) for p in M.PARTS}
    out = {}
    for name in M.BACKENDS:
        r = M.run_backend(name, ref)
        out[name] = {"error": r["error"], "error_by_time": r["error_by_time"]}
        print(json.dumps({name: {p: {k: round(v, 4) for k, v in e.items()} for p, e in r["error"].items()}}),
              flush=True)
        for p in ("A", "B"):
            k = "filtered_readout" if p == "A" else "reweighted_readout"
            print(f"  {p} gain raw/{k} by time:",
                  [round(a / b, 2) for a, b in zip(r["error_by_time"][p]["raw"], r["error_by_time"][p][k])],
                  flush=True)
    with open(os.path.join(HERE, "data", "neutrino_collective_qg_posthoc.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
