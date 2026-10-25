**Title (discussion):** Reweighting decay faults by the final data readout when T1 dominates

Not a bug, and not a request to change defaults: a decoding heuristic we tested in simulation, shared in case it is
useful or already known.

**Idea.** Under amplitude damping, a data qubit read as |0> at the end is more likely to have suffered a decay than one
read as |1> is to have suffered an excitation. Before matching, we reweight the edges of decay faults on each data qubit
by its final readout (lower weight where the readout makes a decay plausible), then run ordinary minimum-weight matching.

**What we found (simulations, PyMatching 2.x).**

- With T1-dominated noise: logical error 2.1-2.5x lower at the code-capacity level and 2.2-3.3x lower at the circuit level
  than the same decoder without the reweighting.
- With depolarizing-dominated noise it is about 15 % worse, so it needs a switch (we used a witness from the readout
  statistics to decide when to turn it on).
- For erasure-type heralds we used an analogous (1 - h) rule, with h the herald reliability (exact weights in chapter 26).

**Question.** Is per-shot edge reweighting from final readouts something PyMatching users already do with the existing API
(for example by building a `Matching` per syndrome class), or would a documented pattern or example be welcome? We are not
proposing a code change unless you see a natural place for it. If issue #172 (per-shot `edge_reweights` in the decode
methods) goes ahead, this would be a concrete use case for it: the reweighting changes only the decay edges of each data
qubit, shot by shot.

Details and code: chapter 26 of https://github.com/Viny2030/qang (book) and `qang.integrations.stim` for the Stim side.
