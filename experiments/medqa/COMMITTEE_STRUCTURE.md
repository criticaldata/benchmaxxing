# MedQA committee-structure experiments (text lane)

New-run experiments that vary the STRUCTURE of the committee (who asserts, at what authority, how
many peers) rather than the cue. All reuse the referee/cascade harness (a flash peer asserts, the
flash-lite holdout is measured), the real MedQA manifest, temperature 0, and a per-experiment
call cache so every run is resumable and reproduces keyless once the cache is committed.

## True-peer negative control (`true_peer_control.py`, #180)

Every seed in the project is a wrong answer. This adds the missing control: on cases the holdout
gets WRONG alone, a single confident peer asserts either the usual wrong distractor or the CORRECT
answer, each with the same case-anchored rationale. If a correct peer is adopted at nearly the
same rate as a wrong peer, the committee is deferring to a confident peer's position, not
discerning content.

```bash
python -m experiments.medqa.true_peer_control --manifest <medqa_manifest.csv> \
    --out experiments/medqa/results --n 150
```

| peer asserts | adoption (on solo-wrong cases, n=24) |
|---|---|
| a WRONG answer | 0.42 |
| the CORRECT answer | 0.58 |

**Read.** A correct peer is adopted somewhat more than a wrong peer (0.58 vs 0.42), so the
flash-lite holdout shows *some* discernment, but it still adopts a confidently-asserted WRONG
answer 42% of the time. Most of the movement is deference to a confident peer's position rather
than to its content: the gap between right and wrong assertions (~0.17) is far smaller than the
overall willingness to defer. This is the project's first true-peer control, and it reframes the
cascade as substantially generic conformity, not purely error injection. 84-114 new API calls
(bare read plus two seeded arms per solo-wrong case); verified keyless afterward.

## Authority gradient on a matched ladder (`authority_ladder.py`, #181)

The same wrong answer, held to identical sentence structure, asserted at four authority rungs
(colleague / senior attending / automated system / clinical guideline) plus a no-assertion
control, on the flash-lite holdout alone, with an adjacent-rung paired McNemar. Results filled in
once the live run (about 600 calls, rate-limited) completes.

```bash
python -m experiments.medqa.authority_ladder --manifest <medqa_manifest.csv> \
    --out experiments/medqa/results --n 120
```
