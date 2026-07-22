# Model-dependence of the plausibility cascade (C)

The anchored-vs-generic plausibility cascade moves the weak holdout **gemini-2.5-flash-lite** from
0.73 to 0.85 conformity (paired McNemar p=0.041 at n=85; `scale_c_summary.json`, PR #141,
post-parser-fix). Does it also move the stronger **gemini-2.5-flash**, or does the stronger tier
resist the way it resisted the system-flag contamination in clean-A (flash +0.03 vs flash-lite
+0.33)?

Same C design (two senior peers assert a plausible wrong answer; a generic "textbook" rationale vs
a case-anchored one), with the holdout set to flash and hard cases defined by flash (it gets the
clean question wrong). Selection is deterministic (first hard cases found in manifest order, up to
`--target`); the fixed parser reduces the hard-case count because fewer of flash's answers are
genuinely wrong once the stray-leading-"A" bug is gone.

## Result (28 flash hard cases, corrected parser)

| holdout | generic | anchored | anchored minus generic | McNemar p |
|---|---|---|---|---|
| gemini-2.5-flash-lite (n=85) | 0.729 | 0.847 | +0.118 | 0.041 |
| **gemini-2.5-flash (n=28)** | **0.714** | **0.679** | **-0.036** | **1.0** |

**Read.** Both tiers conform substantially to a bare confident peer assertion (flash 0.71, flash-lite
0.73, essentially the same baseline). The difference is what happens when the rationale becomes
case-anchored: flash-lite's conformity climbs a further +0.12 (p=0.041), while flash's does not move
(-0.04, n.s.). So the anchored-rationale *lever* is what is model-dependent, not conformity to a peer
per se: the stronger model is just as willing to defer to an assertion as the weaker one, but only
the weaker model can be pushed further by dressing the same wrong answer in a plausible rationale.
This is a more precise reading than "flash resists and flash-lite is susceptible": flash's resistance
is specifically to the *anchoring manipulation*, not to peer conformity in general.

## Reproduce

```bash
python -m experiments.model_dependence.cascade_C_flash \
    --manifest <medqa_manifest.csv> \
    --cache experiments/model_dependence/results/call_cache.jsonl \
    --out experiments/model_dependence/results --target 60 \
    --scale-c-summary <path to scale_c_summary.json, PR #141>
```

A fully cached run reproduces `results/cascade_C_flash_summary.json` with **zero API calls and no
key** (verified: `new_api_calls_this_run = 0`); selection is deterministic (manifest order). The
`flash_lite_reference` block is read from `--scale-c-summary` rather than hardcoded, so it tracks
scale_c's own committed numbers instead of drifting out of sync with a future fix there.

## Files

- `cascade_C_flash.py`, the runner (generic vs anchored conformity with the flash holdout + McNemar;
  reads the flash-lite reference from scale_c's committed summary).
- `results/cascade_C_flash_summary.json`, the scored summary.
- `results/cascade_C_flash.jsonl`, per-case rows (generic / anchored conformity per hard case).
- `results/call_cache.jsonl`, raw model calls, so every number reproduces offline.
