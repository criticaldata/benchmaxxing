# Model-dependence of the plausibility cascade (C)

The anchored-vs-generic plausibility cascade moves the weak holdout **gemini-2.5-flash-lite** from
0.33 to 0.51 conformity (paired McNemar p<1e-4 at n=150). Does it also move the stronger
**gemini-2.5-flash**, or does the stronger tier resist, the way it resisted the system-flag
contamination in clean-A (flash +0.03 vs flash-lite +0.33)?

Same C design (two senior peers assert a plausible wrong answer; a generic "textbook" rationale vs
a case-anchored one), with the holdout set to flash and hard cases defined by flash (it gets the
clean question wrong). Selection is deterministic (first 60 hard cases in manifest order).

## Result (60 flash hard cases)

| holdout | generic | anchored | anchored − generic | McNemar p |
|---|---|---|---|---|
| gemini-2.5-flash-lite (n=150) | 0.33 | 0.51 | +0.18 | <1e-4 |
| **gemini-2.5-flash (n=60)** | **0.10** | **0.083** | **-0.017** | **1.0** |

**Read.** The plausibility lever does **not** move the stronger model: anchored conformity (0.083)
is statistically indistinguishable from generic (0.10), and both are far below the weak model's
rates. So susceptibility to a plausible shortcut is concentrated in the weaker tier, exactly as the
system-flag contamination was in clean-A. Across both channels, the stronger model resists and the
weaker model is moved; the weaker the model, the more the shortcut spreads.

## Reproduce

```bash
python -m experiments.model_dependence.cascade_C_flash \
    --manifest <medqa_manifest.csv> \
    --cache experiments/model_dependence/results/call_cache.jsonl \
    --out experiments/model_dependence/results --target 60
```

A fully cached run reproduces `results/cascade_C_flash_summary.json` with **zero API calls and no
key** (verified: `new_api_calls_this_run = 0`); selection is deterministic (manifest order).

## Files

- `cascade_C_flash.py`, the runner (generic vs anchored conformity with the flash holdout + McNemar).
- `results/cascade_C_flash_summary.json`, the scored summary.
- `results/cascade_C_flash.jsonl`, per-case rows (generic / anchored conformity per hard case).
- `results/call_cache.jsonl`, raw model calls, so every number reproduces offline.
