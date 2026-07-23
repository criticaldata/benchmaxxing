# MedQA (Lane B) full run: same-lineage Gemini arm

- Started 2026-07-20 00:45:30; git 3273c011 (feat/adapters-runners)
- 100 solo cases, 20 cascade cases (seed 0)
- Models gemini-2.5-flash, gemini-2.5-flash-lite; cues longest_option, option_order, lexical_overlap

## Solo susceptibility (flip rate, and flip above the noise floor)
- **gemini-2.5-flash**: raw flip 0.787, noise floor 0.000, flip-above-noise +0.787
- **gemini-2.5-flash-lite**: raw flip 0.893, noise floor 0.000, flip-above-noise +0.893

### per model x cue flip rate
| model | lexical_overlap | longest_option | option_order |
|---|---|---|---|
| gemini-2.5-flash | 0.74 | 0.74 | 0.88 |
| gemini-2.5-flash-lite | 0.85 | 0.84 | 0.99 |

## Same-lineage overlap (control)
- within-phi 0.5581129694273874, p nan (cross term undefined with a single lineage; add the open-weights arm for a real cross value)

## Cascade (seeded shortcut, shared vs isolated)
- onset detected in 4/20 cases
- mean contagion (shared shortcut share minus isolated): **-0.005555555555555552**

_Everything here is recoverable: `call_cache.jsonl` holds every model call, so re-running this script reproduces the results with no new API calls; transcripts/ replays each cascade offline._

## Findings and caveats (read before citing any number)

**Solo susceptibility is solid.** Both tiers are fully self-consistent at temperature 0
(noise floor 0.000 on the control), so the flip rates are genuine cue effects, not
instability. option_order is the strongest cue (flash 0.88, flash-lite 0.99). flip-above-noise
equals raw flip. This is the headline, defensible result for the text lane.

**The cascade contagion number (-0.006) is NOT interpretable yet.** Root cause found by
inspecting transcripts: the seeded "shortcut" answer (`experiments._shortcut_answer`, the
first distractor) coincides with the committee's own independent baseline answer in **16/20
cases** (the pre-seed turn-0 answer already equals the seed). When the seed equals the
baseline there is no counterfactual gap, so shared and isolated both sit at 1.0 and contagion
is undefined. Even on the 4 cases where the seed differs, shared == isolated, so no contagion
is visible in this same-lineage Gemini MCQ setup. The pilot's earlier +0.19 was small-sample
noise.

**Fix (sharpens issue #104):** the seed must be chosen RELATIVE TO each committee's isolated
baseline, a wrong answer the committee would not independently give, not just "the first
distractor". Only then does adopting it under shared context measure contagion. Until that
lands, treat the cascade arm as not-yet-measured; the solo arm stands on its own.
