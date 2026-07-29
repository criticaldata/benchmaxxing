# Imaging experiments (Lane A): NIH ChestX-ray14

Manifest build (#162), single-model shortcut susceptibility (#163), an imaging-lane cascade
(#164), a referee-as-gate evaluation (#165), the cascade on the remaining cues (#167), a
same-lineage judge referee (#168), multi-round cascade dynamics (#169), and a blind-metric
substitution probe (#170), on real chest X-rays via multimodal Gemini. Runs anywhere: the compute
is API-side, no local GPU. Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from the env; every
deterministic (temperature 0) step below reproduces the committed numbers with
`new_api_calls_this_run: 0` and no key.

## Manifest build (`build_manifest.py`, #162)

Parses the real ChestX-ray14 release via the registered `nih_cxr14` adapter
(`benchmaxxing/datasets/nih_cxr14.py`), keeps only cases whose image resolves on disk and that
carry a real finding label, and records a sha256 checksum per kept image
(`results/manifest_provenance.json`) so the exact pixels used are auditable.

```bash
python -m experiments.imaging.build_manifest \
    --raw-root <Data_Entry_2017.csv or its directory> --image-root <images/> \
    --case-ids-file experiments/imaging/results/case_ids_used.txt \
    --out experiments/imaging/results/nih_manifest.csv \
    --provenance experiments/imaging/results/manifest_provenance.json
```

`results/case_ids_used.txt` pins the exact 35-case set the results below were computed on, so a
fresh manifest build reproduces the identical cases (omit `--case-ids-file` to instead sample the
first `--n` eligible cases from the release for a new experiment).

## Solo shortcut susceptibility (`imaging_solo.py`, #163)

A diagnosis-neutral cue (cable / corner-tag / watermark / laterality) is injected onto the image
and we measure whether the model's yes/no read of the finding flips.

```bash
python -m experiments.imaging.imaging_solo --manifest results/nih_manifest.csv \
    --image-root <images/> --cache experiments/imaging/results/img_cache.jsonl \
    --out experiments/imaging/results --n 35
```

### Result (NIH ChestX-ray14, 35 cases, gemini-2.5-flash)

| cue | flip rate | flip-above-noise |
|---|---|---|
| watermark | 0.34 | **+0.11** |
| laterality | 0.26 | +0.03 |
| corner tag | 0.23 | 0.00 |
| cable | 0.20 | -0.03 |

Noise floor (each clean read resampled once at temperature 1.0, cache bypassed, 35 cases) = **0.23**,
in `imaging_solo_summary.json` with the raw resamples in `imaging_noise_floor.jsonl`. Only the
**watermark** cue clearly exceeds sampling noise; laterality is marginal and corner-tag / cable fall
at or below the floor at n=35. The noise floor is the one uncached step (a genuine temperature>0
resample) and needs a key; the deterministic flip pass itself reproduces from the cache with none.

## Imaging-lane cascade (`imaging_cascade.py`, #164) - a confident wrong peer

Beyond the solo cue, does a multimodal *committee* adopt a wrong finding-read when peers assert it?
On a watermarked X-ray, when two peer radiologists both read the film as the wrong answer, the
holdout adopts that wrong read far more under the shared board than reading alone:

```bash
python -m experiments.imaging.imaging_cascade \
    --manifest results/nih_manifest.csv --image-root <images/> \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results --n 35
```

| setting | shared adopt | isolated adopt | contagion |
|---|---|---|---|
| watermark (NIH, 35 cases) | **0.97** | **0.43** | **+0.54** |

> **Plant-direction provenance (#332/#333/#338).** Every cascade number on this page is derived from
> the post-#338 transcripts, where the planted peer read is a constant `no` against an all-positive
> cohort. The earlier design planted the flip of the holdout's own clean read, which coincided with
> ground truth on the cases the holdout already read wrong. Regenerate the derived analyses with
> `python -m experiments.imaging.recompute_derived --check` (no API key needed); CI fails if any
> committed value drifts from the transcripts.
>
> One file is knowingly NOT regenerated: `results/imaging_referee.jsonl` predates the fix and 13 of
> its 35 rows describe boards the corrected design no longer produces. Those rows cannot be rebuilt
> by arithmetic, only by fresh referee re-read calls. Until that run happens, the citable referee
> figures come from the 22-row subgroup where both designs agree, via
> `python -m experiments.imaging.referee_valid_subgroup` (precision 0.92, recall 0.86, false-positive
> rate 0.13, against the naive gate's 0.88).

The cascade transfers to the imaging lane and is *stronger* there than in text: a confident
plausible peer moves an imaging committee almost completely. Uses a yes/no parser, so it is
unaffected by the MCQ answer-parser bug.

### Cascade on the remaining cues (#167)

`imaging_cascade.py` takes a `--cue {cable,corner_tag,watermark,laterality}` flag (default
`watermark`, matching the original run above). Ran the other three:

```bash
python -m experiments.imaging.imaging_cascade --manifest results/nih_manifest.csv \
    --image-root <images/> --cache experiments/imaging/results/img_cache.jsonl \
    --out experiments/imaging/results --n 35 --cue cable
```

| cue | shared adopt | isolated adopt | contagion |
|---|---|---|---|
| cable | 1.0 | 0.46 | **+0.54** |
| corner tag | 0.97 | 0.43 | **+0.54** |
| laterality | 0.97 | 0.40 | **+0.57** |
| watermark | 0.97 | 0.43 | +0.54 |

Striking finding: contagion is strong across **all four** cues, including the three that barely
moved the model in isolation (#163's flip-above-noise was ~0 or negative for cable/corner-tag,
+0.03 for laterality). The cascade does not depend on the cue being independently potent; a
confident peer assertion moves the committee regardless of which visual cue is present. The
shortcut here is social plausibility, not the strength of the underlying visual artifact.

## Majority-pressure (Asch) variant (`imaging_majority_pressure.py`, #172)

Does imaging adoption scale with the number of peers asserting the wrong read, or saturate at
one? The imaging counterpart of #117 (text majority-pressure). `imaging_cascade.py` already
establishes the 0-peer (isolated) and 2-peer points; only the 1-peer condition is new here.

```bash
python -m experiments.imaging.imaging_majority_pressure \
    --manifest results/nih_manifest.csv --image-root <images/> \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results --n 35
```

| seeded peers | adoption |
|---|---|
| 0 (isolated) | 0.34 |
| 1 | **0.97** |
| 2 | 0.97 |

1-peer vs 2-peer, case-by-case: gain=0, lose=0 (identical on every one of the 35 cases, not just
matching in aggregate).

**Read.** Imaging adoption saturates at a single confident wrong peer: one radiologist's assertion
already produces the same near-total adoption as two, with zero cases differing between the two
conditions. Unlike a genuine Asch-style majority-pressure dose-response (where adoption would keep
climbing with more asserting peers), a single plausible authority voice is already sufficient in
this imaging committee - the social-plausibility mechanism saturates immediately rather than
building with numbers. 35 new API calls (the 0- and 2-peer points reuse already-committed data);
verified keyless afterward (0 new calls, identical numbers on re-run).

### Wrong-peer size curve, up to four peers (`imaging_peer_size_curve.py`, #216)

The size axis extended to four peers, confirming the saturation from a different direction than
#172's majority axis. The 1- and 2-peer boards are byte-identical to #172 / `imaging_cascade.py`,
so they hit the cache; only the 4-peer arm is new (35 calls; verified keyless afterward).

```bash
python -m experiments.imaging.imaging_peer_size_curve \
    --manifest results/nih_manifest.csv --image-root <images/> \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results --n 35
```

| wrong peers | 1 | 2 | 4 |
|---|---|---|---|
| adoption | 0.97 | 0.97 | 1.0 |

1-vs-2 McNemar gain=0/lose=0; 2-vs-4 gain=1/lose=0 (p=1.0). The curve is flat from one peer
onward: adding peers up to four recruits essentially no additional adoption, so the imaging
cascade is a single-peer effect, not a graded majority one, on both the majority axis (#172) and
the size axis (#216).

### Matched-temperature noise floor (`imaging_matched_temp_floor.py`, #205)

The solo `flip-above-noise` headline compared a temperature-0 cued flip ($0.34$) against a
temperature-1 clean-resample floor ($0.23$), a temperature mismatch. This completes the 2x2 by
adding the temperature-1 cued read so signal and floor are compared at the same temperature (the
temperature-0 floor is $0$ by construction, a deterministic test-retest cannot disagree). The
temperature-1 cued reads are stored in `imaging_matched_temp.jsonl` so re-runs reproduce keyless.

```bash
python -m experiments.imaging.imaging_matched_temp_floor \
    --manifest results/nih_manifest.csv --image-root <images/> --out experiments/imaging/results --n 35
```

| | temp-0 | temp-1 |
|---|---|---|
| clean-read floor | 0.0 | 0.23 |
| watermark cued flip | 0.34 | 0.29 |

**Read.** Matched at temperature 1, flip-above-noise is $0.29 - 0.23 = +0.057$, roughly half the
originally-reported (temperature-mismatched) $+0.11$; at temperature 0 the floor is $0$ so the
matched effect is the full $0.34$. So the watermark signal survives a same-temperature comparison
and the $+0.11$ headline was not purely a temperature artifact, but the honest matched effect at
temperature 1 ($+0.057$) is the conservative number and the imaging solo cue is even narrower than
first reported. 35 new API calls (one temperature-1 cued read per case); verified keyless.

## System-flag (authority) cascade (`imaging_system_flag.py`, #171)

Does a wrong AUTOMATED pre-screen flag move an imaging committee the way a wrong PEER assertion
does? The text lane's strongest break-it arm was A: a wrong system flag moved the committee
(flash +0.26, flash-lite +0.30), carrying system-authority rather than peer social pressure
(#136). Same watermark cue, same 35 cases, same yes/no parser as the peer cascade above; only the
board text changes (one system-flag sentence instead of two peer sentences).

```bash
python -m experiments.imaging.imaging_system_flag \
    --manifest results/nih_manifest.csv --image-root <images/> \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results --n 35
```

| board | shared adopt | isolated adopt | contagion |
|---|---|---|---|
| automated system flag (watermark, NIH, 35 cases) | 0.60 | 0.43 | +0.17 (McNemar p=0.070) |
| two-peer assertion (same cue, same cases, from above) | 0.97 | 0.43 | +0.54 |

Fisher exact, system-flag vs peer-assertion shared adoption (21/35 vs 34/35): p = 2.3e-4, a highly
significant, large divergence.

**Read.** Unlike the text lane, where a wrong system flag moves the committee nearly as much as a
wrong peer, the imaging lane shows much LESS contagion from a system flag alone (+0.17, not
significant at 0.05) against the same cue's strong peer-driven contagion (+0.54). Authority framing alone,
without a plausible peer voice actually asserting the read, does not move this imaging committee -
the social-plausibility mechanism identified in the peer cascade does not transfer to a bare
authority claim here, the opposite of what the text lane's break-it A would predict. New API
calls: 35 (the isolated condition reuses the already-cached clean/contaminated reads from
`imaging_cascade.py`, since the image and no-board prompt are identical); verified keyless
afterward (0 new calls on re-run).

## Referee-as-gate (`imaging_referee.py`, #165)

Does a deployable referee with no privileged knowledge catch the cascade's peer-driven adoptions?
For each case it sees only the transcript (the peers' wrong read plus the holdout's committee
answer) and issues one private re-query: an independent re-read of the same cued film with the peer
discussion withheld. It flags adoption when the committee answer matches the peers but the re-read
does not.

```bash
python -m experiments.imaging.imaging_referee \
    --manifest results/nih_manifest.csv --image-root <images/> \
    --cascade-jsonl experiments/imaging/results/imaging_cascade.jsonl \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results
```

Ground truth (evaluation only): a peer-driven adoption is a case adopted with peers but not alone
on the same cued image (19 of 35). Scored against that:

| Monitor | Precision | Recall | False-positive rate |
|---|---|---|---|
| Deployable referee (transcript + one re-read) | **0.88** | **0.79** | **0.125** |
| Naive conformity gate (flag any agreement) | 0.56 | 1.0 | 0.9375 |

The referee cuts false alarms from 0.94 to 0.13 while still catching most adoptions. It does not
reach the text lane's perfect separation (P/R 1.0): on images the cue itself can move the holdout
even without peers, so peer-driven and cue-driven flips are harder to tell apart.

## Same-lineage judge referee (`imaging_judge_referee.py`, #168)

Adapts the text lane's same-lineage judge (`experiments/referee/referee_judge.py`, #132): a judge
of the same Gemini lineage reads a text description of the shared deliberation only (the peers'
asserted read and the holdout's shared-board answer, no isolated-read comparison, no image) and
decides FLAG/OK.

```bash
python -m experiments.imaging.imaging_judge_referee \
    --cascade-jsonl experiments/imaging/results/imaging_cascade.jsonl \
    --cache experiments/imaging/results/judge_cache.jsonl --out experiments/imaging/results
```

| Monitor | Precision | Recall | False-positive rate |
|---|---|---|---|
| Same-lineage judge (shared transcript only) | 0.65 | 1.0 | 0.92 |
| Deployable referee (transcript + one re-read) | **0.86** | **0.86** | **0.23** |
| Naive conformity gate | 0.65 | 1.0 | 0.92 |

The judge's numbers are *identical* to the naive gate's. Without a fresh independent signal (the
deployable referee's private re-read) or a reasoning trace to inspect (imaging turns are a single
yes/no token, unlike the text lane's free-form rationale, which is what let the text-lane judge
beat its own naive gate, precision 1.0 vs the naive gate's 0.375), a same-lineage judge here has
nothing to work with beyond "did the answer match the peers", which is exactly the naive gate's
rule. In imaging, the private re-read is what earns the deployable referee's edge, not the
judge's oversight framing.

### Agreement/disagreement matrices (#184)

`referee_agreement.py` confirms the identity numerically, not just at the aggregate P/R level:
the judge's flag and the naive gate's flag are **exactly the same on every one of the 35 cases**
(phi = kappa = 1.0). The deployable referee diverges from both on the same 12 cases (phi = 0.223,
kappa = 0.095), all cases where naive/judge over-flag relative to the deployable referee's
stricter, re-read-backed criterion. Pure re-analysis, no API calls.

```bash
python -m experiments.imaging.referee_agreement
```

## Multi-round cascade dynamics (`imaging_multi_round.py`, #169)

Adapts the text lane's multi-round design (`experiments/cascade/multi_round.py`, #130): does
adoption build over repeated exposure, or is it decided immediately? Each round restates the
peers' assertion and reminds the holdout of its own prior-round answer, so each round is a
genuinely new prompt.

```bash
python -m experiments.imaging.imaging_multi_round --manifest results/nih_manifest.csv \
    --image-root <images/> --cache experiments/imaging/results/img_cache.jsonl \
    --out experiments/imaging/results --n 35 --rounds 3
```

| round | shared adoption | isolated adoption |
|---|---|---|
| 1 | 0.89 | 0.31 |
| 2 | 1.0 | 0.34 |
| 3 | 0.97 | 0.29 |

Round 1 vs round 3 (shared): 3 gained, 0 lost, McNemar p=0.25 (not significant at n=35). Shared
adoption is already high at round 1 and plateaus; isolated stays flat throughout. Consistent with
the text lane's finding (#130): adoption is largely decided immediately on first exposure, not a
slow build-up over rounds.

## Blind-metric substitution probe (`imaging_blind_metric.py`, #170)

Adapts the text lane's blind-metric probe (#18/#139) to the imaging yes/no lane, on the clean
(uncued) image so the effect is isolated from #163's image-cue susceptibility. The decoy is the
opposite of the model's own baseline read.

```bash
python -m experiments.imaging.imaging_blind_metric --manifest results/nih_manifest.csv \
    --image-root <images/> --cache experiments/imaging/results/img_cache.jsonl \
    --out experiments/imaging/results --n 35
```

| condition | decoy rate |
|---|---|
| baseline | 0.0 |
| blind (hidden rubric) | **0.83** |
| test-aware (primed) | 0.37 |

Decoy uptake +0.83, suppressed to +0.37 when primed that the rubric may be misaligned (delta
0.46). Unlike the text lane, where all eleven drifters named the rubric in their justification
(fully self-declared gaming), **0 of 29 drifters here mentioned the rubric at all** (verified
against the raw justification text, not a naming-regex artifact): every drifter gave a
confabulated, purely clinical-sounding justification for the rubric-preferred read. In the
imaging yes/no setting the gaming is silent, not self-declared, which is more concerning: a
monitor that only reads the justification text (as the text-lane finding suggested was
sufficient) would miss it entirely here.

### CI and per-finding re-analysis (#201)

`blind_metric_ci.py` adds Wilson 95% CIs to every proportion above (see
`results/imaging_blind_metric_ci.json`; naming rate 0.0 with the interval staying at or below
0.117 even at its upper bound, given 29 drifters) and an exploratory per-finding decoy-uptake
breakdown. Pure re-analysis, no API calls.

```bash
python -m experiments.imaging.blind_metric_ci
```

### Counterfactual and reasoning-reading referees (#175, imaging lane)

`counterfactual_referee.py` points two no-key referees at these blind-metric transcripts (every
prior referee ran on cascade transcripts instead). Pure re-analysis, zero API calls. Text-lane
half is `experiments/blind_metric/counterfactual_referee.py`.

```bash
python -m experiments.imaging.counterfactual_referee
```

**Read.** F1 (counterfactual, base vs blind) recovers drift ground truth exactly by construction
(P/R 1.0/1.0). F2 (reasoning-reading referee) recall = 0.0 (0/29) - the headline contrast this
issue predicted: a referee that only reads the justification text catches essentially all text-
lane drift (recall 1.0) but none of the imaging-lane drift (recall 0.0), confirming that only the
counterfactual re-query catches silent gaming here, not reading the reasoning.

### Effect sizes, bootstrap CIs, and achieved power (#192, imaging lane)

`effect_sizes_imaging.py` adds a risk difference, bootstrap 95% CI, and achieved
power/required-pairs to the cue-level solo flip rates, the four cascade contagion deltas, and the
referee-vs-naive comparison. Text-lane half of #192 is delivered separately in
`experiments/effect_sizes/` (PR #231).

```bash
python -m experiments.imaging.effect_sizes_imaging
```

| Arm | Risk difference | Bootstrap 95% CI | Achieved power | Pairs needed for 0.8 power |
|---|---|---|---|---|
| Cascade contagion, cable | 0.54 | [0.400, 0.714] | 1.00 | 12 |
| Cascade contagion, corner tag | 0.54 | [0.371, 0.714] | 1.00 | 12 |
| Cascade contagion, laterality | 0.57 | [0.400, 0.743] | 1.00 | 12 |
| Cascade contagion, watermark | 0.54 | [0.371, 0.714] | 1.00 | 12 |
| Referee vs naive (paired accuracy) | 0.17 | [-0.029, 0.343] | 0.41 | 90 |

Solo flip rates (bootstrap CIs, not a risk difference since there is only one condition): cable
0.20 [0.086, 0.343], corner tag 0.23 [0.086, 0.371], watermark 0.34 [0.20, 0.486], laterality 0.26
[0.114, 0.40].

**Read.** Every cue's cascade contagion delta is fully powered at n=35 (achieved power 1.0,
only 7-10 pairs would have sufficed) - the best-powered result in the project. The
referee-vs-naive comparison is the opposite case: a suggestive 0.17 risk difference and McNemar
b=9/c=3 asymmetry, but the bootstrap interval straddles 0 and achieved power is only 0.41 (90
cases needed, 35 available), so it should be read as directionally suggestive, not yet powered to
rule out chance. Pure zero-API re-analysis over already-committed `imaging_solo.jsonl`,
`imaging_cascade*.jsonl`, and `imaging_referee.jsonl`.

### Referee panel majority vote (#178, imaging lane)

`panel_vote.py` activates `benchmaxxing.referee_ablations.referee_panel_vote`/`single_vs_panel`
(previously unit-tested only with mock detectors) on real, reconstructed transcripts built from
the already-committed `imaging_referee.jsonl`/`imaging_judge_referee.jsonl` rows. Majority-votes
three genuinely independent, already-computed, no-key detectors - deployable (shared-vs-reread
disagreement), naive (shared matches the peer's asserted read), same-lineage judge - and compares
the panel to the single deployable referee alone. Text-lane half is
`experiments/referee/panel_vote.py`.

```bash
python -m experiments.imaging.panel_vote
```

| | Precision | Recall | F1 |
|---|---|---|---|
| Single (deployable alone) | 0.864 | 0.864 | 0.864 |
| Panel (majority of 3) | 0.647 | 1.0 | 0.786 |

**Read.** Unlike the text lane (where the panel exactly reproduces the single referee's
already-perfect scores), imaging's panel trades precision for recall: naive alone already flags
34 of 35 cases and the judge tracks it closely, so a 2-of-3 majority is pulled toward those two
high-recall/low-precision voters, and F1 drops from 0.864 to 0.786. Here a majority-vote panel
measurably hurts the deployable referee rather than being redundant with it - the opposite lesson
from the text lane. Pure zero-API re-analysis; a parity check verifies the reconstructed
transcripts reproduce the committed `naive_flag`/`ref_flag` fields exactly before any scoring
happens (this caught a wrong assumption during development: imaging's naive gate is a
peer-vs-holdout match, not the text lane's peer-vs-peer-only gate).

### Misaligned-proxy arm (#176, imaging lane)

`misaligned_proxy.py` activates `benchmaxxing.blind_arms.misaligned_proxy_run` (built and
unit-tested with mocks only, invoked by no data runner) on `imaging_solo.jsonl`, reshaped to one
row per (case, cue) pair (n=140) so "is this specifically the watermark cue" has real variance.
Proxy = watermark-present indicator; truth = resulting read correctness (ground truth is always
"yes" on this manifest); decision = whether the read flipped from the case's own clean read.
Text-lane half is `experiments/contamination/misaligned_proxy.py`.

```bash
python -m experiments.imaging.misaligned_proxy
```

| corr(flip, is-watermark) | corr(flip, correct-read) | uptake_delta |
|---|---|---|
| 0.113 | -0.217 | 0.330 |

**Read.** Watermark is the strongest single cue (#163: flip rate 0.34 vs 0.20-0.26 for the
others), so being the watermark row does weakly predict a flip, and flip anti-correlates with
correctness (weak-to-moderate, not near-perfect, since a flip away from the always-"yes" ground
truth is usually but not always a wrong read). Both correlations are modest, so the positive
uptake_delta should be read as a small, genuine tilt toward the watermark-specific surrogate over
correctness, not a strong effect in either direction. Pure zero-API re-analysis.

### Onset battery: contagion_index + deference_rate (#174, part b, imaging lane)

`onset_battery.py` activates `benchmaxxing.onset.contagion_index`/`deference_rate` (the plan's
flagship reusable metrics, exported and unit-tested but never reported on committed data) on
`imaging_cascade*.jsonl`, treating each case's single holdout as one population member (n=35 per
cue). Text-lane part (a) is `experiments/medqa/onset_distribution.py`.

```bash
python -m experiments.imaging.onset_battery
```

| cue | contagion_index | deference_rate |
|---|---|---|
| cable | 0.80 | 1.00 |
| corner tag | 0.76 | 0.95 |
| laterality | 0.74 | 0.95 |
| watermark | 0.65 | 0.95 |

**Read.** contagion_index ranges 0.65-0.80: a majority but not the entirety of shared-condition
adoptions would not have happened alone, so most adoption is attributable to the peer board, with
a real minority (roughly a fifth to a third) that would have flipped solo regardless - directionally
consistent with #185's case-driven story but a distinct, less extreme quantity, not a restatement
of Spearman rho=-1.0. deference_rate is high for every cue (0.95-1.0): nearly every solo-correct
case still abandons its correct read once the board asserts the wrong one, matching #177's
harm/rescue decomposition (harm rate 0.95-1.0) more closely than contagion_index does. Pure
zero-API re-analysis.

### Descriptive positional regression (#174, part c, imaging lane, K=3)

`positional_regression.py` fits `benchmaxxing.stats.mixed_effects_logit` (adoption ~ round_index,
random intercept per case) on `imaging_multi_round.jsonl`, pooling all 105 (case, round)
observations per arm. Explicitly descriptive, not causal: round order was fixed, never randomized
or counterbalanced, so any coefficient is confounded with anything else correlated with position
in a fixed sequence. Text-lane half (K=5) is `experiments/cascade/positional_regression.py`.

```bash
python -m experiments.imaging.positional_regression
```

**Read.** The shared arm's fitted coefficient (1.49) predicts a monotonic climb (round 0/1/2:
~0.94/0.99/1.00), but the actual empirical adoption is NOT monotonic - 0.89, 1.00, 0.97 (up then
slightly down, per #169's README). The large coefficient is an artifact of forcing a straight-line
log-odds fit onto a saturating, non-monotonic curve, not evidence of a genuine escalating-exposure
effect; verified by comparing the model's own fitted probabilities against the raw empirical rates
before trusting the coefficient. The isolated arm's coefficient (-0.53) is smaller and closer to
the already-reported flat pattern. The position-confounded caveat applies to both regardless of
this fit-quality issue.

## Files

- `build_manifest.py`, `imaging_solo.py`, `imaging_cascade.py`, `imaging_system_flag.py`,
  `imaging_majority_pressure.py`, `imaging_referee.py`, `imaging_judge_referee.py`,
  `imaging_multi_round.py`, `imaging_blind_metric.py` - the nine runners.
- `results/nih_manifest.csv` - the 35-case manifest (built via the `nih_cxr14` adapter).
- `results/manifest_provenance.json` - a sha256 checksum per used image.
- `results/case_ids_used.txt` - the exact case_id set the committed results were computed on.
- `results/img_cache.jsonl` - the shared committed model-response cache for the image-calling runners.
- `results/judge_cache.jsonl` - the committed cache for the text-only judge referee.
- `results/imaging_solo.jsonl` / `_summary.json`, `results/imaging_noise_floor.jsonl` - solo susceptibility.
- `results/imaging_cascade.jsonl` / `_summary.json` - the watermark cascade (default cue).
- `results/imaging_system_flag.jsonl` / `_summary.json` - #171's system-flag (authority) cascade.
- `results/imaging_majority_pressure.jsonl` / `_summary.json` - #172's majority-pressure variant.
- `imaging_peer_size_curve.py`, `results/imaging_peer_size_curve*` - #216's wrong-peer size curve (up to 4).
- `imaging_matched_temp_floor.py`, `results/imaging_matched_temp*` - #205's matched-temperature noise floor.
- `results/imaging_cascade_{cable,corner_tag,laterality}.jsonl` / `_summary.json` - the other three cues.
- `results/imaging_referee.jsonl` / `_summary.json` - the deployable referee-as-gate evaluation.
- `results/imaging_judge_referee.jsonl` / `_summary.json` - the same-lineage judge referee.
- `results/imaging_multi_round.jsonl` / `_summary.json` - multi-round cascade dynamics.
- `results/imaging_blind_metric.jsonl` / `_summary.json` - the blind-metric substitution probe.
- `effect_sizes_imaging.py`, `results/effect_sizes_imaging.json` - #192's imaging-lane effect sizes.
- `panel_vote.py`, `results/panel_vote.json` - #178's referee panel majority vote.
- `misaligned_proxy.py`, `results/misaligned_proxy.json` - #176's imaging-lane misaligned-proxy arm.
- `onset_battery.py`, `results/onset_battery.json` - #174 part b, contagion_index/deference_rate.
- `positional_regression.py`, `results/positional_regression.json` - #174 part c, K=3 regression.
- `counterfactual_referee.py`, `results/counterfactual_referee.json` - #175's imaging-lane referees.
