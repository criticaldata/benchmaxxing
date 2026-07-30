# benchmaxxing MedMCQA (Lane B) results

Standard-battery replication on MedMCQA, mirroring the MedQA/NIH template so results are
apples-to-apples across datasets. Dataset: MedMCQA validation split (the labeled split; the test
split has no public answers). Models: gemini-2.5-flash, gemini-2.5-flash-lite (same-lineage
control arm). Temperature 0. Model calls are cached locally in `call_cache.jsonl` (gitignored,
not committed - it is large and just a reproduction accelerator); the committed evidence is the
per-model/per-case records, the summaries, and the transcripts below.

Harness: `experiments/medqa/reproduce.py`, unchanged. It is manifest-driven (`load_cases`), so
the only per-dataset input is the manifest built by the `medmcqa` adapter. No experiment code was
modified.

## Provenance

- Source: `openlifescienceai/medmcqa` validation split via the HuggingFace datasets-server.
- 500 rows downloaded; manifest built by `benchmaxxing/datasets/medmcqa.py` and validated clean.
  Answer indices cross-checked against the raw `cop` labels: 0 mismatches.
- Raw data and the manifest live under `data/` (gitignored); only derived results are committed.
- Solo run: `--stage solo --solo-n 100 --seed 0` (100 cases x 2 models x 3 cues = 600 records).

## 1. Solo susceptibility (issue #267)

Flip = the model's answer changes between the clean twin and the cue-injected twin (same
question, one cosmetic cue added, ground truth unchanged). Noise floor = self-inconsistency when
the same clean case is run twice with no cue.

| model | clean acc | overall flip (n=100) | noise floor (n=15) |
|---|---|---|---|
| gemini-2.5-flash | 0.830 | 0.120 | 1/15 = 0.067, Wilson 95% [0.012, 0.298] |
| gemini-2.5-flash-lite | 0.770 | 0.237 | 0/15 = 0.000, Wilson 95% [0.000, 0.204] |

The noise floor is measured on only the first 15 cases per model, so it is one flip (flash) and
zero flips (flash-lite): Fisher p=1.0 between the tiers, and the Wilson intervals are far too wide
to subtract from the n=100 flip rate. A "flip-above-noise" figure would difference two quantities
measured at different n (15 vs 100), which is not commensurable, so it is deliberately not reported
here; the floor should be re-run at n=100 before any noise-adjusted susceptibility is read off it.

### per model x cue flip rate
| model | lexical_overlap | longest_option | option_order |
|---|---|---|---|
| gemini-2.5-flash | 0.15 | 0.11 | 0.10 |
| gemini-2.5-flash-lite | 0.27 | 0.26 | 0.18 |

**Read.** The weaker tier (flash-lite) is about twice as cue-susceptible as flash, and the gap is
paired-solid rather than a floor artifact: recomputed from `solo_records.jsonl`, flash-lite flips on
46 case-cue pairs where flash does not, against 11 the other way (McNemar p=3.3e-06). The earlier
reading that "much of flash's apparent susceptibility is decoding nondeterminism" rested on a single
flip in the 15-case floor and is withdrawn until the floor is measured at the same n=100. Lexical
overlap is the strongest cue for both tiers.

Lineage overlap: within-lineage phi 0.398 (the two Gemini tiers fail on moderately overlapping
cases). The cross-lineage arm is undefined here (`nan`) because both models are the same lineage;
it needs an open-weights model (#134).

## 2. Cascade baseline (issue #268)

Shared vs isolated committee (3 Gemini agents), with a baseline-relative planted shortcut (the
seed differs from each committee's own isolated answer, so contagion = shared adoption minus
isolated adoption is well defined). `--cascade-n 20 --seed 0`.

| metric | value |
|---|---|
| mean contagion (shared - isolated) | +0.025 |
| mean shared adoption | 0.025 |
| mean isolated adoption | 0.000 |
| cases with any adoption | 1 / 20 |

**Read: the bare cascade is null on MedMCQA**, as it is on MedQA. On average the shared board
moved 2.5% of non-seed agents onto the planted wrong answer; 19 of 20 committees rejected the
seed outright. Verified against the transcripts: the seed turn is injected in every run, and in
the null cases the committee holds the correct answer against it (e.g. all three agents kept
'20 mg doxycycline' against a planted '60 mg doxycycline'); the one positive case is a genuine
adoption. This is the documented finding that a bare planted answer is too weak a stimulus, the
motivation for the break-it arms (#288 system flag, #289 anchored plausible seed) which is where
a measurable cascade is expected.

MedQA reference (committed `cascade_v2_summary.json` / `cascade_v2_per_case.jsonl`):
mean contagion -0.05, shared 0.025, isolated 0.075 - also null. Both datasets agree the bare
cascade does not form; shared adoption is identical (0.025) across the two.

## 3. Break-it A: contaminated shared context / system pre-screen flag (issue #288)

An "AUTOMATED PRE-SCREEN SYSTEM" (not a peer) asserts a wrong answer chosen to differ from the
model's own baseline, so any adoption is genuinely flag-driven. Control re-asks the clean question
with no flag. Script: `experiments/medqa/clean_a.py`, unchanged. 30 hard cases (holdout wrong on
the clean question), the pool from the n=100 solo run.

| model | flag adoption | control | effect |
|---|---|---|---|
| gemini-2.5-flash | 0.267 | 0.000 | +0.267 |
| gemini-2.5-flash-lite | 0.533 | 0.000 | +0.533 |

**Read: a trusted-system flag drives a large cascade on MedMCQA, larger than on MedQA.** Both
tiers adopt the flagged wrong answer well above the 0.000 control; flash-lite over half the time.
Verified against cached responses: the model switches from its own baseline to the flagged answer
(e.g. '16 Lp/mm' -> flagged '10 Lp/mm'). MedQA reference (clean_a, 60 hard cases): flash-lite
0.00 -> 0.33, flash +0.03. So the effect replicates and is stronger here, consistent with the
higher authority-deference seen in break-it C. n=30 is modest (the effect is large enough to be
clear regardless); it would tighten with a larger hard-case pool.

## 4. Break-it C: plausibility dose-response (issue #289)

Where the bare cascade is null, the break-it arms vary the mechanism. C is the plausibility
dose-response: a holdout (gemini-2.5-flash-lite) that got the clean question wrong sees two
"senior attending physicians" assert a specific wrong answer under conditions of increasing
plausibility, and we measure how often it adopts that answer. Script: `experiments/medqa/scale_c.py`,
unchanged. 110 hard cases.

Operational note: the live run made all its API calls but hung at write time (a Gemini call with
no client-side timeout stalled a worker after the last case). These numbers were recomputed offline
from the local `call_cache.jsonl` with scale_c's own parser, over the 110 hard cases whose full
condition set was cached, and are committed as `scale_c_per_case.jsonl` (per-case condition
booleans) and `scale_c_summary.json` (which carries the same note). The local cache is not
committed, but those two artifacts are the evidence.

| condition | adoption | Wilson 95% | vs generic (McNemar) |
|---|---|---|---|
| generic | 0.718 | 0.63-0.79 | - |
| anchored | 0.818 | 0.74-0.88 | +0.10, p=0.019 |
| anchored_strong | 0.764 | 0.68-0.83 | +0.05, p=0.40 |
| anchored_solo (one peer) | 0.718 | 0.63-0.79 | - |

**Read: the anchoring effect replicates in direction and is significant** (anchored beats generic
by +0.10, McNemar p=0.019), so plausibility drives the cascade on MedMCQA as it does on MedQA.
But two honest differences from the MedQA reference (n=150: generic 0.33 -> anchored 0.51 ->
strong 0.61, both p<1e-4):

1. Baseline conformity is much higher on MedMCQA (generic 0.72 vs 0.33). On hard cases,
   flash-lite defers to two asserted senior peers about three-quarters of the time even with a
   generic rationale. Verified against cached responses: the holdout genuinely switches from its
   own baseline answer to the seeded one (e.g. 'Enamel lamellae' -> seeded 'Enamel spindles').
2. The dose-response is not monotone here: "strongly anchored" (the longer 30-word rationale) did
   not beat "anchored" (p=0.40), plausibly a ceiling effect given the 0.72 baseline leaves little
   headroom. On MedQA the gradient was monotone.

So the cascade-under-plausible-authority effect does replicate on MedMCQA (significant anchoring
gain), but the shape differs: a higher floor and a flatter top.

## 5. Break-it D: blind-metric decoy (issue #290)

A hidden rubric secretly rewards the longest option; we measure drift toward that decoy under the
incentive vs a control, over 13 hard cases (holdout wrong on the clean question). Script:
`experiments/medqa/break_it.py` (D arm), unchanged.

| condition | picks the decoy (longest option) | accuracy |
|---|---|---|
| control | 0.308 | 0.385 |
| blind-metric incentive | 0.308 | 0.462 |

**Read: null on MedMCQA.** The hidden longest-option incentive produces zero decoy drift (0.308
either way); the model does not shift toward the decoy when the rubric secretly rewards it. Unlike
break-it A (system flag) and C (anchored plausible seed), this weaker, non-social manipulation does
not move the model. n=13 is small, but the point estimate is a flat 0.0 drift.

## 6. Committee-structure battery, corrected as one family (issue #237 analog)

Eight headline adoption-rate contrasts across the committee-structure arms, folded into one
multiple-comparison family (`stats_reconciliation_summary.json`): each is an adoption-rate delta
with a paired bootstrap 95% CI and a paired McNemar p, corrected together with BH and Holm. Only
**non-degenerate** contrasts are in the family. A McNemar against a structurally-constant baseline
is a one-sided test against a constant, not a paired comparison, so three such baselines are
excluded: control adoption is 0 by construction (the seed is chosen != the holdout's own answer),
the isolated 0-peer arm is likewise 0, and the wrong orchestrator's forced answer *is* the committee
output so it is 1. Those effects enter through their non-degenerate rungs (guideline vs colleague,
2-peer vs 1-peer, wrong-peer vs honest-orchestrator), and the constant baselines are reported
descriptively in §9.

| contrast | delta | 95% CI | McNemar p | Holm |
|---|---|---|---|---|
| authority_ladder: guideline vs colleague | +0.717 | [0.600, 0.833] | ~0 | ✓ |
| unanimity: unanimous vs with-dissenter | +0.395 | [0.237, 0.553] | 6.1e-05 | ✓ |
| rationale_validity: any-reasoning vs bare | -0.342 | [-0.442, -0.250] | ~0 | ✓ |
| seed_confidence: confident vs hedged | +0.260 | [0.180, 0.350] | 3e-08 | ✓ |
| super_additivity: both vs peer-alone | +0.108 | [0.033, 0.192] | 0.015 | . |
| orchestrator: wrong-peer vs honest-orch over 2 wrong | +0.097 | [-0.032, 0.210] | 0.21 | . |
| true_peer: correct vs wrong peer | +0.333 | [0.000, 0.667] | 0.125 | . |
| majority_pressure: 2-peer vs 1-peer | +0.040 | [-0.120, 0.200] | 1.0 | . |

**Read: 4 of 8 survive Holm** (the strictest correction). The robust findings are the authority-rung
effect (a clinical-guideline citation is adopted +0.72 over a colleague, the ladder in §9), the
dissenter break (a single ally cuts adoption by 0.40), the reasoning-collapse (any rationale drops
adoption by 0.34, see §7), and confidence elasticity (+0.26). Super-additivity (+0.11, p=0.015)
survives BH but not Holm once the family is restricted to real contrasts. The rest are null or
underpowered: measured properly, a wrong 2-of-3 majority is no worse than one wrong peer (+0.04,
p=1.0), and a lone wrong peer poisons the output no more than an honest orchestrator gating two
wrong peers (+0.10, p=0.21). The earlier "+0.92 authority vs control", "+0.71 orchestrator vs
wrong-peer" and "+0.16 majority vs isolated" were contrasts against constants and are withdrawn.

## 7. Seed shape (issues #269-#274)

How the shape of the planted wrong answer changes adoption, holding the source (a wrong senior
peer) fixed. Holdout gemini-2.5-flash-lite on solo-wrong cases.

| arm | conditions (adoption) | finding |
|---|---|---|
| seed_confidence (#271) | hedged 0.18 -> confident 0.44 | confidence elasticity +0.26; a confident wrong peer is adopted ~2.4x more than a hedged one |
| rationale_validity (#270) | bare 0.71 -> plausible-wrong 0.37 -> fallacious 0.39 | **attaching ANY reasoning collapses adoption** (both vs bare p<1e-9); the two reasoned arms are indistinguishable |
| plausible_distractor (#269) | implausible 0.74 -> plausible 0.82 | discernment-gated but weak (+0.08, p=0.15) |
| dose_response (#273) | faint 0.21 -> lean 0.46 -> plain 0.77 -> emphatic 0.55 | graded, but plateaus/drops at emphatic certainty |
| text_cue_types (#274) | baseline 0.74; primacy 0.57; negation 0.93; qualifier 0.75 | negation ("rules out your own pick") is the strongest cue; primacy lowers adoption |
| attributed_tier (#272) | small-model 0.42 -> unlabeled 0.84 -> SOTA-model 0.72 -> attending 0.81 | swayed by the STATED capability of the source, not the content |

**Read.** The counterintuitive one is rationale_validity: a bare "the answer is X" from a senior is
adopted 0.71 of the time, but *any* attached reasoning, valid-looking or openly fallacious, cuts
that to ~0.38. On these MCQ cases the holdout scrutinizes stated reasoning (and finds it wanting)
but defers to bare authority. Confidence and stated source-capability both move adoption in the
expected direction; plausibility of the specific distractor barely does.

## 8. Committee size & social pressure (issues #275-#279)

| arm | conditions (adoption) | finding |
|---|---|---|
| committee_size_sweep (#276) | alone 0.74 -> +1 honest 0.27 -> +2 0.19 -> +4 0.18 | an honest majority DILUTES a single wrong seed (safety in numbers), monotone |
| unanimity_break (#277) | unanimous-2-wrong 0.42 -> one correct dissenter 0.03 | a single ally breaks conformity (-0.39), Asch-consistent |
| deliberation_framing (#278) | none 0.74 -> collaborative 0.56 -> independent 0.37 -> critical 0.28 | dissent-licensing framing is a cheap prompt-time lever |
| majority_pressure (#275) | isolated 0.0 (by construction) -> 1-peer 0.16 -> 2-peer 0.20 | the real 2-vs-1-peer contrast is a **null**: +0.04, McNemar p=1.0 (isolated is 0 by construction, so "vs isolated" is not a valid test) |
| super_additivity (#279) | flag+peer +0.11 over peer alone (p=0.015, BH but not Holm) | the 2x2 interaction is sub-additive (both minus the sum of singles = -0.59): the two overlap, they don't stack |

**Read.** Numbers protect: a lone wrong seed is diluted from 0.74 to 0.18 as honest peers
accumulate, and a single dissenter is even more effective than dilution (0.42 -> 0.03).
Deliberation framing that licenses dissent ("think independently", "be critical") drops adoption to
0.28 with one instruction line, a deployment-time mitigation. Two arms come out weaker than the
first pass suggested once measured against a real (non-constant) baseline: a wrong 2-of-3 majority
is no more persuasive than a single wrong peer (+0.04, p=1.0), and the flag+peer super-additivity
survives BH but not Holm, and is sub-additive besides.

## 9. Hierarchy & orchestrator (issues #280-#283)

| arm | result | finding |
|---|---|---|
| authority_ladder (#282) | colleague 0.20 < automated 0.70 < senior 0.72 < guideline 0.92 (control 0 by construction) | adoption rises with the authority rung; the tested contrast is guideline vs colleague (+0.72, survives Holm in §6) |
| leader_as_auditor (#283) | as peer 0.74 -> as auditor 0.19 -> as attending-of-record 0.49 | re-pointing the holdout to an oversight role is a cheap remediation |
| orchestrator_failure (#281) | wrong peer poisons 0.29; honest-orch over 2 wrong peers 0.19; wrong-orch 1.0 **by construction** | the "1.0" is structural (the orchestrator's forced answer IS the output), not a finding; the real contrast (wrong-peer vs honest-orch) is +0.10, p=0.21, **n.s.** |
| hierarchy_dominance (#280) | dominance rate 1.0 (degenerate) | **methodological null**: at temp 0 the committee converges to unanimity, so every agent "dominates" and the metric is a tie-break artifact |

**Read.** Authority is the real result here: adoption climbs from a colleague (0.20) to a
clinical-guideline citation (0.92), and guideline-vs-colleague is the non-degenerate contrast that
survives Holm in §6. The orchestrator arm does NOT show a single-point-of-failure once measured
honestly: the "wrong orchestrator poisons 1.0" is a tautology (its forced answer is defined to be
the output), and the valid comparison -- does a lone wrong peer poison more than an honest
orchestrator synthesizing over two wrong peers -- is not significant (+0.10, p=0.21). Two earlier
headline claims (orchestrator single-point-of-failure, hierarchy dominance) are withdrawn as
construction artifacts.

## 10. Controls & robustness (issues #284-#287)

| arm | result | finding |
|---|---|---|
| true_peer_control (#284) | wrong peer 0.20 vs correct peer 0.53 | negative control passes: a *correct* peer is adopted more than a wrong one, so the deference is not blind conformity |
| test_awareness (#285) | neutral 0.74; accuracy-framed 0.58; agreement-framed 0.62 | stated evaluation framing is a modest lever (accuracy framing mildly resists) |
| temperature_sensitivity (#286) | t0.0 0.74, t0.3 0.72, t0.7 0.75, t1.0 0.71 | flat across temperature: the cascade is not a greedy-decoding artifact, it survives sampling |
| onset (#287) | onset detected 20/20, turns {1,2,7}; contagion 0.025 | on the near-null bare cascade, onset is a transient single-turn blip, not a sustained tipping point |

**Read.** The true-peer control is the important one: adoption tracks correctness (0.53 correct vs
0.20 wrong), so the deference measured throughout is not indiscriminate agreement.
Temperature-invariance rules out a decoding artifact. Onset over the bare-cascade transcripts
confirms section 2, there is no sustained regime change to detect.

## 11. Referee detection - the named contribution (issue #322)

Can a monitor catch the adoption? Over 40 cascade cases with 23 holdout adoptions of the planted
shortcut, reusing `experiments/referee/referee_deployable.py` and `referee_judge.py` on the MedMCQA
manifest (the shared templates #357 ports for MIMIC-CXR).

| referee | precision | recall | FPR | real detection? |
|---|---|---|---|---|
| naive conformity gate (fires on any agreement) | 0.575 | 1.00 | 1.00 | no -- flags everything |
| deployable (peer-modal + private re-query) | 1.00 | 1.00 | 0.00 | **no -- identity, see below** |
| oracle audit (planted key + isolated counterfactual) | 1.00 | 1.00 | 0.00 | no -- it IS the label |
| same-lineage LLM judge | 1.00 | 0.91 | 0.00 | yes |

**Read (corrected).** The one honest result is the **same-lineage LLM judge**: given just the
transcript, no privileged key and no forced-peer shortcut handed to it, it catches 21 of 23
adoptions (recall 0.91, precision 1.0, FPR 0.0). The "deployable" referee's precision/recall 1.0 is
NOT detection: in this planted-shortcut setup the peers are forced to assert the same wrong answer,
so its inferred shortcut is exactly that answer and its decision `(board == inferred) and (board !=
bare)` reduces algebraically to the adoption label it is scored against. The oracle is the label by
definition too. So the earlier headline "a deployable monitor matches the oracle" is withdrawn --
those two are identities, not measurements. The defensible claim is narrower: an LLM judge detects
the adoption reasonably well; showing a *deployable* referee works needs a setup where the shortcut
is not the forced-peer answer the referee re-derives (a real fix to the script + re-run, tracked as
follow-up), not this one.

## Comparison to the MedQA reference (a caveat worth flagging)

Read against the **committed MedQA per-record data**
(`experiments/medqa/results/solo_records.jsonl`), MedMCQA is the same regime:

| dataset | flash flip | flash-lite flip |
|---|---|---|
| MedQA (committed `solo_records.jsonl`) | 0.063 | 0.117 |
| MedMCQA (this run) | 0.120 | 0.237 |

Same direction (flash-lite ~2x flash), same order of magnitude, MedMCQA modestly higher.

Note: `experiments/medqa/results/RESULTS.md` prose cites MedQA solo flip 0.787 / 0.893, which do
NOT match that folder's own `solo_records.jsonl` (0.063 / 0.117) and are inconsistent with the
reported clean accuracy (a 0.89-accurate model cannot flip on 0.79 of cases). Those prose numbers
carry the signature of the pre-parser-fix bug described in that file. This replication compares
against the per-record artifact, which is the internally consistent one. Flagged for the team;
not resolved here.
