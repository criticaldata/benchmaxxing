# Pre-registration: natural-cue imaging cascade on CheXpert Support Devices

Addendum to [`docs/PREREGISTRATION.md`](../PREREGISTRATION.md) for the redesigned cascade
(issue #133), scoped to the confirmatory CheXpert run tracked in #120. Freeze this document
(commit hash, date) before that run starts. Everything below is informed by, but distinct
from, the already-completed injected-cue pilot on NIH ChestX-ray14 (PR #166): that pilot is
what makes this design confirmatory rather than a guess, but it is exploratory work relative
to this study, not the confirmatory result itself.

## 1. Study metadata

- Title: Natural-cue imaging cascade on CheXpert Support Devices
- Authors / contributors: `<names, DOJO author roster>`
- Frozen at: commit `<git-sha>`, date `<YYYY-MM-DD>`
- Code version: `benchmaxxing <version>`; cue-set: natural (CheXpert `Support Devices` label,
  no injection) — see `benchmaxxing/datasets/chexpert.py` for the label encoding
- Dataset and revision: CheXpert (or CheXpert-small) @ `<dataset_revision>`, gated on #94
  (Stanford licence); the pilot that shaped this design ran on the open NIH ChestX-ray14
  release (see PR #166)
- Master seed(s): project default `seed=0`, threaded through `AgentView.seed` per the
  blackboard harness contract

## 2. Background and hypotheses

- H1 (shortcut reliance): solo flip rate is higher on cases where a support device is
  visibly present than on cases where it is absent. Mirrors the pilot's solo lane, where the
  strongest injected cue (watermark) cleared the noise floor by +0.11 at n=35.
- H2 (conformity cascade, PRIMARY): a committee holdout's rate of adopting a
  peer-asserted **wrong** finding-read is higher when two confident peers assert it on the
  SHARED board than when the holdout reads the same image ISOLATED (no board). This is the
  central confirmatory claim of the imaging-cascade story, and it is not a guess: the
  identical paired design, run on NIH ChestX-ray14 with an injected cue, gave
  `contagion = shared_adopt - isolated_adopt = +0.629` for watermark (PR #166,
  `imaging_cascade_summary.json`) and an even larger effect for three independent cue types
  (cable +0.80, corner_tag +0.74, laterality +0.71) — the cascade held across four different
  injected cues, so it is not an artifact of one cue's phrasing.
- H3 (emergent dominance / cross-lineage): explicitly **out of scope** for this run. The
  cross-lineage arm needs an open-weights model on the cluster (#95, #134); deferred until
  that unblocks.

## 3. Design

- Units and conditions: individual CheXpert cases with a real, positive `Support Devices`
  finding. Each case's designated holdout committee member reads it twice: once with the
  same board (two committee peers assert a confident wrong read) and once isolated (no
  board) — a paired, within-case design, matching `imaging_cascade.py`'s validated pattern
  exactly (PR #166).
- Modality in scope: image only.
- Committees: same-lineage Gemini (`gemini-2.5-flash` roster, 3 members: 2 assert, 1 is
  scored, per the validated pilot's design). Cross-lineage arm deferred (H3, out of scope).
- Sample size and power: computed with `benchmaxxing.stats.required_pairs`, anchored on the
  pilot's observed discordant proportion (`psi ≈ 0.629`, from n=35 with shared adopt 0.971 /
  isolated adopt 0.343). Natural device cues may be subtler than an injected watermark, so
  three effect-size scenarios are reported rather than assuming the pilot's own effect
  transfers unchanged:

  | assumed contagion effect | required pairs (80% power, α=0.05) |
  |---|---|
  | 0.629 (matches the pilot's smallest injected-cue effect, watermark) | 10 |
  | 0.40 (meaningfully attenuated) | 29 |
  | 0.25 (conservative floor, ~40% of the pilot effect) | 77 |

  **Target: n=90 hard cases** (real, positive `Support Devices` label; frontal view;
  non-uncertain), which keeps power ≥ 80% even under the conservative 0.25 scenario
  (`achieved_power(90, 0.629, 0.25)` — verify at freeze time against the actual staged case
  count).
- Randomisation: case sampling order fixed by `seed=0` (project default); committee speaking
  order follows the blackboard harness's own within-run randomisation; the assignment of
  which 2 of 3 committee members serve as the asserting peers vs. the scored holdout is
  fixed per case (not re-randomised), matching the validated pilot design so results are
  directly comparable to it.

## 4. Confirmatory endpoints (primary)

| ID | Endpoint (metric) | Hypothesis | Estimand | Direction |
|----|-------------------|------------|----------|-----------|
| E1 | solo flip rate, device-present vs device-absent cases | H1 | risk difference (independent groups; CheXpert has no natural twin pairs) | higher (present > absent) |
| E2 | contagion = shared_adopt − isolated_adopt on the natural cue | H2 (primary) | paired risk difference (same case, same image, two board conditions) | higher than 0 |

## 5. Statistical tests

- E1: `benchmaxxing.stats.mcnemar` is a **paired** test and CheXpert's natural cue has no
  twin-pair structure, so E1 is instead an independent-groups comparison (Fisher's exact
  test via `benchmaxxing.stats.fisher_exact` on the 2x2 flip-vs-no-flip by
  present-vs-absent table), with `bootstrap_ci` for the risk-difference CI. This is a
  deliberate deviation from the twin-pair paired design used everywhere else in this
  project, and is noted here so it isn't mistaken for an oversight at analysis time.
- E2 (primary): `benchmaxxing.stats.mcnemar` on the paired shared-vs-isolated adoption per
  case, exactly reproducing the pilot's design (PR #166, `imaging_cascade.py`). Effect size
  reported as contagion (risk difference) with a `bootstrap_ci`. As a secondary robustness
  check (not a replacement for the primary McNemar test), fit
  `benchmaxxing.stats.mixed_effects_logit(adoption ~ condition + (1|case))` per issue #131;
  this is complementary here since the primary paired design already controls for case
  identity by construction.
- Confidence level: two-sided 95% CI throughout; point estimate, CI, and test statistic
  reported for both endpoints, not just p-values.

## 6. Exclusion and inclusion rules

- Case inclusion: `Support Devices` label is positive (`1.0`; not `-1.0` uncertain, blank
  unmentioned, or `0.0` negative — see the CheXpert adapter's uncertainty policy in
  `benchmaxxing/datasets/chexpert.py`); frontal view only (avoids a lateral-view confound);
  image resolves on disk with a recorded sha256 checksum (per `build_manifest.py`'s
  provenance pattern in PR #166); a real, non-"No Finding" primary label, so there is a
  concrete wrong read for the peers to assert.
- Run/transcript exclusion: an unparseable yes/no read (the yes/no parser returns `"?"`),
  an API error surviving retries, a timeout, or a refusal.
- Agent-turn exclusion: none beyond the above — the design already reduces each case to
  exactly two reads (shared, isolated), per the validated pilot script.
- Handling of missing data: complete-case. A case excluded on either read (unparseable,
  error) is dropped entirely from **both** E1 and E2, and the drop count is reported in the
  results.
- Any case excluded by a rule here is excluded for ALL endpoints, and counts are reported.

## 7. Multiple-comparison policy

- Confirmatory family: {E1, E2}, `k=2`.
- Correction: Holm-Bonferroni via `benchmaxxing.stats.multiple_comparison(method="holm")`
  at family-wise alpha 0.05. Holm (not Benjamini-Hochberg) is chosen deliberately: `k` is
  small, E2 is the central claim of the paper's imaging story, and the text-lane Break-it C
  dose-response (generic 0.729 → anchored 0.847, McNemar p=0.041) did **not** survive BH
  correction (see the 2026-07-21 re-grade, PR #141) — this study should not repeat that
  fragility on its primary result.
- Any exploratory sub-analysis (a per-cue breakdown if a second natural cue becomes
  available; a per-tier breakdown across model sizes) is explicitly labelled exploratory
  and reported with uncorrected p-values, never folded into the confirmatory family.
- No endpoint is added, dropped, or reclassified after unblinding. Any deviation is
  documented in a dated amendment below.

## 8. Reproducibility

- Every model call is cached in a committed `call_cache.jsonl` keyed by
  `(model, image_bytes, prompt)`, matching the pattern already validated in PR #166
  (`imaging_cascade.py`, `imaging_referee.py`). A fully cached re-run reproduces every
  number with zero new API calls and no key.
- Analysis code path: `experiments/imaging_chexpert/` (to be created off the PR #166
  pattern once CheXpert access lands per #120); command:
  `python experiments/imaging_chexpert/cascade.py --manifest <chexpert_manifest.csv> --image-root <path>`.
- Provenance: a sha256 checksum per used image, recorded in a provenance JSON, per
  `build_manifest.py`'s existing pattern (PR #166).

## 9. Amendments (append-only, dated)

- `<YYYY-MM-DD, freeze date>`: Document created and frozen before the CheXpert run, per
  issue #133. Confirmatory endpoints and the sample-size target are informed by the
  already-completed, keyless-reproducible NIH ChestX-ray14 injected-cue pilot (PR #166),
  which shaped this design but is exploratory relative to this study — the confirmatory
  result is the CheXpert natural-cue run once #94 access lands. All pilot numbers cited
  above are post the 2026-07-21 answer-parser fix (#161) and were verified against the
  committed summary JSON files in PR #166 at the time this document was written.
