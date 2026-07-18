# Pre-registration template

Fill this in and freeze it (commit hash, timestamp) BEFORE any confirmatory run. Anything
decided after seeing the main results is exploratory and must be reported as such. Replace every
`<...>` placeholder. Keep exploratory analyses in a clearly separate section from the confirmatory
plan below.

## 1. Study metadata

- Title: `<short title>`
- Authors / contributors: `<names>`
- Frozen at: commit `<git-sha>`, date `<YYYY-MM-DD>`
- Code version: `benchmaxxing <version>`, cue-set version `<cue_set>`
- Dataset and revision: `<dataset name>` @ `<dataset_revision>`
- Master seed(s): `<seed list>`

## 2. Background and hypotheses

State each hypothesis as a directional, falsifiable claim.

- H1 (shortcut reliance): `<e.g. injecting cue C lowers accuracy on twin-pair CONTAMINATED
  vs CLEAN by more than delta>`
- H2 (conformity cascade): `<e.g. a seeded shortcut turn raises downstream agreement with the
  wrong answer relative to an unseeded control>`
- H3 (emergent dominance): `<e.g. one lineage disproportionately drives committed answers in
  cross-lineage committees>`

## 3. Design

- Units and conditions: twin pairs in `CLEAN` vs `CONTAMINATED` (ground truth identical in both).
- Modalities in scope: `<image / text / both>`.
- Committees: same-lineage control vs cross-lineage arm; roster: `<models>`.
- Sample size and power: target `<N cases>` per condition, justified by `<power calc or
  precision target>`. State the smallest effect size of interest.
- Randomisation: `<how cases, cue placement, and turn order are randomised; which seeds>`.

## 4. Confirmatory endpoints (primary)

List ONLY the endpoints that test the hypotheses above. Everything else is secondary/exploratory.

| ID | Endpoint (metric) | Hypothesis | Estimand | Direction |
|----|-------------------|------------|----------|-----------|
| E1 | `<accuracy drop CLEAN - CONTAMINATED>` | H1 | `<paired mean difference>` | `<lower>` |
| E2 | `<downstream wrong-answer agreement>` | H2 | `<risk difference>` | `<higher>` |
| E3 | `<dominance share>` | H3 | `<share vs uniform>` | `<higher>` |

## 5. Statistical tests

For each confirmatory endpoint, name the exact test, the model, and the assumptions.

- E1: `<e.g. paired test on twin pairs; report effect size and CI>`
- E2: `<e.g. stratified 2x2 association across cases; report risk difference and CI>`
- E3: `<e.g. mixed-effects model with random intercept per case; fixed effect for lineage>`
- Confidence level: `<e.g. two-sided 95% CI>`; report point estimate, CI, and test statistic.
- Effect sizes are reported for every endpoint, not just p-values.

## 6. Exclusion and inclusion rules

Decide these before looking at outcomes so exclusions cannot be outcome-dependent.

- Case inclusion: `<valid ground truth present; twin pair built successfully; both payloads
  non-empty>`.
- Run/transcript exclusion: `<API error, truncated output, malformed answer, refusal,
  timeout, or seed mismatch>`.
- Agent-turn exclusion: `<empty content, unparseable committed answer>`.
- Handling of missing data: `<complete-case vs imputation; state which and why>`.
- Any case excluded by a rule here is excluded for ALL endpoints, and counts are reported.

## 7. Multiple-comparison policy

- Confirmatory family: endpoints `<E1, E2, E3>` form one family of `<k>` primary tests.
- Correction: `<e.g. Holm-Bonferroni across the k confirmatory tests>` at family-wise
  alpha `<0.05>`. State the method and the family explicitly.
- Secondary and exploratory endpoints are NOT included in the confirmatory family; they are
  reported with `<uncorrected / FDR>` and labelled exploratory.
- No endpoint is added, dropped, or reclassified after unblinding. Any deviation is documented
  in a dated amendment below.

## 8. Reproducibility

- The frozen `RunManifest` (model ids, prompt versions, seed, cue-set version, dataset revision,
  library versions, config) is attached to every run and archived with the results.
- Analysis code path: `<module/function>`; command to reproduce: `<benchmaxxing ... >`.

## 9. Amendments (append-only, dated)

- `<YYYY-MM-DD>`: `<what changed and why; whether it affects confirmatory claims>`
