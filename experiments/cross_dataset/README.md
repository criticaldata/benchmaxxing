# Cross-dataset cue consistency: MedQA vs MedMCQA (issue #129)

A shortcut result is more convincing if it replicates. This runner scores the **same** text cue
set on **two** text datasets with **one** fixed solo model and reports the per-cue flip rate side
by side, so a cue that bites on MedQA can be checked for a comparable effect on MedMCQA.

The scoring lives in `benchmaxxing.cross_dataset.run_cross_dataset_cue`; it composes the existing
cue builders, `solo_evaluate` / flip counting, `stats.bootstrap_ci`, and `stats.fisher_exact`.
Per cue it returns:

- each dataset's flip rate with a bootstrap 95% CI and its `(flips, n)` counts,
- the cross-dataset `spread` of the rate (`max - min`),
- for the two-dataset case, a Fisher exact test of whether the two flip counts are distinguishable
  (a high p-value means the datasets are statistically consistent),

plus a single `agreement.mean_spread` over the compared cues (lower is more consistent).

The result also carries a `clean_correct` block: the same per-cue table and `agreement`, same
bootstrap CI and same Fisher exact test, restricted to the twins the model answers correctly on the
clean (unperturbed) side, with `n_twins` (how many those are) and `clean_accuracy` per dataset. A
dataset that is harder for the model flips more for that reason alone, and this block is the
comparison with that removed.

**What counts as a flip.** A flip is a different option on the cued twin than on the clean twin
(`benchmaxxing.analysis.chose_same_option`). Each cue row also carries `text_changed`, the count a
plain comparison of the two answer strings gives, which is what a flip was before. The two differ
for `longest_option` and `lexical_overlap` only: both rewrite the text of one wrong option, so a
model that picks that option on both twins returns two strings for one choice. `text_changed` is
kept so that the earlier numbers stay on record; no interval or test is computed on it.

## Run

```bash
python experiments/cross_dataset/run_cross_dataset.py \
    --medqa-manifest path/to/medqa.csv \
    --medmcqa-manifest path/to/medmcqa.csv \
    --limit 200 \
    --cache experiments/cross_dataset/results/call_cache.jsonl \
    --out experiments/cross_dataset/results/medqa_vs_medmcqa.json
```

Pass `--cache` to record every model reply to a JSONL file (wrapping `gateway.CachedBackend`); a
later run with the same path reuses those replies, so the rates and CIs reproduce exactly instead
of drifting with model non-determinism.

Build the two manifests first with the dataset adapters (`benchmaxxing datasets`), and set
`GEMINI_API_KEY`. With no key or a missing manifest the script prints a `[skip]` line and exits 0:
it is a real-model run and never fabricates numbers, so **no results are checked in until a keyed
run produces them.**

## Answer parsing (why `option_order` is honest here)

Flip is defined on the model's chosen **option identity**, not its position. The backend is a
`payload -> option-text` callable: it prompts for the letter and full text of the best option and
resolves the reply against that payload's options with `extract.parse_mcq_choice`, falling back to
a conservative letter parse for replies whose signal is the declared letter (`"B"`, `"Answer: C"`,
and the `$\boxed{\text{D. ...}}$` / "final answer is D" box that Gemini 2.5 reasoning replies end
with). So when `option_order` permutes the options, a model that keeps picking the same clinical
answer registers as **no flip** (robust), and only a genuinely position-driven change counts. This
mirrors the solo normalizer in `experiments/medqa/reproduce.py`. The option text is then mapped to
the option it names before the two twins are compared, because `longest_option` and
`lexical_overlap` change the text of the option they target: the same option, rewritten, is not a
flip.

Handling the LaTeX box matters in practice: Gemini 2.5 Flash answers with chain-of-thought that
quotes every option and ends in `\boxed{...}`. Without the box-aware fallback, `parse_mcq_choice`
finds several option strings, returns "ambiguous", and the decisive answer collapses to an
abstention. On the run below that mis-scored **~40% of replies** as abstentions and inflated every
flip rate.

The `--out` JSON writes non-finite values (an unbuildable cue's `nan` rate, an undefined Fisher
odds ratio) as `null`, so the artifact is strict-JSON parseable downstream. It also records the
model id (`model`) and the sha256 of each manifest file the run read (`manifest_sha256`), so a
result file identifies its own inputs.

## Results (first keyed run)

`gemini-2.5-flash`, 50 cases per dataset (150 twins each), temperature 0. Replies are cached in
`results/cache_medqa_vs_medmcqa.jsonl`, so the table below reproduces from `results/medqa_vs_medmcqa.json`
without re-querying the model.

| cue | MedQA flip | MedMCQA flip | spread | Fisher p |
|-----|-----------:|-------------:|-------:|---------:|
| `option_order`   | 0.08 (4/50) | 0.08 (4/50) | 0.00 | 1.00 |
| `longest_option` | 0.10 (5/50) | 0.16 (8/50) | 0.06 | 0.55 |
| `lexical_overlap`| 0.08 (4/50) | 0.16 (8/50) | 0.08 | 0.36 |

`agreement.mean_spread = 0.047`. Every cue's flip counts are statistically **indistinguishable**
across the two datasets (all Fisher p >= 0.36, spreads <= 0.08): the solo flip rate is low and
consistent across MedQA and MedMCQA, i.e. the cue is not a one-dataset artifact. Bootstrap 95% CIs
per dataset are in the `--out` JSON. This is a 50-case pilot; a larger `--limit` tightens the CIs.

This pilot file predates the option-based flip and is not regenerated: its counts are answer-string
comparisons (`text_changed` in the newer files). It cannot be replayed, because its MedQA cases are
not the first 50 of the test manifest and 11 of its MedMCQA `lexical_overlap` prompts changed with
the cue builder. See `docs/FLIP_SCORING_AUDIT.md`.

## Results at 500 cases per dataset

`gemini-2.5-flash`, temperature 0, the first 500 cases of each manifest: the MedQA manifest that
`benchmaxxing.datasets.medqa` builds from the public `test.jsonl` (sha256 prefix 0f2076a4) and
MedMCQA `medmcqa_dev_500.csv` (sha256 prefix 5b2ce971, built by `benchmaxxing.datasets.medmcqa`).
The result file records the model id and both hashes in full (`model`, `manifest_sha256`), as does
the 150-case file. An earlier build of the MedQA manifest (sha256 prefix 6a2ac665, the one this
section named before; it joins the options with `|` and has no `meta` column) parses to the same
1,273 cases and gives the same prompts and the same numbers; a replay on it records that hash
instead. Result: `results/medqa_vs_medmcqa_n500.json`; cache:
`results/cache_medqa_vs_medmcqa_n500.jsonl`, which starts from the 1,200 replies of the 150-case run
(same prompts) and adds 2,799 new ones. One MedMCQA case (`26a82f11-4261-4975-bee9-ae8d4e74cdb7`) is
excluded with `--exclude-case`: its `longest_option` twin returned 504 DEADLINE_EXCEEDED on every
attempt, so MedMCQA has 499 cases per cue.

| cue | MedQA flip | MedMCQA flip | spread | Fisher p |
|-----|-----------:|-------------:|-------:|---------:|
| `option_order`   | 0.062 (31/500) | 0.128 (64/499) | 0.066 | 0.00035 |
| `longest_option` | 0.058 (29/500) | 0.110 (55/499) | 0.052 | 0.0030 |
| `lexical_overlap`| 0.058 (29/500) | 0.124 (62/499) | 0.066 | 0.00027 |

`agreement.mean_spread = 0.062`. Over all cases MedMCQA flips about twice as often as MedQA on
every cue (ratios 2.1, 1.9 and 2.1), the direction seen at 150 cases. That comparison is not
adjusted for difficulty; the next table is.

Under the earlier answer-string comparison these rows read 41/500 against 73/499 for
`longest_option` and 38/500 against 84/499 for `lexical_overlap` (`text_changed` in the result
file; `option_order` is the same under both). The difference, 12 and 18 cases for `longest_option`
and 9 and 22 for `lexical_overlap`, is the model picking on both twins the wrong option the cue
rewrote.

### Adjusted for difficulty: cases answered correctly on the clean twin

MedMCQA is harder for this model. It answers the clean twin correctly on 451 of 500 MedQA cases
(0.902) and on 401 of 499 MedMCQA cases (0.804), and a case it already gets wrong changes its
answer under a cue far more often than one it gets right: under `option_order`, 17 of 49 and 38 of
98 wrong cases flip (0.35 and 0.39), against 14 of 451 and 26 of 401 right ones. So part of the gap
above is a difference in difficulty. The `clean_correct` block of the result file gives the same
comparison on the cases answered correctly on the clean twin, here beside the unrestricted one:

| cue | all cases: MedQA | all cases: MedMCQA | Fisher p | correct on clean: MedQA | correct on clean: MedMCQA | Fisher p |
|-----|-----------------:|-------------------:|---------:|------------------------:|--------------------------:|---------:|
| `option_order`   | 0.062 (31/500) | 0.128 (64/499) | 0.00035 | 0.031 (14/451) | 0.065 (26/401) | 0.023 |
| `longest_option` | 0.058 (29/500) | 0.110 (55/499) | 0.0030  | 0.038 (17/451) | 0.047 (19/401) | 0.50 |
| `lexical_overlap`| 0.058 (29/500) | 0.124 (62/499) | 0.00027 | 0.029 (13/451) | 0.070 (28/401) | 0.0061 |

`clean_correct.agreement.mean_spread = 0.028`. What the restricted numbers support:

- On `option_order` and `lexical_overlap`, MedMCQA still flips about twice as often as MedQA among
  the cases the model answers correctly without the cue (ratios 2.1 and 2.4). The evidence is weaker
  than over all cases because the flips are few (14 against 26, and 13 against 28): Fisher p 0.023
  and 0.0061, or 0.046 and 0.018 after a Holm correction over the three cues. The `option_order`
  difference is borderline; the `lexical_overlap` one is the firmer of the two.
- On `longest_option` the two datasets are not distinguishable once restricted (0.038 against
  0.047, Fisher p 0.50). Its gap over all cases comes from the cases the model gets wrong on the
  clean twin: MedMCQA has twice as many (98 against 49), and they change option far more often than
  the right ones (12 of 49 and 36 of 98).

The restricted counts are the same under the option comparison and under the earlier answer-string
comparison: a model that is right on the clean twin picked the correct option, whose text no cue
rewrites, so every case the two comparisons disagree on is wrong on the clean twin.

At 150 cases (`results/medqa_vs_medmcqa_n150.json`) the restricted rates point the same way on all
three cues and none of the differences is significant: 4/140 against 9/123, 4/140 against 8/123 and
3/140 against 9/123 (Fisher p 0.15, 0.24, 0.072). Over all cases at 150 the counts are 8/150
against 18/150, 7/150 against 15/150 and 5/150 against 15/150 (Fisher p 0.063, 0.12, 0.035).

Replay with no key and zero calls:

```bash
python experiments/cross_dataset/run_cross_dataset.py --medqa-manifest <medqa> --medmcqa-manifest <medmcqa> \
    --limit 500 --cache experiments/cross_dataset/results/cache_medqa_vs_medmcqa_n500.jsonl \
    --exclude-case 26a82f11-4261-4975-bee9-ae8d4e74cdb7 --replay --out /tmp/replay.json
```

`--workers N` asks a run's prompts N at a time into the cache before the scoring pass; the prompts
do not depend on any answer, so the numbers are those of a sequential run over the same replies.

## Cue set

The default cues are the zero-parameter text cues that `build_twins` can construct for every case:
`option_order`, `longest_option`, `lexical_overlap`. Cues that need parameters (for example
`demographic_hint`) are skipped and reported with a `nan` rate rather than silently dropped.

## Tests

The scoring is covered offline by `tests/test_cross_dataset.py` (deterministic pick-first backend
over the real cue payloads), and the entrypoint's wiring, skip paths, and answer normalizer by
`tests/test_cross_dataset_run.py`. The real flip rates need a key and are deferred to a staged run.
