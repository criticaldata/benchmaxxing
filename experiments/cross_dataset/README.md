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
mirrors the solo normalizer in `experiments/medqa/reproduce.py`.

Handling the LaTeX box matters in practice: Gemini 2.5 Flash answers with chain-of-thought that
quotes every option and ends in `\boxed{...}`. Without the box-aware fallback, `parse_mcq_choice`
finds several option strings, returns "ambiguous", and the decisive answer collapses to an
abstention. On the run below that mis-scored **~40% of replies** as abstentions and inflated every
flip rate.

The `--out` JSON writes non-finite values (an unbuildable cue's `nan` rate, an undefined Fisher
odds ratio) as `null`, so the artifact is strict-JSON parseable downstream.

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

## Cue set

The default cues are the zero-parameter text cues that `build_twins` can construct for every case:
`option_order`, `longest_option`, `lexical_overlap`. Cues that need parameters (for example
`demographic_hint`) are skipped and reported with a `nan` rate rather than silently dropped.

## Tests

The scoring is covered offline by `tests/test_cross_dataset.py` (deterministic pick-first backend
over the real cue payloads), and the entrypoint's wiring, skip paths, and answer normalizer by
`tests/test_cross_dataset_run.py`. The real flip rates need a key and are deferred to a staged run.
