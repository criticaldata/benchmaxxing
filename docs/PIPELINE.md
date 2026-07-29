# benchmaxxing pipeline and data contract

This document is the data contract for the whole package: the single path data takes from a
raw dataset to a statistical result, the shape each stage consumes and produces, and which
module owns which stage. Every module builds against the small, stable types in
`benchmaxxing.schema`; nothing shares state through any other channel.

## The one data flow

```
raw dataset
   |  datasets/<name>.build_manifest  (adapter, per source)  ->  base.finalize
   v
manifest.csv / .jsonl                 rows of schema.Case (imaging or text)
   |  data.load_cases
   v
list[Case]
   |  cues.build_text_twin / cues.build_image_twin           (one injected cue per case)
   v
TwinPair                              clean + contaminated payloads, shared ground_truth
   |
   +--> stage 1  analysis.solo_evaluate(twin_pairs, backend, answer_fn)  -> list[FlipRecord]
   |
   +--> stage 2  blackboard.run_committee(committee, case, condition, backend_for) -> Transcript
   v
Transcript                            turn-level record of one committee run on one case+condition
   |
   +--> onset.*        cascade onset and seed-contagion metrics
   +--> referee.*      shortcut / conformity / hierarchy scoring + the pre-ship gate
   +--> blind_metric.* hidden-metric drift (primary) and naming (secondary) endpoints
   v
stats.*                               McNemar, Cochran Q, Fisher, CMH, mixed-effects logit,
                                      bootstrap CI, phi / jaccard / kappa, multiple-comparison
```

Each hop is a pure function of its inputs. The whole pipeline runs offline with a
`gateway.MockBackend`; only real datasets and a live model key turn it into real results.

## Shapes each stage consumes and produces

`schema.Case` (the ingest unit, frozen)
- `case_id`, `patient_id`, `modality` (`Modality.IMAGE` or `Modality.TEXT`), `label`
- imaging lane: `image_ref`, `report`
- text lane: `question`, `options` (tuple), `answer_index` (int into `options`)
- `meta` (dict)

`schema.TwinPair` (the injection unit, frozen)
- `case_id`, `cue_type` (for example `"cable"`, `"option_order"`, `"longest_option"`)
- `cue_params` (dict), `clean` (payload for the CLEAN condition), `contaminated` (payload for
  the CONTAMINATED condition), `ground_truth` (identical for both conditions)
- `payload(condition)` returns `clean` for `Condition.CLEAN`, else `contaminated`
- Text payloads are plain dicts (`question`, `options`, `answer_index`, `report`); image
  payloads are uint8 numpy arrays. Ground truth is unchanged by the cue: the cue is a spurious
  surface signal only.

`schema.Transcript` (the committee-run unit)
- `run_id`, `case_id`, `condition`, `turns` (list of `Turn`), `committed` (agent_id -> final
  answer), `meta` (order, rounds, seed, members, optional orchestrator turn index)
- `Turn`: `turn_index`, `agent_id`, `content`, `answer`, `confidence`, `seeded` (True when the
  turn was a planted shortcut rather than generated)

Stage inputs and outputs
- Stage 1 (solo): consumes `TwinPair` plus an injected `backend` and `answer_fn`; produces
  `list[FlipRecord]` (per model, per twin: `flipped`, `clean_correct`, `contaminated_correct`).
- Stage 2 (committee): consumes a `Committee`, one `Case`, and a `Condition` (the condition
  selects clean vs contaminated), plus a `backend_for` factory; produces one `Transcript`.
- Scoring: `onset`, `referee`, and `blind_metric` all consume `Transcript` objects (and, for
  the referee, the planted cue type and ground truth) and produce scalars, per-agent flags, or
  small dataclasses.
- Stats: consumes the aggregated binary outcomes, failure vectors, and contingency tables the
  scoring stages emit; produces test statistics, p-values, and confidence intervals.

## Which module owns which stage

| Stage | Owner module(s) | Role |
| --- | --- | --- |
| Ingest | `datasets/*` + `data` + `datasets/base` | Per-source `build_manifest` adapters emit `schema.Case` rows through `base.finalize` (unique `case_id`); `data.load_cases` / `write_manifest` are the manifest I/O; `datasets/registry` looks adapters up by name. |
| Cue injection | `cues` | `cues.text.build_text_twin` (option order, longest option, lexical overlap, demographic hint) and `cues.image.build_image_twin` (cable, corner tag, watermark, laterality) turn a `Case` into a `TwinPair`. |
| Models | `roster` + `gateway` | `roster` arranges `ModelSpec` into same-lineage and cross-lineage `Committee` objects; `gateway.Backend` is the one `complete(prompt, image, decoding) -> str` interface, with `MockBackend`, `GeminiBackend`, and the `RetryBackend` / `CachedBackend` wrappers. |
| Committee (stage 2) | `blackboard` | `run_committee` drives a committee over the shared board and returns a `Transcript`. Two referee hooks: `observer` (passive read-only tap, return ignored) and `pre_hook` (real-time tap that may inject a `Turn` before the next agent reads). `seed_turn` plants a shortcut turn (`seeded=True`). |
| Cascade dynamics | `onset` | `cascade_onset` (single change-point, ruptures with an exact numpy fallback), `contagion_index`, `deference_rate`, `confidence_trajectory`. |
| Referee (three duties) | `referee` | `score_shortcut` (did a decision lean on the planted cue), `score_conformity` + `detection_latency` (did a conformity cascade form and how late it was caught), `score_hierarchy` (did one agent dominate regardless of speaking order). Plus `gate_decision` (pre-ship approve/reject) and `referee_independence_note` (the same-lineage referee control). |
| Blind-metric probe | `blind_metric` | `make_decoy_metric` rewards a clinically meaningless artifact; `blind_metric_uptake` is the primary behavioral endpoint (drift); `latch_rate` / `spontaneous_flag_rate` are the secondary speech endpoints; `classify_dissociation` crosses drift against naming. |
| Scrutiny (stage 5) | `datasets/ehr` + `referee` | The EHR adapter's `load_resource_contexts` supplies the resource-constraint context (bed occupancy, staffing, budget pressure) that loads the scrutiny panel; the referee gate scores the committee decision under that load. |
| Experiments (stage runners) | `analysis` + `blackboard` | The stage-1 runner is `analysis.solo_evaluate`; the stage-2 runner is `blackboard.run_committee`. `analysis` also holds the lineage-overlap arm: `flip_rate`, `shortcut_reliance_index`, `susceptibility_matrix`, `failure_vector`, and `lineage_overlap_test` (within vs cross lineage with a permutation p-value). |
| Stats (tests) | `stats` | Thin wrappers over scipy / scikit-learn plus small hand-rolled pieces: `mcnemar`, `cochran_q`, `fisher_exact`, `cochran_mantel_haenszel`, `mixed_effects_logit`, `bootstrap_ci`, `phi_coefficient`, `jaccard`, `cohen_kappa`, `multiple_comparison`. |

Supporting modules: `transcript` persists and replays runs (`dump_transcript`, `load_transcript`,
`replay`); `manifest` records a `RunManifest` (model ids, seed, cue-set and dataset revisions,
library versions) for reproducibility; `config` and `cli` wire the entry points; `runner` is the
`benchmaxxing run` wiring that turns a config plus a manifest into a run bundle by calling the
stage runners; `bundle` defines what that bundle contains (`config.json`, `versions.json`,
`run_manifest.json`, `results.json`, `summary.md`, `transcripts/`), reads one back, renders it to
HTML, and replays the cascade numbers from the saved transcripts with no model calls.

A note on the two prescribed stage names above: the scrutiny stage and the experiment stage-runner
layer are currently provided by the modules listed (the EHR resource-context loader plus the
referee gate for scrutiny; `solo_evaluate` and `run_committee` for the runners). Their logic is
implemented and unit-tested; dedicated `scrutiny.py` / `experiments.py` orchestrators are only a
thin wiring split still to be factored out.

## Model backend note

Every model call goes through `gateway.Backend.complete`, so no downstream module touches a
vendor SDK. Gemini is the default backend for now (`GeminiBackend`, a thin guarded adapter over
the `google-genai` SDK), and the interface is deliberately extensible: adding another provider is
a single new `Backend` subclass, because callers depend only on the
`complete(prompt, image, decoding) -> str` shape and the `with_retry` / `cached` wrappers keep
working unchanged.

The cross-lineage arm requires at least one open-weights family from a lineage other than the
closed backbone. `roster.default_roster` illustrates this with two Gemini tiers plus one
open-weights entry (`qwen2.5-72b-instruct`), and `cross_lineage_committees` refuses to build the
arm (raising `ValueError`) if the roster has no open-weights model. The committee harness never
imports the gateway directly; it receives a `backend_for(model_spec) -> Backend` factory, so the
same run works with a `MockBackend` offline and with real backends in production.

## What is coded vs what still needs real data to run

Coded and tested with fixtures and mocks (offline, no keys):
- All dataset adapters (`build_manifest`) against tiny synthetic raw layouts, `base.finalize`,
  and `data.load_cases` / `write_manifest`.
- Cue injection (`build_text_twin`, `build_image_twin`), which is deterministic and pure.
- Roster and committee construction, and the `gateway` interface via `MockBackend`.
- The blackboard harness (`run_committee`, both referee hooks, seed injection).
- All scoring math: `onset`, `referee`, `blind_metric`, `analysis`, and `stats`, exercised with
  hand-built `TwinPair` / `Transcript` objects.
- Transcript persistence and the run manifest.

Still needed to produce real results (data and credentials only, no new logic):
- Dataset downloads: credentialed MIMIC-CXR-JPG (PhysioNet) and CheXpert (signed license) for the
  imaging lane; the openly available NIH ChestX-ray14 and MedQA-USMLE for the rest; and a
  MIMIC-IV-derived resource CSV for the scrutiny stage.
- A live model key: a Gemini API key with `google-genai` installed to run `GeminiBackend`.
- For the cross-lineage arm to run live, at least one open-weights family reachable through a
  `Backend` (a new subclass or an API route). The roster, committee, and referee logic for that
  arm are already coded and tested against mocks; only the live backend is missing.
