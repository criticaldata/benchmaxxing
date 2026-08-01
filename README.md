<p align="center">
  <img src="assets/benchmaxxing.gif" alt="benchmaxxing: a committee shares context, one agent's shortcut cascades, a referee flags it" width="760">
</p>

# benchmaxxing

Referee agents that catch benchmark gaming in clinical multi-agent systems: shortcut cascades, blind metrics, and mutual oversight, measured.

A benchmark score tells you an agent hit the target, not whether the target was right or whether it got there honestly. When agents share context, one agent's shortcut can propagate through the whole group before any human notices. This repo builds **referee agents** (agents whose task is the assessment of other agents) and a reproducible sandbox to measure three failure patterns: shortcut reliance, conformity cascades, and emergent dominance.

## Design principle: reuse the originals

Every line we write ourselves is a line that can be wrong and a line a reviewer can attribute the result to instead of the phenomenon. So the pipeline is thin wrappers over established, version-pinned libraries (numpy/scipy/scikit-learn for statistics, a change-point library for cascade onset, image libraries for cue injection, a unified gateway for models), and the bespoke core is kept small and tested hardest. The measure of progress is how little of the pipeline is ours.

## Model backend

Agents use the **Gemini API** (multimodal) for now, behind one gateway wrapper so the roster can be extended to other model APIs later without changing experiment code. No fine-tuning; models are used off-the-shelf. Model lineage is a first-class variable: Gemini-only committees are the same-lineage control, and Gemini-plus-open-weights committees are the cross-lineage arm.

## Install

```bash
pip install -e ".[dev]"                 # pure-Python core + test tooling
pip install -e ".[dev,stats,changepoint,image,config]"   # add the reuse extras
```

## Layout

- `benchmaxxing/schema.py`: the shared data contract every module builds against.
- `benchmaxxing/stats.py`: statistical tests (wrappers over scipy/sklearn/statsmodels).
- `benchmaxxing/onset.py`: cascade-onset (change-point), contagion, deference.
- `benchmaxxing/cues/`: image and text cue injection, twin-pair builder.
- `benchmaxxing/blackboard.py`: the shared-context committee harness and referee hooks.
- `benchmaxxing/gateway.py`, `roster.py`: model access and same/cross-lineage committees.
- `benchmaxxing/datasets/`: dataset adapters into the shared case schema; dataset staged/coded/blocked status lives in `benchmaxxing/datasets/status.py` and is surfaced by `benchmaxxing datasets`.
- `benchmaxxing/referee.py`, `blind_metric.py`: the referee duties and the blind-metric probe.

The build plan and the issue tracker for this repo mirror a six-stage program (stages 0-5).

## Data access

No dataset is redistributed here. Each has its own access terms, and the repo carries only derived
per-case outcomes and model responses.

| Dataset | Access | What this repo contains |
|---|---|---|
| MedQA-USMLE | public | per-case rows, summaries, and the call caches, so every arm replays |
| NIH ChestX-ray14 | public | per-case rows, summaries, image-keyed caches; images not redistributed |
| MedMCQA | public | per-case rows, summaries, call cache |
| CheXpert | Stanford research agreement | per-case rows and summaries; no images |
| SUPPORT2 | public (UCI) | per-case rows, summaries, manifest, provenance, call cache |
| MIMIC-CXR and MIMIC-CXR-JPG | **PhysioNet credentialed** | de-identified per-case rows only. No report text, no images, no call caches, and no MIMIC identifiers |

**On the MIMIC lanes specifically.** PhysioNet's terms do not allow redistributing the data, so those
rows are keyed on a `case_index` local to this repo rather than on MIMIC subject and study ids. Every
rate and paired test in the paper is computed from the outcome flags, so the numbers stay recomputable;
what is removed is only the linkage back to source records. The imaging lane additionally ships
`results/deid/*.csv` and gitignores its transcripts and caches. `tests/test_no_credentialed_identifiers.py`
enforces this on every run, because the runners still emit raw ids locally.

## Reproducing the results

Two levels, and the repo is explicit about which lanes support which.

**Recompute the numbers, no API key, no dataset access.** Every published rate, difference and exact
test is derived from the committed per-case rows under `experiments/*/results/`. This works for every
lane including the MIMIC ones.

**Replay the runs, no API key, dataset access needed.** Where the call cache is committed, a runner
re-executes end to end and makes zero API calls, because every response is served from cache. This
holds for MedQA, NIH ChestX-ray14, CheXpert and SUPPORT2. It does not hold for the MIMIC lanes, whose
caches cannot be committed, or for MedMCQA, whose cache is committed but whose sampling manifest is not.

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q                      # 1160 passed, no key required
python -m experiments.support2.support2_referee --manifest ... --out ...   # replays at 0 API calls
```

A run that hits an uncached prompt exits rather than silently calling the API, so a "cached" claim
cannot quietly become a paid one.

## Guards

Several properties are enforced by tests rather than by convention, because each of them failed at
least once during the work:

- `tests/test_degeneracy_guard.py`: screens every committed per-case file for metrics that cannot fail
  by construction, such as a comparator that is constant, a predicate identical to the label it is
  scored against, or two reads that never diverge.
- `tests/test_no_credentialed_identifiers.py`: no MIMIC identifier may be committed.
- `tests/test_board_render.py`: every committee runner renders the shared board through one function,
  so a seeded rationale cannot be silently dropped before it reaches the holdout.

## Licence

**Not yet set.** Until a licence file is added the default is exclusive copyright, which means this
code cannot be reused. See the release checklist before publishing.

