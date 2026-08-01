# Contributing to benchmaxxing

Thanks for helping build the referee-agent sandbox. This guide is how to reproduce the code, run the checks, and land a change. The conceptual plan lives in the project docs; this file is the practical how-to.

## 1. Setup

Python 3.10+ . The pure-Python core installs and tests anywhere; the heavy pieces are optional extras.

```bash
git clone https://github.com/criticaldata/benchmaxxing
cd benchmaxxing
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# core + dev tooling (numpy/scipy/scikit-learn + pytest/ruff): runs the whole test suite offline
pip install -e ".[dev]"

# add the reuse extras when you need them:
#   image        -> cue injection (pillow, opencv)
#   changepoint  -> cascade-onset (ruptures)
#   stats        -> mixed-effects + Cochran-Mantel-Haenszel (statsmodels)
#   models       -> the model backends (Gemini API, transformers, litellm)
#   config       -> YAML config (pyyaml)
pip install -e ".[dev,image,changepoint,stats,config]"
```

### Environment variables (API key)

The default model backend is the **Gemini API**. Configure your key through the environment; never commit a key or paste it into a tracked file.

```bash
export GEMINI_API_KEY=your-key-here      # google-genai reads this (GOOGLE_API_KEY also works)
```

You do **not** need a key for most work: the full test suite and `benchmaxxing smoke` use mock backends and synthetic data, so they run entirely offline. A key (plus the `models` extra) is only needed to run the pipeline on real cases through Gemini. For the required cross-lineage arm, also make at least one open-weights vision-language model available locally (via the `models` extra / HuggingFace); those need no key.

To make the key persist, add the `export` line to your shell profile, or keep it in a local `.env` file that you do not commit (add `.env` to your local git ignore).

## 2. Run the checks locally (the gate)

There is no hosted CI; local `ruff` + `pytest` is the gate, plus the offline smoke.

```bash
ruff check .
pytest -q
```

The full suite runs offline with no API keys and no real data (mock backends and synthetic fixtures). A green suite plus a clean ruff is the bar for a PR.

## 3. Testing policy (real data for behavior, tiny synthetic for shape)

Two tiers, and both matter:

- **Contract / shape tests** may use tiny synthetic arrays to pin the input/output shapes of a pure function. These run anywhere.
- **Behavioral claims** are validated on real data or on deterministic mock backends, never on fabricated data. Tests that need an uninstalled optional dependency (statsmodels, ruptures, torch, the Gemini client) must `pytest.importorskip` and skip cleanly, so the core suite always passes.

If you add a module, add its unit tests in `tests/`. If your change crosses module boundaries, extend the end-to-end offline smoke (see below) so an interface drift between two modules is caught.

### Prompt golden files

Every core prompt template in `benchmaxxing/prompts.py` (`DEFAULT_REGISTRY`) is pinned by a golden file under `tests/goldens/prompts/`, checked by `tests/test_prompts_golden.py`: a template rendered against a fixed synthetic input must match the checked-in text exactly, so an accidental edit to a prompt is caught in review instead of silently changing agent behavior. After an *intentional* prompt change, regenerate the affected goldens and review the diff before committing it:

```bash
BENCHMAXXING_UPDATE_GOLDENS=1 python -m pytest tests/test_prompts_golden.py -q
git diff tests/goldens/prompts/
```

A new prompt template needs a matching fixed input added to `_FIXED_INPUTS` in that test file, or the suite fails loudly rather than silently skipping coverage for it.

### Metrics that cannot fail (#374)

`benchmaxxing/degeneracy.py` screens the whole repo for reported numbers whose predicate could not have come out otherwise, and `tests/test_degeneracy_guard.py` fails when it finds one that is not exempted. Three screens: binary per-case columns that are constant across every row, significance verdicts hardcoded as literals in the same interpolation that reports a p-value, and committed p-values of exactly 0.0 or 1.0. Run it standalone for a report:

```bash
python -m benchmaxxing.degeneracy
```

Exemptions live in `tests/degeneracy_exemptions.json`, in two maps with different meanings. `allowlist` is for findings verified legitimate, and the reason must say what was checked. `preexisting` records what was already in the tree when the guard landed and is not an endorsement of any of it. Both require a written reason, and an entry that no longer matches a finding fails the suite, so deleting the line is the last step of a fix.

What it does not catch, so a green run is not an all-clear: a near-constant predicate that is true on 199 of 200 rows, an algebraic reduction between two columns that both vary, and anything in an artifact that is not committed. Add a real check rather than an exemption when you hit one of those.

## 4. Reproduce the pipeline (CLI)

```bash
# print the version
benchmaxxing version

# list the registered dataset adapters
benchmaxxing datasets

# summarize and sanity-check a manifest (row/modality counts, MCQ shape, label
# distribution; pass --image-root to check that image_ref paths resolve on disk)
benchmaxxing datasets stats path/to/manifest.csv [--image-root path/to/images]

# stage a downloaded raw release: build the manifest, validate it, checksum it, and write
# its provenance record (source, access level, licence, counts). See docs/DATASETS.md.
benchmaxxing datasets stage nih_cxr14 --check-images

# show the resolved default config
benchmaxxing config-show

# render a clean/contaminated cue twin to disk to eyeball it (defaults to a bundled
# sample if --case is omitted; the image lane needs the `image` extra)
benchmaxxing cues preview --lane {image,text} --cue NAME --out path/to/dir [--case path/to/manifest.csv]

# run the offline end-to-end pipeline smoke on synthetic data:
#   cue injection -> solo baselines -> shared vs isolated committee with a seeded shortcut
#   -> cascade onset -> referee scoring + gate -> blind-metric probe
benchmaxxing smoke

# run one experiment stage from a config plus a manifest, and write the run directory
# (results.json, summary.md, config.json, run_manifest.json) into --out
benchmaxxing run --stage {pilot,solo,overlap,cascade} --manifest path/to/manifest.csv \
    --dataset medqa --config path/to/config.json --out runs/my-run

# see what a run would do (models, twins, estimated model calls) without spending any calls
benchmaxxing run --stage cascade --manifest path/to/manifest.csv --out runs/my-run --dry-run

# same command, no key and no data: the mock backend is the offline stand-in
benchmaxxing run --stage solo --manifest path/to/manifest.csv --out runs/mock --backend mock

# add the decoding-noise control: re-ask each clean payload once, so the flip rate can be
# read against the rate at which the model changes its mind with no cue present
benchmaxxing run --stage solo --manifest path/to/manifest.csv --out runs/x --noise-floor

# the imaging lane (Lane A): cue_set "image-v1" in the config, plus the image root that
# image_ref paths resolve against. Needs the `image` extra.
benchmaxxing run --stage solo --manifest cxr.csv --image-root /data/images --out runs/cxr

# run the cascade only where the committee was uncertain: a confident, correct committee is
# the worst place to look for one. Ranks the cases from a previous solo run's records.
benchmaxxing run --stage cascade --manifest medqa.csv --out runs/hard \
    --hard-cases experiments/medqa/results/solo_records.jsonl --hard-k 20

# render a finished run bundle into a standalone HTML report, and recompute the cascade
# numbers from the saved transcripts (no model calls, no key)
benchmaxxing report runs/hard --replay
```

The smoke is the fastest way to see the whole pipeline compose and to confirm your change did not break a seam. It needs no data and no keys.

`benchmaxxing run` is the reproducible entry point for a stage: it resolves the config (models, cue set, seed), loads the manifest, injects the cues, calls the matching runner in `benchmaxxing/experiments.py`, and writes a self-describing run directory. A config file may be JSON (no extra needed) or YAML (needs the `config` extra). The default backend is Gemini, so a real run needs `GEMINI_API_KEY` and the `models` extra; `--backend mock` runs the same code path offline. The cascade stage also saves every shared and isolated transcript under `--out/transcripts/` for re-analysis.

`summary.md` reports three things: the headline point estimates, the same estimates with a 95% bootstrap CI (seeded from the config seed, so a rerun reproduces them), and every p-value the stage produced corrected once across the whole family with Benjamini-Hochberg, with the family size stated. A p-value that cannot be adjusted (an undefined overlap, for example) is listed as dropped rather than quietly left out.

### The run bundle

Every run writes one self-contained directory, named `<stage>-<dataset>-<date>` under the config's `out_dir` unless `--out` says otherwise:

```
runs/cascade-medqa-20260723/
  config.json         resolved config: models, cue set, seed, limits
  versions.json       package version, git SHA, python, platform, pinned library versions
  run_manifest.json   the RunManifest (model ids, prompt versions, dataset revision, roster)
  results.json        structured output, plus the plan, the estimates and the corrected family
  summary.md          the artifact a collaborator reads first
  transcripts/        one saved transcript per case and condition (cascade only)
```

`benchmaxxing report <dir>` renders it to standalone HTML from the directory alone, and `--replay` recomputes the cascade adoption numbers from `transcripts/` and prints them next to the reported ones, so a reviewer can check a result rather than take it on faith. Do not commit real-run bundles that contain dataset-derived text; keep those outside the repo.
## 5. Reuse the originals (the design rule)

The pipeline is thin wrappers over established, version-pinned libraries (numpy/scipy/scikit-learn for statistics, a change-point library for cascade onset, image libraries for cue injection, a single gateway for models). The bespoke core is kept small and tested hardest. Two rules follow:

- **Do not re-implement anything a maintained library already does.** If a library exists, wrap it and pin the version.
- **Verify before relying.** Confirm a dependency's current API against its own docs (not memory) before wiring it in, and let the run manifest record the resolved versions so a result is reproducible against a known environment.

The measure of progress is how little of the pipeline is ours. If the bespoke surface grows, that is a signal to look for a library you missed.

## 6. Add a dataset adapter (one owner each)

Everything downstream of a shared `Case` manifest is dataset-agnostic and already built, so a dataset adapter is the only per-dataset code. Each dataset has one owner and one module under `benchmaxxing/datasets/`.

To add or complete an adapter:

1. Implement `build_manifest(raw_root, out, limit=None)` in your `benchmaxxing/datasets/<name>.py`: map the raw layout into `list[benchmaxxing.schema.Case]`, then call `finalize(cases, out)` from `benchmaxxing/datasets/base.py`.
2. Keep imaging cases keyed by patient so the pairing step can pick a same-patient swap image.
3. Produce even a one-row real manifest and confirm it loads: `benchmaxxing datasets` lists it, and the dataset smoke in `tests/test_datasets.py` should pass for it.

The adapters currently ship as `NotImplementedError` stubs with a pointer to where the raw data lives; that is the work the open dataset issues track.

Once an adapter exists, [docs/DATASETS.md](docs/DATASETS.md) is the acquisition side: where each release comes from, what it costs to get (open, registration, credentialed), and the `benchmaxxing datasets stage` command that turns a download into a validated manifest with a provenance record. Raw data and credentials never go in the repo.

## 7. Model backend (Gemini for now)

Agents use the **Gemini API** (multimodal) for now, behind one gateway wrapper (`benchmaxxing/gateway.py`), so the roster can be extended to other model APIs later without changing experiment code. Adding a backend is a new `Backend` subclass, nothing else. Model lineage is a first-class variable: Gemini-only committees are the same-lineage control, and Gemini-plus-open-weights committees are the cross-lineage arm, so at least one open-weights family is required for the cross-lineage experiments. No fine-tuning; models are used off-the-shelf.

## 8. Contribution workflow

```bash
git checkout main && git pull
git checkout -b feat/<short-name>
# ... code + tests ...
ruff check . && pytest -q
git commit -m "feat: <what changed>"
git push -u origin feat/<short-name>
# open a PR against main; put "Closes #<issue>" in the description; request reviewers
```

Keep PRs scoped to one issue where possible, and make sure the suite is green and ruff is clean before requesting review. If your change adds prose, keep it plain (no em dashes).

## 9. Reviewers

When a change needs eyes, request review from the core team. Assign the issue you are working on to yourself so ownership is clear on the tracker.

## 10. Data access

No dataset is redistributed in this repository. Each carries its own terms, and what is committed
here is derived per-case outcomes and model responses.

| Dataset | Access | What this repo contains |
|---|---|---|
| MedQA-USMLE | public | per-case rows, summaries, call caches |
| NIH ChestX-ray14 | public | per-case rows, summaries, image-keyed caches; no images |
| MedMCQA | public | per-case rows, summaries, call cache; no sampling manifest |
| CheXpert | Stanford research agreement | per-case rows, summaries; no images |
| SUPPORT2 | public (UCI) | per-case rows, summaries, manifest, provenance, call cache |
| MIMIC-CXR, MIMIC-CXR-JPG | **PhysioNet credentialed** | de-identified per-case rows only. No report text, no images, no caches, no MIMIC identifiers |

**The MIMIC lanes need care.** PhysioNet's terms do not permit redistributing the data. The imaging
lane gitignores its transcripts and caches and ships `results/deid/*.csv`; the text lane's rows are
keyed on a `case_index` local to this repository rather than on MIMIC subject and study ids. Every
rate and paired test is computed from the outcome flags, so the numbers stay recomputable and only the
linkage to source records is removed.

The runners still emit real identifiers when you run them locally. `tests/test_no_credentialed_identifiers.py`
fails if any reach a tracked file, so de-identify before committing rather than relying on care.

## 11. Reproducing the published results

Two levels, and they are not the same claim.

**Recompute** every rate, difference and exact test from the committed per-case rows under
`experiments/*/results/`. Needs no API key and no dataset access, and works for all seven cohorts.

**Replay** a runner end to end at zero API calls, every response served from the committed
content-addressed cache. Available for MedQA, NIH ChestX-ray14, CheXpert and SUPPORT2. Not available
for the MIMIC lanes, whose caches cannot be committed, nor fully for MedMCQA, which has its cache but
not its sampling manifest.

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q     # whole suite, offline, no key
```

A run that reaches an uncached prompt exits instead of calling the API, so a cached claim cannot
quietly become a billed one.

## 12. Guards that are tests, not conventions

Each of these pins a property that failed at least once during the work, which is why it is a test:

- `tests/test_degeneracy_guard.py`: screens every committed per-case file for metrics that cannot fail
  by construction, a comparator constant across all rows, a predicate identical to the label it is
  scored against, two reads that never diverge, a significance verdict hardcoded as a string. New
  findings need an entry in `tests/degeneracy_exemptions.json` explaining why they are legitimate, and
  the pre-existing count is ratcheted so it can only shrink.
- `tests/test_no_credentialed_identifiers.py`: no MIMIC identifier may be committed.
- `tests/test_board_render.py`: every committee runner renders the shared board through one function,
  so a seeded rationale cannot be dropped before it reaches the holdout.

