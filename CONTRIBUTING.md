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
```

The smoke is the fastest way to see the whole pipeline compose and to confirm your change did not break a seam. It needs no data and no keys.

`benchmaxxing run` is the reproducible entry point for a stage: it resolves the config (models, cue set, seed), loads the manifest, injects the cues, calls the matching runner in `benchmaxxing/experiments.py`, and writes a self-describing run directory. A config file may be JSON (no extra needed) or YAML (needs the `config` extra). The default backend is Gemini, so a real run needs `GEMINI_API_KEY` and the `models` extra; `--backend mock` runs the same code path offline. The cascade stage also saves every shared and isolated transcript under `--out/transcripts/` for re-analysis.

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
