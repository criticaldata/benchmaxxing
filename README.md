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
- `benchmaxxing/datasets/`: dataset adapters into the shared case schema.
- `benchmaxxing/referee.py`, `blind_metric.py`: the referee duties and the blind-metric probe.

The build plan and the issue tracker for this repo mirror a six-stage program (stages 0-5).
