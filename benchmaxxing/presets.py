"""Ready-to-edit config presets, one per stage of the six-stage program (stages 0-5).

Each preset is a plain, documentation-quality config dict that a contributor copies, edits,
and runs once real data is present. The presets resolve to a :class:`benchmaxxing.config.Config`
via :func:`get_preset`, so they are guaranteed loadable and round-trippable through the same
config layer the rest of the pipeline uses. Nothing here calls a model or touches the network:
a preset is just resolved configuration.

The six stages mirror the build plan and the offline smoke pipeline
(cue injection -> solo baselines -> committee -> cascade onset -> referee -> blind metric):

* ``stage0`` Foundations. Build clean/contaminated twin pairs from a dataset manifest
  (``benchmaxxing.cues`` + ``benchmaxxing.datasets``). Same-lineage roster is enough here.
* ``stage1`` Solo baselines and the lineage-overlap test (``benchmaxxing.analysis``): does a
  single model lean on the injected cue, and do same-lineage models fail on the same cases?
* ``stage2`` Committee blackboard (``benchmaxxing.blackboard``): shared vs isolated context with
  a seeded shortcut turn. Uses the cross-lineage roster so oversight is not one backbone.
* ``stage3`` Cascade-onset dynamics (``benchmaxxing.onset``): where a seeded shortcut tips a
  committee from following the evidence to following the first committed answer.
* ``stage4`` Referee scoring and the gate (``benchmaxxing.referee``): shortcut reliance,
  conformity cascade, and emergent dominance scored over the transcripts.
* ``stage5`` Blind-metric probe (``benchmaxxing.blind_metric``): does the committee's behavior
  drift toward a hidden, meaningless decoy metric, with or without naming it?

How to edit a preset:

* ``models``: gateway ids for the roster. The defaults are the illustrative ids from
  ``roster.default_roster``; swap in real ids once the backends are wired. Same-lineage stages
  (0, 1) ship Gemini-only; the committee/oversight stages (2-5) ship the cross-lineage roster
  (Gemini plus one open-weights family), which the cross-lineage arm requires.
* ``dataset``: a registered adapter name (see ``benchmaxxing datasets``). Every preset defaults
  to the text/MCQ lane (``medqa``) because that lane runs fully offline. To switch a preset to
  the imaging lane (Lane A), set ``dataset`` to an imaging adapter (``mimic_cxr``, ``chexpert``,
  ``nih_cxr14``, or ``openi_cxr``) and ``cue_set`` to ``"image-v1"``.
* ``cue_set``: the cue-set version to inject. ``"text-v1"`` covers the text cues
  (``option_order``, ``longest_option``, ``lexical_overlap``, ``demographic_hint``);
  ``"image-v1"`` covers the image cues (``cable``, ``corner_tag``, ``watermark``, ``laterality``).
* ``seed``: the master seed for reproducibility (one per stage so runs stay distinct); run the
  same preset under several seeds when you want variance.
* ``out_dir``: where this stage's artifacts are written (one subdirectory per stage).
"""

from __future__ import annotations

from benchmaxxing import roster
from benchmaxxing.config import Config, load_config

__all__ = ["STAGE_PRESETS", "get_preset", "list_presets"]

# Model roster hints, derived from the illustrative default roster so the preset ids stay in
# sync with roster.default_roster(). ``list(...)`` gives every preset its own copy, so editing
# one preset's ``models`` never aliases another's.
_ROSTER = roster.default_roster()
_GEMINI_ONLY = [spec.name for spec in _ROSTER if spec.lineage == "gemini"]
_CROSS_LINEAGE = [spec.name for spec in _ROSTER]


# One preset per stage. Insertion order is stage0..stage5, which is the order list_presets and
# the config layer report them in. Every value is a plain dict of Config fields, so each preset
# resolves cleanly through config.load_config with no unknown keys dropped.
STAGE_PRESETS: dict[str, dict] = {
    # Stage 0 - Foundations: build clean/contaminated twin pairs from a dataset manifest.
    "stage0": {
        "models": list(_GEMINI_ONLY),
        "dataset": "medqa",
        "cue_set": "text-v1",
        "seed": 0,
        "out_dir": "runs/stage0",
    },
    # Stage 1 - Solo baselines and the same-lineage overlap test (same-lineage control roster).
    "stage1": {
        "models": list(_GEMINI_ONLY),
        "dataset": "medqa",
        "cue_set": "text-v1",
        "seed": 1,
        "out_dir": "runs/stage1",
    },
    # Stage 2 - Committee blackboard: shared vs isolated context with a seeded shortcut turn.
    "stage2": {
        "models": list(_CROSS_LINEAGE),
        "dataset": "medqa",
        "cue_set": "text-v1",
        "seed": 2,
        "out_dir": "runs/stage2",
    },
    # Stage 3 - Cascade-onset dynamics: when a seeded shortcut tips the committee.
    "stage3": {
        "models": list(_CROSS_LINEAGE),
        "dataset": "medqa",
        "cue_set": "text-v1",
        "seed": 3,
        "out_dir": "runs/stage3",
    },
    # Stage 4 - Referee scoring and the gate over the committee transcripts.
    "stage4": {
        "models": list(_CROSS_LINEAGE),
        "dataset": "medqa",
        "cue_set": "text-v1",
        "seed": 4,
        "out_dir": "runs/stage4",
    },
    # Stage 5 - Blind-metric probe: does behavior drift toward a hidden decoy metric?
    "stage5": {
        "models": list(_CROSS_LINEAGE),
        "dataset": "medqa",
        "cue_set": "text-v1",
        "seed": 5,
        "out_dir": "runs/stage5",
    },
}


def list_presets() -> list[str]:
    """Return the preset names in stage order (``stage0`` .. ``stage5``)."""
    return list(STAGE_PRESETS)


def get_preset(name: str) -> Config:
    """Resolve the named stage preset into a :class:`~benchmaxxing.config.Config`.

    The preset dict is applied through ``config.load_config`` (which delegates to
    ``Config.from_dict``), so the returned config is independent of the stored preset and safe
    to mutate. Raises ``KeyError`` (listing the available names) for an unknown preset.
    """
    try:
        preset = STAGE_PRESETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {list_presets()}"
        ) from None
    return load_config(preset)
