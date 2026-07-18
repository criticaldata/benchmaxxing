"""Shared data contract for benchmaxxing. Every module builds against these types.

Keep this small and stable: it is the interface the whole pipeline agrees on. Downstream
modules (cues, blackboard, stats, onset, referee, ...) import from here and nowhere else for
their shared types, so the pieces stay compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Condition(str, Enum):
    """The two variants of a cue-injection twin pair (ground truth identical in both)."""

    CLEAN = "clean"                 # original case, no injected cue
    CONTAMINATED = "contaminated"   # identical case with the spurious cue injected


class Modality(str, Enum):
    IMAGE = "image"     # chest X-ray (Lane A)
    TEXT = "text"       # MCQ / report (Lane B)


@dataclass(frozen=True)
class Case:
    """One clinical case in the shared schema (imaging or text)."""

    case_id: str
    patient_id: str
    modality: Modality
    label: str | None = None
    image_ref: str | None = None      # path/handle for an image case
    report: str | None = None         # free-text report (imaging) or context
    question: str | None = None       # MCQ stem (text lane)
    options: tuple[str, ...] | None = None
    answer_index: int | None = None   # index into options of the correct answer
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TwinPair:
    """A clean/contaminated pair: identical except one injected cue. Ground truth unchanged."""

    case_id: str
    cue_type: str                      # e.g. "cable", "option_order", "longest_option"
    cue_params: dict = field(default_factory=dict)
    clean: object = None               # payload for the CLEAN condition (image array / text)
    contaminated: object = None        # payload for the CONTAMINATED condition
    ground_truth: object = None        # the correct answer, identical for both conditions

    def payload(self, condition: Condition):
        return self.clean if condition == Condition.CLEAN else self.contaminated


@dataclass(frozen=True)
class ModelSpec:
    """One model in the roster. ``lineage`` is the family/provider used for the
    same-lineage vs cross-lineage split (the backbone of the overlap and referee arms)."""

    name: str                          # gateway id, e.g. "gemini-2.5-flash"
    lineage: str                       # provider/family, e.g. "gemini", "qwen", "llama"
    tier: str = "default"              # within-lineage size/version axis
    is_open_weights: bool = False


@dataclass(frozen=True)
class Committee:
    """A set of agents on the blackboard, tagged by whether it spans >1 lineage."""

    members: tuple[ModelSpec, ...]

    @property
    def lineages(self) -> set[str]:
        return {m.lineage for m in self.members}

    @property
    def is_cross_lineage(self) -> bool:
        return len(self.lineages) > 1


@dataclass
class Turn:
    """One agent's turn on the blackboard."""

    turn_index: int
    agent_id: str
    content: str
    answer: object = None              # the committed answer this turn (if any)
    confidence: float | None = None
    seeded: bool = False               # True if this turn was injected (planted shortcut)


@dataclass
class Transcript:
    """The full turn-level record of one committee run on one case+condition. Replayable."""

    run_id: str
    case_id: str
    condition: Condition
    turns: list[Turn] = field(default_factory=list)
    committed: dict = field(default_factory=dict)   # agent_id -> final answer
    meta: dict = field(default_factory=dict)


@dataclass
class RunManifest:
    """Everything needed to reproduce a run. Stamped with time/versions by the caller."""

    run_id: str
    model_ids: list[str] = field(default_factory=list)
    prompt_versions: dict = field(default_factory=dict)
    seed: int = 0
    cue_set_version: str = ""
    dataset_revision: str = ""
    library_versions: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
