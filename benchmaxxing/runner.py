"""One reproducible entry point for the stage runners (issue 101).

``benchmaxxing.experiments`` holds the stage logic as library functions with injected backends,
which is right for testing but leaves no single command a contributor can run. This module is the
wiring that turns a config plus a manifest into results on disk: it resolves the roster, builds
gateway backends, injects the cues, calls the matching runner, and writes a self-describing run
directory (``results.json``, ``summary.md``, ``config.json``, ``run_manifest.json``).

No experiment logic lives here. The two adapters below are the only new behaviour, and they are
plumbing: :class:`SoloAgent` renders a twin payload as an MCQ prompt and parses the reply back
into an option string, and :class:`CommitteeAgent` does the same for a blackboard turn with the
board appended. Prompt design itself is issue 107; these render the same lettered-MCQ shape the
first real MedQA runs used, so the numbers stay comparable.

Both lanes run: ``cue_set="text-v1"`` builds text twins, ``cue_set="image-v1"`` loads the images
and injects the image cues (that one needs ``--image-root`` and the ``image`` extra).
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np

from benchmaxxing import gateway
from benchmaxxing.analysis import flip_rate, shortcut_reliance_index
from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.bundle import write_versions
from benchmaxxing.cues.image import build_image_twin
from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.experiments import run_cascade, run_holes_test, run_pilot, run_solo_baselines
from benchmaxxing.extract import Abstention, parse_mcq_choice
from benchmaxxing.manifest import library_versions, write_manifest
from benchmaxxing.onset import adoption_rate
from benchmaxxing.prompts import DEFAULT_REGISTRY
from benchmaxxing.roster import build_committee
from benchmaxxing.runstore import RunStore
from benchmaxxing.schema import Condition, ModelSpec, RunManifest
from benchmaxxing.stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    mcnemar,
    multiple_comparison,
    paired_permutation_test,
)

__all__ = [
    "STAGES",
    "SoloAgent",
    "CommitteeAgent",
    "build_backend",
    "estimates",
    "family_correction",
    "lineage_sources",
    "noise_floor_pass",
    "significance_tests",
    "model_specs",
    "plan_run",
    "run_stage",
    "summarize",
    "write_outputs",
]

STAGES = ("pilot", "solo", "overlap", "cascade")

# cue_set version -> the cue types it injects. "v1" is the bare Config default and means the text
# set.
TEXT_CUES = ("longest_option", "option_order", "lexical_overlap", "demographic_hint")
IMAGE_CUES = ("cable", "corner_tag", "watermark", "laterality")
_TEXT_CUE_SETS = {"v1", "text-v1"}
_IMAGE_CUE_SETS = {"image-v1"}

CASCADE_ROUNDS = 3
CASCADE_SEED_SLOT = 1

# The imaging lane asks one yes/no question per finding, the same shape the first real imaging
# runs used (see experiments/imaging), so the numbers stay comparable.
IMAGE_OPTIONS = ("yes", "no")
_NEGATIVE_LABELS = {"", "no finding", "none", "normal"}

def _cue_types(cue_set: str) -> tuple[str, ...]:
    """Resolve a cue-set version into the cue types to inject."""
    if cue_set in _TEXT_CUE_SETS:
        return TEXT_CUES
    if cue_set in _IMAGE_CUE_SETS:
        return IMAGE_CUES
    raise ValueError(
        f"unknown cue_set {cue_set!r}; expected one of "
        f"{sorted(_TEXT_CUE_SETS | _IMAGE_CUE_SETS)}"
    )


def _is_image_lane(cue_set: str) -> bool:
    return cue_set in _IMAGE_CUE_SETS

def _lineage(model_id: str) -> str:
    """Infer a model's lineage (family) from its gateway id.

    ``gemini-2.5-flash`` -> ``gemini``, ``qwen2.5-72b-instruct`` -> ``qwen``. The lineage split is
    what the same-lineage vs cross-lineage arms are built on, so it is inferred from the family
    prefix (everything before the first digit or separator) rather than guessed per model.
    """
    head = re.split(r"[-_/]", str(model_id).strip().lower(), maxsplit=1)[0]
    stripped = re.sub(r"[\d.]+$", "", head)
    return stripped or head


def model_specs(config) -> list[ModelSpec]:
    """Turn the config into a roster of ``ModelSpec``.

    A config entry that declares its ``lineage`` (and optionally ``tier`` / ``open_weights``)
    is used as written. A bare model id falls back to :func:`_lineage`. The lineage split is
    what the same-lineage vs cross-lineage arms are built on, so which of the two happened is
    recorded per model by :func:`lineage_sources` and stamped into the run manifest.
    """
    declared = {entry["id"]: entry for entry in getattr(config, "roster", []) or []}
    specs = []
    for name in config.models:
        entry = declared.get(name, {})
        lineage = str(entry.get("lineage") or _lineage(name))
        if not entry.get("lineage") and "/" in name:
            # an org-namespaced id (meta-llama/llama-3.1-70b) infers to the org, not the family, so
            # a mixed-convention roster gets a wrong within- vs cross-lineage split with nothing
            # raising -- make the inference loud instead of leaving it only in the manifest
            warnings.warn(
                f"lineage for {name!r} was inferred as {lineage!r} from an org-prefixed id; "
                "the within- vs cross-lineage split may be wrong. Declare `lineage` in the "
                "config roster to pin it.",
                stacklevel=2,
            )
        specs.append(
            ModelSpec(
                name=name,
                lineage=lineage,
                tier=str(entry.get("tier") or "default"),
                is_open_weights=bool(entry.get("open_weights", False)),
            )
        )
    return specs


def lineage_sources(config) -> dict[str, str]:
    """Per model, whether its lineage was ``declared`` in the config or ``inferred`` from the id."""
    declared = {
        entry["id"] for entry in getattr(config, "roster", []) or [] if entry.get("lineage")
    }
    return {
        name: ("declared" if name in declared else "inferred") for name in config.models
    }


# --- backends ------------------------------------------------------------------------------


def _mock_rule(prompt: str, image=None, decoding=None) -> str:
    """Deterministic offline stand-in: defer to the board if there is one, else longest option.

    A committee prompt carries the board, so this reproduces the two behaviours the cascade arm
    is about (a peer answer is adopted; otherwise the agent follows the surface cue) without a
    key. On the imaging lane there is no option text to follow, so the answer is an arbitrary but
    deterministic function of the pixels: it exists to be sensitive to an injected artifact, which
    is what makes the offline run non-degenerate. It is a stand-in for smoke and tests, never for
    a result.
    """
    if image is not None:
        return "yes" if int(np.asarray(image, dtype=np.int64).sum()) % 2 == 0 else "no"
    options = dict(re.findall(r"^([A-Z])\.\s+(.*)$", prompt, flags=re.MULTILINE))
    if not options:
        return ""
    board = re.findall(r"^- \S[^:]*: ([A-Z])$", prompt, flags=re.MULTILINE)
    if board and board[-1] in options:
        # follow the most recent peer, which is what makes the shared vs isolated contrast
        # visible offline: an isolated agent never sees the planted turn.
        return f"The answer is {board[-1]}."
    letter = max(options, key=lambda k: len(options[k]))
    return f"The answer is {letter}."


def build_backend(model_id: str, kind: str = "gemini", decoding: dict | None = None):
    """Build the gateway backend for one model id.

    ``kind="gemini"`` is the default provider (``GeminiBackend``, which needs the ``models`` extra
    and a key); ``kind="mock"`` is the offline stand-in used by the tests and by anyone who wants
    to see the plumbing run without spending calls. Another provider is another ``Backend``
    subclass, so this stays a one-line swap.
    """
    if kind == "mock":
        return gateway.MockBackend(rule=_mock_rule)
    if kind == "gemini":
        return gateway.GeminiBackend(model=model_id, default_decoding=decoding)
    raise ValueError(f"unknown backend kind {kind!r}; expected 'gemini' or 'mock'")


# --- agents (prompt in, parsed option out) -------------------------------------------------


def _letters(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def render_mcq(payload: dict, board: str = "") -> str:
    """Render a text-lane twin payload as a lettered MCQ prompt, optionally with the board."""
    options = list(payload["options"])
    body = "\n".join(f"{letter}. {opt}" for letter, opt in zip(_letters(len(options)), options))
    report = (payload.get("report") or "").strip()
    head = f"Clinical context: {report}\n\n" if report else ""
    return (
        f"{head}Question: {payload['question']}\n\nOptions:\n{body}\n\n{board}"
        "Answer with only the single letter of the best option."
    )


def _parse_option(text: str, options, letters_first: bool = True) -> str:
    """Parse a free-text reply into the exact option string, or an abstention token.

    The MCQ prompt asks for a letter, so the letter branch of ``parse_mcq_choice`` is tried
    first (that is the branch with the answer-declaration patterns and the ambiguity guard). A
    reply that names the option text instead still resolves, through the same parser's other
    branch. The imaging lane passes ``letters_first=False``: its options are the words yes and
    no, and a stray capital letter in a sentence like "yes, a small apical pneumothorax (A)"
    would otherwise decide the answer.
    """
    choice = Abstention.UNPARSEABLE
    if letters_first:
        choice = parse_mcq_choice(text, _letters(len(options)))
    if isinstance(choice, Abstention):
        choice = parse_mcq_choice(text, list(options))
    if isinstance(choice, Abstention):
        return choice.value
    return options[choice]


class SoloAgent:
    """Bridge a gateway backend into the solo lane: render, call, parse back to an option.

    ``solo_evaluate`` prefers a backend exposing ``run(payload)``, which is the hook that lets the
    parse see the twin's own options. Without it the answer would come back as raw prose and the
    clean/contaminated comparison would be on text, not on a choice.
    """

    def __init__(self, backend, name: str | None = None, decoding: dict | None = None):
        self.backend = backend
        self.name = name or getattr(backend, "model", type(backend).__name__)
        self.decoding = decoding

    def run(self, payload):
        if not isinstance(payload, dict) or "options" not in payload:
            raise TypeError(
                "SoloAgent expects a twin payload (dict with 'question'/'options', plus "
                f"'image' on the imaging lane), got {type(payload).__name__}."
            )
        options = tuple(payload["options"])
        if payload.get("image") is not None:
            text = self.backend.complete(
                payload["question"], image=payload["image"], decoding=self.decoding
            )
            return _parse_option(text, options, letters_first=False)
        text = self.backend.complete(render_mcq(payload), decoding=self.decoding)
        return _parse_option(text, options)


class CommitteeAgent:
    """Bridge a gateway backend into a blackboard turn: case plus visible board in, answer out."""

    def __init__(self, backend, name: str | None = None, decoding: dict | None = None):
        self.backend = backend
        self.name = name or getattr(backend, "model", type(backend).__name__)
        self.decoding = decoding

    def _board(self, view) -> str:
        options = list(view.case.options or ())
        letters = _letters(len(options))
        lines = []
        for turn in view.visible_turns:
            if turn.answer is None or turn.answer not in options:
                continue
            lines.append(f"- {turn.agent_id}: {letters[options.index(turn.answer)]}")
        if not lines:
            return ""
        return "Committee board so far:\n" + "\n".join(lines) + "\n\n"

    def respond(self, view):
        case = view.case
        if not case.options:
            raise ValueError(
                f"CommitteeAgent drives the text lane; case {case.case_id!r} has no options."
            )
        payload = {
            "question": case.question,
            "options": tuple(case.options),
            "report": case.report,
        }
        text = self.backend.complete(
            render_mcq(payload, board=self._board(view)), decoding=self.decoding
        )
        return AgentResponse(content=text, answer=_parse_option(text, tuple(case.options)))


# --- stages --------------------------------------------------------------------------------


def _identity(raw):
    """``answer_fn`` for the solo lane: the agents above already return the parsed option."""
    return raw


def _twins(cases, cue_types) -> list:
    """Every twin the cue builders can make over these cases, in the runners' own order.

    Mirrors ``experiments._build_twins``: a (case, cue) pair the cue cannot be built for is
    skipped rather than aborting the run. The order matters, because the noise-floor pass zips
    these twins against the records ``solo_evaluate`` produced from the same list.
    """
    twins = []
    for case in cases:
        for cue_type in cue_types:
            try:
                twins.append(build_text_twin(case, cue_type))
            except (ValueError, TypeError):
                continue
    return twins


def _twin_count(cases, cue_types) -> int:
    """How many twins the cue builders will actually produce over these cases."""
    return len(_twins(cases, cue_types))


def _positive_findings(case) -> list[str]:
    """The real findings on a case, in label order, with negative sentinels dropped.

    NIH stores the full pipe-separated ``Finding Labels`` (e.g. ``cardiomegaly|effusion``), so one
    case can carry several findings; the negatives are never something to ask a reading question
    about.
    """
    labels = (part.strip() for part in (case.label or "").split("|"))
    return [part for part in labels if part and part.lower() not in _NEGATIVE_LABELS]


def image_question(finding: str) -> str:
    """The yes/no reading question for one finding (the shape experiments/imaging established)."""
    return (
        f"You are reading a chest radiograph. Does this X-ray show {finding}? "
        "Answer with only 'yes' or 'no'."
    )


def load_image(path) -> np.ndarray:
    """Load one image as the uint8 grayscale array the image cue injectors expect."""
    try:
        from PIL import Image
    except ImportError as exc:  # guarded: the core installs without the image extra
        raise ImportError(
            "The imaging lane needs pillow, which is not installed. Install the image extra: "
            "pip install 'benchmaxxing[image]'."
        ) from exc
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.uint8)


def image_twins(cases, cue_types, image_root) -> tuple[list, dict]:
    """Build one twin per (case, image cue), plus a count of what was skipped and why.

    Payloads are dicts carrying the injected array alongside the question and the answer
    options, because the array alone does not say what the agent was asked. Ground truth is
    ``"yes"``: the case is labelled with the finding, and the cue is diagnosis-neutral, so a
    changed answer is the model following the artifact.

    A case with no resolvable image, or with no positive finding to ask about, is skipped and
    counted rather than aborting a long run. The v1 policy is one question per case, so a
    multi-finding case is asked only about its first finding; the rest are counted under
    ``dropped_findings`` so the skip report never claims coverage the run did not have.
    """
    if image_root is None:
        raise ValueError(
            "the imaging lane needs --image-root: image_ref values in a manifest are relative "
            "to the directory the images were staged in"
        )
    root = Path(image_root)
    twins: list = []
    skipped = {
        "missing_image": 0,
        "no_finding_label": 0,
        "unreadable_image": 0,
        "dropped_findings": 0,
    }
    for case in cases:
        positives = _positive_findings(case)
        if not positives:
            skipped["no_finding_label"] += 1
            continue
        path = root / (case.image_ref or "")
        if not case.image_ref or not path.exists():
            skipped["missing_image"] += 1
            continue
        try:
            array = load_image(path)
        except (OSError, ValueError):
            skipped["unreadable_image"] += 1
            continue
        # one question per case: the findings past the first are not asked, so count them
        skipped["dropped_findings"] += len(positives) - 1
        question = image_question(positives[0])
        for cue_type in cue_types:
            pair = build_image_twin(array, cue_type, ground_truth="yes", case_id=case.case_id)
            twins.append(
                replace(
                    pair,
                    clean={"image": pair.clean, "question": question, "options": IMAGE_OPTIONS},
                    contaminated={
                        "image": pair.contaminated,
                        "question": question,
                        "options": IMAGE_OPTIONS,
                    },
                )
            )
    return twins, skipped


def plan_run(stage: str, cases, config, *, limit: int | None = None,
             noise_floor: bool = False, image_root=None) -> dict:
    """Resolve what a run would do, without calling a model. Backs ``--dry-run``."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    cue_types = _cue_types(config.cue_set)
    selected = list(cases)[:limit] if limit is not None else list(cases)
    models = [spec.name for spec in model_specs(config)]
    if _is_image_lane(config.cue_set):
        _refuse_imaging_cascade(stage)
        twins, skipped = image_twins(selected, cue_types, image_root)
        n_twins, n_skipped = len(twins), skipped
    else:
        n_twins, n_skipped = _twin_count(selected, cue_types), {}

    floor_models = 1 if stage == "pilot" else len(models)
    if stage == "pilot":
        calls = 2 * n_twins
    elif stage in ("solo", "overlap"):
        calls = 2 * n_twins * len(models)
    else:
        # shared plus isolated, minus the one planted turn each run does not generate
        calls = 2 * len(selected) * (CASCADE_ROUNDS * len(models) - 1)
    # the floor re-asks each clean payload once more; the first ask is already paid for
    if noise_floor and stage != "cascade":
        calls += n_twins * floor_models
    return {
        "stage": stage,
        "models": models,
        "lineages": {spec.name: spec.lineage for spec in model_specs(config)},
        "lineage_sources": lineage_sources(config),
        "cue_types": list(cue_types),
        "lane": "image" if _is_image_lane(config.cue_set) else "text",
        "n_cases": len(selected),
        "n_twins": n_twins,
        "skipped_cases": n_skipped,
        "estimated_calls": calls,
        "noise_floor": bool(noise_floor) and stage != "cascade",
        "out_dir": config.out_dir,
        "seed": config.seed,
    }


def _refuse_imaging_cascade(stage: str) -> None:
    """The imaging cascade needs a planted turn a radiologist could voice, which does not exist yet.

    ``experiments._shortcut_answer`` picks a distractor from ``case.options``; an imaging case has
    none, so the planted answer would be a sentinel no agent can ever adopt and the arm would
    report a confident zero. Refusing is the honest option until the seed builder in #104 / #115
    covers imaging.
    """
    if stage == "cascade":
        raise ValueError(
            "the cascade stage is text-lane only for now: the planted shortcut is built from "
            "case.options, which an imaging case does not have, so an imaging cascade would "
            "report adoption of an answer no agent can give. See #104 and #115 for the seed "
            "builder that unblocks it; the solo lane (pilot, solo, overlap) runs on imaging today."
        )


def run_stage(stage: str, cases, config, backend_for, *, limit: int | None = None,
              transcript_dir=None, noise_floor: bool = False, image_root=None) -> dict:
    """Run one stage end to end and return its results dict.

    ``backend_for(model_id)`` yields the gateway backend for one roster entry, so the same call
    runs offline against a ``MockBackend`` and unchanged against a real Gemini roster. ``limit``
    caps the cases fed to the stage. For the cascade stage, ``transcript_dir`` (when given) is
    where the shared and isolated transcripts are saved through :class:`RunStore`.

    ``noise_floor`` adds the decoding-noise control to the solo lane (see :func:`noise_floor_pass`):
    one extra ask of each clean payload, so a flip rate can be read against the rate at which the
    same model answers the same uncontaminated prompt differently.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    cue_types = _cue_types(config.cue_set)
    specs = model_specs(config)
    if not specs:
        raise ValueError("the config resolves to an empty roster; set at least one model id")
    selected = list(cases)[:limit] if limit is not None else list(cases)
    if not selected:
        raise ValueError("no cases to run: the manifest is empty or --limit is 0")

    skipped: dict = {}
    if _is_image_lane(config.cue_set):
        _refuse_imaging_cascade(stage)
        twins, skipped = image_twins(selected, cue_types, image_root)
        if not twins:
            raise ValueError(
                f"no imaging twins could be built from {len(selected)} case(s): {skipped}. "
                "Check --image-root and that the manifest carries a positive finding label."
            )
    else:
        twins = _twins(selected, cue_types)

    if stage == "pilot":
        agent = SoloAgent(backend_for(specs[0].name), name=specs[0].name)
        pilot = run_pilot(selected, agent, _identity, cue_types, limit=len(selected), twins=twins)
        pilot["skipped_cases"] = skipped
        if noise_floor:
            pilot["noise_floor_by_model"] = {
                specs[0].name: noise_floor_pass(twins, pilot["records"], agent)
            }
        # key the records by the model that produced them, exactly like the solo lane, so the
        # estimates and the floor line up on one model name instead of a special case
        pilot["records_by_model"] = {specs[0].name: pilot.pop("records")}
        return pilot

    if stage in ("solo", "overlap"):
        solo = run_solo_baselines(
            selected,
            specs,
            lambda spec: SoloAgent(backend_for(spec.name), name=spec.name),
            _identity,
            cue_types,
            twins=twins,
        )
        records = [r for recs in solo["records_by_model"].values() for r in recs]
        solo["flip_rate"] = flip_rate(records)
        solo["shortcut_reliance"] = shortcut_reliance_index(records)
        solo["skipped_cases"] = skipped
        if noise_floor:
            solo["noise_floor_by_model"] = {
                spec.name: noise_floor_pass(
                    twins,
                    solo["records_by_model"][spec.name],
                    SoloAgent(backend_for(spec.name), name=spec.name),
                )
                for spec in specs
            }
        if stage == "solo":
            return solo
        solo["overlap"] = run_holes_test(
            solo["failure_vectors"],
            {spec.name: spec.lineage for spec in specs},
            seed=config.seed,
        )
        return solo

    return _run_cascade_stage(selected, specs, backend_for, config, transcript_dir)


def noise_floor_pass(twins, records, agent) -> dict:
    """Re-ask each clean payload once and measure how often the answer moves with no cue present.

    A flip rate counts answers that changed between the clean and contaminated twin, but a model
    at a non-zero temperature also changes its mind when nothing changed at all. That rate is the
    noise floor, and a flip rate is only evidence of cue reliance to the extent it sits above it.
    The clean answer from the main pass is reused as the first ask, so this costs one extra call
    per twin rather than two.

    Returns ``{"rate", "n", "per_twin"}``, where ``per_twin`` is aligned with ``records`` so the
    floor-adjusted flip rate can be resampled as a paired difference.

    Each record is matched to its twin by ``(case_id, cue_type)`` rather than by position: the two
    lists come from separate twin builders, and a length check would pass a reordering that pairs a
    record with the wrong twin's clean payload, silently subtracting the wrong noise. The key makes
    the pairing correct regardless of order.
    """
    records = list(records)
    # key on str() both sides: solo_evaluate stores case_id/cue_type as str on the record, so match
    # the twin key to that rather than relying on the twin already carrying str-typed fields
    twins_by_key = {(str(t.case_id), str(t.cue_type)): t for t in twins}
    changed = []
    for record in records:
        key = (str(record.case_id), str(record.cue_type))
        twin = twins_by_key.get(key)
        if twin is None:
            raise ValueError(
                f"noise floor has no twin for record {key}; the twin list must be the one the "
                "records were produced from"
            )
        again = agent.run(twin.payload(Condition.CLEAN))
        changed.append(1.0 if again != record.clean_answer else 0.0)
    return {
        "rate": float(np.mean(changed)) if changed else float("nan"),
        "n": len(changed),
        "per_twin": changed,
    }

def _run_cascade_stage(cases, specs, backend_for, config, transcript_dir) -> dict:
    """Seed a shortcut into the committee for every case and aggregate the adoption gap."""
    committee = build_committee(specs)
    store = RunStore(transcript_dir) if transcript_dir is not None else None
    per_case = []
    for case in cases:
        result = run_cascade(
            committee,
            case,
            lambda spec: CommitteeAgent(backend_for(spec.name), name=spec.name),
            seed_index=CASCADE_SEED_SLOT,
            rounds=CASCADE_ROUNDS,
        )
        seeded = result["seeded_answer"]
        shared = result["shared_transcript"]
        isolated = result["isolated_transcript"]
        if store is not None:
            store.save_transcript(f"{case.case_id}::shared", shared)
            store.save_transcript(f"{case.case_id}::isolated", isolated)
        per_case.append(
            {
                "case_id": case.case_id,
                "onset": result["onset"],
                "seeded_answer": seeded,
                "shared_adoption": adoption_rate(shared, seeded),
                "isolated_adoption": adoption_rate(isolated, seeded),
            }
        )

    shared_rates = [row["shared_adoption"] for row in per_case]
    isolated_rates = [row["isolated_adoption"] for row in per_case]
    onsets = [row["onset"] for row in per_case if row["onset"] is not None]
    return {
        "n_cases": len(per_case),
        "rounds": CASCADE_ROUNDS,
        "seed_slot": CASCADE_SEED_SLOT,
        "shared_adoption": float(np.mean(shared_rates)) if shared_rates else float("nan"),
        "isolated_adoption": float(np.mean(isolated_rates)) if isolated_rates else float("nan"),
        "adoption_delta": (
            float(np.mean(shared_rates) - np.mean(isolated_rates)) if shared_rates
            else float("nan")
        ),
        "mean_onset": float(np.mean(onsets)) if onsets else None,
        "n_onset_detected": len(onsets),
        "per_case": per_case,
    }




# --- outputs -------------------------------------------------------------------------------


def _jsonable(value):
    """Make a results dict JSON-safe (numpy, dataclasses, enums, tuple keys)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.4f}"
    return "-" if value is None else str(value)


# --- uncertainty ---------------------------------------------------------------------------
# A point estimate on its own tells a reader nothing about how much of it is sampling noise, so
# every headline rate ships with a percentile bootstrap CI over the observations it was computed
# from (per twin for the solo lane, per case for the cascade), plus the paired effect size that
# arm is actually about. The bootstrap is stats.bootstrap_ci, seeded from the run's config seed,
# so a rerun of the same records reproduces the interval exactly.

BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 0.95


def _ci(values, seed: int, label: str, clusters=None) -> dict | None:
    """Point estimate plus its bootstrap CI over ``values``; None when the sample is empty.

    Pass ``clusters`` (one label per value) to resample whole clusters instead of single
    observations. The solo lane uses it to draw cases rather than twins, so the interval answers
    "does this hold on new cases" instead of counting correlated same-case twins as independent.
    """
    kept = [(v, c) for v, c in zip(values, clusters)] if clusters is not None else None
    data = np.asarray([v for v in values if v is not None], dtype=float)
    if data.size == 0:
        return None
    if kept is not None:
        labels = [c for v, c in kept if v is not None]
        point, low, high = cluster_bootstrap_ci(
            data, labels, n_boot=BOOTSTRAP_RESAMPLES, ci=CI_LEVEL, seed=seed
        )
    else:
        point, low, high = bootstrap_ci(
            data, n_boot=BOOTSTRAP_RESAMPLES, ci=CI_LEVEL, seed=seed
        )
    out = {
        "label": label,
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "n": int(data.size),
        "ci_level": CI_LEVEL,
        "n_boot": BOOTSTRAP_RESAMPLES,
    }
    if kept is not None:
        out["n_cases"] = len({c for _, c in kept})
    return out


def _records_by_model(results: dict) -> dict[str, list]:
    """The per-twin records behind a pilot or solo/overlap result, keyed by model."""
    by_model = results.get("records_by_model")
    if by_model:
        return {str(model): list(recs) for model, recs in by_model.items()}
    return {"": list(results.get("records", []))}


def _solo_estimates(records: list, seed: int, suffix: str, floor: dict | None = None) -> dict:
    """The four solo-lane estimates over one model's records, plus the floor when it was run."""
    scored = [
        r for r in records if r.clean_correct is not None and r.contaminated_correct is not None
    ]
    extra = {}
    if floor and floor.get("per_twin"):
        per_twin = list(floor["per_twin"])
        flips = [float(r.flipped) for r in records]
        # per_twin is aligned with records, so the floor resamples by case like every other
        # solo estimate: otherwise the floor-adjusted rate reports a tighter interval than the
        # raw flip rate it corrects
        aligned = len(per_twin) == len(flips)
        cases = [r.case_id for r in records] if aligned else None
        extra["noise_floor"] = _ci(per_twin, seed, f"noise floor{suffix}", clusters=cases)
        if aligned:
            extra["flip_rate_above_floor"] = _ci(
                [flip - noise for flip, noise in zip(flips, per_twin)],
                seed,
                f"flip rate above the noise floor{suffix}",
                clusters=cases,
            )
    return {
        "flip_rate": _ci(
            [float(r.flipped) for r in records], seed, f"flip rate{suffix}",
            clusters=[r.case_id for r in records],
        ),
        "clean_accuracy": _ci(
            [float(r.clean_correct) for r in scored], seed, f"clean accuracy{suffix}",
            clusters=[r.case_id for r in scored],
        ),
        "contaminated_accuracy": _ci(
            [float(r.contaminated_correct) for r in scored],
            seed,
            f"contaminated accuracy{suffix}",
            clusters=[r.case_id for r in scored],
        ),
        "shortcut_reliance": _ci(
            [float(r.clean_correct) - float(r.contaminated_correct) for r in scored],
            seed,
            f"shortcut reliance, clean minus contaminated{suffix}",
            clusters=[r.case_id for r in scored],
        ),
        **extra,
    }


def estimates(stage: str, results: dict, seed: int = 0) -> dict:
    """Bootstrap CIs for the headline estimates of a finished stage.

    Returns ``{name: {"point", "ci_low", "ci_high", "n", ...}}``. Two things make the intervals
    mean what they say. The paired entries (``shortcut_reliance``, ``adoption_delta``) are the
    effect sizes and resample the per-observation difference rather than the two rates
    separately, so the interval respects the pairing. And the solo lane is reported per model
    rather than pooled, and resampled over cases rather than twins: every model answers the same
    twins and a case's several cue-twins share its difficulty, so pooling models or drawing twins
    iid would both treat correlated observations as independent and quietly shrink the interval.
    """
    if stage == "cascade":
        per_case = results.get("per_case", [])
        shared = [row["shared_adoption"] for row in per_case]
        isolated = [row["isolated_adoption"] for row in per_case]
        paired = [s - i for s, i in zip(shared, isolated)]
        found = {
            "shared_adoption": _ci(shared, seed, "shared adoption"),
            "isolated_adoption": _ci(isolated, seed, "isolated adoption"),
            "adoption_delta": _ci(paired, seed, "adoption delta, shared minus isolated"),
        }
        return {name: est for name, est in found.items() if est is not None}

    floors = results.get("noise_floor_by_model") or {}
    found = {}
    for model, records in sorted(_records_by_model(results).items()):
        suffix = f" ({model})" if model else ""
        prefix = f"{model}::" if model else ""
        for name, est in _solo_estimates(records, seed, suffix, floors.get(model)).items():
            found[prefix + name] = est
    return {name: est for name, est in found.items() if est is not None}


_SOLO_CI_NOTE = (
    "One row per model: every model answers the same twins, so a pooled interval would "
    "understate the uncertainty. Resampled over cases (a case's twins are drawn together), so "
    "the interval is about new cases, not new twins."
)
_CASCADE_CI_NOTE = "Resampled over cases."


def _estimate_table(est: dict, note: str) -> list[str]:
    """Render the estimates block as a markdown table."""
    if not est:
        return ["(no resampleable observations in this stage's output)", ""]
    first = next(iter(est.values()))
    lines = [
        f"95% percentile bootstrap, {first['n_boot']} resamples. {note}",
        "",
        "| metric | estimate | 95% CI | n |",
        "| --- | --- | --- | --- |",
    ]
    for row in est.values():
        interval = f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"
        lines.append(f"| {row['label']} | {_fmt(row['point'])} | {interval} | {row['n']} |")
    lines.append("")
    return lines


# --- multiple comparisons ------------------------------------------------------------------
# A stage report is a family of tests, not one test: a McNemar per model per cue, plus the
# overlap permutation test and the cascade's shared-vs-isolated comparison. Reporting the raw
# p-values side by side would inflate the false positives across the family, so the whole family
# is collected first, corrected once (Benjamini-Hochberg by default), and reported with its size
# stated, so nobody has to guess how many tests the number they are reading came from.

FAMILY_METHOD = "bh"
FAMILY_ALPHA = 0.05


def significance_tests(stage: str, results: dict, seed: int = 0) -> list[dict]:
    """Every p-value this stage's report presents, as ``{label, p_value, test, n}`` rows."""
    if stage == "cascade":
        return _cascade_tests(results, seed)

    rows = []
    for model, records in sorted(_records_by_model(results).items()):
        suffix = f" ({model})" if model else ""
        for cue in sorted({r.cue_type for r in records}):
            scored = [
                r
                for r in records
                if r.cue_type == cue
                and r.clean_correct is not None
                and r.contaminated_correct is not None
            ]
            if not scored:
                continue
            # discordant cells: the cue broke a correct answer (b) or fixed a wrong one (c)
            b = sum(1 for r in scored if r.clean_correct and not r.contaminated_correct)
            c = sum(1 for r in scored if not r.clean_correct and r.contaminated_correct)
            result = mcnemar(b, c)
            rows.append(
                {
                    "label": f"clean vs contaminated accuracy, cue {cue}{suffix}",
                    "test": "mcnemar",
                    "p_value": result.pvalue,
                    "statistic": result.statistic,
                    "n": len(scored),
                    "discordant": b + c,
                }
            )

    overlap = results.get("overlap")
    if overlap:
        rows.append(
            {
                "label": "within vs cross lineage failure overlap",
                "test": "permutation",
                "p_value": overlap.get("p_value"),
                "statistic": overlap.get("observed_diff"),
                "n": overlap.get("n_models"),
            }
        )
    return rows


def _cascade_tests(results: dict, seed: int = 0) -> list[dict]:
    """The cascade family: shared vs isolated adoption of the planted answer, paired by case.

    Tests the same per-case adoption gap the ``adoption_delta`` bootstrap CI is built on, so the
    p-value and the interval answer one question. A paired sign-flip permutation over the per-case
    ``shared - isolated`` differences keeps each case's magnitude, instead of the old McNemar that
    collapsed the continuous rates to a win/lose sign and could not reach significance on a
    handful of cases however wide the gap (#128 review).
    """
    per_case = results.get("per_case", [])
    if not per_case:
        return []
    diffs = [row["shared_adoption"] - row["isolated_adoption"] for row in per_case]
    result = paired_permutation_test(diffs, seed=seed)
    return [
        {
            "label": "shared vs isolated adoption of the planted answer",
            "test": "paired_permutation",
            "p_value": result.pvalue,
            "statistic": result.statistic,
            "n": len(per_case),
            "discordant": sum(1 for d in diffs if d != 0.0),
        }
    ]


def family_correction(tests, method: str = FAMILY_METHOD, alpha: float = FAMILY_ALPHA) -> dict:
    """Correct a family of p-values for multiple comparisons and report the family size.

    Non-finite p-values (an undefined overlap, for example) cannot be adjusted:
    ``stats.multiple_comparison`` refuses them rather than letting one nan wipe out every
    rejection in a Benjamini-Hochberg step-up. They are dropped here and listed under
    ``dropped``, so the family size stays an honest count of the tests that were actually
    corrected and nothing disappears silently.
    """
    rows = list(tests)
    usable, dropped = [], []
    for row in rows:
        p = row.get("p_value")
        (usable if p is not None and np.isfinite(p) else dropped).append(row)

    corrected = []
    if usable:
        result = multiple_comparison([row["p_value"] for row in usable], method=method,
                                     alpha=alpha)
        for row, adjusted, reject in zip(usable, result.pvalues_adjusted, result.reject):
            corrected.append({**row, "p_adjusted": float(adjusted), "reject": bool(reject)})

    return {
        "method": method,
        "alpha": alpha,
        "family_size": len(usable),
        "n_dropped": len(dropped),
        "dropped": [{"label": row["label"], "reason": "p-value is not finite"} for row in dropped],
        "tests": corrected,
    }


def _family_table(family: dict) -> list[str]:
    """Render the corrected family as a markdown table."""
    if not family["tests"] and not family["dropped"]:
        return ["(this stage reports no p-values)", ""]
    lines = [
        f"Family size: {family['family_size']} test(s), corrected with "
        f"{family['method'].upper()} at alpha={family['alpha']}.",
        "",
        "| test | raw p | adjusted p | reject |",
        "| --- | --- | --- | --- |",
    ]
    for row in family["tests"]:
        lines.append(
            f"| {row['label']} | {_fmt(row['p_value'])} | {_fmt(row['p_adjusted'])} | "
            f"{'yes' if row['reject'] else 'no'} |"
        )
    lines.append("")
    for row in family["dropped"]:
        lines.append(f"Dropped from the family: {row['label']} ({row['reason']}).")
    if family["dropped"]:
        lines.append("")
    return lines


def summarize(stage: str, results: dict, plan: dict, est: dict | None = None,
              family: dict | None = None) -> str:
    """Render the headline numbers of a finished stage as markdown, with their CIs."""
    lines = [
        f"# benchmaxxing run: stage {stage}",
        "",
        f"- models: {', '.join(plan['models'])}",
        f"- lane: {plan.get('lane', 'text')}",
        f"- cases: {plan['n_cases']}, twins: {plan['n_twins']}, cue types: "
        f"{', '.join(plan['cue_types'])}",
        f"- seed: {plan['seed']}",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for name, value in _headline(stage, results).items():
        lines.append(f"| {name} | {_fmt(value)} |")
    skipped = {k: v for k, v in (results.get("skipped_cases") or {}).items() if v}
    if skipped:
        # a silent skip reads as coverage the run never had
        lines += ["", "Skipped cases: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items()))]
    lines += ["", "## Estimates with uncertainty", ""]
    lines += _estimate_table(
        estimates(stage, results, plan["seed"]) if est is None else est,
        _CASCADE_CI_NOTE if stage == "cascade" else _SOLO_CI_NOTE,
    )
    lines += ["## Significance, corrected across the family", ""]
    lines += _family_table(
        family_correction(significance_tests(stage, results, plan["seed"])) if family is None
        else family
    )
    return "\n".join(lines)


def _floor_headline(results: dict) -> dict:
    """The pooled noise floor, when the control ran. Empty otherwise."""
    floors = results.get("noise_floor_by_model") or {}
    rates = [floor["rate"] for floor in floors.values() if floor.get("n")]
    if not rates:
        return {}
    return {"noise floor": float(np.mean(rates))}


def _headline(stage: str, results: dict) -> dict:
    """The few numbers a reader wants first, per stage."""
    if stage == "pilot":
        headline = {
            "flip rate": results["flip_rate"]["overall"],
            "twins": results["n"],
            "cues bite": results["cues_bite"],
        }
        headline.update(_floor_headline(results))
        return headline
    if stage in ("solo", "overlap"):
        headline = {
            "flip rate": results["flip_rate"]["overall"],
            "clean accuracy": results["shortcut_reliance"]["clean_accuracy"],
            "contaminated accuracy": results["shortcut_reliance"]["contaminated_accuracy"],
            "shortcut reliance": results["shortcut_reliance"]["overall"],
            "twins per model": results["n_twins"],
        }
        headline.update(_floor_headline(results))
        overlap = results.get("overlap")
        if overlap:
            headline["within-lineage overlap"] = overlap.get("within_mean")
            headline["cross-lineage overlap"] = overlap.get("cross_mean")
            headline["overlap p-value"] = overlap.get("p_value")
        return headline
    return {
        "shared adoption": results["shared_adoption"],
        "isolated adoption": results["isolated_adoption"],
        "adoption delta": results["adoption_delta"],
        "mean onset turn": results["mean_onset"],
        "cases": results["n_cases"],
    }


def _resolved_roster(config) -> list[dict]:
    """The roster as the run actually used it, with the provenance of each lineage."""
    sources = lineage_sources(config)
    return [
        {
            "id": spec.name,
            "lineage": spec.lineage,
            "tier": spec.tier,
            "open_weights": spec.is_open_weights,
            "lineage_source": sources.get(spec.name, "inferred"),
        }
        for spec in model_specs(config)
    ]


def write_outputs(out_dir, stage: str, config, results: dict, plan: dict, *,
                  manifest_path=None, run_id: str | None = None) -> dict:
    """Write the run bundle and return the paths written.

    ``results.json`` and ``summary.md`` are the run; ``config.json``, ``versions.json`` and
    ``run_manifest.json`` are what make it self-describing (resolved config, environment, model
    ids, seed, cue-set and dataset revision). See :mod:`benchmaxxing.bundle` for the bundle
    contract and for reading one back.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"{stage}-{config.dataset}-seed{config.seed}"

    paths = {
        "results": out / "results.json",
        "summary": out / "summary.md",
        "config": out / "config.json",
        "run_manifest": out / "run_manifest.json",
        # what the run ran on: package version, git SHA, interpreter, pinned library versions
        "versions": write_versions(out),
    }
    est = estimates(stage, results, config.seed)
    family = family_correction(significance_tests(stage, results, config.seed))
    payload = {
        "run_id": run_id,
        "stage": stage,
        "plan": plan,
        "estimates": _jsonable(est),
        "family_correction": _jsonable(family),
        "results": _jsonable(results),
    }
    paths["results"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"].write_text(summarize(stage, results, plan, est, family), encoding="utf-8")
    paths["config"].write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_manifest(
        RunManifest(
            run_id=run_id,
            model_ids=list(config.models),
            prompt_versions=dict(DEFAULT_REGISTRY.all_ids()),
            seed=config.seed,
            cue_set_version=config.cue_set,
            dataset_revision=str(manifest_path) if manifest_path else config.dataset,
            library_versions=library_versions(),
            # the resolved roster, not just the ids: a rerun needs to know which lineage each
            # model was treated as, and whether that came from the config or from the id
            config={
                **config.to_dict(),
                "resolved_roster": _resolved_roster(config),
                "noise_floor": bool(plan.get("noise_floor")),
                "case_selection": plan.get("case_selection"),
            },
        ),
        paths["run_manifest"],
    )
    return paths
