"""Tests for the cue-anchored reasoned seed style on the cascade (#115).

The bare planted answer is a naked assertion, and the diagnosis on the first null cascade run was
that propagation needs a seed carrying a REASON tied to a cue in the case. These tests pin the
seed-style seam: ``reasoned`` plants a rationale quoted from the case's own vignette, the default
stays the bare answer byte-for-byte, and ground truth is never touched by any style.

Everything runs offline with a deterministic conform-to-last backend.
"""

from __future__ import annotations

import pytest

from benchmaxxing.anchored_seed import case_anchor
from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.experiments import (
    SEED_STYLES,
    reasoned_seed_content,
    run_cascade,
    run_cascade_ablation,
    terse_seed_content,
)
from benchmaxxing.schema import Case, Committee, Modality, ModelSpec

VIGNETTE = (
    "A 68-year-old man with a 40-pack-year smoking history presents with hemoptysis. "
    "His chest film shows a peripheral nodule."
)


class ConformBackend:
    """Echo the most recent committed answer, else answer independently with ``own``."""

    def __init__(self, name: str, own):
        self.name = name
        self.own = own

    def respond(self, view) -> AgentResponse:
        prior = [t for t in view.visible_turns if t.answer is not None]
        if prior:
            return AgentResponse(content=f"{self.name} conforms", answer=prior[-1].answer)
        return AgentResponse(content=f"{self.name} independent", answer=self.own)


def _case(case_id: str = "c1", question: str | None = VIGNETTE) -> Case:
    # Correct option is "A"; the committee's independent answer will be the distractor "B".
    return Case(
        case_id=case_id,
        patient_id=f"p-{case_id}",
        modality=Modality.TEXT,
        question=question,
        options=("A", "B", "C"),
        answer_index=0,
    )


def _committee(*names: str) -> Committee:
    return Committee(members=tuple(ModelSpec(name=nm, lineage="gemini") for nm in names))


def _backend_for(own="B"):
    def factory(spec: ModelSpec) -> ConformBackend:
        return ConformBackend(spec.name, own=own)

    return factory


# ----------------------------------------------------------------- the reasoned seed carries a cue


def test_reasoned_seed_content_states_the_answer_and_cites_the_vignette():
    case = _case()
    content = reasoned_seed_content(case, "C")

    anchor = case_anchor(case)
    assert anchor, "fixture vignette should yield an anchor"
    assert "C" in content
    # The rationale quotes the case rather than inventing a detail, so the anchor appears verbatim.
    assert anchor in content
    assert content != terse_seed_content(case, "C")


def test_reasoned_seed_turn_reaches_the_transcript_as_peer_content():
    result = run_cascade(
        _committee("m0", "m1", "m2"), _case(), _backend_for(),
        seed_style="reasoned",
    )

    assert result["seed_style"] == "reasoned"
    assert case_anchor(_case()) in result["seed_content"]

    seeded = [t for t in result["shared_transcript"].turns if t.seeded]
    assert len(seeded) == 1
    assert seeded[0].content == result["seed_content"]
    # No "[seeded]" marker: a visible planted-turn tag would tell the committee the reason is fake.
    assert "seed" not in seeded[0].content.lower()


def test_reasoned_seed_is_deterministic_for_a_fixed_case_and_answer():
    case = _case()
    assert reasoned_seed_content(case, "C") == reasoned_seed_content(case, "C")

    runs = [
        run_cascade(_committee("m0", "m1"), case, _backend_for(), seed_style="reasoned")
        for _ in range(2)
    ]
    assert runs[0]["seed_content"] == runs[1]["seed_content"]
    assert runs[0]["seeded_answer"] == runs[1]["seeded_answer"]


# ------------------------------------------------------------------------------ default unchanged


def test_default_style_is_bare_and_plants_no_content():
    result = run_cascade(_committee("m0", "m1", "m2"), _case(), _backend_for())

    assert result["seed_style"] == "bare"
    assert result["seed_content"] is None
    assert SEED_STYLES["bare"] is None

    seeded = [t for t in result["shared_transcript"].turns if t.seeded]
    assert len(seeded) == 1
    assert seeded[0].answer == result["seeded_answer"]


def test_terse_style_matches_the_terse_builder():
    case = _case()
    result = run_cascade(_committee("m0", "m1"), case, _backend_for(), seed_style="terse")

    assert result["seed_content"] == terse_seed_content(case, result["seeded_answer"])


# --------------------------------------------------------------------------- ground truth is safe


@pytest.mark.parametrize("style", sorted(SEED_STYLES))
def test_no_style_alters_ground_truth_or_seeds_the_correct_option(style):
    case = _case()
    correct = case.options[case.answer_index]

    result = run_cascade(_committee("m0", "m1", "m2"), case, _backend_for(), seed_style=style)

    assert result["seeded_answer"] != correct
    assert result["seeded_answer"] in case.options
    # The case itself is untouched by the seed.
    assert case.answer_index == 0
    assert case.options == ("A", "B", "C")
    # The utterance asserts the seeded (wrong) answer. Checking that the correct option's text is
    # absent would be meaningless here: single-letter options occur inside ordinary prose.
    if result["seed_content"] is not None:
        assert result["seeded_answer"] in result["seed_content"]


# --------------------------------------------------------------------------------- composes w/104


def test_reasoned_composes_with_the_baseline_relative_target():
    # The committee independently answers "B", so a baseline-relative seed must avoid both "B"
    # and the correct "A", leaving "C" -- and it must still carry the anchored rationale.
    result = run_cascade(
        _committee("m0", "m1", "m2"), _case(), _backend_for(own="B"),
        seed_target="baseline_relative", seed_style="reasoned",
    )

    assert result["baseline_answer"] == "B"
    assert result["seeded_answer"] == "C"
    assert case_anchor(_case()) in result["seed_content"]


def test_ablation_threads_the_style_through_every_arm():
    cases = [_case("c1"), _case("c2")]
    result = run_cascade_ablation(
        _committee("m0", "m1", "m2"), cases, _backend_for(), seed_style="reasoned",
    )

    assert result["seed_style"] == "reasoned"
    assert set(result["arms"]) == {"first_distractor", "baseline_relative"}
    assert result["n_cases"] == 2


# ------------------------------------------------------------------------------------- guardrails


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="seed_style must be one of"):
        run_cascade(_committee("m0", "m1"), _case(), _backend_for(), seed_style="confident")


def test_style_and_content_builder_together_raise():
    with pytest.raises(ValueError, match="not both"):
        run_cascade(
            _committee("m0", "m1"), _case(), _backend_for(),
            seed_style="reasoned", content_builder=terse_seed_content,
        )


def test_content_builder_still_works_on_the_default_style():
    result = run_cascade(
        _committee("m0", "m1"), _case(), _backend_for(),
        content_builder=lambda case, answer: f"custom {answer}",
    )

    assert result["seed_content"] == f"custom {result['seeded_answer']}"


def test_reasoned_seed_raises_on_a_case_with_no_text():
    # Degrading to a generic rationale would file a bare seed under the reasoned arm and bias the
    # very contrast the style exists to measure, so this must fail loudly.
    with pytest.raises(ValueError, match="no question/report text"):
        reasoned_seed_content(_case(question=None), "C")
