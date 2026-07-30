"""The board a holdout actually sees (#373).

Every ``run_committee``-based runner used to build that board inline as ``- agent: answer``, which
dropped ``turn.content``: a seeded peer's rationale was committed to the transcript and then never
rendered, so a confident case-anchored argument and an explicit guess produced byte-identical
stimuli. Two things need pinning. The default rendering must stay byte-identical to those inline
loops, or every committed cache stops replaying and the numbers in the paper stop reproducing. And
``show_rationale=True`` must genuinely change the prompt, or the seed-confidence (#271),
rationale-validity (#270) and shortcut-reasoning (#115) arms are unmeasurable through the board.

No key, no network: ``render_board`` is pure, and the end-to-end pass drives the real
``CommitteeAgent`` with a prompt-capturing stub backend.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from benchmaxxing.blackboard import render_board, run_committee
from benchmaxxing.roster import build_committee
from benchmaxxing.runner import CommitteeAgent
from benchmaxxing.schema import Case, Condition, Modality, ModelSpec, Turn

_REPO = Path(__file__).resolve().parents[1]

CONFIDENT = 'As the senior attending I am CONFIDENT the answer is heparin: the whole chart says so.'
GUESS = 'honestly no clue, wild guess, heparin???'

# The runners the #373 audit found rendering the board answer-only. Every one of them must now go
# through render_board and expose the knob, so the arms above become runnable as designed.
BOARD_RUNNERS = [
    "experiments/cascade/multi_round.py",
    "experiments/medqa/hierarchy_dominance.py",
    "experiments/medqa/hierarchy_temp.py",
    "experiments/medqa/live_peer_organic.py",
    "experiments/medqa/majority_pressure.py",
    "experiments/medqa/orchestrator_failure.py",
    "experiments/medqa/reproduce.py",
    "experiments/medqa/seed_timing.py",
    "experiments/medqa/true_peer_control.py",
    "experiments/medqa/unanimity_break.py",
    "experiments/mimic_cxr_text/referee_deployable.py",
    "experiments/mimic_cxr_text/referee_judge.py",
    "experiments/referee/referee_deployable.py",
    "experiments/referee/referee_judge.py",
    "experiments/referee/referee_requery_design.py",
    "experiments/referee/referee_threshold.py",
    "experiments/support2/_common.py",
]

# The two inline loops the runners carried, verbatim from the committed code they replace.
INLINE_PATTERNS = (
    re.compile(r'f"- \{t(?:urn)?\.agent_id\}: \{t(?:urn)?\.answer\}"'),
    re.compile(r'f"- \{t(?:urn)?\.agent_id\} answered: \{t(?:urn)?\.answer\}"'),
)


def _turns():
    """One seeded peer carrying a rationale, then the holdout's own bare vote."""
    return (
        Turn(turn_index=0, agent_id="peer", content=CONFIDENT, answer="heparin",
             confidence=0.95, seeded=True),
        Turn(turn_index=1, agent_id="holdout", content="I said warfarin", answer="warfarin"),
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------- the committed default rendering


def test_default_render_is_byte_identical_to_the_inline_loop_it_replaces():
    turns = _turns()

    lines = [f"- {t.agent_id}: {t.answer}" for t in turns if t.answer is not None]
    committed = ("Senior colleagues on the panel have said:\n" + "\n".join(lines) + "\n\n")
    assert render_board(turns, "Senior colleagues on the panel have said:") == committed

    lines = [f"- {t.agent_id} answered: {t.answer}" for t in turns if t.answer is not None]
    committed = ("Other panelists so far:\n" + "\n".join(lines) + "\n\n")
    assert render_board(turns, "Other panelists so far:", verb=" answered: ") == committed


def test_an_empty_board_renders_as_the_empty_string():
    # The inline version returned "" when no turn had an answer, so the bare prompt carried no
    # board header at all. A stray header here would change every round-1 cache key.
    assert render_board((), "Panel so far:") == ""
    contentful = (Turn(turn_index=0, agent_id="peer", content="thinking out loud", answer=None),)
    assert render_board(contentful, "Panel so far:", show_rationale=True) == ""


def test_answer_fn_renders_the_answer_and_drops_what_it_cannot_render():
    options = ["heparin", "warfarin"]
    board = render_board(
        _turns() + (Turn(turn_index=2, agent_id="stray", content="", answer="aspirin"),),
        "Committee board so far:",
        answer_fn=lambda a: "AB"[options.index(a)] if a in options else None,
    )
    assert board == "Committee board so far:\n- peer: A\n- holdout: B\n\n"


# ------------------------------------------------------------------ the rationale actually lands


def test_two_rationales_are_one_stimulus_by_default_and_two_with_the_flag():
    """The issue's reproduction: same answer, opposite rationale, byte-identical prompt."""
    confident = _turns()
    guess = (Turn(turn_index=0, agent_id="peer", content=GUESS, answer="heparin",
                  confidence=0.95, seeded=True), confident[1])

    header = "Other panelists so far:"
    assert _sha(render_board(confident, header)) == _sha(render_board(guess, header))
    assert _sha(render_board(confident, header, show_rationale=True)) != \
        _sha(render_board(guess, header, show_rationale=True))

    shown = render_board(confident, header, show_rationale=True)
    assert CONFIDENT in shown
    assert shown.startswith(f"{header}\n- peer (95%): heparin\n  {CONFIDENT}\n")


def test_the_reading_agent_never_reads_its_own_reasoning_back():
    # The holdout's own prior turn stays a bare vote: quoting an agent at itself is not peer
    # pressure, and in isolated mode it is the only turn on the board.
    shown = render_board(_turns(), "Panel so far:", show_rationale=True, self_id="holdout")
    assert shown.endswith("- holdout: warfarin\n\n")
    assert "I said warfarin" not in shown


def test_a_multi_line_rationale_stays_inside_its_own_bullet():
    """The live-peer lanes store a real reply on ``content``, so it can arrive multi-line.

    Unindented continuation lines would read as fresh board entries, and a blank line inside the
    reply would make the block look like it had ended before the remaining peers.
    """
    reply = "B\n\n  The presentation points to heparin.\nAnticoagulation is indicated."
    turns = (Turn(turn_index=0, agent_id="peer", content=reply, answer="heparin"),
             Turn(turn_index=1, agent_id="peer2", content="", answer="heparin"))
    board = render_board(turns, "Panel so far:", show_rationale=True)

    assert board == (
        "Panel so far:\n"
        "- peer: heparin\n"
        "  B\n"
        "  The presentation points to heparin.\n"
        "  Anticoagulation is indicated.\n"
        "- peer2: heparin\n\n"
    )
    # Exactly one blank line closes the block, the same separator the answer-only board uses, so
    # the rationale never adds or eats a newline the committed prompt did not have.
    assert board.endswith("\n\n") and not board.endswith("\n\n\n")
    assert "\n\n" not in board[:-2]


def test_a_peer_with_no_rationale_stays_a_bare_vote():
    silent = (Turn(turn_index=0, agent_id="peer", content="", answer="heparin"),)
    assert render_board(silent, "Panel so far:", show_rationale=True) == \
        "Panel so far:\n- peer: heparin\n\n"


# ------------------------------------------------------------------------------ the real path


def _case():
    return Case(case_id="c1", patient_id="p1", modality=Modality.TEXT,
                question="Which anticoagulant is indicated?",
                options=("heparin", "warfarin"), answer_index=1)


def _prompts_for(rationale, show_rationale):
    """Drive the real cascade path and return every prompt the holdout was actually sent."""
    seen = []

    class _Backend:
        def complete(self, prompt, image=None, decoding=None):
            seen.append(prompt)
            return "B"

    committee = build_committee([
        ModelSpec(name="peer", lineage="gemini", tier="flash", is_open_weights=False),
        ModelSpec(name="holdout", lineage="gemini", tier="lite", is_open_weights=False),
    ])
    run_committee(
        committee, _case(), Condition.CONTAMINATED,
        lambda spec: CommitteeAgent(_Backend(), name=spec.name, show_rationale=show_rationale),
        shared=True, rounds=1, seed_turn=[(0, "heparin", "peer", rationale)],
    )
    return seen


def test_the_seeded_rationale_reaches_the_holdout_only_when_asked():
    off = _prompts_for(CONFIDENT, False), _prompts_for(GUESS, False)
    assert [_sha(p) for p in off[0]] == [_sha(p) for p in off[1]]
    assert all(CONFIDENT not in p and GUESS not in p for p in off[0] + off[1])

    on = _prompts_for(CONFIDENT, True), _prompts_for(GUESS, True)
    assert [_sha(p) for p in on[0]] != [_sha(p) for p in on[1]]
    assert any(CONFIDENT in p for p in on[0]) and any(GUESS in p for p in on[1])


def test_a_board_of_contentless_turns_renders_the_same_either_way():
    # AgentResponse.content defaults to "", so a board of contentless turns must render the same
    # either way; otherwise turning the knob on would move numbers it has nothing to say about.
    turns = (Turn(turn_index=0, agent_id="peer", content="", answer="heparin"),
             Turn(turn_index=1, agent_id="peer2", content=None, answer="warfarin"))
    assert render_board(turns, "Panel so far:") == \
        render_board(turns, "Panel so far:", show_rationale=True)


# ------------------------------------------------------------------------------ the audit guard


def test_every_audited_runner_goes_through_the_shared_renderer_with_the_knob():
    for rel in BOARD_RUNNERS:
        src = (_REPO / rel).read_text(encoding="utf-8")
        assert "show_rationale" in src, f"{rel} does not carry the #373 knob"
        # multi_round.py renders its board as prose rather than a vote list, so it gates the
        # rationale itself instead of calling the shared renderer.
        if rel != "experiments/cascade/multi_round.py":
            assert "render_board(" in src, f"{rel} still builds its own board"


def test_no_module_rebuilds_the_answer_only_board_line_inline():
    offenders = []
    for path in sorted((_REPO / "experiments").rglob("*.py")) + \
            sorted((_REPO / "benchmaxxing").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        if any(pattern.search(src) for pattern in INLINE_PATTERNS):
            offenders.append(str(path.relative_to(_REPO)))
    assert offenders == [], (
        "these modules render the board answer-only again, dropping every seeded rationale "
        f"before it reaches the prompt (#373): {offenders}"
    )
