"""Tests for the shared-context committee harness (benchmaxxing.blackboard).

Everything runs offline with a MockBackend, so there is no optional-dependency path here (the
harness itself needs only numpy-free pure Python). We verify the four referee behaviours the
spec calls out: isolated vs shared visibility, the passive observer tap, the pre-emptive
pre_hook injection, the seeded shortcut injection, plus speaking-order permutation and the
orchestrator synthesizer turn.
"""

from __future__ import annotations

import copy

from benchmaxxing.blackboard import AgentResponse, AgentView, run_committee
from benchmaxxing.schema import Case, Committee, Condition, Modality, ModelSpec, Turn


# --------------------------------------------------------------------------- helpers


class MockBackend:
    """A deterministic offline backend.

    Its content records how many turns it was allowed to see, so shared vs isolated visibility
    is directly observable in the transcript. It also records every view it received.
    """

    def __init__(self, name: str, answer=None, confidence: float = 0.5):
        self.name = name
        self.answer = answer if answer is not None else name
        self.confidence = confidence
        self.views: list[AgentView] = []

    def respond(self, view: AgentView) -> AgentResponse:
        self.views.append(view)
        n_visible = len(view.visible_turns)
        role = "orchestrator" if view.is_orchestrator else "member"
        content = f"{self.name}[{role}] saw {n_visible} turns"
        return AgentResponse(content=content, answer=self.answer, confidence=self.confidence)


def make_committee(*names: str) -> Committee:
    members = tuple(
        ModelSpec(name=nm, lineage="gemini" if i % 2 == 0 else "qwen")
        for i, nm in enumerate(names)
    )
    return Committee(members=members)


def make_case() -> Case:
    return Case(
        case_id="case-1",
        patient_id="p-1",
        modality=Modality.TEXT,
        question="Which is correct?",
        options=("A", "B", "C"),
        answer_index=0,
    )


def backend_factory(backends: dict[str, MockBackend]):
    def backend_for(spec: ModelSpec) -> MockBackend:
        return backends[spec.name]

    return backend_for


# --------------------------------------------------------------------------- tests


def test_shared_lets_later_agent_condition_on_prior_turns():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       shared=True)

    assert [t.agent_id for t in tr.turns] == ["A", "B"]
    # A speaks first and sees nothing; B sees A's committed turn.
    assert tr.turns[0].content == "A[member] saw 0 turns"
    assert tr.turns[1].content == "B[member] saw 1 turns"
    assert len(backends["B"].views[0].visible_turns) == 1


def test_isolated_hides_other_agents():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       shared=False)

    # In isolated mode B cannot see A: the solo counterfactual.
    assert tr.turns[1].content == "B[member] saw 0 turns"
    assert backends["B"].views[0].visible_turns == ()


def test_isolated_vs_shared_change_what_second_agent_sees():
    committee = make_committee("A", "B")

    shared_backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    tr_shared = run_committee(committee, make_case(), Condition.CLEAN,
                              backend_factory(shared_backends), shared=True)

    iso_backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    tr_iso = run_committee(committee, make_case(), Condition.CLEAN,
                           backend_factory(iso_backends), shared=False)

    assert tr_shared.turns[1].content != tr_iso.turns[1].content


def test_observer_sees_every_turn_and_does_not_alter_run():
    committee = make_committee("A", "B")

    seen: list[Turn] = []

    def observer(turn: Turn):
        seen.append(turn)
        return "IGNORE ME"        # a return value must never influence the run

    backends_obs = {"A": MockBackend("A"), "B": MockBackend("B")}
    tr_obs = run_committee(committee, make_case(), Condition.CLEAN,
                           backend_factory(backends_obs), observer=observer)

    backends_plain = {"A": MockBackend("A"), "B": MockBackend("B")}
    tr_plain = run_committee(committee, make_case(), Condition.CLEAN,
                             backend_factory(backends_plain))

    # Observer saw exactly the committed turns.
    assert len(seen) == len(tr_obs.turns) == 2
    assert [t.agent_id for t in seen] == ["A", "B"]
    # Presence of the observer changed nothing about the transcript.
    assert [(t.agent_id, t.content, t.answer) for t in tr_obs.turns] == \
           [(t.agent_id, t.content, t.answer) for t in tr_plain.turns]


def test_pre_hook_injects_turn_before_next_agent():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    state_box = {"fired": False}

    def pre_hook(state):
        # Inject a single referee note right after the first turn commits.
        if not state_box["fired"] and state.last_turn is not None \
                and state.last_turn.agent_id == "A":
            state_box["fired"] = True
            return Turn(turn_index=-1, agent_id="referee",
                        content="referee flag", answer="flagged")
        return None

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       pre_hook=pre_hook)

    ids = [t.agent_id for t in tr.turns]
    assert ids == ["A", "referee", "B"]
    # The injected referee turn is visible to B (shared mode): B saw A + referee = 2 turns.
    assert tr.turns[2].content == "B[member] saw 2 turns"
    # The injected turn got a real sequential turn_index assigned at commit time.
    assert tr.turns[1].turn_index == 1
    assert tr.committed["referee"] == "flagged"


def test_pre_hook_non_turn_return_is_noop():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       pre_hook=lambda state: None)

    assert [t.agent_id for t in tr.turns] == ["A", "B"]


def test_seed_turn_marks_and_plants_the_turn():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       seed_turn=(0, "PLANTED", "planter"))

    planted = tr.turns[0]
    assert planted.seeded is True
    assert planted.agent_id == "planter"
    assert planted.answer == "PLANTED"
    # The member that would have spoken in slot 0 (A) was never queried.
    assert backends["A"].views == []
    # The non-seeded turn is not flagged.
    assert tr.turns[1].seeded is False
    assert tr.committed["planter"] == "PLANTED"
    # B, in shared mode, conditioned on the planted shortcut.
    assert tr.turns[1].content == "B[member] saw 1 turns"


def test_order_permutation_changes_speaking_order():
    committee = make_committee("A", "B", "C")

    backends_default = {nm: MockBackend(nm) for nm in ("A", "B", "C")}
    tr_default = run_committee(committee, make_case(), Condition.CLEAN,
                               backend_factory(backends_default))

    backends_perm = {nm: MockBackend(nm) for nm in ("A", "B", "C")}
    tr_perm = run_committee(committee, make_case(), Condition.CLEAN,
                            backend_factory(backends_perm), order=[2, 0, 1])

    assert [t.agent_id for t in tr_default.turns] == ["A", "B", "C"]
    assert [t.agent_id for t in tr_perm.turns] == ["C", "A", "B"]


def test_orchestrator_adds_final_synthesizer_turn():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       orchestrator=True)

    assert [t.agent_id for t in tr.turns] == ["A", "B", "B"]
    assert tr.meta["orchestrator_turn_index"] == 2
    # The leader's final turn was flagged as the orchestrator via the view it received.
    last_view = backends["B"].views[-1]
    assert last_view.is_orchestrator is True
    # The orchestrator sees the full board (both member turns).
    assert len(last_view.visible_turns) == 2


def test_rounds_repeat_the_speaking_order():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       rounds=2)

    assert [t.agent_id for t in tr.turns] == ["A", "B", "A", "B"]
    assert [t.turn_index for t in tr.turns] == [0, 1, 2, 3]


def test_isolated_agent_sees_only_its_own_prior_turns_across_rounds():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       shared=False, rounds=2)

    # Round 2 turns: each agent sees only its own single prior turn, not the other agent's.
    assert tr.turns[2].content == "A[member] saw 1 turns"
    assert tr.turns[3].content == "B[member] saw 1 turns"


def test_committed_holds_final_answers_per_agent():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A", answer="ans-A"), "B": MockBackend("B", answer="ans-B")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends))

    assert tr.committed == {"A": "ans-A", "B": "ans-B"}


def test_transcript_carries_run_metadata():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    case = make_case()

    tr = run_committee(committee, case, Condition.CONTAMINATED, backend_factory(backends),
                       order=[1, 0], seed=7)

    assert tr.case_id == case.case_id
    assert tr.condition == Condition.CONTAMINATED
    assert tr.run_id == "cmte-case-1-contaminated-seed7"
    assert tr.meta["order"] == [1, 0]
    assert tr.meta["seed"] == 7


def test_callable_backend_is_accepted():
    committee = make_committee("A")

    def callable_backend(view: AgentView) -> AgentResponse:
        return AgentResponse(content="hi", answer="X", confidence=0.9)

    def backend_for(spec: ModelSpec):
        return callable_backend

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_for)
    assert tr.turns[0].content == "hi"
    assert tr.turns[0].answer == "X"


def test_string_and_dict_responses_are_coerced():
    committee = make_committee("A", "B")

    def backend_for(spec: ModelSpec):
        if spec.name == "A":
            return lambda view: "plain string answer"
        return lambda view: {"content": "dict answer", "answer": 2, "confidence": 0.3}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_for)
    assert tr.turns[0].content == "plain string answer"
    assert tr.turns[0].answer is None
    assert tr.turns[1].content == "dict answer"
    assert tr.turns[1].answer == 2
    assert tr.turns[1].confidence == 0.3


def test_observer_snapshot_matches_committed_turn_state():
    # Guards against the observer being handed a turn whose fields mutate afterwards.
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    snapshots: list[tuple] = []

    def observer(turn: Turn):
        snapshots.append((turn.turn_index, turn.agent_id, copy.copy(turn.content)))

    run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                  observer=observer)

    assert snapshots == [(0, "A", "A[member] saw 0 turns"),
                         (1, "B", "B[member] saw 1 turns")]


def test_bad_order_index_raises():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    try:
        run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                      order=[0, 5])
        raise AssertionError("expected ValueError for out-of-range order index")
    except ValueError:
        pass


# --------------------------------------------------------------------------- board_seed


def test_board_seed_places_context_artifact_before_first_agent():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    artifact = Turn(turn_index=-1, agent_id="workspace", content="note", answer="IMPLIED")

    tr = run_committee(committee, make_case(), Condition.CONTAMINATED, backend_factory(backends),
                       board_seed=artifact)

    assert [t.agent_id for t in tr.turns] == ["workspace", "A", "B"]
    art = tr.turns[0]
    assert art.context is True and art.seeded is True
    # A speaks first but the artifact is already on the board, so A conditions on it.
    assert tr.turns[1].content == "A[member] saw 1 turns"
    # The artifact never enters the committed map (it is not a member decision).
    assert "workspace" not in tr.committed
    assert tr.committed == {"A": "A", "B": "B"}
    assert tr.meta["board_seed"] == ["IMPLIED"]


def test_board_seed_context_is_visible_in_isolated_mode():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    artifact = Turn(turn_index=-1, agent_id="workspace", content="note", answer="IMPLIED")

    tr = run_committee(committee, make_case(), Condition.CONTAMINATED, backend_factory(backends),
                       board_seed=artifact, shared=False)

    # Isolation hides peers but not the workspace artifact: each agent sees exactly the artifact.
    assert tr.turns[1].content == "A[member] saw 1 turns"
    assert tr.turns[2].content == "B[member] saw 1 turns"
    b_view = backends["B"].views[0]
    assert [t.agent_id for t in b_view.visible_turns] == ["workspace"]


def test_board_seed_does_not_shift_seed_turn_index():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    artifact = Turn(turn_index=-1, agent_id="workspace", content="note", answer="CTX")

    # seed_turn index 0 still targets the FIRST member slot (A), not the board seed.
    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       board_seed=artifact, seed_turn=(0, "PLANTED", "planter"))

    assert [t.agent_id for t in tr.turns] == ["workspace", "planter", "B"]
    assert tr.turns[1].seeded is True and tr.turns[1].answer == "PLANTED"
    # A occupied member slot 0, which the plant replaced, so A was never queried.
    assert backends["A"].views == []


def test_board_seed_does_not_mutate_caller_turn_or_alias_across_runs():
    committee = make_committee("A")
    backends1 = {"A": MockBackend("A")}
    backends2 = {"A": MockBackend("A")}
    artifact = Turn(turn_index=-1, agent_id="workspace", content="note", answer="IMPLIED")

    tr1 = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends1),
                        board_seed=artifact)
    tr2 = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends2),
                        board_seed=artifact)

    # The caller's Turn is untouched: flags and turn_index are stamped only on the committed copy.
    assert artifact.turn_index == -1
    assert artifact.context is False and artifact.seeded is False
    # Each run committed an independent copy, so re-stamping run 2 never moved run 1's artifact.
    assert tr1.turns[0].turn_index == 0
    assert tr2.turns[0].turn_index == 0
    assert tr1.turns[0] is not tr2.turns[0]


def test_board_seed_accepts_multiple_artifacts():
    committee = make_committee("A")
    backends = {"A": MockBackend("A")}
    seeds = [
        Turn(turn_index=-1, agent_id="workspace", content="n1", answer="X"),
        Turn(turn_index=-1, agent_id="workspace", content="n2", answer="Y"),
    ]

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       board_seed=seeds)

    assert [t.agent_id for t in tr.turns] == ["workspace", "workspace", "A"]
    assert tr.meta["board_seed"] == ["X", "Y"]
    assert tr.turns[2].content == "A[member] saw 2 turns"


def test_board_seed_attributed_to_member_raises():
    # A board seed carrying a member's id would be a peer utterance leaking into isolated mode.
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    artifact = Turn(turn_index=-1, agent_id="A", content="note", answer="IMPLIED")

    try:
        run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                      board_seed=artifact)
        raise AssertionError("expected ValueError for a member-attributed board seed")
    except ValueError:
        pass


def test_pre_hook_member_attributed_context_turn_raises():
    # A context turn attributed to A would be visible to B even in isolated mode: a peer leak.
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    def leaky_hook(state):
        if state.last_turn is not None and state.last_turn.agent_id == "A":
            return Turn(turn_index=-1, agent_id="A", content="secret", answer="SECRET",
                        context=True)
        return None

    try:
        run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                      pre_hook=leaky_hook, shared=False)
        raise AssertionError("expected ValueError for a member-attributed context turn")
    except ValueError:
        pass


# --------------------------------------------------------------------------- multi-seed specs


def test_seed_turn_accepts_several_specs_and_custom_content():
    # A 2-peer majority: both peers are planted, each with its own utterance, and only the
    # holdout (C) is generated by a backend.
    committee = make_committee("A", "B", "C")
    backends = {nm: MockBackend(nm) for nm in ("A", "B", "C")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       seed_turn=[(0, "PLANTED", "A", "I read it as PLANTED."),
                                  (1, "PLANTED", "B", "Agreed, PLANTED.")])

    assert [t.agent_id for t in tr.turns] == ["A", "B", "C"]
    assert [t.seeded for t in tr.turns] == [True, True, False]
    assert tr.turns[0].content == "I read it as PLANTED."
    assert tr.turns[1].content == "Agreed, PLANTED."
    assert backends["A"].views == [] and backends["B"].views == []
    # C deliberated against both planted peers.
    assert tr.turns[2].content == "C[member] saw 2 turns"


def test_seed_turn_without_content_keeps_the_bare_assertion():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    tr = run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                       seed_turn=[(0, "PLANTED", "planter")])

    assert tr.turns[0].content == "[seeded] planted answer: 'PLANTED'"


def test_duplicate_seed_slot_raises():
    # Two specs on one slot would silently drop one and un-match a paired arm.
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    try:
        run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                      seed_turn=[(0, "X", "A"), (0, "Y", "B")])
        raise AssertionError("expected ValueError for two specs on the same member slot")
    except ValueError:
        pass


def test_malformed_seed_spec_raises():
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    for bad in [(0, "X"), (0, "X", "A", "content", "extra"), (-1, "X", "A")]:
        try:
            run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                          seed_turn=bad)
            raise AssertionError(f"expected ValueError for seed spec {bad!r}")
        except ValueError:
            pass


def test_nested_seed_spec_shape_raises():
    # A single spec is told from a sequence of specs by seed_turn[0] being an int, so an extra
    # layer of nesting parses as a sequence whose members are themselves sequences of specs.
    # Each arity below must raise rather than seed some misread slot.
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}
    one, two, three = (0, "X", "A"), (1, "Y", "B"), (2, "Z", "A")

    for bad in [[[one]], [[one, two]], [[one, two, three]], [[(0, "X", "A", "c"), two]]]:
        try:
            run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                          seed_turn=bad)
            raise AssertionError(f"expected ValueError for nested seed_turn {bad!r}")
        except ValueError:
            pass


def test_non_int_seed_index_raises():
    # A stringified index makes the spec look like a sequence of specs; it must not seed slot 0.
    committee = make_committee("A", "B")
    backends = {"A": MockBackend("A"), "B": MockBackend("B")}

    try:
        run_committee(committee, make_case(), Condition.CLEAN, backend_factory(backends),
                      seed_turn=("0", "X", "A"))
        raise AssertionError("expected ValueError for a non-int seed_turn index")
    except ValueError:
        pass
