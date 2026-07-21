"""The shared-context committee harness (the blackboard) and its referee hooks.

A committee of agents takes turns writing to a shared board. In ``shared`` mode each agent
reads the turns already on the board before it answers, so one agent's answer can condition
the next (this is how a shortcut or a conformity cascade propagates). In ``isolated`` mode an
agent sees only its own prior turns and never the other agents, which is the solo
counterfactual we compare against.

Everything here is a pure function driven by an injected backend factory, so the whole harness
runs offline with a ``MockBackend`` and no API keys. Model access lives behind
``benchmaxxing.gateway``; this module never imports it directly. Instead ``backend_for`` is
passed in and is expected to return an object conforming to the :class:`Backend` protocol
below (a ``respond(view) -> AgentResponse`` method, or any callable of the same shape). That
keeps the harness decoupled from the concrete gateway and trivially testable.

The referee hooks:

* ``observer`` is a passive read-only tap: it is called once for every committed turn and its
  return value is ignored, so an observer can never change the run.
* ``pre_hook`` is the real-time referee tap: it is called AFTER a turn commits and BEFORE the
  next agent reads the board. If it returns a :class:`~benchmaxxing.schema.Turn`, that turn is
  injected onto the board (and observed) so the next agent sees it. Returning anything else is
  a no-op.
* ``seed_turn`` is the shortcut injection: the turn at a given position is replaced by a
  planted answer rather than being generated, and that turn is flagged ``seeded=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable
from benchmaxxing.schema import Case, Committee, Condition, Transcript, Turn


@dataclass
class AgentResponse:
    """What a backend returns for one turn: free-text plus an optional committed answer."""

    content: str = ""
    answer: Any = None
    confidence: float | None = None


@dataclass
class AgentView:
    """The read-only view handed to a backend for a single turn.

    ``visible_turns`` is exactly what this agent is allowed to condition on: the whole board in
    shared mode, or only this agent's own prior turns in isolated mode. The orchestrator always
    receives the full board because its job is to synthesize.
    """

    case: Case
    condition: Condition
    agent_id: str
    round_index: int
    position: int                       # position of this agent within the speaking order
    shared: bool
    visible_turns: tuple[Turn, ...] = ()
    is_orchestrator: bool = False
    role: str = "member"
    seed: int = 0
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """The contract a gateway backend must satisfy to drive a blackboard turn.

    A backend may either expose ``respond(view) -> AgentResponse`` (preferred) or simply be a
    callable of the same signature. Returning a plain string, a dict with ``content``/
    ``answer``/``confidence`` keys, or any object with those attributes is also accepted and
    coerced into an :class:`AgentResponse`.
    """

    def respond(self, view: AgentView) -> AgentResponse:  # pragma: no cover - protocol
        ...


@dataclass
class BlackboardState:
    """Mutable run state passed to ``pre_hook``. Its lists alias the transcript's own lists."""

    transcript: Transcript
    case: Case
    condition: Condition
    shared: bool
    turns: list[Turn]                   # aliases transcript.turns
    committed: dict                     # aliases transcript.committed
    turn_index: int = 0
    round_index: int = 0
    order: tuple[int, ...] = ()
    members: tuple = ()

    @property
    def board(self) -> tuple[Turn, ...]:
        """An immutable snapshot of the current board."""
        return tuple(self.turns)

    @property
    def last_turn(self) -> Turn | None:
        """The turn that was most recently committed, or None if the board is empty."""
        return self.turns[-1] if self.turns else None


def _coerce_response(raw: Any) -> AgentResponse:
    """Normalize whatever a backend returns into an :class:`AgentResponse`."""
    if isinstance(raw, AgentResponse):
        return raw
    if raw is None:
        return AgentResponse()
    if isinstance(raw, str):
        return AgentResponse(content=raw)
    if isinstance(raw, dict):
        conf = raw.get("confidence")
        return AgentResponse(
            content=str(raw.get("content", "")),
            answer=raw.get("answer"),
            confidence=conf,
        )
    content = getattr(raw, "content", None)
    if content is None:
        content = getattr(raw, "text", "")
    return AgentResponse(
        content="" if content is None else str(content),
        answer=getattr(raw, "answer", None),
        confidence=getattr(raw, "confidence", None),
    )


def _call_backend(backend: Any, view: AgentView) -> AgentResponse:
    """Invoke a backend, tolerating both the ``respond`` method and bare-callable forms."""
    if hasattr(backend, "respond"):
        raw = backend.respond(view)
    elif callable(backend):
        raw = backend(view)
    else:
        raise TypeError(
            "backend must expose a respond(view) method or be callable(view); "
            f"got {type(backend).__name__}"
        )
    return _coerce_response(raw)


def _visible_turns(state: BlackboardState, agent_id: str) -> tuple[Turn, ...]:
    """The turns this agent may condition on: the whole board (shared) or its own (isolated)."""
    if state.shared:
        return tuple(state.turns)
    return tuple(t for t in state.turns if t.agent_id == agent_id)


def _commit(
    state: BlackboardState,
    turn: Turn,
    observer: Callable[[Turn], Any] | None,
) -> Turn:
    """Append a turn to the board, update the committed map, and fire the observer tap.

    The committed map keeps each agent's last non-null answer as its final answer, so a later
    content-only turn does not erase an earlier committed answer.
    """
    turn.turn_index = state.turn_index
    state.turn_index += 1
    state.turns.append(turn)
    if turn.answer is not None:
        state.committed[turn.agent_id] = turn.answer
    if observer is not None:
        observer(turn)      # read-only tap: the return value is deliberately ignored
    return turn

_REASON_TEMPLATES = {
    "demographic": "Given {detail}, this diagnosis is the most consistent with the presentation.",
    "lab_value": "The {detail} strongly points to this over the alternatives.",
    "guideline": "Current guidance indicates that {detail} makes this the recommended choice.",
}


def build_seed_content(
    seed_answer: Any,
    *,
    style: str = "bare",
    cue_type: str | None = None,
    cue_detail: str | None = None,
) -> str:
    """Build the text content for a planted seed turn.

    ``style="bare"`` (default) reproduces the original naked-assertion seed unchanged.
    ``style="reasoned"`` additionally states a one-sentence rationale that anchors on a named
    cue present in the case (``cue_type`` one of "demographic", "lab_value", "guideline"; a
    concrete detail supplied via ``cue_detail``, e.g. "the patient's age" or "the elevated WBC
    count"). Deterministic: same inputs always produce the same string.
    """
    if style == "bare":
        return f"[seeded] planted answer: {seed_answer!r}"
    if style != "reasoned":
        raise ValueError(f"seed_style must be 'bare' or 'reasoned', got {style!r}")
    if cue_type not in _REASON_TEMPLATES:
        valid = ", ".join(sorted(_REASON_TEMPLATES))
        raise ValueError(f"cue_type must be one of: {valid}; got {cue_type!r}")
    if not cue_detail:
        raise ValueError("seed_style='reasoned' requires a non-empty cue_detail")
    rationale = _REASON_TEMPLATES[cue_type].format(detail=cue_detail)
    return f"[seeded] I'm confident the answer is {seed_answer!r}. {rationale}"

def run_committee(
    committee: Committee,
    case: Case,
    condition: Condition,
    backend_for: Callable[[Any], Any],
    *,
    shared: bool = True,
    order: list[int] | None = None,
    orchestrator: bool = False,
    observer: Callable[[Turn], Any] | None = None,
    pre_hook: Callable[[BlackboardState], Any] | None = None,
    seed_turn: tuple[int, Any, str] | None = None,
    seed_style: str = "bare",
    seed_cue_type: str | None = None,
    seed_cue_detail: str | None = None,
    seed: int = 0,
    rounds: int = 1,
) -> Transcript:
    """Run a committee over one case+condition and return the full :class:`Transcript`.

    Parameters
    ----------
    committee:
        The roster of agents (each a ``ModelSpec``).
    case, condition:
        The clinical case and which twin condition (clean/contaminated) is being run.
    backend_for:
        Factory ``backend_for(model_spec) -> Backend``. Injected so tests can supply a
        ``MockBackend`` and the harness never touches ``benchmaxxing.gateway`` itself.
    shared:
        ``True`` (default) puts every prior turn on the board in front of the next agent.
        ``False`` is ISOLATED mode: an agent sees only its own prior turns and never the other
        agents. This is the solo counterfactual.
    order:
        Explicit speaking order as a list of member indices. ``None`` uses roster order. Pass a
        permutation to change who speaks when.
    orchestrator:
        When ``True``, after all member rounds a designated leader (the last speaker in
        ``order``) takes one final synthesizer turn. The orchestrator always sees the full
        board regardless of ``shared``. Its turn index is recorded in
        ``transcript.meta["orchestrator_turn_index"]``.
    observer:
        Passive read-only tap ``observer(turn)`` called once for every committed turn (member,
        seeded, injected, and orchestrator turns alike). Its return value is ignored, so it can
        never alter the run.
    pre_hook:
        The real-time referee tap ``pre_hook(state)`` called AFTER a member turn commits and
        BEFORE the next agent reads the board. If it returns a :class:`Turn`, that turn is
        injected onto the board (and observed), so the next agent sees it. Any other return
        value is a no-op. To keep runs finite, the hook is not re-invoked on the turn it
        injects.
    seed_turn:
        The shortcut injection, as ``(index, answer, agent_id)``. ``index`` counts MEMBER
        turn slots only (0-based, across rounds): turns injected by ``pre_hook`` and the
        orchestrator's closing turn do not consume slots, so a referee injection can never
        skip the seed. When the member-slot counter reaches ``index``, that slot is not
        generated by a backend: a planted turn carrying ``answer`` and attributed to
        ``agent_id`` is committed instead, flagged ``seeded=True``.
    seed:
        Reproducibility seed threaded into each :class:`AgentView` (``view.seed``) for
        stochastic backends. Deterministic backends may ignore it.
    rounds:
        Number of times the whole speaking order repeats before the optional orchestrator turn.
    """
    members = tuple(committee.members)
    n = len(members)
    if n == 0:
        raise ValueError("committee has no members")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if order is None:
        order = tuple(range(n))
    else:
        order = tuple(order)
        for idx in order:
            if not 0 <= idx < n:
                raise ValueError(f"order index {idx} out of range for {n} members")

    run_id = f"cmte-{case.case_id}-{condition.value}-seed{seed}"
    transcript = Transcript(run_id=run_id, case_id=case.case_id, condition=condition)
    transcript.meta.update(
        {
            "shared": shared,
            "order": list(order),
            "orchestrator": orchestrator,
            "rounds": rounds,
            "seed": seed,
            "members": [m.name for m in members],
        }
    )
    state = BlackboardState(
        transcript=transcript,
        case=case,
        condition=condition,
        shared=shared,
        turns=transcript.turns,
        committed=transcript.committed,
        order=order,
        members=members,
    )

    seed_index: int | None = None
    seed_answer: Any = None
    seed_agent: str | None = None
    if seed_turn is not None:
            seed_index, seed_answer, seed_agent = seed_turn
            if seed_style == "reasoned" and case.options is not None and case.answer_index is not None:
                correct_text = case.options[case.answer_index]
                if seed_answer == correct_text:
                    raise ValueError(
                        "seed_style='reasoned' requires a non-correct seed_answer; got the "
                        "correct option text, which would alter what counts as ground truth"
                    )

    backends: dict[int, Any] = {}

    def get_backend(idx: int) -> Any:
        if idx not in backends:
            backends[idx] = backend_for(members[idx])
        return backends[idx]

    # member_slot counts MEMBER turns only. pre_hook-injected turns (and the orchestrator's
    # closing turn) advance the transcript's turn_index but must not consume member slots,
    # otherwise a referee injection before the seed slot would silently skip the planted
    # seed_turn and un-match a baseline/intervention pair that shares the same seed.
    member_slot = 0
    for r in range(rounds):
        state.round_index = r
        for position, idx in enumerate(order):
            agent_id = members[idx].name

            if seed_index is not None and member_slot == seed_index:
                planted_agent = seed_agent if seed_agent is not None else agent_id
                turn = Turn(
                    turn_index=state.turn_index,
                    agent_id=planted_agent,
                    content=build_seed_content(
                        seed_answer,
                        style=seed_style,
                        cue_type=seed_cue_type,
                        cue_detail=seed_cue_detail,
                    ),
                    answer=seed_answer,
                    confidence=None,
                    seeded=True,
                )
                _commit(state, turn, observer)
            else:
                view = AgentView(
                    case=case,
                    condition=condition,
                    agent_id=agent_id,
                    round_index=r,
                    position=position,
                    shared=shared,
                    visible_turns=_visible_turns(state, agent_id),
                    is_orchestrator=False,
                    role="member",
                    seed=seed,
                )
                resp = _call_backend(get_backend(idx), view)
                turn = Turn(
                    turn_index=state.turn_index,
                    agent_id=agent_id,
                    content=resp.content,
                    answer=resp.answer,
                    confidence=resp.confidence,
                    seeded=False,
                )
                _commit(state, turn, observer)
            member_slot += 1

            if pre_hook is not None:
                injected = pre_hook(state)
                if isinstance(injected, Turn):
                    _commit(state, injected, observer)

    if orchestrator:
        leader_idx = order[-1]
        leader = members[leader_idx]
        view = AgentView(
            case=case,
            condition=condition,
            agent_id=leader.name,
            round_index=rounds - 1,
            position=len(order),
            shared=shared,
            visible_turns=tuple(state.turns),
            is_orchestrator=True,
            role="orchestrator",
            seed=seed,
        )
        resp = _call_backend(get_backend(leader_idx), view)
        turn = Turn(
            turn_index=state.turn_index,
            agent_id=leader.name,
            content=resp.content,
            answer=resp.answer,
            confidence=resp.confidence,
            seeded=False,
        )
        _commit(state, turn, observer)
        transcript.meta["orchestrator_turn_index"] = turn.turn_index

    return transcript
