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
  planted answer rather than being generated, and that turn is flagged ``seeded=True``. Pass a
  sequence of specs to plant several positions (a 2-peer majority), and a fourth element to write
  the planted turn's content yourself (a peer that argues for the shortcut rather than asserting
  it bare, which is what the anchored-vs-generic contrast in
  :mod:`benchmaxxing.anchored_seed` varies).
* ``board_seed`` is the shared-context injection: a spurious artifact (a "pre-screen flag",
  a "prior similar cases were all X" note, a leaked proxy metric) placed on the board BEFORE
  any agent deliberates and NOT attributed to a committee member. Unlike ``seed_turn`` (a peer
  utterance that only propagates in shared mode), a board seed is ambient workspace context: it
  is flagged ``context=True`` so every agent sees it in isolated mode too, which isolates the
  shared-context channel from peer conformity. It consumes no member slot and stays out of the
  per-agent committed map.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, runtime_checkable, Sequence

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
    """The turns this agent may condition on: the whole board (shared) or its own (isolated).

    In isolated mode an agent also sees ``context`` turns (ambient workspace artifacts): isolation
    hides peer utterances, not the shared workspace, so a board seed anchors each agent solo.
    """
    if state.shared:
        return tuple(state.turns)
    return tuple(t for t in state.turns if t.agent_id == agent_id or t.context)


def _commit(
    state: BlackboardState,
    turn: Turn,
    observer: Callable[[Turn], Any] | None,
) -> Turn:
    """Append a turn to the board, update the committed map, and fire the observer tap.

    The committed map keeps each agent's last non-null answer as its final answer, so a later
    content-only turn does not erase an earlier committed answer. Context artifacts (board seeds)
    are ambient workspace signals, not agent decisions, so they never write to committed.
    """
    turn.turn_index = state.turn_index
    state.turn_index += 1
    state.turns.append(turn)
    if turn.answer is not None and not turn.context:
        state.committed[turn.agent_id] = turn.answer
    if observer is not None:
        observer(turn)      # read-only tap: the return value is deliberately ignored
    return turn


def _seed_specs(seed_turn: Any) -> dict[int, tuple[Any, str | None, str | None]]:
    """Normalize ``seed_turn`` into ``{member_slot: (answer, agent_id, content)}``.

    Accepts one spec or a sequence of them; a spec is ``(index, answer, agent_id)`` or
    ``(index, answer, agent_id, content)``. A single spec is told from a sequence of specs by its
    first element being the integer slot index. Two specs on the same slot would silently drop one,
    so a duplicate index raises.
    """
    if seed_turn is None:
        return {}
    specs = [seed_turn] if seed_turn and isinstance(seed_turn[0], int) else list(seed_turn)
    normalized: dict[int, tuple[Any, str | None, str | None]] = {}
    for spec in specs:
        parts = tuple(spec)
        if len(parts) == 3:
            index, answer, agent_id = parts
            content = None
        elif len(parts) == 4:
            index, answer, agent_id, content = parts
        else:
            raise ValueError(
                "a seed_turn spec must be (index, answer, agent_id) or "
                f"(index, answer, agent_id, content); got {len(parts)} elements: {parts!r}"
            )
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"seed_turn index must be a non-negative member slot, got {index!r}")
        if index in normalized:
            raise ValueError(f"two seed_turn specs target member slot {index}; slots must be unique")
        normalized[index] = (answer, agent_id, None if content is None else str(content))
    return normalized


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
    seed_turn: tuple | Sequence[tuple] | None = None,
    board_seed: Turn | Sequence[Turn] | None = None,
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
        The shortcut injection, as ``(index, answer, agent_id)`` or, to write the planted turn's
        text yourself, ``(index, answer, agent_id, content)``; pass a sequence of specs to plant
        several slots (a 2-peer majority). ``index`` counts MEMBER turn slots only (0-based,
        across rounds): turns injected by ``pre_hook`` and the orchestrator's closing turn do not
        consume slots, so a referee injection can never skip the seed. When the member-slot
        counter reaches ``index``, that slot is not generated by a backend: a planted turn
        carrying ``answer`` and attributed to ``agent_id`` is committed instead, flagged
        ``seeded=True``. Omitted ``content`` falls back to the bare planted-answer assertion, so
        what a peer ARGUES is the caller's variable while the flag still marks the turn planted.
    board_seed:
        A spurious context artifact (or a sequence of them) placed on the board BEFORE the first
        member speaks and NOT attributed to any committee member. Each is committed as a fresh
        copy forced to ``context=True`` and ``seeded=True``, so it is visible in both shared and
        isolated mode, stays out of the committed map, and consumes no member slot. This is the
        shared-context channel (ambient workspace signal) as opposed to ``seed_turn``'s peer
        utterance. A board seed must NOT be attributed to a committee member (that would make it a
        peer utterance visible in isolated mode); a member ``agent_id`` raises ``ValueError``. The
        implied answers are recorded in ``transcript.meta["board_seed"]``.
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

    member_names = {m.name for m in members}

    # Board seeds: ambient context artifacts committed before any member speaks. Fresh copies are
    # forced context=True/seeded=True so they never mutate the caller's Turn and never leak into
    # the committed map, and they precede the member loop so member_slot (and thus seed_turn
    # indexing) is untouched.
    if board_seed is not None:
        seeds = [board_seed] if isinstance(board_seed, Turn) else list(board_seed)
        for artifact in seeds:
            if artifact.agent_id in member_names:
                raise ValueError(
                    "a board_seed artifact must not be attributed to a committee member "
                    f"(got agent_id={artifact.agent_id!r}); it is ambient workspace context, not "
                    "an agent utterance, and a member id would leak it as a peer in isolated mode"
                )
            _commit(state, replace(artifact, context=True, seeded=True), observer)
        transcript.meta["board_seed"] = [t.answer for t in state.turns if t.context]

    seed_specs = _seed_specs(seed_turn)

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

            if member_slot in seed_specs:
                seed_answer, seed_agent, seed_content = seed_specs[member_slot]
                planted_agent = seed_agent if seed_agent is not None else agent_id
                turn = Turn(
                    turn_index=state.turn_index,
                    agent_id=planted_agent,
                    content=(
                        seed_content
                        if seed_content is not None
                        else f"[seeded] planted answer: {seed_answer!r}"
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
                    if injected.context and injected.agent_id in member_names:
                        raise ValueError(
                            "a pre_hook must not inject a context turn attributed to a committee "
                            f"member (got agent_id={injected.agent_id!r}); context turns are "
                            "ambient and visible in isolated mode, so this would leak a peer"
                        )
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
