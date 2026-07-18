"""Offline tests for the cascade ablation sweeps (benchmaxxing.ablations, issues 75-78).

Everything runs against small synthetic committees/cases and deterministic mock backends, so
there are no API keys and no real data. The image dose-response test skips cleanly if PIL is not
installed (the only optional dependency on any path here).
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing.ablations import (
    OrderPoint,
    SeedingPoint,
    SizeSweepPoint,
    StrengthPoint,
    committee_size_sweep,
    cue_strength_sweep,
    order_permutation_run,
    orchestrator_vs_peer,
)
from benchmaxxing.blackboard import AgentResponse, AgentView
from benchmaxxing.roster import build_committee
from benchmaxxing.schema import Case, Committee, Modality, ModelSpec


# --------------------------------------------------------------------------- fixtures


def make_case() -> Case:
    return Case(
        case_id="case-abl",
        patient_id="p-1",
        modality=Modality.TEXT,
        question="Which is correct?",
        options=("A", "B", "C"),
        answer_index=0,
    )


def make_roster(n: int) -> list[ModelSpec]:
    """A synthetic roster of ``n`` distinct models spanning two lineages."""
    return [
        ModelSpec(
            name=f"m{i}",
            lineage="gemini" if i % 2 == 0 else "qwen",
            tier=str(i),
            is_open_weights=(i % 2 == 1),
        )
        for i in range(n)
    ]


class ConformityBackend:
    """Position-based cascade mock: conform to the seed once past the tipping position.

    An agent speaking at ``position < tip`` reports the evidence (ground truth); at or past ``tip``
    it reports the planted seed. Combined with a seeded turn at position 0, this produces a clean
    tug-of-war series (evidence early, seed late) with a detectable cascade onset.
    """

    def __init__(self, name, seed_answer, evidence, tip=2):
        self.name = name
        self.seed_answer = seed_answer
        self.evidence = evidence
        self.tip = tip

    def respond(self, view: AgentView) -> AgentResponse:
        if view.position >= self.tip:
            return AgentResponse(content=f"{self.name} conforms", answer=self.seed_answer,
                                 confidence=0.6)
        return AgentResponse(content=f"{self.name} evidence", answer=self.evidence,
                             confidence=0.6)


class HerdBackend:
    """Herding mock: copy the first answer already on the board, else lead with your own name."""

    def __init__(self, name):
        self.name = name

    def respond(self, view: AgentView) -> AgentResponse:
        for turn in view.visible_turns:
            if turn.answer is not None:
                return AgentResponse(content=f"{self.name} copies", answer=turn.answer,
                                     confidence=0.5)
        return AgentResponse(content=f"{self.name} leads", answer=self.name, confidence=0.5)


class BrightnessBackend:
    """Solo image mock: answer 1 once the (possibly cued) image is bright enough, else 0."""

    def __init__(self, threshold=5.0):
        self.threshold = threshold

    def __call__(self, img):
        return 1 if float(np.asarray(img).mean()) > self.threshold else 0


def conformity_backend_for(seed_answer, evidence, tip=2):
    def backend_for(spec: ModelSpec) -> ConformityBackend:
        return ConformityBackend(spec.name, seed_answer, evidence, tip=tip)

    return backend_for


def herd_backend_for():
    def backend_for(spec: ModelSpec) -> HerdBackend:
        return HerdBackend(spec.name)

    return backend_for


# --------------------------------------------------------------------------- issue 75


def test_committee_size_sweep_reports_onset_and_shortcut_fraction():
    roster = make_roster(6)
    backend_for = conformity_backend_for("SEED", "TRUTH", tip=2)

    points = committee_size_sweep(
        [3, 5],
        make_case(),
        backend_for,
        seed_turn=(0, "SEED", "planter"),
        rounds=1,
        roster=roster,
        cue_type="",
        ground_truth="TRUTH",
    )

    assert [p.size for p in points] == [3, 5]
    assert all(isinstance(p, SizeSweepPoint) for p in points)
    for p in points:
        # A cascade forms (evidence early, seed late), so onset is a real turn index.
        assert isinstance(p.onset, int)
        assert 0 <= p.shortcut_fraction <= 1.0
        # The planter plus the conformers endorse the seed, so the fraction is positive.
        assert p.shortcut_fraction > 0.0
    # Size 3 = 3 turns (seed + 2 members); size 5 = 5 turns.
    assert points[0].n_turns == 3
    assert points[1].n_turns == 5


def test_committee_size_sweep_accepts_explicit_committees():
    roster = make_roster(4)
    committees = [build_committee(roster[:2]), build_committee(roster[:4])]
    backend_for = conformity_backend_for("SEED", "TRUTH", tip=2)

    points = committee_size_sweep(
        [2, 4],
        make_case(),
        backend_for,
        seed_turn=(0, "SEED", "planter"),
        rounds=1,
        committees=committees,
        ground_truth="TRUTH",
    )

    assert [p.size for p in points] == [2, 4]
    assert points[1].n_turns == 4


def test_committee_size_sweep_rejects_oversize_request():
    roster = make_roster(2)
    backend_for = conformity_backend_for("SEED", "TRUTH")
    with pytest.raises(ValueError):
        committee_size_sweep(
            [5],
            make_case(),
            backend_for,
            seed_turn=(0, "SEED", "planter"),
            roster=roster,
        )


# --------------------------------------------------------------------------- issue 76


def test_cue_strength_sweep_is_a_dose_response():
    pytest.importorskip("PIL")
    img = np.zeros((20, 20), dtype=np.uint8)
    backend = BrightnessBackend(threshold=5.0)

    points = cue_strength_sweep(
        img,
        "cable",
        [0.0, 0.5, 1.0],
        backend,
        answer_fn=lambda raw: raw,
        strength_param="opacity",
        ground_truth=0,
        thickness=3,
    )

    assert [round(p.strength, 3) for p in points] == [0.0, 0.5, 1.0]
    assert all(isinstance(p, StrengthPoint) for p in points)
    # No cue (opacity 0) leaves the image untouched, so the answer cannot flip.
    assert points[0].flip_rate == 0.0
    assert points[0].flipped is False
    # A full-strength cable is bright enough to move the answer: a flip.
    assert points[-1].flip_rate == 1.0
    assert points[-1].flipped is True
    # Dose-response is monotone non-decreasing in strength.
    rates = [p.flip_rate for p in points]
    assert rates == sorted(rates)


def test_cue_strength_sweep_records_are_carried_through():
    pytest.importorskip("PIL")
    img = np.zeros((16, 16), dtype=np.uint8)
    points = cue_strength_sweep(
        img,
        "cable",
        [1.0],
        BrightnessBackend(threshold=5.0),
        answer_fn=lambda raw: raw,
        ground_truth=0,
    )
    assert len(points) == 1
    assert len(points[0].records) == 1
    assert points[0].records[0].cue_type == "cable"


# --------------------------------------------------------------------------- issue 77


def test_order_permutation_verdict_tracks_first_speaker():
    roster = make_roster(3)
    committee: Committee = build_committee(roster)  # m0, m1, m2
    backend_for = herd_backend_for()

    orders = [[0, 1, 2], [2, 1, 0], [1, 2, 0]]
    points = order_permutation_run(
        committee,
        make_case(),
        backend_for,
        orders,
        rounds=1,
    )

    assert [p.order for p in points] == [(0, 1, 2), (2, 1, 0), (1, 2, 0)]
    assert all(isinstance(p, OrderPoint) for p in points)
    for p in points:
        # Everyone herds onto the first speaker, so the verdict is the first speaker's own answer.
        assert p.verdict == p.first_speaker
    # The verdict tracks position, not correctness: different orders give different verdicts.
    verdicts = {p.verdict for p in points}
    assert verdicts == {"m0", "m2", "m1"}
    assert [p.first_speaker for p in points] == ["m0", "m2", "m1"]


# --------------------------------------------------------------------------- issue 78


def test_orchestrator_vs_peer_reports_both_regimes():
    roster = make_roster(4)
    committee = build_committee(roster)
    backend_for = conformity_backend_for("SEED", "TRUTH", tip=2)

    result = orchestrator_vs_peer(
        committee,
        make_case(),
        backend_for,
        seed_index=0,
        seed_answer="SEED",
        seed_agent="planter",
        rounds=1,
        cue_type="",
        ground_truth="TRUTH",
    )

    assert set(result) == {"peer", "orchestrator"}
    peer, orch = result["peer"], result["orchestrator"]
    assert isinstance(peer, SeedingPoint) and isinstance(orch, SeedingPoint)
    assert peer.label == "peer" and orch.label == "orchestrator"
    # The orchestrator toggle adds exactly one final synthesizer turn.
    assert orch.n_turns == peer.n_turns + 1
    for point in (peer, orch):
        assert point.onset is None or isinstance(point.onset, int)
        assert 0 <= point.shortcut_fraction <= 1.0
        assert point.shortcut_fraction > 0.0


def test_orchestrator_amplifies_seed_endorsement():
    # With a leader that also conforms, the orchestrator run endorses the seed at least as much.
    roster = make_roster(4)
    committee = build_committee(roster)
    backend_for = conformity_backend_for("SEED", "TRUTH", tip=2)

    result = orchestrator_vs_peer(
        committee,
        make_case(),
        backend_for,
        seed_index=0,
        seed_answer="SEED",
        seed_agent="planter",
        rounds=1,
        ground_truth="TRUTH",
    )
    assert result["orchestrator"].shortcut_fraction >= result["peer"].shortcut_fraction
