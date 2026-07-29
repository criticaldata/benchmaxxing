"""Tests for the stage runners in benchmaxxing.experiments.

Everything runs offline. The solo lane uses deterministic callable backends that model a
surface shortcut over the real text-cue payloads (a dict with ``options``); the committee lane
uses a blackboard-style backend that conforms to the last committed answer, so a planted
shortcut cascades in shared mode but not in isolated mode. On real data with a Gemini roster
the same runners execute unchanged: only the injected backends change.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing import roster
from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.experiments import (
    run_cascade,
    run_holes_test,
    run_pilot,
    run_solo_baselines,
)
from benchmaxxing.schema import Case, Committee, Modality, ModelSpec

# --------------------------------------------------------------------------- solo lane


def _text_case(case_id: str) -> Case:
    """A text MCQ whose correct option is (originally) the strictly longest option."""
    return Case(
        case_id=case_id,
        patient_id=f"p-{case_id}",
        modality=Modality.TEXT,
        question="Which finding is most specific for pneumonia on chest imaging?",
        options=(
            "Diffuse bilateral airspace consolidation with air bronchograms",  # longest, correct
            "Mild basal atelectasis",
            "Normal study",
        ),
        answer_index=0,
    )


def _longest_option_backend(payload) -> str:
    """A 'pick the longest option' shortcut. Flips only when a distractor is padded longest."""
    options = payload["options"]
    best = max(range(len(options)), key=lambda i: len(options[i]))
    return options[best]


def _shortest_option_backend(payload) -> str:
    """A 'pick the shortest option' rule. Cue-blind for these twins: never flips."""
    options = payload["options"]
    best = min(range(len(options)), key=lambda i: len(options[i]))
    return options[best]


def _constant_backend(payload) -> str:
    return "constant"


def _identity(raw):
    return raw


PILOT_CUES = ["longest_option", "option_order"]


def test_run_pilot_cues_bite():
    cases = [_text_case("c0"), _text_case("c1")]
    res = run_pilot(cases, _longest_option_backend, _identity, PILOT_CUES)

    assert res["n"] == 4  # 2 cases x 2 cues
    assert res["cue_types"] == PILOT_CUES
    assert res["cues_bite"] is True
    # The longest-option cue flips the longest-option backend; option order does not.
    assert res["flip_rate"]["overall"] == pytest.approx(0.5)
    assert res["flip_rate"]["per_cue"]["longest_option"] == pytest.approx(1.0)
    assert res["flip_rate"]["per_cue"]["option_order"] == pytest.approx(0.0)


def test_run_pilot_limit_caps_cases():
    cases = [_text_case("c0"), _text_case("c1"), _text_case("c2")]
    res = run_pilot(cases, _longest_option_backend, _identity, PILOT_CUES, limit=1)
    assert res["n"] == 2  # 1 case x 2 cues


def test_run_pilot_no_bite_when_backend_is_cue_blind():
    cases = [_text_case("c0"), _text_case("c1")]
    res = run_pilot(cases, _constant_backend, _identity, PILOT_CUES)
    assert res["cues_bite"] is False
    assert res["flip_rate"]["overall"] == pytest.approx(0.0)


def test_run_solo_baselines_matrix_and_vectors():
    cases = [_text_case("c0"), _text_case("c1")]
    models = [
        ModelSpec(name="m-long", lineage="gemini"),
        ModelSpec(name="m-short", lineage="qwen"),
    ]
    backends = {"m-long": _longest_option_backend, "m-short": _shortest_option_backend}

    res = run_solo_baselines(cases, models, lambda m: backends[m.name], _identity, PILOT_CUES)

    sm = res["susceptibility_matrix"]
    assert sm["models"] == ["m-long", "m-short"]
    assert sm["cues"] == ["longest_option", "option_order"]
    assert sm["matrix"].shape == (2, 2)
    # Every cell has records, so nothing is nan and every rate is a probability.
    assert np.all(np.isfinite(sm["matrix"]))
    assert np.all((sm["matrix"] >= 0.0) & (sm["matrix"] <= 1.0))

    ci = {c: j for j, c in enumerate(sm["cues"])}
    assert sm["matrix"][0, ci["longest_option"]] == pytest.approx(1.0)
    assert sm["matrix"][0, ci["option_order"]] == pytest.approx(0.0)
    assert sm["matrix"][1, ci["longest_option"]] == pytest.approx(0.0)

    fv = res["failure_vectors"]
    assert set(fv) == {"m-long", "m-short"}
    # Two cases x two cues -> length-4 vectors, aligned across models.
    assert fv["m-long"].shape == (4,)
    assert fv["m-short"].tolist() == [0, 0, 0, 0]
    # m-long flips on the longest_option cue only (ordered by case_id, cue_type).
    assert fv["m-long"].tolist() == [1, 0, 1, 0]
    assert res["n_twins"] == 4


def test_run_solo_baselines_runs_on_a_gemini_roster():
    # The same call shape works against roster.default_roster (Gemini + one open-weights entry).
    cases = [_text_case("c0"), _text_case("c1")]
    models = roster.default_roster()
    backend = _longest_option_backend  # one stand-in backend for every roster member

    res = run_solo_baselines(cases, models, lambda m: backend, _identity, PILOT_CUES)
    sm = res["susceptibility_matrix"]
    assert sm["matrix"].shape == (len(models), 2)
    assert set(res["failure_vectors"]) == {m.name for m in models}




def test_run_pilot_progress_emits_start_and_updates():
    class Recorder:
        def __init__(self):
            self.started = []
            self.updates = []

        def start(self, detail=None):
            self.started.append(detail)

        def update(self, completed):
            self.updates.append(completed)

    cases = [_text_case("c0"), _text_case("c1")]
    progress = Recorder()

    res = run_pilot(
        cases,
        _longest_option_backend,
        _identity,
        PILOT_CUES,
        progress=progress,
        dataset_name="medqa",
    )

    assert res["n"] == 4
    assert progress.started == ["dataset=medqa cases=2 twins=4 cues=2"]
    assert progress.updates == [1, 2, 3, 4]


# --------------------------------------------------------------------------- holes test


def test_run_holes_test_within_beats_cross():
    a = np.array([1, 0, 1, 0, 1, 0])
    b = np.array([0, 1, 0, 1, 0, 1])
    vectors = {"g1": a, "g2": a, "q1": b, "q2": b}
    lineages = {"g1": "gemini", "g2": "gemini", "q1": "qwen", "q2": "qwen"}

    res = run_holes_test(vectors, lineages, seed=0)
    assert res["metric"] == "phi"
    assert res["n_models"] == 4
    # Identical within-lineage vectors: perfect within-lineage overlap, positive gap.
    assert res["within_mean"] == pytest.approx(1.0)
    assert res["observed_diff"] > 0
    assert 0.0 <= res["p_value"] <= 1.0


def test_run_holes_test_forwards_metric_and_seed():
    a = np.array([1, 0, 1, 0, 1, 0])
    b = np.array([0, 1, 0, 1, 0, 1])
    vectors = {"g1": a, "g2": a, "q1": b, "q2": b}
    lineages = {"g1": "gemini", "g2": "gemini", "q1": "qwen", "q2": "qwen"}

    res1 = run_holes_test(vectors, lineages, metric="jaccard", seed=7)
    res2 = run_holes_test(vectors, lineages, metric="jaccard", seed=7)
    assert res1["metric"] == "jaccard"
    # Same seed reproduces the permutation p-value exactly.
    assert res1["p_value"] == res2["p_value"]


# --------------------------------------------------------------------------- cascade lane


class ConformBackend:
    """Blackboard backend: echo the most recent committed answer, else answer independently.

    In shared mode a later agent sees the planted shortcut and adopts it (a cascade). In
    isolated mode it sees only its own turns, so the shortcut cannot spread.
    """

    def __init__(self, name: str, own):
        self.name = name
        self.own = own

    def respond(self, view) -> AgentResponse:
        prior = [t for t in view.visible_turns if t.answer is not None]
        if prior:
            return AgentResponse(content=f"{self.name} conforms", answer=prior[-1].answer,
                                 confidence=0.6)
        return AgentResponse(content=f"{self.name} independent", answer=self.own, confidence=0.6)


def _cascade_case() -> Case:
    return Case(
        case_id="cascade-1",
        patient_id="p-1",
        modality=Modality.TEXT,
        question="Which is correct?",
        options=("A", "B", "C"),
        answer_index=0,  # correct option is "A"; the planted shortcut is the distractor "B"
    )


def _make_committee(*names: str) -> Committee:
    members = tuple(
        ModelSpec(name=nm, lineage="gemini" if i % 2 == 0 else "qwen")
        for i, nm in enumerate(names)
    )
    return Committee(members=members)


def _conform_backend_for(own="A"):
    def backend_for(spec: ModelSpec) -> ConformBackend:
        return ConformBackend(spec.name, own=own)

    return backend_for


def test_run_cascade_shortcut_spreads_only_in_shared_mode():
    committee = _make_committee("A", "B", "C")
    res = run_cascade(committee, _cascade_case(), _conform_backend_for(own="A"),
                      seed_index=1, rounds=3)

    # The planted shortcut is the distractor option "B".
    assert res["seeded_answer"] == "B"
    # Shared: the shortcut cascades from the seed turn onward.
    assert res["series"] == [0.0] + [1.0] * 8
    assert res["onset"] == 1
    assert all(a == "B" for a in res["shared_answers"][2:])
    # Isolated: only the seed turn itself carries the shortcut; it never spreads.
    assert res["isolated_answers"].count("B") == 1
    assert all(a == "A" for a in res["isolated_answers"][2:])
    # The shared committee is dominated by the shortcut; the isolated one is not.
    assert res["shared_answers"].count("B") > res["isolated_answers"].count("B")


def test_run_cascade_no_onset_when_shortcut_dominates_from_the_start():
    committee = _make_committee("A", "B", "C")
    # Seeding at turn 0 means every later (conforming) agent adopts the shortcut immediately,
    # so the running-shortcut series is flat and there is no tipping point to locate.
    res = run_cascade(committee, _cascade_case(), _conform_backend_for(own="A"),
                      seed_index=0, rounds=2)
    assert all(v == 1.0 for v in res["series"])
    assert res["onset"] is None
    assert all(a == "B" for a in res["shared_answers"])


def test_run_cascade_runs_on_a_cross_lineage_committee():
    # The same runner drives a real roster committee (Gemini + an open-weights lineage).
    committee = roster.cross_lineage_committees(roster.default_roster(), size=2)[0]
    assert committee.is_cross_lineage
    res = run_cascade(committee, _cascade_case(), _conform_backend_for(own="A"),
                      seed_index=1, rounds=3)
    assert res["onset"] == 1
    assert res["shared_answers"].count("B") > res["isolated_answers"].count("B")
