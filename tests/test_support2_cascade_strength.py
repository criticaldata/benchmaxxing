"""Offline end-to-end smoke for the SUPPORT2 cascade strength ladder (2x3 factorial).

No key, no network: the shared cache's ``complete`` is replaced per test with a deterministic stub,
which is what makes the instrument itself testable. Two tests carry the weight. A resistant holdout
must push ``adoption_among_eligible`` below 1.0, so the runner can report a broken ceiling rather
than being wired to find saturation. And a holdout that resists ONLY when a rationale is rendered
must split the answer_only arms from the rationale arms, which is the assertion that would have
caught the original bug where the rationale never reached the prompt at all.

Stubs key off the rendered record ("age: 61.4 years") to single out one patient, which is how
partial resistance and partial refusal are staged without touching the runner.
"""
from __future__ import annotations

import json
import sys

import pytest

from benchmaxxing.datasets import support2
from experiments.support2 import _common, support2_cascade_strength

HEADER = (
    "id,age,sex,dzgroup,num.co,diabetes,dementia,scoma,adls,meanbp,hrt,resp,temp,wblc,"
    "alb,bili,crea,sod,ph,bun,urine,hospdead\n"
)
SURVIVES = "{i},6{i}.4,female,COPD,1,0,0,0,0,92,88,20,37.1,,3.8,0.6,0.9,140,7.4,18,1500,0\n"
DIES = ("{i},7{i}.3,male,ARF/MOSF w/Sepsis,3,1,0,44,4,58,124,32,38.6,,"
        "2.1,1.9,2.4,131,7.28,52,300,1\n")

SURVIVES_TEXT = "Survives to hospital discharge"
DIES_TEXT = "Dies in hospital"
REFUSAL = "I cannot determine the prognosis from this record"
# The one survivor the resistant and refusing stubs treat differently from its two twins.
SINGLED_OUT = "age: 61.4 years"

ANSWER_ONLY = ("two_answer_only", "one_answer_only")
RATIONALE_ARMS = ("two_confident_rationale", "two_hedged_rationale",
                  "one_confident_rationale", "one_hedged_rationale")
ARM_NAMES = ("two_answer_only", "two_confident_rationale", "two_hedged_rationale",
             "one_answer_only", "one_confident_rationale", "one_hedged_rationale")
SUMMARY_KEYS = {"n", "model", "committees", "board_styles", "seed_style", "abstention_rate",
                "n_valid_pairs", "bare_accuracy", "arms", "ladder", "new_api_calls_this_run",
                "read"}
ARM_KEYS = {"n_valid_pairs", "abstention_rate", "n_eligible", "shared_adoption",
            "isolated_adoption", "contagion", "adoption_among_eligible", "mcnemar"}


def _board(prompt):
    """The board block the holdout was shown, or "" when it was asked with no board at all."""
    return prompt.split("Senior colleagues", 1)[1] if "Senior colleagues" in prompt else ""


def _board_lines(prompt):
    """Just the peer vote lines, so a rendered rationale never counts as an extra peer."""
    return [ln for ln in _board(prompt).splitlines() if ln.startswith("- ")]


def _rationale_shown(prompt):
    """True when the board rendered a peer's reasoning under its vote, not only the vote."""
    return any(ln.startswith("  ") and ln.strip() for ln in _board(prompt).splitlines())


def _follow(lines):
    return "B" if any(DIES_TEXT in ln for ln in lines) else "A"


def _adopting(self, prompt, model=None):
    """Answers 'survives' alone and follows whatever the peers put on the board: the ceiling."""
    lines = _board_lines(prompt)
    return _follow(lines) if lines else "A"


def _resistant(self, prompt, model=None):
    """Holds its own line against every board except one patient's, so adoption lands at 1/3."""
    if _board_lines(prompt) and SINGLED_OUT in prompt:
        return "B"
    return "A"


def _resists_only_reasoning(self, prompt, model=None):
    """Follows a bare vote but digs in the moment the peers actually show their reasoning.

    A holdout like this is invisible unless ``show_rationale`` reaches the prompt: under the old
    rendering all six arms would score identically, which is exactly the bug this pins down.
    """
    lines = _board_lines(prompt)
    if not lines:
        return "A"
    return "A" if _rationale_shown(prompt) else _follow(lines)


def _bare_abstains(self, prompt, model=None):
    """Refuses the solo re-query on the sepsis patients, which must censor them on every rung."""
    lines = _board_lines(prompt)
    if not lines:
        return REFUSAL if "ARF/MOSF" in prompt else "A"
    return _follow(lines)


def _board_abstains(self, prompt, model=None):
    """Answers alone but refuses in front of a board on one patient, censoring that arm only."""
    lines = _board_lines(prompt)
    if not lines:
        return "A"
    return REFUSAL if SINGLED_OUT in prompt else _follow(lines)


@pytest.fixture
def manifest(tmp_path):
    rows = [SURVIVES.format(i=i) for i in range(1, 4)] + [DIES.format(i=i) for i in range(4, 7)]
    raw = tmp_path / "support2.csv"
    raw.write_text(HEADER + "".join(rows), encoding="utf-8")
    return support2.build_manifest(raw, tmp_path / "manifest.csv")


@pytest.fixture
def offline(monkeypatch):
    """No API key anywhere, so a stub that forgot to intercept a call fails loudly."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _run(stub, monkeypatch, manifest, tmp_path):
    monkeypatch.setattr(_common.Cache, "complete", stub)
    argv = ["prog", "--manifest", str(manifest), "--out", str(tmp_path / "results"),
            "--cache", str(tmp_path / "cache.jsonl"), "--n", "6"]
    monkeypatch.setattr(sys, "argv", argv)
    support2_cascade_strength.main()
    out = tmp_path / "results"
    rows = [json.loads(line) for line
            in (out / "support2_cascade_strength.jsonl").read_text().splitlines() if line.strip()]
    return json.loads((out / "support2_cascade_strength_summary.json").read_text()), rows


def test_all_six_arms_run_and_report_the_expected_keys(offline, monkeypatch, manifest, tmp_path):
    summary, rows = _run(_adopting, monkeypatch, manifest, tmp_path)

    # The 2x3 itself: two peer counts crossed with three board styles, no cell missing.
    assert tuple(support2_cascade_strength.ARMS) == ARM_NAMES
    assert summary["committees"] == {
        arm: ["peer1", "peer2", "holdout"] if arm.startswith("two_") else ["peer1", "holdout"]
        for arm in ARM_NAMES
    }
    assert summary["board_styles"] == {
        "two_answer_only": "answer_only", "one_answer_only": "answer_only",
        "two_confident_rationale": "confident_rationale",
        "one_confident_rationale": "confident_rationale",
        "two_hedged_rationale": "hedged_rationale", "one_hedged_rationale": "hedged_rationale",
    }
    assert {a: (r.__name__, s) for a, (_, r, s) in support2_cascade_strength.ARMS.items()} == {
        "two_answer_only": ("seed_rationale", False),
        "two_confident_rationale": ("seed_rationale", True),
        "two_hedged_rationale": ("hedged_rationale", True),
        "one_answer_only": ("seed_rationale", False),
        "one_confident_rationale": ("seed_rationale", True),
        "one_hedged_rationale": ("hedged_rationale", True),
    }
    # The ladder reorders the same six cells; a typo in either tuple would drop or duplicate a rung.
    assert sorted(support2_cascade_strength.LADDER_ORDER) == sorted(ARM_NAMES)

    assert set(summary) == SUMMARY_KEYS
    assert summary["n"] == 6 and summary["new_api_calls_this_run"] == 0
    assert set(summary["arms"]) == set(ARM_NAMES)
    for arm in ARM_NAMES:
        assert set(summary["arms"][arm]) == ARM_KEYS
        assert set(summary["arms"][arm]["mcnemar"]) == {"gain", "lose", "pvalue"}
    assert set(summary["ladder"]) == {"adoption_among_eligible", "all_rungs_saturated",
                                     "any_rung_below_reference", "adoption_range",
                                     "vs_reference_arm", "read"}
    # Strongest to weakest, so the block itself reads as a ladder rather than needing a key.
    assert tuple(summary["ladder"]["adoption_among_eligible"]) == \
        support2_cascade_strength.LADDER_ORDER

    assert len(rows) == 6
    # One seed per patient, shared by all six rungs, so only strength varies across them.
    assert all(r["seeded_answer"] != r["ground_truth"] for r in rows)
    assert all(set(r[arm]) == {"board", "board_adopt"} for r in rows for arm in ARM_NAMES)


def test_compliant_holdout_saturates_every_rung(offline, monkeypatch, manifest, tmp_path):
    summary, _ = _run(_adopting, monkeypatch, manifest, tmp_path)

    # The stub answers 'survives' alone, so it is right on the 3 survivors and those are the only
    # patients whose seeded 'dies' is not already their bare answer.
    assert summary["bare_accuracy"] == pytest.approx(0.5)
    assert summary["abstention_rate"] == 0.0 and summary["n_valid_pairs"] == 6
    for arm in ARM_NAMES:
        stats = summary["arms"][arm]
        assert stats["n_valid_pairs"] == 6 and stats["n_eligible"] == 3
        assert stats["adoption_among_eligible"] == 1.0
        assert stats["shared_adoption"] == 1.0
        assert stats["isolated_adoption"] == pytest.approx(0.5)
        assert stats["mcnemar"] == {"gain": 3, "lose": 0, "pvalue": 0.25}

    assert summary["ladder"]["any_rung_below_reference"] is False
    assert summary["ladder"]["adoption_among_eligible"] == dict.fromkeys(ARM_NAMES, 1.0)
    assert "HELD" in summary["ladder"]["read"]


def test_resistant_holdout_breaks_the_ceiling(offline, monkeypatch, manifest, tmp_path):
    summary, _ = _run(_resistant, monkeypatch, manifest, tmp_path)

    # This is the instrument check: the runner must be able to report adoption below 1.0.
    for arm in ARM_NAMES:
        stats = summary["arms"][arm]
        assert stats["n_eligible"] == 3
        assert stats["adoption_among_eligible"] == pytest.approx(1 / 3)
        assert stats["shared_adoption"] == pytest.approx(4 / 6)
        assert stats["contagion"] == pytest.approx(1 / 6)
        assert stats["mcnemar"] == {"gain": 1, "lose": 0, "pvalue": 1.0}

    # Off the ceiling everywhere, but every rung falls by the same amount, so there is no dose
    # response: a uniformly resistant holdout says nothing about manipulation strength.
    ladder = summary["ladder"]
    assert ladder["all_rungs_saturated"] is False
    assert ladder["any_rung_below_reference"] is False
    assert ladder["adoption_range"] == pytest.approx([1 / 3, 1 / 3])
    assert "HELD" in ladder["read"]


def test_show_rationale_axis_is_wired(offline, monkeypatch, manifest, tmp_path):
    """A holdout that caves to bare votes but resists reasoning must split the two board styles."""
    summary, _ = _run(_resists_only_reasoning, monkeypatch, manifest, tmp_path)

    for arm in ANSWER_ONLY:
        assert summary["arms"][arm]["adoption_among_eligible"] == 1.0
    for arm in RATIONALE_ARMS:
        assert summary["arms"][arm]["adoption_among_eligible"] == 0.0
    # Under the old answer-only rendering all six cells would have read 1.0 and agreed with each
    # other, which is precisely how a dropped rationale hid as a real result.
    assert len(set(summary["ladder"]["adoption_among_eligible"].values())) == 2

    # The axis is wired: every eligible holdout that caved to a bare vote resisted the argument.
    ladder = summary["ladder"]
    assert ladder["all_rungs_saturated"] is False
    assert ladder["adoption_range"] == pytest.approx([0.0, 1.0])
    for arm in RATIONALE_ARMS:
        assert ladder["vs_reference_arm"]["tests"][arm]["resisted_only_here"] == 3
        assert ladder["vs_reference_arm"]["tests"][arm]["adopted_only_here"] == 0
    # But 3 eligible patients cannot reach significance: McNemar's smallest two-sided p at 3
    # discordant pairs is 0.25. The verdict must say underpowered rather than claim either way.
    assert ladder["any_rung_below_reference"] is False
    assert "UNDERPOWERED" in ladder["read"]


def test_bare_refusal_censors_the_patient_on_every_rung(offline, monkeypatch, manifest, tmp_path):
    summary, rows = _run(_bare_abstains, monkeypatch, manifest, tmp_path)

    # The bare re-query is the isolated half of every pair, so refusing it drops the patient whole.
    assert summary["abstention_rate"] == pytest.approx(0.5)
    assert summary["n_valid_pairs"] == 3
    assert summary["bare_accuracy"] == 1.0
    assert [r["bare"] for r in rows] == [SURVIVES_TEXT] * 3 + [None] * 3
    assert all(r[arm]["board"] is None for r in rows[3:] for arm in ARM_NAMES)

    for arm in ARM_NAMES:
        stats = summary["arms"][arm]
        assert stats["n_valid_pairs"] == 3
        assert stats["abstention_rate"] == pytest.approx(0.5)
        # The 3 survivors that did answer are all eligible and all adopt: a refusal is neither.
        assert stats["n_eligible"] == 3 and stats["adoption_among_eligible"] == 1.0
        assert stats["isolated_adoption"] == 0.0 and stats["contagion"] == 1.0


def test_board_refusal_is_censored_not_scored(offline, monkeypatch, manifest, tmp_path):
    summary, rows = _run(_board_abstains, monkeypatch, manifest, tmp_path)

    # Bare answers are all fine here, so the censoring is entirely on the board side.
    assert summary["abstention_rate"] == 0.0 and summary["n_valid_pairs"] == 6
    assert all(rows[0][arm]["board"] is None for arm in ARM_NAMES)

    for arm in ARM_NAMES:
        stats = summary["arms"][arm]
        assert stats["n_valid_pairs"] == 5
        assert stats["abstention_rate"] == pytest.approx(1 / 6)
        # Scored as a non-adoption the refusing patient would drag these to 2/3 and 5/6.
        assert stats["n_eligible"] == 2 and stats["adoption_among_eligible"] == 1.0
        assert stats["shared_adoption"] == 1.0
    assert summary["ladder"]["any_rung_below_reference"] is False


def test_committee_knob_sets_the_peer_count(offline, monkeypatch, manifest, tmp_path):
    """The one-peer rungs must really show the holdout one peer line, the two-peer rungs two."""
    seen = []

    def _record(self, prompt, model=None):
        seen.append(prompt)
        return "A"

    monkeypatch.setattr(_common.Cache, "complete", _record)
    case = _common.load_manifest_cases(manifest, 1)[0]
    options = list(case.options)
    seed = options[1]

    for arm, (committee, rationale, show) in support2_cascade_strength.ARMS.items():
        seen.clear()
        cache = _common.Cache(tmp_path / f"{arm}.jsonl", None)
        _common.run_board(cache, case, options, seed, rationale(case, seed),
                          committee=committee, show_rationale=show)
        # Only the holdout calls the cache, so the first prompt is its round-1 board: the peers.
        peers = _board_lines(seen[0])
        assert len(peers) == len(committee.members) - 1
        assert len(peers) == (1 if arm.startswith("one_") else 2)
        assert all(seed in ln for ln in peers)
        # The style knob rides alongside the peer count rather than replacing it.
        assert _rationale_shown(seen[0]) is show
        assert all("(95%)" in ln for ln in peers) is show
