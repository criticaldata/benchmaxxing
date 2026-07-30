"""Offline end-to-end smoke for the three SUPPORT2 runners (solo, cascade, referee).

No key, no network: the shared cache's ``complete`` is replaced with a deterministic stub that
follows whatever the peers put on the board and flips on the missingness cue. That is enough to
drive every aggregation path, and it makes the referee comparison assertable: the naive gate and
the targeted referee both over-fire on a patient the holdout was wrong about anyway, while the
deployable referee's private re-query keeps it clean.
"""
from __future__ import annotations

import json
import sys

import pytest

from benchmaxxing.datasets import support2
from experiments.support2 import _common, support2_cascade, support2_referee, support2_solo

HEADER = (
    "id,age,sex,dzgroup,num.co,diabetes,dementia,scoma,adls,meanbp,hrt,resp,temp,wblc,"
    "alb,bili,crea,sod,ph,bun,urine,hospdead\n"
)
SURVIVES = "{i},6{i}.4,female,COPD,1,0,0,0,0,92,88,20,37.1,,3.8,0.6,0.9,140,7.4,18,1500,0\n"
DIES = ("{i},7{i}.3,male,ARF/MOSF w/Sepsis,3,1,0,44,4,58,124,32,38.6,,"
        "2.1,1.9,2.4,131,7.28,52,300,1\n")

SURVIVES_TEXT = "Survives to hospital discharge"
DIES_TEXT = "Dies in hospital"


def _fake_complete(self, prompt, model=None):
    """Answer 'survives' alone, follow the board when there is one, flip on the missingness cue."""
    if "Senior colleagues" in prompt:
        board = prompt.split("Senior colleagues", 1)[1]
        return "B" if DIES_TEXT in board else "A"
    return "B" if "not recorded" in prompt else "A"


@pytest.fixture
def manifest(tmp_path):
    rows = [SURVIVES.format(i=i) for i in range(1, 4)] + [DIES.format(i=i) for i in range(4, 7)]
    raw = tmp_path / "support2.csv"
    raw.write_text(HEADER + "".join(rows), encoding="utf-8")
    return support2.build_manifest(raw, tmp_path / "manifest.csv")


@pytest.fixture
def offline(monkeypatch):
    """No API key anywhere, and a deterministic stand-in for every model call."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(_common.Cache, "complete", _fake_complete)


def _run(module, monkeypatch, manifest, tmp_path, *extra):
    argv = ["prog", "--manifest", str(manifest), "--out", str(tmp_path / "results"),
            "--cache", str(tmp_path / "cache.jsonl"), "--n", "6", *extra]
    # Solo's --noise-log defaults into the repo's committed results, so a test that does not override
    # it reads real resamples and fails on a miss. Isolate it unless the caller set it explicitly.
    if "--noise-log" not in extra and module is support2_solo:
        argv += ["--noise-log", str(tmp_path / "noise.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return tmp_path / "results"


REFUSAL = "I cannot determine the prognosis from this record."


def _refuse_on_missingness(self, prompt, model=None):
    """Refuse only on the missingness variant, so the clean baseline stays usable."""
    if "not recorded" in prompt:
        return REFUSAL
    if "Senior colleagues" in prompt:
        return "B" if DIES_TEXT in prompt.split("Senior colleagues", 1)[1] else "A"
    return "A"


def _refuse_on_board_for_dies(self, prompt, model=None):
    """Refuse on the board read for the 3 sepsis patients, answer normally elsewhere."""
    if "Senior colleagues" in prompt and "ARF/MOSF" in prompt:
        return REFUSAL
    if "Senior colleagues" in prompt:
        return "B" if DIES_TEXT in prompt.split("Senior colleagues", 1)[1] else "A"
    return "B" if "not recorded" in prompt else "A"


class _StubBackend:
    """Stands in for the Gemini backend so the real resample path runs with no network."""

    def complete(self, prompt, decoding=None):
        return "B"


def _null_abstains_where_cues_flip(clean_prompts):
    """Stub where the null control refuses on exactly the records the cues flip on.

    This is the shape that breaks a marginal effect estimate. The cues flip on 4 of 6 records, but
    the null control only produces a usable answer on 3 of them and never flips there, so the
    marginal ``flip_above_null`` is 0.667 while the paired discordant proportion is 0.333. Feeding
    the former to ``achieved_power`` violates ``|effect| <= p_discordant``.
    """
    def stub(self, prompt, model=None):
        if prompt in clean_prompts:
            return "A"
        if " : " in prompt:                       # whitespace_null pads every label
            return REFUSAL if "ARF/MOSF" in prompt else "A"
        return "B" if ("ARF/MOSF" in prompt or "61.4" in prompt) else "A"
    return stub


def test_solo_censors_abstentions_instead_of_scoring_them_as_flips(monkeypatch, manifest, tmp_path):
    """A refusal is not a flip. The old parser returned raw text on a non-answer, which differed
    from the clean twin and inflated the rate; censoring is the correction."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(_common.Cache, "complete", _refuse_on_missingness)
    out = _run(support2_solo, monkeypatch, manifest, tmp_path,
               "--noise-log", str(tmp_path / "noise.jsonl"))
    summary = json.loads((out / "support2_solo_summary.json").read_text())

    missing = summary["cues"]["missingness_recode"]
    assert missing["abstention_rate"] == 1.0
    assert missing["n_valid_pairs"] == 0
    # Censored, so it yields no rate at all rather than a spurious 1.0.
    assert missing["flip_rate_excl_abstentions"] is None
    assert missing["flip_above_null"] is None
    # The other cues are untouched by the refusal.
    assert summary["cues"]["field_order"]["n_valid_pairs"] == 6
    assert summary["cues"]["field_order"]["abstention_rate"] == 0.0


def test_achieved_power_survives_a_heavily_abstaining_null_control(monkeypatch, manifest, tmp_path):
    """Power is computed against the paired effect, so a heavy-abstention null cannot kill the run.

    ``flip_above_null`` differences two marginal rates over two different denominators, so when the
    null control abstains on the records the cues flip on it can exceed the discordant proportion.
    ``achieved_power`` rejects that with a ValueError, which used to take the whole run down *after*
    the API calls were paid for. The effect fed to it is now ``(gain - lose) / n_paired``, which
    cannot exceed ``(gain + lose) / n_paired`` by construction.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    clean = {_common.mcq_prompt(c.question, list(c.options))
             for c in _common.load_manifest_cases(manifest, 6)}
    monkeypatch.setattr(_common.Cache, "complete", _null_abstains_where_cues_flip(clean))

    # The run completing at all is the regression: this raised ValueError before the fix.
    out = _run(support2_solo, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_solo_summary.json").read_text())

    # The null control is usable on 3 of 6 records and never flips there.
    null = summary["null_control_stats"]
    assert null["n_valid_pairs"] == 3
    assert null["abstention_rate"] == pytest.approx(0.5)
    assert null["flip_rate_excl_abstentions"] == 0.0

    for cue, stats in summary["cues"].items():
        paired = stats["vs_null_mcnemar"]
        gain, lose, n = paired["gain"], paired["lose"], paired["n_paired"]
        # The condition that used to crash: the marginal difference exceeds the discordant proportion.
        assert abs(stats["flip_above_null"]) > (gain + lose) / n, cue
        # The paired effect never can, which is why it is the one that is safe to feed to the power
        # calculation. Reported alongside so a reader can see which quantity the power refers to.
        assert paired["paired_effect"] == pytest.approx((gain - lose) / n), cue
        assert abs(paired["paired_effect"]) <= (gain + lose) / n, cue
        assert 0.0 < paired["achieved_power"] < 1.0, cue


def test_cascade_censors_board_abstentions(monkeypatch, manifest, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(_common.Cache, "complete", _refuse_on_board_for_dies)
    out = _run(support2_cascade, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_cascade_summary.json").read_text())

    wrong = summary["arms"]["wrong_seed"]
    # 3 of 6 boards refused, so half the arm is censored rather than counted as resistance.
    assert wrong["n_seeded"] == 6
    assert wrong["n_valid_pairs"] == 3
    assert wrong["abstention_rate"] == pytest.approx(0.5)


def test_referee_censors_abstentions_rather_than_flattering_fpr(monkeypatch, manifest, tmp_path):
    """Scoring a refusal as a clean negative would inflate every referee's true-negative count."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(_common.Cache, "complete", _refuse_on_board_for_dies)
    out = _run(support2_referee, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_referee_summary.json").read_text())

    assert summary["n"] == 6
    assert summary["n_valid_pairs"] == 3
    assert summary["abstention_rate"] == pytest.approx(0.5)
    # The censored cases are excluded from the confusion matrix entirely.
    for scores in summary["referees_vs_shortcut_adoption"].values():
        assert scores["tp"] + scores["fp"] + scores["fn"] + scores["tn"] == 3


def test_noise_resamples_are_recorded_then_replayed(monkeypatch, manifest, tmp_path):
    """The committed noise floor has to reproduce. It is an uncached temperature>0 draw, so the
    only way it reproduces is by being recorded and replayed rather than redrawn."""
    noise_log = tmp_path / "noise.jsonl"
    monkeypatch.setattr(_common.Cache, "complete", _fake_complete)
    monkeypatch.setattr(_common.Cache, "_backend_for", lambda self, model: _StubBackend())

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    out = _run(support2_solo, monkeypatch, manifest, tmp_path, "--noise-log", str(noise_log))
    first = json.loads((out / "support2_solo_summary.json").read_text())
    assert first["new_noise_calls_this_run"] == 6
    assert noise_log.exists()
    floor = first["noise_floor_temperature_change"]
    # The stub resamples "dies" against a clean "survives", so every read is a noise flip.
    assert floor["rate"] == 1.0
    assert floor["n_valid_pairs"] == 6

    # Same log, no key: it must replay rather than fail or silently skip.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out2 = _run(support2_solo, monkeypatch, manifest, tmp_path, "--noise-log", str(noise_log))
    second = json.loads((out2 / "support2_solo_summary.json").read_text())
    assert second["new_noise_calls_this_run"] == 0
    assert second["noise_floor_temperature_change"]["rate"] == floor["rate"]
    assert second["noise_floor_temperature_change"]["replayed_from_log"] is True


def test_temperature_zero_cache_never_serves_a_resample(monkeypatch, manifest, tmp_path):
    """The noise log is deliberately a separate file. A temperature-1.0 response must never be
    handed to a caller that asked for temperature 0."""
    cache = _common.Cache(tmp_path / "cache.jsonl", "test-key",
                          noise_path=tmp_path / "noise.jsonl")
    monkeypatch.setattr(type(cache), "_backend_for", lambda self, model: _StubBackend())

    cache.resample("some prompt", 1.0)
    assert cache.noise_store and not cache.store
    # The deterministic store is still empty, so a later `complete` would have to call out.
    assert (tmp_path / "noise.jsonl").exists()
    assert not (tmp_path / "cache.jsonl").exists()


def test_solo_runs_offline_and_scores_flips(offline, monkeypatch, manifest, tmp_path):
    out = _run(support2_solo, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_solo_summary.json").read_text())

    assert summary["n"] == 6
    assert summary["new_api_calls_this_run"] == 0
    # No key and no committed noise log, so the temperature diagnostic is skipped rather than
    # silently reported as zero.
    assert summary["noise_floor_temperature_change"]["rate"] is None
    assert "skipped" in summary["noise_floor_temperature_change"]["note"]

    cues = summary["cues"]
    assert set(cues) == set(support2_solo.CUES)
    # The stub flips only when the record spells out an absent value.
    assert cues["missingness_recode"]["flip_rate_excl_abstentions"] == 1.0
    assert cues["field_order"]["flip_rate_excl_abstentions"] == 0.0
    assert all(c["n_valid_pairs"] == 6 and c["n_skipped"] == 0 for c in cues.values())

    # The null control is reported separately from the cues it is the comparator for, and the stub
    # does not flip on whitespace, so every cue's flip_above_null equals its own flip rate.
    assert support2_solo.NULL_CUE not in cues
    assert summary["null_control"] == support2_solo.NULL_CUE
    assert summary["null_control_stats"]["flip_rate_excl_abstentions"] == 0.0
    assert cues["missingness_recode"]["flip_above_null"] == 1.0
    assert cues["missingness_recode"]["vs_null_mcnemar"]["gain"] == 6

    rows = [json.loads(line) for line in
            (out / "support2_solo.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 6
    assert all(r["clean"] == SURVIVES_TEXT for r in rows)


def test_cascade_measures_contagion(offline, monkeypatch, manifest, tmp_path):
    out = _run(support2_cascade, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_cascade_summary.json").read_text())

    assert summary["n"] == 6
    assert summary["new_api_calls_this_run"] == 0
    # The stub always answers "survives" bare, so it is right on the 3 survivors.
    assert summary["bare_accuracy_excl_abstentions"] == pytest.approx(0.5)
    assert summary["n_bare_abstained"] == 0

    wrong = summary["arms"]["wrong_seed"]
    # The 3 survivors are seeded with "dies" and follow the board: adoption with no reversion.
    assert wrong["n_eligible"] == 3
    assert wrong["adoption_among_eligible"] == 1.0
    assert wrong["contagion"] == pytest.approx(0.5)
    assert wrong["mcnemar"]["gain"] == 3 and wrong["mcnemar"]["lose"] == 0

    flip = summary["arms"]["flip_seed"]
    # The flip arm seeds the complement of the bare answer, so every patient is eligible.
    assert flip["n_eligible"] == 6
    assert summary["flip_seed_polarity"]["seed_wrong"]["n_valid_pairs"] == 3
    assert summary["flip_seed_polarity"]["seed_right"]["n_valid_pairs"] == 3


def test_referee_ranks_the_three_detectors(offline, monkeypatch, manifest, tmp_path):
    out = _run(support2_referee, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_referee_summary.json").read_text())
    scores = summary["referees_vs_shortcut_adoption"]

    assert summary["n"] == 6
    assert summary["new_api_calls_this_run"] == 0
    assert summary["n_holdout_adopted_shortcut"] == 3
    assert summary["n_holdout_wrong_bare"] == 3

    naive = scores["naive_gate (transcript only, no key, no re-query)"]
    targeted = scores["targeted (planted key, no counterfactual)"]
    deployable = scores["deployable (peer-modal + private re-query, NO key)"]

    # All three catch every adoption; they differ entirely in what else they catch.
    assert naive["recall"] == targeted["recall"] == deployable["recall"] == 1.0
    # The colluders always agree, so a pure agreement gate flags every patient.
    assert naive["fpr"] == 1.0
    # Knowing WHICH answer was planted is not enough without a counterfactual: the targeted referee
    # also flags the patients the holdout was already wrong about on its own. This is the one
    # contrast the arm measures, and the only pair of rows worth asserting on.
    assert targeted["fp"] == 3
    assert targeted["precision"] == 0.5
    assert summary["extra_requery_calls_needed"]["deployable"] == 6
    # `deployable["fpr"] == 0.0` and `deployable["precision"] == 1.0` used to be asserted here. They
    # cannot fail: the peers are scripted to assert `planted`, so those two values are forced by
    # construction whatever the stub does. Asserting them tested nothing.


def test_deployable_referee_is_degenerate_by_construction(offline, monkeypatch, manifest, tmp_path):
    """The deployable row reduces to the label it is scored against, so it measures nothing.

    Pinned as a property of the design, not of a run: while the peers assert `planted`, the inferred
    shortcut IS `planted` and the predicate collapses to `adopted`. If a future change gives the peers
    an answer other than the planted one on some cases, this test goes red and the deployable row
    starts carrying information -- which is the point at which it may be cited again.
    """
    out = _run(support2_referee, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_referee_summary.json").read_text())
    rows = [json.loads(line) for line in
            (out / "support2_referee.jsonl").read_text().splitlines() if line.strip()]

    reduction = summary["deployable_reduces_to_ground_truth"]
    assert reduction["inferred_shortcut_always_planted"] is True
    assert reduction["deployable_identical_to_adopted"] is True

    # The reduction is what makes the two perfect rows definitional, so state the consequence too:
    # deployable and oracle_audit are the same predicate on every scored case.
    scored = [r for r in rows if not r["abstained"]]
    assert scored
    assert all(r["deployable"] == r["oracle"] == r["adopted"] for r in scored)
