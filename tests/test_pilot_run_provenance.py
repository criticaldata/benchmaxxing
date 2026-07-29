"""Guard the committed `benchmaxxing run --stage pilot` artifact (#101, #251).

#251 was held under the real-data rule: it shipped a CLI with no real run behind it. This pins the
committed run so the artifact cannot silently become a mock, and so the cross-check against the paper's
MedQA solo numbers stays honest.
"""
import json
from pathlib import Path

RUN = Path(__file__).parent.parent / "experiments" / "runs" / "pilot-medqa-seed0"


def _results():
    return json.loads((RUN / "results.json").read_text())


def test_the_committed_pilot_is_a_real_run_not_a_mock():
    man = json.loads((RUN / "run_manifest.json").read_text())
    assert man["model_ids"] == ["gemini-2.5-flash"], man["model_ids"]
    assert "mock" not in json.dumps(man).lower()
    assert man["seed"] == 0


def test_pilot_reproduces_the_committed_medqa_solo_rates():
    """The whole point: an independent code path must land on the paper's numbers."""
    x = _results()["cross_check_against_committed_solo"]
    for cue in ("longest_option", "option_order"):
        assert x["this_pilot"][cue] == x["committed"][cue], cue
    assert abs(x["this_pilot"]["lexical_overlap"] - x["committed"]["lexical_overlap"]) <= 0.02
    assert abs(x["this_pilot"]["overall"] - x["committed"]["overall"]) <= 0.02


def test_cues_actually_bit():
    """A pilot where no cue moves anything would pass vacuously, so assert the flip rate is nonzero."""
    r = _results()["results"]
    assert r["cues_bite"] is True
    assert r["flip_rate"]["overall"] > 0
    assert r["flip_rate"]["n"] == 60
