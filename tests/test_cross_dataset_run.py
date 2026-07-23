"""Offline tests for the cross-dataset real-run entrypoint (issue #129).

The script needs a key and staged manifests to produce real numbers, so these tests exercise only
the parts that run offline: the skip paths, the manifest-loading + scoring wiring (with an injected
loader and backend), and the ``payload -> option-text`` normalizer that keeps ``option_order``
scored by answer identity. The real flip rates are deferred to a keyed run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from benchmaxxing import gateway
from benchmaxxing.schema import Case, Modality

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "cross_dataset" / "run_cross_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_cross_dataset", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_cross = _load_module()


def _case(cid: str) -> Case:
    return Case(
        case_id=cid,
        patient_id=f"p-{cid}",
        modality=Modality.TEXT,
        question=f"Question {cid}?",
        options=(f"{cid}-o0", f"{cid}-o1", f"{cid}-o2"),
        answer_index=0,
    )


def test_main_skips_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rc = run_cross.main(["--medqa-manifest", "a.csv", "--medmcqa-manifest", "b.csv"])
    assert rc == 0
    assert "[skip]" in capsys.readouterr().out


def test_main_skips_when_manifest_missing(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    missing = str(tmp_path / "nope.csv")
    rc = run_cross.main(["--medqa-manifest", missing, "--medmcqa-manifest", missing])
    assert rc == 0
    assert "manifest not found" in capsys.readouterr().out


def test_run_wires_loader_and_backend_offline():
    # Inject a fake loader and a pick-first backend so no network or manifest is touched.
    datasets = {"medqa": [_case("a0")], "medmcqa": [_case("b0")]}
    result = run_cross.run(
        "medqa.csv",
        "medmcqa.csv",
        model="mock",
        api_key="k",
        loader=lambda path: datasets["medqa"] if "medqa" in path else datasets["medmcqa"],
        backend_factory=lambda model, key: (lambda payload: payload["options"][0]),
    )
    assert result["datasets"] == ["medqa", "medmcqa"]
    assert set(result["per_cue"]) == set(run_cross.CUES)
    assert result["n_twins"]["medqa"] >= 1


def test_backend_resolves_choice_to_option_text():
    # The parser resolves a "letter. text" reply (what the prompt asks for) to the option text.
    raw = gateway.MockBackend(rule=lambda prompt, image, decoding: "B. opt1")
    backend = run_cross.make_backend("m", "k", raw=raw)
    payload = {"options": ["opt0", "opt1", "opt2"], "question": "q", "report": ""}
    assert backend(payload) == "opt1"


def test_backend_maps_abstention_to_sentinel():
    raw = gateway.MockBackend(rule=lambda prompt, image, decoding: "")
    backend = run_cross.make_backend("m", "k", raw=raw)
    payload = {"options": ["opt0", "opt1", "opt2"], "question": "q", "report": ""}
    assert backend(payload) == run_cross._ABSTAIN


def test_format_table_reads_the_result():
    result = {
        "datasets": ["medqa", "medmcqa"],
        "cue_types": ["option_order"],
        "per_cue": {
            "option_order": {
                "rate": {"medqa": 0.4, "medmcqa": 0.5},
                "spread": 0.1,
                "fisher": {"oddsratio": 1.0, "pvalue": 0.8},
            }
        },
        "agreement": {"mean_spread": 0.1, "n_cues_compared": 1},
    }
    table = run_cross.format_table(result)
    assert "option_order" in table
    assert "0.400" in table and "0.500" in table
    assert "mean_spread" in table


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
