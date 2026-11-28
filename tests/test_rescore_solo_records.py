"""Saved solo rows under the option-identity flip: the rescoring script, the regenerated files, and
the list of files that still carry the earlier string comparison.

``experiments/medqa/rescore_solo_records.py`` recomputes ``flipped`` from the two answers each saved
row already stores, so it needs no cache and no model. Here it is pinned on synthetic rows (a same
option whose text the cue rewrote, a real move, an answer text made by an earlier cue builder), and
the committed solo files are checked against their own rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Case, Condition, Modality

ROOT = Path(__file__).resolve().parents[1]
MEDQA = ROOT / "experiments/medqa/results"
MEDMCQA = ROOT / "experiments/medmcqa/results"
REWRITING_CUES = ("longest_option", "lexical_overlap")


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_").removesuffix(".py"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "experiments"))
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


rescore = _load("experiments/medqa/rescore_solo_records.py")


def _case(case_id: str, question: str | None = None) -> Case:
    return Case(
        case_id=case_id, patient_id=f"p-{case_id}", modality=Modality.TEXT,
        question=question or "Which drug is first line for this presentation of severe community pneumonia?",
        options=("alpha one", "beta two three", "gamma", "delta four five six"), answer_index=2)


def _row(case_id, cue, clean, cued, model="m"):
    # the layout reproduce.py wrote before the fix: `flipped` is the string comparison
    return {"case_id": case_id, "cue": cue, "model": model, "clean": clean, "contaminated": cued,
            "flipped": clean != cued, "clean_correct": clean == "gamma", "contaminated_correct": cued == "gamma"}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# --- the script on synthetic rows -----------------------------------------------------------------


def test_rescoring_separates_a_rewritten_text_from_a_changed_option(tmp_path, capsys):
    cases = [_case(f"c{i}") for i in range(4)]
    manifest = write_manifest(cases, tmp_path / "manifest.csv")
    cued = {cue: build_text_twin(cases[0], cue).contaminated["options"] for cue in (*REWRITING_CUES, "option_order")}
    rows = []
    for cue in REWRITING_CUES:
        rows += [
            _row("c0", cue, "alpha one", cued[cue][0]),             # same option, text rewritten
            _row("c1", cue, "gamma", cued[cue][0]),                 # right answer -> cued distractor
            _row("c2", cue, "gamma", "gamma"),                      # held the right answer
            _row("c3", cue, "alpha one", "alpha one from an earlier builder"),  # text no builder makes now
        ]
    rows += [_row("c0", "option_order", "gamma", "gamma"), _row("c1", "option_order", "gamma", "alpha one"),
             _row("c2", "option_order", "", "gamma"), _row("c3", "option_order", "", "")]
    records = tmp_path / "solo_records.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out, summary = tmp_path / "rescored.jsonl", tmp_path / "summary.json"
    assert rescore.main(["--manifest", str(manifest), "--records", str(records),
                         "--out-records", str(out), "--out-summary", str(summary)]) == 0
    capsys.readouterr()
    got = {(r["case_id"], r["cue"]): r for r in _rows(out)}
    for cue in REWRITING_CUES:
        assert (got["c0", cue]["flipped"], got["c0", cue]["text_changed"]) == (False, True)
        assert (got["c1", cue]["flipped"], got["c1", cue]["text_changed"]) == (True, True)
        assert (got["c2", cue]["flipped"], got["c2", cue]["text_changed"]) == (False, False)
        assert (got["c3", cue]["flipped"], got["c3", cue]["text_changed"]) == (False, True)  # by prefix
    assert [got[f"c{i}", "option_order"]["flipped"] for i in range(4)] == [False, True, True, False]
    assert all(r["flipped"] == r["text_changed"] for r in got.values() if r["cue"] == "option_order")
    # the saved column order, with the earlier comparison right after the flip
    assert list(_rows(out)[0]) == ["case_id", "cue", "model", "clean", "contaminated", "flipped",
                                   "text_changed", "clean_correct", "contaminated_correct"]

    s = json.loads(summary.read_text())
    for cue in REWRITING_CUES:
        c = s["counts"][f"m|{cue}"]
        assert (c["n"], c["text_changed"], c["flipped"]) == (4, 3, 1)
        assert c["same_option_text_rewritten"] == 2 and c["matched_by_prefix"] == 1
        assert c["stored_flag_is_not_the_text_comparison"] == 0
    assert s["counts"]["m|option_order"]["names_no_option"] == 2
    assert s["flip_rate_by_model"]["m"]["per_cue"] == {"lexical_overlap": 0.25, "longest_option": 0.25, "option_order": 0.5}
    assert s["text_changed_rate_by_model"]["m"]["per_cue"] == {"lexical_overlap": 0.75, "longest_option": 0.75,
                                                               "option_order": 0.5}

    # rescoring rescored rows changes nothing: the earlier comparison is read from text_changed
    again = tmp_path / "again.jsonl"
    assert rescore.main(["--manifest", str(manifest), "--records", str(out), "--out-records", str(again)]) == 0
    assert again.read_text() == out.read_text()


def test_rows_from_another_manifest_are_refused(tmp_path):
    manifest = write_manifest([_case("c0")], tmp_path / "manifest.csv")
    records = tmp_path / "solo_records.jsonl"
    records.write_text(json.dumps(_row("not-in-the-manifest", "longest_option", "gamma", "gamma")) + "\n")
    with pytest.raises(SystemExit) as exc:
        rescore.main(["--manifest", str(manifest), "--records", str(records)])
    assert "not in the manifest" in str(exc.value)


# --- the MIMIC-CXR text re-analysis uses the same comparison --------------------------------------


def test_the_mimic_text_reanalysis_counts_options_and_keeps_the_text_count(tmp_path, monkeypatch, capsys):
    """A third copy of the string comparison lived in refusal_aware_reanalysis.py. Its inputs are not
    in the repository (PhysioNet terms), so it is exercised here on a synthetic manifest and cache."""
    mod = _load("experiments/mimic_cxr_text/refusal_aware_reanalysis.py")
    # one stem per case: the cache is keyed on the prompt, so two cases must not share one
    cases = [_case(f"c{i}", f"Report number {i + 1}: what is the primary finding described here?") for i in range(4)]
    manifest = write_manifest(cases, tmp_path / "manifest.csv")
    model = "some-lab/some-model"

    def letter_for(payload, option_text):
        return "ABCD"[list(payload["options"]).index(option_text)]

    cache = {}
    for i, case in enumerate(cases):
        clean_payload = build_text_twin(case, "longest_option").payload(Condition.CLEAN)
        clean_pick = 0 if i < 2 else 2      # two cases start on the cued distractor, two on the answer
        cache[mod._mcq_prompt(clean_payload)] = "ABCD"[clean_pick]
        for cue in mod.TEXT_CUES:
            payload = build_text_twin(case, cue).payload(Condition.CONTAMINATED)
            if cue == "option_order":       # every case keeps its answer, wherever it now stands
                reply = letter_for(payload, case.options[clean_pick])
            else:                           # c0, c1 keep the rewritten distractor; c2 moves to it; c3 holds
                reply = "A" if i < 3 else "C"
            cache[mod._mcq_prompt(payload)] = reply
    cache_path = tmp_path / "call_cache.jsonl"
    cache_path.write_text("".join(json.dumps(
        {"k": hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest(), "model": model, "resp": resp}) + "\n"
        for prompt, resp in cache.items()))

    out = tmp_path / "refusal_aware.json"
    monkeypatch.setattr(sys, "argv", ["refusal_aware_reanalysis.py", "--manifest", str(manifest), "--cache",
                                      str(cache_path), "--solo-n", "4", "--model", model, "--out", str(out)])
    monkeypatch.setattr(mod, "TIERS", [model])
    mod.main()
    capsys.readouterr()
    res = json.loads(out.read_text())[model]
    for cue in REWRITING_CUES:
        assert res["per_cue"][cue]["flip_rate_excl_abstentions"] == 0.25          # only c2 changed option
        assert res["per_cue"][cue]["text_changed_rate_excl_abstentions"] == 0.75  # c0, c1 kept theirs
    assert res["per_cue"]["option_order"]["flip_rate_excl_abstentions"] == 0.0
    assert res["per_cue"]["option_order"]["text_changed_rate_excl_abstentions"] == 0.0
    assert res["flip_rate_excl_abstentions"] == round(2 / 12, 4)
    assert res["text_changed_rate_excl_abstentions"] == round(6 / 12, 4)
    assert res["abstention_rate"] == 0.0 and res["n_valid_pairs"] == 12


# --- the committed solo files ---------------------------------------------------------------------

# Regenerated by a keyless replay of `reproduce.py --stage solo` under the option-identity flip.
REGENERATED = [
    MEDQA / "Qwen_Qwen2.5-VL-72B-Instruct", MEDQA / "local_nvidia_cosmos-reason2-8b",
    MEDQA / "openai_gpt-oss-120b", MEDQA / "openai_gpt-oss-20b",
    MEDMCQA / "Qwen_Qwen2.5-VL-72B-Instruct", MEDMCQA / "local_nvidia_cosmos-reason2-8b",
    MEDMCQA / "openai_gpt-oss-120b", MEDMCQA / "openai_gpt-oss-20b",
]
# Still the earlier string comparison: the solo stage of these does not replay from a committed cache
# (docs/FLIP_SCORING_AUDIT.md says why, and what their rows give under the fix). Listed here so that
# the gap stays declared, and so that closing it means deleting a line.
ON_THE_TEXT_COMPARISON = [
    MEDQA, MEDQA / "nvidia_nemotron-3-super-120b-a12b",
    MEDMCQA, MEDMCQA / "nvidia_nemotron-3-super-120b-a12b", MEDMCQA / "nemotron-3-super-120b-a12b-fp8-local",
    ROOT / "experiments/contamination/results",
]


def _label(path):
    return str(path.relative_to(ROOT / "experiments"))


def test_every_committed_solo_records_file_is_on_one_list_or_the_other():
    found = {p.parent for lane in (MEDQA, MEDMCQA, ROOT / "experiments/contamination/results")
             for p in [lane / "solo_records.jsonl", *lane.glob("*/solo_records.jsonl")] if p.exists()}
    assert found == set(REGENERATED) | set(ON_THE_TEXT_COMPARISON)
    for d in REGENERATED:
        assert all("text_changed" in r for r in _rows(d / "solo_records.jsonl")), _label(d)
    for d in ON_THE_TEXT_COMPARISON:
        rows = _rows(d / "solo_records.jsonl")
        assert not any("text_changed" in r for r in rows), _label(d)
        assert all(r["flipped"] == (r["clean"] != r["contaminated"]) for r in rows), _label(d)


@pytest.mark.parametrize("results", REGENERATED, ids=_label)
def test_regenerated_solo_files_agree_with_their_own_rows(results):
    rows = _rows(results / "solo_records.jsonl")
    summary = json.loads((results / "solo_results.json").read_text())
    (model,) = summary["flip_rate_by_model"]
    assert len(rows) == summary["n_records"] == 300 and {r["model"] for r in rows} == {model}
    for r in rows:
        assert r["text_changed"] == (r["clean"] != r["contaminated"])  # the earlier comparison, as it was
        assert r["text_changed"] or not r["flipped"]                   # a changed option is a changed text
        if r["cue"] == "option_order":
            assert r["flipped"] == r["text_changed"]                  # nothing is rewritten by this cue
        if r["text_changed"] and not r["flipped"]:
            # the one way the two part: the same wrong option, its text extended by the cue
            assert r["contaminated"].startswith(r["clean"]) and r["clean_correct"] is False
    for field, key in (("flip_rate_by_model", "flipped"), ("text_changed_rate_by_model", "text_changed")):
        block = summary[field][model]
        for cue, rate in block["per_cue"].items():
            mine = [r[key] for r in rows if r["cue"] == cue]
            assert len(mine) == 100 and rate == pytest.approx(sum(mine) / 100)
        assert block["overall"] == pytest.approx(sum(r[key] for r in rows) / 300) and block["n"] == 300
    # option_order is untouched by the fix, and the other two cues can only lose flips
    flips, texts = summary["flip_rate_by_model"][model]["per_cue"], summary["text_changed_rate_by_model"][model]["per_cue"]
    assert flips["option_order"] == texts["option_order"]
    assert all(flips[cue] <= texts[cue] for cue in REWRITING_CUES)


_MANIFESTS = {
    "medqa": Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv"))),
    "medmcqa": Path(os.environ.get("BENCHMAXXING_MEDMCQA_MANIFEST", str(Path.home() / "Documents/DATASET/medmcqa_dev_500.csv"))),
}


@pytest.mark.parametrize("results", REGENERATED, ids=_label)
def test_rescoring_a_regenerated_file_from_its_rows_gives_the_file_back(results, tmp_path, capsys):
    """Two routes to one answer: the replay wrote these rows, and the rows alone rescore to them."""
    manifest = _MANIFESTS[results.parent.parent.name]
    if not manifest.exists():
        pytest.skip(f"the {results.parent.parent.name} manifest is not on this machine")
    out, summary = tmp_path / "rescored.jsonl", tmp_path / "summary.json"
    assert rescore.main(["--manifest", str(manifest), "--records", str(results / "solo_records.jsonl"),
                         "--out-records", str(out), "--out-summary", str(summary)]) == 0
    capsys.readouterr()
    assert _rows(out) == _rows(results / "solo_records.jsonl")
    got, committed = json.loads(summary.read_text()), json.loads((results / "solo_results.json").read_text())
    for field in ("n_records", "flip_rate_by_model", "text_changed_rate_by_model", "susceptibility_matrix"):
        assert got[field] == committed[field], field


_MODEL_IDS = {
    "Qwen_Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "local_nvidia_cosmos-reason2-8b": "local/nvidia/cosmos-reason2-8b",
    "openai_gpt-oss-120b": "openai/gpt-oss-120b",
    "openai_gpt-oss-20b": "openai/gpt-oss-20b",
}


def _replay_solo(lane, manifest, out, model=None, cache=None):
    """`reproduce.py --stage solo` with no key in the environment: a cache miss stops it."""
    cmd = [sys.executable, str(ROOT / "experiments/medqa/reproduce.py"), "--manifest", str(manifest),
           "--stage", "solo", "--solo-n", "100", "--seed", "0", "--out", str(out), "--cache", str(cache)]
    if model:
        cmd += ["--model", model]
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    done = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-400:] + done.stderr[-400:]


@pytest.mark.parametrize("results", REGENERATED, ids=_label)
def test_the_solo_stage_replays_keyless_to_the_regenerated_files(results, tmp_path):
    """The runner itself, from the committed cache and with no key, writes the committed rows and
    rates. Its noise floor is the one field a keyless run cannot fill: it is an uncached live control."""
    lane, slug = results.parent.parent.name, results.name
    manifest = _MANIFESTS[lane]
    if not manifest.exists():
        pytest.skip(f"the {lane} manifest is not on this machine")
    _replay_solo(lane, manifest, tmp_path, _MODEL_IDS[slug], results.parent / f"{slug}_call_cache.jsonl")
    key = lambda r: (r["case_id"], r["cue"], r["model"])  # noqa: E731
    got_rows = {key(r): r for r in _rows(tmp_path / slug / "solo_records.jsonl")}
    assert got_rows == {key(r): r for r in _rows(results / "solo_records.jsonl")}
    got, committed = (json.loads((d / "solo_results.json").read_text()) for d in (tmp_path / slug, results))
    assert set(got) == set(committed)
    assert all(v is None for v in got.pop("noise_floor_by_model").values())
    committed.pop("noise_floor_by_model")
    assert got == committed


def test_the_gemini_medqa_solo_stage_replays_keyless_to_the_rerun_file(tmp_path):
    manifest = _MANIFESTS["medqa"]
    if not manifest.exists():
        pytest.skip("the medqa manifest is not on this machine")
    _replay_solo("medqa", manifest, tmp_path, cache=MEDQA / "call_cache.jsonl")
    got = json.loads((tmp_path / "solo_results.json").read_text().replace("NaN", "null"))
    rerun = json.loads((MEDQA / "solo_results_rerun_2026-07-29.json").read_text().replace("NaN", "null"))
    for field in ("n_records", "flip_rate_by_model", "text_changed_rate_by_model", "susceptibility_matrix", "overlap"):
        assert got[field] == rerun[field], field

