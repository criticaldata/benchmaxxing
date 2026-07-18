"""Round-trip tests for transcript JSONL and manifest JSON persistence (issues 23, 24, 25)."""

import json

from benchmaxxing.manifest import library_versions, read_manifest, write_manifest
from benchmaxxing.schema import Condition, RunManifest, Transcript, Turn
from benchmaxxing.transcript import dump_transcript, load_transcript, replay


def _sample_transcript() -> Transcript:
    return Transcript(
        run_id="run-001",
        case_id="case-abc",
        condition=Condition.CONTAMINATED,
        turns=[
            Turn(turn_index=0, agent_id="alpha", content="opening", answer="A", confidence=0.5),
            Turn(
                turn_index=1,
                agent_id="beta",
                content="rebuttal",
                answer="B",
                confidence=None,
                seeded=True,
            ),
            Turn(turn_index=2, agent_id="alpha", content="final", answer="B", confidence=0.9),
        ],
        committed={"alpha": "B", "beta": "B"},
        meta={"note": "twin", "rounds": 3},
    )


def test_transcript_round_trip_lossless(tmp_path):
    original = _sample_transcript()
    path = tmp_path / "transcript.jsonl"
    dump_transcript(original, path)
    loaded = load_transcript(path)
    assert loaded == original
    # condition is rebuilt as the enum, not a bare string
    assert isinstance(loaded.condition, Condition)
    assert loaded.condition is Condition.CONTAMINATED


def test_replay_matches_load(tmp_path):
    original = _sample_transcript()
    path = tmp_path / "t.jsonl"
    dump_transcript(original, path)
    assert replay(path) == load_transcript(path) == original


def test_transcript_header_is_first_line(tmp_path):
    original = _sample_transcript()
    path = tmp_path / "t.jsonl"
    dump_transcript(original, path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # one header line plus one line per turn
    assert len(lines) == 1 + len(original.turns)
    first = json.loads(lines[0])
    assert first["kind"] == "header"
    assert first["run_id"] == original.run_id
    assert first["case_id"] == original.case_id
    assert first["condition"] == original.condition.value


def test_empty_turns_round_trip(tmp_path):
    original = Transcript(run_id="r", case_id="c", condition=Condition.CLEAN)
    path = tmp_path / "empty.jsonl"
    dump_transcript(original, path)
    loaded = load_transcript(path)
    assert loaded == original
    assert loaded.turns == []


def test_load_missing_header_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"kind": "turn", "turn_index": 0, "agent_id": "a",
                                "content": "x"}) + "\n", encoding="utf-8")
    try:
        load_transcript(path)
        raise AssertionError("expected ValueError for a file with no header")
    except ValueError:
        pass


def test_manifest_round_trip(tmp_path):
    manifest = RunManifest(
        run_id="run-001",
        model_ids=["gemini-2.5-flash", "qwen-max"],
        prompt_versions={"referee": "v3"},
        seed=7,
        cue_set_version="cues-1.2",
        dataset_revision="rev-9",
        library_versions={"numpy": "1.26.0"},
        config={"rounds": 3, "temperature": 0.0},
    )
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    loaded = read_manifest(path)
    assert loaded == manifest


def test_manifest_ignores_unknown_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"run_id": "x", "seed": 1, "bogus": 42}), encoding="utf-8")
    loaded = read_manifest(path)
    assert loaded.run_id == "x"
    assert loaded.seed == 1


def test_library_versions_includes_numpy():
    versions = library_versions()
    assert isinstance(versions, dict)
    assert "numpy" in versions
    assert isinstance(versions["numpy"], str)
    assert versions["numpy"]
