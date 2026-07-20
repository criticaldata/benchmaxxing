"""Tests for the resumable on-disk run store (issue 89).

Covers lossless round-trip, has()/load()/keys() bookkeeping, and the resume path: after an
interrupted run only the missing keys come back from iter_pending. Everything runs offline with
synthetic transcripts, no backends or data.
"""

from __future__ import annotations

from benchmaxxing.runstore import RunStore
from benchmaxxing.schema import Condition, Transcript, Turn


def _transcript(
    run_id: str, case_id: str, condition: Condition = Condition.CONTAMINATED
) -> Transcript:
    """Build a small, fully-populated transcript so the round-trip is meaningfully exercised."""
    return Transcript(
        run_id=run_id,
        case_id=case_id,
        condition=condition,
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


def test_init_creates_directory(tmp_path):
    target = tmp_path / "nested" / "store"
    assert not target.exists()
    store = RunStore(target)
    assert store.dir.is_dir()


def test_save_then_has_and_load_round_trip(tmp_path):
    store = RunStore(tmp_path / "runs")
    key = "case-abc::contaminated"
    original = _transcript("run-001", "case-abc")

    assert store.has(key) is False
    store.save_transcript(key, original)

    assert store.has(key) is True
    loaded = store.load(key)
    assert loaded == original
    # condition comes back as the enum, not a bare string
    assert isinstance(loaded.condition, Condition)
    assert loaded.condition is Condition.CONTAMINATED


def test_keys_lists_completed_sorted(tmp_path):
    store = RunStore(tmp_path / "runs")
    for i in range(3):
        store.save_transcript(f"k{i}", _transcript(f"run-{i}", f"case-{i}"))
    assert store.keys() == ["k0", "k1", "k2"]


def test_load_missing_key_raises(tmp_path):
    store = RunStore(tmp_path / "runs")
    try:
        store.load("nope")
        raise AssertionError("expected KeyError for a key that was never saved")
    except KeyError:
        pass


def test_save_overwrites_existing_key(tmp_path):
    store = RunStore(tmp_path / "runs")
    key = "case-x"
    store.save_transcript(key, _transcript("run-1", "case-x", Condition.CLEAN))
    store.save_transcript(key, _transcript("run-2", "case-x", Condition.CONTAMINATED))

    loaded = store.load(key)
    assert loaded.run_id == "run-2"
    assert loaded.condition is Condition.CONTAMINATED
    # still exactly one entry for that key
    assert store.keys() == ["case-x"]


def test_iter_pending_returns_exactly_missing_keys(tmp_path):
    """Simulate an interrupted run: half the plan is saved, the rest must come back pending."""
    store = RunStore(tmp_path / "runs")
    all_keys = [f"case-{i}::contaminated" for i in range(6)]

    # The run got interrupted after finishing indices 0, 2 and 3.
    done = [all_keys[0], all_keys[2], all_keys[3]]
    for key in done:
        store.save_transcript(key, _transcript(key, key))

    pending = list(store.iter_pending(all_keys))
    expected = [all_keys[1], all_keys[4], all_keys[5]]
    assert pending == expected
    # completed set matches what we saved
    assert store.keys() == sorted(done)


def test_iter_pending_empty_when_all_done(tmp_path):
    store = RunStore(tmp_path / "runs")
    all_keys = ["a", "b", "c"]
    for key in all_keys:
        store.save_transcript(key, _transcript(key, key))
    assert list(store.iter_pending(all_keys)) == []


def test_iter_pending_dedups_preserving_order(tmp_path):
    store = RunStore(tmp_path / "runs")
    all_keys = ["a", "a", "b", "c", "b"]
    assert list(store.iter_pending(all_keys)) == ["a", "b", "c"]


def test_iter_pending_is_lazy(tmp_path):
    store = RunStore(tmp_path / "runs")
    result = store.iter_pending(["a", "b"])
    assert hasattr(result, "__next__")
    assert next(result) == "a"


def test_keys_with_unsafe_characters_round_trip(tmp_path):
    """Keys with path separators, colons, spaces and unicode survive filename encoding."""
    store = RunStore(tmp_path / "runs")
    tricky = "case/42::contaminated round 3 éè"
    store.save_transcript(tricky, _transcript("run-tricky", "case/42"))

    assert store.has(tricky) is True
    assert store.keys() == [tricky]
    assert store.load(tricky).run_id == "run-tricky"
    # the key never escapes its directory despite the slash
    assert list(store.dir.iterdir())[0].parent == store.dir


def test_reopen_store_sees_saved_keys(tmp_path):
    """A fresh RunStore on the same directory resumes from what is already on disk."""
    root = tmp_path / "runs"
    first = RunStore(root)
    first.save_transcript("done-1", _transcript("r1", "c1"))
    first.save_transcript("done-2", _transcript("r2", "c2"))

    reopened = RunStore(root)
    assert reopened.has("done-1") is True
    assert reopened.keys() == ["done-1", "done-2"]
    assert list(reopened.iter_pending(["done-1", "done-2", "todo-3"])) == ["todo-3"]


def test_save_leaves_no_temp_files(tmp_path):
    store = RunStore(tmp_path / "runs")
    store.save_transcript("k", _transcript("r", "c"))
    names = [p.name for p in store.dir.iterdir()]
    assert all(n.endswith(".jsonl") for n in names)
    assert not any(".tmp" in n for n in names)
