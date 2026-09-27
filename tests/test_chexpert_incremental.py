from __future__ import annotations

import json
from pathlib import Path

from experiments.imaging_chexpert.incremental import (
    DEFAULT_TARGETS,
    PHASE_FILES,
    build_runner_command,
    child_environment,
    load_api_key,
    seed_checkpoints,
    validate_block,
)


MODEL = "meta/llama-3.2-90b-vision-instruct"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _valid_block(tmp_path: Path, target: int = 20) -> tuple[Path, Path]:
    out_root = tmp_path / "phase2_n20"
    model_dir = out_root / MODEL.replace("/", "_")
    for phase in ("phase1_baseline", "phase2_blind", "phase3_test_aware"):
        _write_jsonl(
            model_dir / f"imaging_blind_metric_{phase}.jsonl",
            [{"case_id": f"case-{i}"} for i in range(target)],
        )
    (model_dir / "imaging_blind_metric_summary.json").write_text(
        json.dumps({"n": target, "model": MODEL})
    )
    cache = tmp_path / "img_cache_llama_n2.jsonl"
    _write_jsonl(cache, [{"k": f"key-{i}", "resp": "yes"} for i in range(3 * target)])
    return out_root, cache


def test_incremental_targets_are_cumulative():
    assert DEFAULT_TARGETS == (20, 40, 60, 80, 100)


def test_load_api_key_prefers_n2_with_spaces_around_equals(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("NVIDIA_API_KEY=old\nN_2 = second\n")

    assert load_api_key(env_path) == ("N_2", "second")


def test_runner_command_pins_safe_rate_and_retry_settings(tmp_path):
    command = build_runner_command(
        python="python",
        runner=Path("experiments/imaging_chexpert/imaging_blind_metric.py"),
        manifest=Path("manifest.csv"),
        image_root=Path("images"),
        model=MODEL,
        cache=Path("cache.jsonl"),
        out=tmp_path / "phase2_n20",
        target=20,
    )

    assert command[command.index("--n") + 1] == "20"
    assert command[command.index("--rpm") + 1] == "20"
    assert command[command.index("--request-retries") + 1] == "2"
    assert command[command.index("--timeout") + 1] == "60"
    assert MODEL in command


def test_validate_block_accepts_three_complete_phases_and_unique_cache(tmp_path):
    out_root, cache = _valid_block(tmp_path)

    report = validate_block(out_root, MODEL, target=20, cache_path=cache)

    assert report["valid"] is True
    assert report["phase_counts"] == {"baseline": 20, "blind": 20, "test_aware": 20}
    assert report["summary_n"] == 20
    assert report["duplicate_cache_keys"] == []


def test_validate_block_rejects_partial_phase_and_duplicate_cache(tmp_path):
    out_root, cache = _valid_block(tmp_path)
    model_dir = out_root / MODEL.replace("/", "_")
    (model_dir / "imaging_blind_metric_phase3_test_aware.jsonl").unlink()
    with cache.open("a") as f:
        f.write(json.dumps({"k": "key-0", "resp": "yes"}) + "\n")

    report = validate_block(out_root, MODEL, target=20, cache_path=cache)

    assert report["valid"] is False
    assert "test_aware" in report["missing_phases"]
    assert report["duplicate_cache_keys"] == ["key-0"]


def test_seed_checkpoints_copies_previous_block_only_when_missing(tmp_path):
    previous = tmp_path / "phase2_n20"
    current = tmp_path / "phase2_n40"
    model_dir = previous / MODEL.replace("/", "_")
    model_dir.mkdir(parents=True)
    for filename in PHASE_FILES.values():
        (model_dir / filename).write_text("previous\n")

    copied = seed_checkpoints(previous, current, MODEL)

    assert copied == len(PHASE_FILES)
    for filename in PHASE_FILES.values():
        assert (current / MODEL.replace("/", "_") / filename).read_text() == "previous\n"

    existing = current / MODEL.replace("/", "_") / PHASE_FILES["baseline"]
    existing.write_text("keep\n")
    for phase in ("blind", "test_aware"):
        (current / MODEL.replace("/", "_") / PHASE_FILES[phase]).unlink()
    assert seed_checkpoints(previous, current, MODEL) == 2
    assert existing.read_text() == "keep\n"


def test_child_environment_prefers_current_repository_on_pythonpath(tmp_path):
    env = child_environment(tmp_path, "secret", {"PYTHONPATH": "/other/repo"})

    assert env["NVIDIA_API_KEY"] == "secret"
    assert env["PYTHONPATH"].split(":", 1) == [str(tmp_path), "/other/repo"]
