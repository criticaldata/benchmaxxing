"""Tests for `benchmaxxing cues preview` (issue #126).

The text lane needs no optional extra; the image lane needs PIL (the `image` extra), so its
tests skip cleanly when PIL is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmaxxing import cli
from benchmaxxing.cues import preview


# --- text lane: pure library calls -------------------------------------------------------

def test_render_text_preview_writes_markdown_for_bundled_sample(tmp_path: Path) -> None:
    result = preview.render_text_preview("longest_option", tmp_path)
    assert result["path"] == tmp_path / "preview.md"
    assert result["path"].exists()
    text = result["path"].read_text(encoding="utf-8")
    assert "cues preview: longest_option" in text
    assert "## clean" in text and "## contaminated" in text
    # the correct option is preserved and marked in both conditions
    assert "*" in text


def test_render_text_preview_from_manifest_selects_case(tmp_path: Path) -> None:
    manifest = tmp_path / "mcq.csv"
    manifest.write_text(
        "case_id,question,options,answer_index\n"
        "q1,Best next step?,Aspirin|Warfarin|Heparin,1\n"
        "q2,Likely diagnosis?,MI|PE,1\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = preview.render_text_preview(
        "option_order", out, manifest_path=manifest, case_id="q2"
    )
    assert result["case"].case_id == "q2"
    assert "Likely diagnosis?" in result["markdown"]


def test_render_text_preview_unknown_case_id_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "mcq.csv"
    manifest.write_text(
        "case_id,question,options,answer_index\nq1,Best next step?,Aspirin|Warfarin,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="case_id"):
        preview.render_text_preview("longest_option", tmp_path, manifest_path=manifest,
                                     case_id="nope")


def test_render_text_preview_unknown_cue_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        preview.render_text_preview("not-a-real-cue", tmp_path)


def test_render_text_preview_side_by_side_has_both_columns(tmp_path: Path) -> None:
    result = preview.render_text_preview("longest_option", tmp_path)
    side_by_side = result["side_by_side"]
    header = side_by_side.splitlines()[0]
    assert "clean" in header and "contaminated" in header
    # the padded distractor text only appears on the contaminated side of the same row; check
    # the unwrapped payload lines directly since side_by_side may hard-wrap long options
    contaminated_lines = preview._payload_lines(result["twin"].contaminated)
    assert any("[additional clinical detail]" in line for line in contaminated_lines)


def test_format_side_by_side_pads_uneven_columns() -> None:
    clean = {"question": "Q?", "options": ("a",), "answer_index": 0}
    contaminated = {
        "question": "Q?",
        "options": ("a", "b padded"),
        "answer_index": 0,
        "report": "extra line only on this side",
    }
    text = preview.format_side_by_side(clean, contaminated)
    lines = text.splitlines()
    # every row (after the header + separator) must be present on both sides, even where one
    # side has fewer lines than the other
    assert len(lines) == 2 + max(
        len(preview._payload_lines(clean)), len(preview._payload_lines(contaminated))
    )
    assert "report: extra line only on this side" in text


# --- image lane: needs PIL --------------------------------------------------------------

def test_render_image_preview_writes_png_pair_for_bundled_sample(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    result = preview.render_image_preview("watermark", tmp_path)
    assert result["clean"] == tmp_path / "clean.png"
    assert result["contaminated"] == tmp_path / "contaminated.png"
    assert result["clean"].exists()
    assert result["contaminated"].exists()
    assert result["clean"].read_bytes() != result["contaminated"].read_bytes()


def test_render_image_preview_from_manifest_loads_real_image(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image
    import numpy as np

    img_path = tmp_path / "a.png"
    Image.fromarray(np.full((40, 40), 90, dtype=np.uint8)).save(img_path)
    manifest = tmp_path / "imaging.csv"
    manifest.write_text(
        "case_id,patient_id,image_ref,report\nc1,p1,a.png,No acute findings.\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = preview.render_image_preview("cable", out, manifest_path=manifest)
    assert "c1" in result["source"]
    clean = np.asarray(Image.open(result["clean"]))
    assert clean.shape == (40, 40)


def test_render_image_preview_missing_image_on_disk_raises(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    manifest = tmp_path / "imaging.csv"
    manifest.write_text(
        "case_id,patient_id,image_ref,report\nc1,p1,missing.png,No findings.\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        preview.render_image_preview("cable", tmp_path, manifest_path=manifest)


# --- CLI wiring --------------------------------------------------------------------------

def test_cli_parses_cues_preview() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["cues", "preview", "--lane", "text", "--cue", "longest_option", "--out", "/tmp/x"]
    )
    assert args.command == "cues"
    assert args.cues_command == "preview"
    assert args.lane == "text"


def test_cli_rejects_invalid_lane() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["cues", "preview", "--lane", "audio", "--cue", "x", "--out", "/tmp/x"]
        )


def test_cli_cues_preview_text_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(
        ["cues", "preview", "--lane", "text", "--cue", "longest_option", "--out", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "preview.md").exists()
    out = capsys.readouterr().out
    assert "clean" in out and "contaminated" in out


def test_cli_cues_preview_image_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("PIL")
    rc = cli.main(
        ["cues", "preview", "--lane", "image", "--cue", "watermark", "--out", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "clean.png").exists()
    assert (tmp_path / "contaminated.png").exists()
    assert "wrote" in capsys.readouterr().out


def test_cli_cues_preview_unknown_cue_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(
        ["cues", "preview", "--lane", "text", "--cue", "not-a-cue", "--out", str(tmp_path)]
    )
    assert rc != 0
    assert "error" in capsys.readouterr().err.lower()


def test_cli_cues_bare_requires_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    # `cues` has one subcommand and no sensible default (unlike bare `datasets`), so it must
    # fail like any other incomplete invocation instead of silently exiting 0.
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["cues"])
    assert exc_info.value.code != 0
    assert "command" in capsys.readouterr().err.lower()
