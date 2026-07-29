"""Tests for the benchmaxxing CLI and config layer.

Everything here runs offline with no API keys. The YAML path uses ``pytest.importorskip``
so it skips cleanly when pyyaml (the optional ``config`` extra) is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmaxxing
from benchmaxxing import cli
from benchmaxxing.config import Config, load_config


def test_build_parser_parses_version() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["version"])
    assert args.command == "version"


def test_main_version_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["version"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == benchmaxxing.__version__

def test_build_parser_parses_version_verbose() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["version", "--verbose"])
    assert args.command == "version"
    assert args.verbose is True


def test_main_version_verbose_includes_sha_and_extras(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["version", "--verbose"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.splitlines()
    assert lines[0] == benchmaxxing.__version__
    assert lines[1].startswith("git SHA: ")
    assert "extras:" in out
    for extra in ("stats", "changepoint", "image", "models", "config"):
        assert extra in out


def test_main_version_without_verbose_omits_sha_and_extras(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "git SHA" not in out
    assert "extras:" not in out


def test_git_sha_returns_string_even_outside_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no .git here
    sha = cli._git_sha()
    assert sha == "unknown"


def test_extras_status_reports_bool_per_known_extra() -> None:
    status = cli._extras_status()
    assert set(status) == {"stats", "changepoint", "image", "models", "config"}
    assert all(isinstance(v, bool) for v in status.values())


def test_main_no_command_returns_zero() -> None:
    assert cli.main([]) == 0


def test_config_defaults_load() -> None:
    config = load_config()
    assert isinstance(config, Config)
    assert config.seed == 0
    assert config.models  # non-empty default roster
    assert config.out_dir


def test_config_from_dict_roundtrip() -> None:
    data = {"models": "solo-model", "seed": "7", "dataset": "demo", "unknown": "ignored"}
    config = Config.from_dict(data)
    assert config.models == ["solo-model"]
    assert config.seed == 7
    assert config.dataset == "demo"
    assert Config.from_dict(config.to_dict()) == config


def test_config_show_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["config-show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "models" in out
    assert "seed" in out


def test_datasets_command_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["datasets"])
    assert rc == 0
    # Either lists registered names or prints a friendly message, but must produce output.
    assert capsys.readouterr().out.strip() != ""


def test_datasets_stats_summarizes_mcq_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "mcq.csv"
    manifest.write_text(
        "case_id,question,options,answer_index,label\n"
        "q1,Best next step?,Aspirin|Warfarin|Heparin,1,cardio\n"
        "q2,Likely diagnosis?,MI|PE,1,resp\n",
        encoding="utf-8",
    )
    rc = cli.main(["datasets", "stats", str(manifest)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rows: 2" in out
    assert "text: 2" in out
    assert "3 options: 1 cases" in out
    assert "2 options: 1 cases" in out
    assert "cardio: 1" in out and "resp: 1" in out


def test_datasets_stats_accepts_answer_index_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Regression: answer_index=0 is falsy in Python and must not be mistaken for "missing".
    manifest = tmp_path / "mcq.csv"
    manifest.write_text(
        "case_id,question,options,answer_index\nq1,Vessel?,Aorta|Vena cava,0\n", encoding="utf-8"
    )
    rc = cli.main(["datasets", "stats", str(manifest)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "error" not in out.lower()


def test_datasets_stats_resolves_images_with_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.png").write_bytes(b"\x00")
    manifest = tmp_path / "imaging.csv"
    manifest.write_text(
        "case_id,patient_id,image_ref,report,label\n"
        "c1,p1,a.png,No acute findings.,No Finding\n"
        "c2,p2,missing.png,Small left effusion.,Effusion\n",
        encoding="utf-8",
    )
    rc = cli.main(["datasets", "stats", str(manifest), "--image-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "image_ref resolves on disk: 1/2" in out


def test_datasets_stats_skips_image_check_without_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "imaging.csv"
    manifest.write_text(
        "case_id,patient_id,image_ref,report\nc1,p1,a.png,No findings.\n", encoding="utf-8"
    )
    rc = cli.main(["datasets", "stats", str(manifest)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipped" in out


def test_datasets_stats_missing_manifest_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["datasets", "stats", str(tmp_path / "nope.csv")])
    assert rc != 0
    assert "error" in capsys.readouterr().err.lower()


def test_datasets_stats_empty_manifest_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "empty.csv"
    manifest.write_text("case_id,question,options,answer_index\n", encoding="utf-8")
    rc = cli.main(["datasets", "stats", str(manifest)])
    assert rc != 0
    assert "error" in capsys.readouterr().err.lower()


def test_datasets_list_subcommand_matches_bare_datasets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc_bare = cli.main(["datasets"])
    out_bare = capsys.readouterr().out
    rc_list = cli.main(["datasets", "list"])
    out_list = capsys.readouterr().out
    assert rc_bare == rc_list == 0
    assert out_bare == out_list


def test_load_config_from_json(tmp_path: Path) -> None:
    # JSON needs no optional extra, so a run is configurable on a bare core install.
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"seed": 5, "dataset": "medqa", "models": ["a", "b"]}))
    config = load_config(str(path))
    assert config.seed == 5
    assert config.models == ["a", "b"]


def test_load_config_from_yaml(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"seed": 3, "dataset": "d", "models": ["a", "b"]}))
    config = load_config(str(path))
    assert config.seed == 3
    assert config.dataset == "d"
    assert config.models == ["a", "b"]
