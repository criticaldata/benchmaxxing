"""Tests for the benchmaxxing CLI and config layer.

Everything here runs offline with no API keys. The YAML path uses ``pytest.importorskip``
so it skips cleanly when pyyaml (the optional ``config`` extra) is not installed.
"""

from __future__ import annotations

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


def test_main_version_verbose_contains_expected_keys(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["version", "--verbose"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == benchmaxxing.__version__
    assert any(line.startswith("git SHA:") for line in lines)
    assert any(line.startswith("Python:") for line in lines)
    for extra in ("image", "changepoint", "stats", "models", "config"):
        assert any(line.startswith(f"extra {extra}:") for line in lines)


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


def test_load_config_from_yaml(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"seed": 3, "dataset": "d", "models": ["a", "b"]}))
    config = load_config(str(path))
    assert config.seed == 3
    assert config.dataset == "d"
    assert config.models == ["a", "b"]
"""Tests for the benchmaxxing CLI and config layer.

Everything here runs offline with no API keys. The YAML path uses ``pytest.importorskip``
so it skips cleanly when pyyaml (the optional ``config`` extra) is not installed.
"""

from __future__ import annotations

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


def test_load_config_from_yaml(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"seed": 3, "dataset": "d", "models": ["a", "b"]}))
    config = load_config(str(path))
    assert config.seed == 3
    assert config.dataset == "d"
    assert config.models == ["a", "b"]
