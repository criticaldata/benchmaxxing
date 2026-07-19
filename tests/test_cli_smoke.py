"""The `benchmaxxing smoke` subcommand: documented in CONTRIBUTING, must exist and pass."""

from __future__ import annotations

from benchmaxxing.cli import _run_smoke, build_parser, main


def test_smoke_subcommand_is_registered():
    # regression: CONTRIBUTING.md advertises `benchmaxxing smoke`; the parser must accept it
    # (previously it errored with "invalid choice: 'smoke'").
    args = build_parser().parse_args(["smoke"])
    assert args.command == "smoke"


def test_smoke_command_runs_and_passes(capsys):
    assert main(["smoke"]) == 0
    out = capsys.readouterr().out
    assert "composes end to end" in out
    assert "cascade onset" in out


def test_run_smoke_invariants():
    r = _run_smoke()
    # the cue must bite the length-follower ...
    assert r["solo_flip_rate"] > 0
    # ... and the planted shortcut must spread MORE on the shared board than in isolation.
    assert r["shared_shortcut_agents"] > r["isolated_shortcut_agents"]
    assert r["n_agents"] >= 2
