"""Command-line entry point for benchmaxxing.

Dependency-light: standard-library argparse only. Optional pieces (the datasets registry,
config resolution) are imported lazily inside each command so ``benchmaxxing --help`` and
``benchmaxxing version`` keep working even while the rest of the package is mid-build.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with the version, datasets, and config-show commands."""
    parser = argparse.ArgumentParser(
        prog="benchmaxxing",
        description=(
            "Referee agents for catching benchmark gaming in clinical multi-agent systems."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    sub.add_parser("version", help="print the installed benchmaxxing version")
    sub.add_parser("datasets", help="list the available dataset adapters")
    sub.add_parser("smoke", help="run the offline end-to-end pipeline smoke on synthetic data")

    p_config = sub.add_parser("config-show", help="print the resolved default config")
    p_config.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="PATH",
        help="optional YAML config file to resolve and print (defaults are shown otherwise)",
    )
    return parser


def _cmd_version(_args: argparse.Namespace) -> int:
    import benchmaxxing

    print(benchmaxxing.__version__)
    return 0


def _dataset_names() -> list[str]:
    """Best-effort list of registered dataset names, empty if none or on any import error."""
    try:
        from benchmaxxing import datasets
    except Exception:  # noqa: BLE001 - datasets may be mid-build; degrade gracefully
        return []
    registry = getattr(datasets, "registry", None)
    if not registry:
        return []
    items = registry.keys() if isinstance(registry, Mapping) else registry
    names = []
    try:
        for item in items:
            label = getattr(item, "name", None)
            names.append(str(label) if label is not None else str(item))
    except TypeError:
        return []
    return sorted(names)


def _cmd_datasets(_args: argparse.Namespace) -> int:
    names = _dataset_names()
    if names:
        for name in names:
            print(name)
    else:
        print("No dataset adapters are registered yet.")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    from benchmaxxing.config import load_config

    config = load_config(getattr(args, "config", None))
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return 0


# --- offline end-to-end smoke -------------------------------------------------------------
# Documented in CONTRIBUTING.md as `benchmaxxing smoke`. It composes the real modules on
# synthetic data with a mock backend, so it needs no dataset and no API key: the point is to
# confirm the plumbing (cue injection -> solo flip -> shared/isolated cascade -> onset)
# composes end to end and to catch an interface drift between two modules.

_CORRECT = "correct-dx"
_SHORTCUT = "shortcut-dx"


def _pick_longest(payload):
    """A length-following mock: pick the longest option (identity answer_fn downstream)."""
    options = list(payload["options"])
    return max(options, key=len) if options else ""


def _run_smoke() -> dict:
    """Run the whole offline pipeline on synthetic data and return a small report dict."""
    from benchmaxxing.analysis import flip_rate, solo_evaluate
    from benchmaxxing.blackboard import AgentResponse, run_committee
    from benchmaxxing.cues.text import build_text_twin
    from benchmaxxing.onset import cascade_onset
    from benchmaxxing.roster import build_committee, default_roster
    from benchmaxxing.schema import Case, Condition, Modality

    # 1. cue injection + solo baseline: the longest-option cue flips a length-follower.
    cases = [
        Case(case_id=f"t{i}", patient_id=f"p{i}", modality=Modality.TEXT,
             question="Most likely diagnosis?",
             options=("Pneumothorax", "A benign incidental finding described at great length",
                      "Rib fracture"),
             answer_index=0)
        for i in range(4)
    ]
    twins = [build_text_twin(c, "longest_option") for c in cases]
    records = solo_evaluate(twins, backend=_pick_longest, answer_fn=lambda x: x, model="mock")

    # 2. shared vs isolated committee. Same-lineage (Gemini) agents are shortcut-prone; the
    #    open-weights (cross-lineage) agent is robust and resists the planted shortcut.
    class _Robust:
        def respond(self, view):
            return AgentResponse(content="on the merits", answer=_CORRECT)

    class _ShortcutProne:
        def respond(self, view):
            on_board = any(t.answer == _SHORTCUT for t in view.visible_turns)
            return AgentResponse(content="deferring" if on_board else "on the merits",
                                 answer=_SHORTCUT if on_board else _CORRECT)

    committee = build_committee(default_roster())
    seed_agent = next(m.name for m in committee.members if not m.is_open_weights)
    seed_index = next(i for i, m in enumerate(committee.members) if m.name == seed_agent)

    def backend_for(spec):
        return _Robust() if spec.is_open_weights else _ShortcutProne()

    case = Case(case_id="c0", patient_id="p0", modality=Modality.IMAGE, label="pneumothorax")
    shared = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=True,
                           seed_turn=(seed_index, _SHORTCUT, seed_agent), seed=0, rounds=3)
    isolated = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=False,
                             seed_turn=(seed_index, _SHORTCUT, seed_agent), seed=0, rounds=3)

    shared_shortcut = sum(1 for a in shared.committed.values() if a == _SHORTCUT)
    isolated_shortcut = sum(1 for a in isolated.committed.values() if a == _SHORTCUT)
    series = [1.0 if t.answer == _SHORTCUT else 0.0 for t in shared.turns]

    return {
        "solo_flip_rate": flip_rate(records)["overall"],
        "n_twins": len(twins),
        "shared_shortcut_agents": shared_shortcut,
        "isolated_shortcut_agents": isolated_shortcut,
        "cascade_onset_turn": cascade_onset(series),
        "n_agents": len(committee.members),
    }


def _cmd_smoke(_args: argparse.Namespace) -> int:
    report = _run_smoke()
    lines = [
        "benchmaxxing end-to-end smoke (offline, synthetic data)",
        "=" * 52,
        f"1. solo flip rate (longest-option cue): {report['solo_flip_rate']} over "
        f"{report['n_twins']} twins",
        f"2. cascade: shared run has {report['shared_shortcut_agents']}/{report['n_agents']} "
        f"agents on the shortcut vs {report['isolated_shortcut_agents']}/{report['n_agents']} "
        "isolated (shared higher: the shortcut spreads only when the board is shared)",
        f"3. cascade onset turn: {report['cascade_onset_turn']}",
    ]
    # the plumbing is healthy only if the cue bit AND the shortcut spread more when shared
    healthy = (
        report["solo_flip_rate"] > 0
        and report["shared_shortcut_agents"] > report["isolated_shortcut_agents"]
    )
    lines.append(
        "all stages ran; the plumbing composes end to end." if healthy
        else "SMOKE FAILED: a stage did not behave as expected (see numbers above)."
    )
    print("\n".join(lines))
    return 0 if healthy else 1


_HANDLERS = {
    "version": _cmd_version,
    "datasets": _cmd_datasets,
    "config-show": _cmd_config_show,
    "smoke": _cmd_smoke,
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (or ``sys.argv``) and dispatch to a subcommand. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
