"""Command-line entry point for benchmaxxing.

Dependency-light: standard-library argparse only. Optional pieces (the datasets registry,
config resolution) are imported lazily inside each command so ``benchmaxxing --help`` and
``benchmaxxing version`` keep working even while the rest of the package is mid-build.
"""

from __future__ import annotations

import argparse
import json
import sys


# Listed here so --help keeps working even when the runner's imports (numpy) are unavailable.
_STAGES = ("pilot", "solo", "overlap", "cascade")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with the version, datasets, and config-show commands."""
    parser = argparse.ArgumentParser(
        prog="benchmaxxing",
        description=(
            "Referee agents for catching benchmark gaming in clinical multi-agent systems."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_version = sub.add_parser("version", help="print the installed benchmaxxing version")
    p_version.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also print the git commit SHA and which optional extras are installed",
    )

    p_datasets = sub.add_parser("datasets", help="dataset readiness commands")
    datasets_sub = p_datasets.add_subparsers(dest="datasets_command", metavar="command")
    datasets_sub.add_parser("list", help="list dataset readiness status (the default)")
    p_stats = datasets_sub.add_parser(
        "stats", help="summarize and sanity-check a manifest"
    )
    p_stats.add_argument("manifest", metavar="PATH", help="path to a CSV/JSONL manifest")
    p_stats.add_argument(
        "--image-root",
        default=None,
        metavar="PATH",
        help="root directory to resolve image_ref paths against (imaging manifests only)",
    )
    p_stage = datasets_sub.add_parser(
        "stage",
        help="build, validate and record a manifest from a staged raw release",
    )
    p_stage.add_argument("name", metavar="NAME", help="a registered dataset adapter name")
    p_stage.add_argument(
        "--raw-root",
        default=None,
        metavar="PATH",
        help="the raw release (defaults to $BENCHMAXXING_DATASET_ROOT/<name>, else data/<name>)",
    )
    p_stage.add_argument(
        "--out", default=None, metavar="PATH", help="where to write the manifest"
    )
    p_stage.add_argument("--limit", type=int, default=None, help="deterministically subsample N cases")
    p_stage.add_argument(
        "--check-images",
        action="store_true",
        help="also confirm every image_ref resolves on disk (imaging datasets)",
    )

    sub.add_parser("smoke", help="run the offline end-to-end pipeline smoke on synthetic data")

    p_config = sub.add_parser("config-show", help="print the resolved default config")
    p_config.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="PATH",
        help="optional YAML config file to resolve and print (defaults are shown otherwise)",
    )

    p_run = sub.add_parser("run", help="run one experiment stage from a config and a manifest")
    p_run.add_argument(
        "--stage", required=True, choices=list(_STAGES), help="which stage runner to drive"
    )
    p_run.add_argument("--manifest", required=True, metavar="PATH", help="case manifest to run on")
    p_run.add_argument(
        "--dataset", default=None, help="dataset adapter name recorded with the run"
    )
    p_run.add_argument(
        "-c", "--config", default=None, metavar="PATH", help="YAML config for models/cues/seed"
    )
    p_run.add_argument("--out", default=None, metavar="DIR", help="run directory to write")
    p_run.add_argument(
        "--backend",
        default="gemini",
        choices=("gemini", "mock"),
        help="model backend: gemini (needs a key) or mock (offline stand-in)",
    )
    p_run.add_argument("--limit", type=int, default=None, help="cap the number of cases")
    p_run.add_argument(
        "--hard-cases",
        default=None,
        metavar="PATH",
        help=(
            "restrict the run to the cases a previous solo run found hardest "
            "(a solo_records.jsonl); a cascade needs an uncertain committee"
        ),
    )
    p_run.add_argument(
        "--hard-k", type=int, default=None, help="how many hard cases to keep (default: all)"
    )
    p_run.add_argument(
        "--hard-max-accuracy",
        type=float,
        default=None,
        help="keep only cases at or below this clean accuracy (e.g. 0.25 for four-way chance)",
    )
    p_run.add_argument(
        "--image-root",
        default=None,
        metavar="PATH",
        help="root directory image_ref paths resolve against (required for the imaging lane)",
    )
    p_run.add_argument(
        "--noise-floor",
        action="store_true",
        help=(
            "also measure the decoding-noise floor (re-ask each clean payload once) and report "
            "the flip rate against it; costs one extra call per twin"
        ),
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan (models, twins, estimated calls) and spend no API calls",
    )

    p_report = sub.add_parser(
        "report", help="render a finished run bundle into a standalone HTML report"
    )
    p_report.add_argument("bundle", metavar="DIR", help="a run bundle directory")
    p_report.add_argument(
        "-o", "--out", default=None, metavar="PATH", help="where to write the HTML"
    )
    p_report.add_argument(
        "--replay",
        action="store_true",
        help="recompute the cascade numbers from the saved transcripts and compare them",
    )
    return parser

def _git_sha() -> str:
    """Short git commit SHA of the working tree, or 'unknown' if unavailable."""
    from benchmaxxing.manifest import git_sha

    return git_sha()


def _extras_status() -> dict[str, bool]:
    """Best-effort check of which optional-dependency extras are importable."""
    import importlib

    # module name to try importing, per extra in pyproject.toml
    probes = {
        "stats": "statsmodels",
        "changepoint": "ruptures",
        "image": "PIL",
        "models": "litellm",
        "config": "yaml",
    }
    status = {}
    for extra, module_name in probes.items():
        try:
            importlib.import_module(module_name)
            status[extra] = True
        except Exception:  # noqa: BLE001 - absence is expected, not an error
            status[extra] = False
    return status

def _cmd_version(args: argparse.Namespace) -> int:
    import benchmaxxing

    print(benchmaxxing.__version__)
    if getattr(args, "verbose", False):
        print(f"git SHA: {_git_sha()}")
        print("extras:")
        for name, available in _extras_status().items():
            print(f"  {name}: {'installed' if available else 'not installed'}")
    return 0


def _dataset_names() -> list[str]:
    """Best-effort list of registered dataset names, empty only on a real import error."""
    try:
        from benchmaxxing.datasets import registry
    except Exception:  # noqa: BLE001 - datasets may be mid-build; degrade gracefully
        return []
    return sorted(registry.names())


def _cmd_datasets_list(_args: argparse.Namespace) -> int:
    from benchmaxxing.datasets import status

    names = status.names()
    if not names:
        print("No dataset status entries are registered yet.")
        return 0

    headers = (
        "dataset",
        "lane",
        "staged",
        "adapter",
        "solo",
        "cascade",
        "plausibility",
        "referee",
        "blocker",
    )
    rows = []
    for name in names:
        entry = status.get(name)
        rows.append(
            (
                name,
                entry.lane,
                entry.staged,
                entry.adapter,
                entry.solo,
                entry.cascade,
                entry.plausibility,
                entry.referee,
                entry.blocker,
            )
        )

    widths = [max(len(str(value)) for value in column) for column in zip(headers, *rows)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(width) for value, width in zip(row, widths)))
    return 0


def _cmd_datasets_stats(args: argparse.Namespace) -> int:
    """Load a manifest and print a summary: row/modality counts, an MCQ shape check, an
    optional on-disk image check, the label distribution, and how many rows carry meta.
    Exits non-zero if the manifest is empty or malformed (the loader itself enforces that)."""
    from collections import Counter
    from pathlib import Path

    from benchmaxxing.data import load_cases
    from benchmaxxing.schema import Modality

    try:
        cases = load_cases(args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n = len(cases)
    print(f"rows: {n}")
    by_modality = Counter(c.modality.value for c in cases)
    for modality, count in sorted(by_modality.items()):
        print(f"  {modality}: {count}")

    text_cases = [c for c in cases if c.modality is Modality.TEXT]
    if text_cases:
        opt_counts = Counter(len(c.options) if c.options else 0 for c in text_cases)
        print("options-per-case distribution:")
        for n_opts in sorted(opt_counts):
            print(f"  {n_opts} options: {opt_counts[n_opts]} cases")
        # load_cases already rejects an out-of-range answer_index at parse time; this is a
        # belt-and-suspenders check in case a future loader relaxes that guarantee.
        bad = [
            c.case_id
            for c in text_cases
            if c.options and not (0 <= (c.answer_index if c.answer_index is not None else -1)
                                   < len(c.options))
        ]
        if bad:
            print(f"error: answer_index out of range for case_id(s): {bad}", file=sys.stderr)
            return 1

    image_cases = [c for c in cases if c.modality is Modality.IMAGE]
    if image_cases:
        if args.image_root:
            root = Path(args.image_root)
            resolved = sum(
                1 for c in image_cases if c.image_ref and (root / c.image_ref).exists()
            )
            print(f"image_ref resolves on disk: {resolved}/{len(image_cases)} (root={root})")
        else:
            print(
                f"image_ref resolution: skipped ({len(image_cases)} image cases; "
                "pass --image-root to check)"
            )

    labels = [c.label for c in cases if c.label]
    if labels:
        print("label distribution:")
        for label, count in Counter(labels).most_common():
            print(f"  {label}: {count}")

    with_meta = sum(1 for c in cases if c.meta)
    print(f"rows with meta: {with_meta}/{n}")
    return 0


def _cmd_datasets_stage(args: argparse.Namespace) -> int:
    """Stage one dataset: build the manifest, validate it, and write its provenance record."""
    from pathlib import Path

    from benchmaxxing.datasets.staging import SOURCES, stage_dataset

    try:
        provenance = stage_dataset(
            args.name,
            raw_root=args.raw_root,
            out=args.out,
            limit=args.limit,
            check_images=args.check_images,
        )
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    source = SOURCES.get(args.name)
    counts = provenance["counts"]
    print(f"staged {args.name}: {counts['n_cases']} cases from {provenance['raw_root']}")
    print(f"  manifest:   {provenance['manifest']}")
    print(f"  sha256:     {provenance['manifest_sha256']}")
    print(f"  provenance: {Path(provenance['manifest']).parent / (args.name + '_SOURCE.txt')}")
    if source:
        print(f"  source:     {source.url} (access: {source.access})")
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    sub_command = getattr(args, "datasets_command", None)
    if sub_command == "stats":
        return _cmd_datasets_stats(args)
    if sub_command == "stage":
        return _cmd_datasets_stage(args)
    return _cmd_datasets_list(args)


def _select_hard_cases(cases, args) -> tuple[list, dict]:
    """Keep only the cases a previous solo run found hardest, and describe the selection.

    A cascade needs an uncertain committee: a confident, correct committee is the worst place to
    look for one. The ranking itself is ``analysis.select_uncertain_cases`` over the records the
    earlier run already wrote, so this only reads a file and filters.
    """
    from pathlib import Path

    from benchmaxxing.analysis import select_uncertain_cases

    path = Path(args.hard_cases)
    if not path.exists():
        raise FileNotFoundError(f"Hard-case records not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Hard-case records file is empty: {path}")

    # rank everything first, then cut: reporting both counts keeps the cut visible instead of
    # letting a --hard-k silently look like the whole hard set
    ranked = select_uncertain_cases(records, max_clean_accuracy=args.hard_max_accuracy)
    # guard the cut the way analysis.select_uncertain_cases guards its own k: a negative --hard-k
    # is a slice that quietly keeps everything but the easiest few at full API cost, and --hard-k 0
    # selects nothing -- both should fail loudly naming k, not slide into the manifest error below
    if args.hard_k is not None and args.hard_k < 0:
        raise ValueError(f"--hard-k must be non-negative, got {args.hard_k}")
    selected = ranked[: args.hard_k] if args.hard_k is not None else ranked
    if args.hard_k is not None and ranked and not selected:
        raise ValueError(
            f"--hard-k={args.hard_k} kept no cases out of {len(ranked)} ranked; "
            "drop --hard-k to keep them all or pass a positive count"
        )
    order = {case_id: rank for rank, case_id in enumerate(selected)}
    kept = sorted((c for c in cases if c.case_id in order), key=lambda c: order[c.case_id])
    if not kept:
        raise ValueError(
            f"none of the {len(selected)} hard case ids from {path} are in this manifest; "
            "the records and the manifest have to come from the same dataset"
        )
    return kept, {
        "source": str(path),
        "k": args.hard_k,
        "max_clean_accuracy": args.hard_max_accuracy,
        "n_ranked": len(ranked),
        "n_selected": len(selected),
        "n_in_manifest": len(kept),
        "case_ids": [c.case_id for c in kept],
    }

def _cmd_run(args: argparse.Namespace) -> int:
    """Drive one stage runner from a config plus a manifest and write the run directory.

    Thin on purpose: resolve the config, load the cases, hand a backend factory to
    ``benchmaxxing.runner``, and report where the artifacts landed. All the stage logic lives in
    ``benchmaxxing.experiments`` and the wiring in ``benchmaxxing.runner``.
    """
    from datetime import datetime
    from pathlib import Path

    from benchmaxxing import runner
    from benchmaxxing.bundle import bundle_dir_name
    from benchmaxxing.config import load_config
    from benchmaxxing.data import load_cases

    try:
        config = load_config(args.config)
        if args.dataset:
            config.dataset = args.dataset
        # --out names the bundle directory; without it, one is named per the bundle convention
        # (<stage>-<dataset>-<date>) under the config's out_dir, so runs do not overwrite
        # each other
        config.out_dir = args.out or str(
            Path(config.out_dir)
            / bundle_dir_name(args.stage, config.dataset, datetime.now().strftime("%Y%m%d"))
        )
        cases = load_cases(args.manifest)
        selection = None
        if args.hard_cases:
            cases, selection = _select_hard_cases(cases, args)
        plan = runner.plan_run(
            args.stage,
            cases,
            config,
            limit=args.limit,
            noise_floor=args.noise_floor,
            image_root=args.image_root,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if selection is not None:
        plan["case_selection"] = selection
    if args.dry_run:
        print(f"plan for stage {plan['stage']} (dry run, no API calls)")
        print(f"  models: {', '.join(plan['models'])}")
        lineages = {
            model: f"{lineage} ({plan['lineage_sources'][model]})"
            for model, lineage in plan["lineages"].items()
        }
        print(f"  lineages: {json.dumps(lineages, sort_keys=True)}")
        print(f"  cues: {', '.join(plan['cue_types'])}")
        print(f"  lane: {plan['lane']}")
        if selection is not None:
            print(
                f"  hard-case selection: {selection['n_in_manifest']} case(s) from "
                f"{selection['source']} ({selection['n_selected']} selected of "
                f"{selection['n_ranked']} ranked)"
            )
        print(f"  cases: {plan['n_cases']}, twins: {plan['n_twins']}")
        if plan["skipped_cases"]:
            skipped = ", ".join(f"{k}={v}" for k, v in sorted(plan["skipped_cases"].items()) if v)
            print(f"  skipped cases: {skipped or 'none'}")
        print(f"  estimated model calls: {plan['estimated_calls']}")
        print(f"  noise-floor control: {'on' if plan['noise_floor'] else 'off'}")
        print(f"  would write to: {plan['out_dir']}")
        return 0

    out_dir = Path(config.out_dir)
    backends: dict[str, object] = {}

    def backend_for(model_id: str):
        if model_id not in backends:
            backends[model_id] = runner.build_backend(model_id, kind=args.backend)
        return backends[model_id]

    try:
        results = runner.run_stage(
            args.stage,
            cases,
            config,
            backend_for,
            limit=args.limit,
            transcript_dir=out_dir / "transcripts" if args.stage == "cascade" else None,
            noise_floor=plan["noise_floor"],
            image_root=args.image_root,
        )
    except (ImportError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    paths = runner.write_outputs(
        out_dir, args.stage, config, results, plan, manifest_path=args.manifest
    )
    print(f"stage {args.stage} finished over {plan['n_cases']} cases")
    for name in ("results", "summary", "config", "versions", "run_manifest"):
        print(f"  {name}: {paths[name]}")
    print(f"render it with: benchmaxxing report {out_dir}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Render a finished run bundle into HTML, optionally replaying it from the transcripts."""
    from benchmaxxing.bundle import load_bundle, render_bundle_report, replay_cascade

    try:
        bundle = load_bundle(args.bundle)
        out = render_bundle_report(args.bundle, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stage = bundle["results"].get("stage")
    print(f"report: {out}")
    if not args.replay:
        return 0

    if stage != "cascade":
        print(f"replay: nothing to replay for stage {stage!r} (no transcripts)")
        return 0
    try:
        replayed = replay_cascade(args.bundle)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    reported = bundle["results"]["results"]
    reported_n = reported.get("n_cases", len(reported.get("per_case", [])))
    print(f"replay: {replayed['n_cases']} case(s) recomputed from transcripts, no model calls")
    ok = True
    # a bundle that lost transcripts replays fewer cases than it reports; the surviving subset's
    # means can still match (pinned adoption, for one), so coverage is checked before the means or
    # a partial bundle certifies numbers it no longer covers
    if replayed["skipped"] or replayed["n_cases"] != reported_n:
        ok = False
        missing = f", {len(replayed['skipped'])} missing transcripts" if replayed["skipped"] else ""
        print(f"  coverage: replayed {replayed['n_cases']} of {reported_n} reported case(s)"
              f"{missing}  MISMATCH")
    for key in ("shared_adoption", "isolated_adoption", "adoption_delta"):
        same = abs(replayed[key] - reported[key]) < 1e-9
        ok = ok and same
        print(f"  {key}: reported {reported[key]:.4f}, replayed {replayed[key]:.4f}"
              f"{'' if same else '  MISMATCH'}")
    return 0 if ok else 1

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
    from benchmaxxing.blind_metric import blind_metric_uptake
    from benchmaxxing.cues.text import build_text_twin
    from benchmaxxing.onset import cascade_onset
    from benchmaxxing.referee import gate_decision, precision_recall, score_shortcut
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

    # 3. referee scoring + gate. Ground truth is INDEPENDENT of the detector: an agent was
    #    gamed if it committed correctly in isolation but the shortcut under shared context.
    gamed = {a: (shared.committed.get(a) == _SHORTCUT and isolated.committed.get(a) == _CORRECT)
             for a in shared.committed}
    flags = score_shortcut(shared, "seeded")
    ref_pr = precision_recall(flags, gamed)
    robust_id = next(m.name for m in committee.members if m.is_open_weights)
    gate = gate_decision(shared, planted_cue_type="seeded")

    # 4. blind-metric probe: decisions that track a hidden decoy more than ground truth give a
    #    positive uptake delta (drift toward the artifact). Synthetic, deterministic.
    ground_truth = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    decoy = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    decisions = [0.9, 0.85, 0.95, 0.15, 0.1, 0.2]  # follow the decoy, not the truth
    uptake = blind_metric_uptake(decisions, decoy, ground_truth)

    return {
        "solo_flip_rate": flip_rate(records)["overall"],
        "n_twins": len(twins),
        "shared_shortcut_agents": shared_shortcut,
        "isolated_shortcut_agents": isolated_shortcut,
        "cascade_onset_turn": cascade_onset(series),
        "n_agents": len(committee.members),
        "referee_precision": ref_pr.get("precision"),
        "referee_recall": ref_pr.get("recall"),
        "robust_agent_flagged": flags.get(robust_id),      # must be False (a true negative)
        "gate_approve": gate.approve,
        "gate_reason": gate.reason,
        "blind_metric_uptake_delta": uptake.uptake_delta,
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
        f"4. referee precision/recall (vs independent cross-condition truth): "
        f"{report['referee_precision']}/{report['referee_recall']}; robust cross-lineage agent "
        f"flagged={report['robust_agent_flagged']} (must be False)",
        f"5. gate: approve={report['gate_approve']} ({report['gate_reason']})",
        f"6. blind-metric uptake delta: {report['blind_metric_uptake_delta']:.3f} "
        "(positive = decisions drift toward the decoy)",
    ]
    # the plumbing is healthy only if the cue bit, the shortcut spread more when shared, the
    # referee did not falsely flag the robust agent, and the blind-metric probe saw drift.
    healthy = (
        report["solo_flip_rate"] > 0
        and report["shared_shortcut_agents"] > report["isolated_shortcut_agents"]
        and report["robust_agent_flagged"] is False
        and report["blind_metric_uptake_delta"] > 0
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
    "run": _cmd_run,
    "report": _cmd_report,
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
