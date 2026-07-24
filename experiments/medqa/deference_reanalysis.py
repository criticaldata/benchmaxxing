"""Offline deference re-analysis over the committed MedQA cascade transcripts.

Uses ``benchmaxxing.transcript_dynamics`` on the three saved transcript families under
``experiments/medqa/results/transcripts/``:

* ``v1_first_distractor``: ``*_shared.jsonl`` / ``*_isolated.jsonl`` (no ``v2``/``repro`` infix)
* ``v2_baseline_relative``: ``*_v2_shared.jsonl`` / ``*_v2_isolated.jsonl``
* ``repro_baseline_relative``: ``*_repro_shared.jsonl`` / ``*_repro_isolated.jsonl``

For each family, report shared vs isolated deference rates, and within shared boards split
deferred turns into seed-sourced (the intervening peer turn was ``seeded=True``) vs
organic-peer (the intervening peer turn was an unplanted agent answer).

Turn-level organic *runs* (``live_peer_organic``) are not available as transcripts, so this
cannot yet compare a fully organic committee to a seeded one at the turn level.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from benchmaxxing.transcript import load_transcript
from benchmaxxing.transcript_dynamics import deference_summary, summarize_transcripts

FAMILIES = (
    "v1_first_distractor",
    "v2_baseline_relative",
    "repro_baseline_relative",
)


def _arm_of(path: Path) -> tuple[str, str]:
    name = path.name
    if "_v2_" in name:
        family = "v2_baseline_relative"
    elif "_repro_" in name:
        family = "repro_baseline_relative"
    else:
        family = "v1_first_distractor"
    mode = "shared" if name.endswith("_shared.jsonl") else "isolated"
    return family, mode


def _classify_events(transcript):
    turns_by_index = {turn.turn_index: turn for turn in transcript.turns}
    summary = deference_summary(transcript)
    seeded_events = 0
    organic_events = 0
    for event in summary.events:
        peer = turns_by_index.get(event.peer_turn_index)
        if peer is not None and peer.seeded:
            seeded_events += 1
        else:
            organic_events += 1
    return summary, seeded_events, organic_events


def _jsonable(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def run_reanalysis(transcript_dir: Path) -> dict:
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for path in sorted(transcript_dir.glob("*.jsonl")):
        groups[_arm_of(path)].append(path)

    arms: dict[str, dict] = {}
    for (family, mode), paths in sorted(groups.items()):
        transcripts = [load_transcript(path) for path in paths]
        stats = summarize_transcripts(transcripts)[mode]
        seeded_ev = organic_ev = 0
        for transcript in transcripts:
            _, seeded, organic = _classify_events(transcript)
            seeded_ev += seeded
            organic_ev += organic
        eligible = int(stats["eligible_turns"])
        arms[f"{family}/{mode}"] = {
            **stats,
            "seed_sourced_deferred_turns": seeded_ev,
            "organic_peer_deferred_turns": organic_ev,
            "seed_sourced_deference_rate": seeded_ev / eligible if eligible else 0.0,
            "organic_peer_deference_rate": organic_ev / eligible if eligible else 0.0,
            "n_files": len(paths),
        }

    comparison = {}
    for family in FAMILIES:
        shared = arms.get(f"{family}/shared")
        isolated = arms.get(f"{family}/isolated")
        if shared is None or isolated is None:
            continue
        comparison[family] = {
            "shared_deference_rate": shared["deference_rate"],
            "isolated_deference_rate": isolated["deference_rate"],
            "shared_minus_isolated": shared["deference_rate"] - isolated["deference_rate"],
            "shared_seed_sourced_rate": shared["seed_sourced_deference_rate"],
            "shared_organic_peer_rate": shared["organic_peer_deference_rate"],
            "n_shared": shared["n_transcripts"],
            "n_isolated": isolated["n_transcripts"],
        }

    return {
        "transcript_dir": str(transcript_dir),
        "arms": arms,
        "comparison": comparison,
        "notes": [
            "All three families are seeded cascade runs; isolated boards have peer visibility off.",
            "seed_sourced = intervening peer turn was seeded; organic_peer = intervening peer was unplanted.",
            "live_peer_organic.jsonl has no turn-level transcripts, so a fully organic committee arm is absent.",
        ],
    }


def format_report(report: dict) -> str:
    lines = [
        "MedQA deference re-analysis (committed cascade transcripts)",
        "=" * 58,
    ]
    for family, row in report["comparison"].items():
        lines.append(
            f"{family}: shared={row['shared_deference_rate']:.3f} "
            f"isolated={row['isolated_deference_rate']:.3f} "
            f"delta={row['shared_minus_isolated']:.3f} "
            f"(seed-sourced={row['shared_seed_sourced_rate']:.3f}, "
            f"organic-peer={row['shared_organic_peer_rate']:.3f})"
        )
    lines.append("")
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript-dir",
        default="experiments/medqa/results/transcripts",
        help="directory of committed cascade transcript JSONL files",
    )
    parser.add_argument(
        "--out",
        default="experiments/medqa/results/deference_reanalysis.json",
        help="where to write the JSON summary",
    )
    args = parser.parse_args()

    report = run_reanalysis(Path(args.transcript_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    print(format_report(report))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
