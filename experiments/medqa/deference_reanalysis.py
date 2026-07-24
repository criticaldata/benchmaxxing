"""Offline deference re-analysis over the committed MedQA cascade transcripts.

Uses ``benchmaxxing.transcript_dynamics`` on the three saved transcript families under
``experiments/medqa/results/transcripts/``:

* ``v1_first_distractor``: ``*_shared.jsonl`` / ``*_isolated.jsonl`` (no ``v2``/``repro`` infix)
* ``v2_baseline_relative``: ``*_v2_shared.jsonl`` / ``*_v2_isolated.jsonl``
* ``repro_baseline_relative``: ``*_repro_shared.jsonl`` / ``*_repro_isolated.jsonl``

For each family, report shared vs isolated deference rates. Within deferred turns, the
headline seed-sourced vs organic-peer split keys on **adopted-answer identity**: whether the
answer the agent switched to matches the transcript's planted seed answer. A seed that is
relayed through an intermediate unplanted peer still counts as seed-sourced. Peer-turn
provenance (whether the immediately preceding matching peer was itself ``seeded=True``) is
kept as a secondary breakdown only.

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


def _answers_equal(left, right) -> bool:
    return left == right

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


def _seeded_answer(transcript):
    """First answer-bearing planted turn, or None when the board has no seed."""
    for turn in sorted(transcript.turns, key=lambda item: item.turn_index):
        if turn.seeded and turn.answer is not None:
            return turn.answer
    return None


def _classify_events(transcript):
    """Split deference events by adopted-answer identity (primary) and peer provenance.

    Returns ``(summary, seed_answer_match, organic_novel, peer_was_seeded, peer_was_unplanted)``.
    """
    turns_by_index = {turn.turn_index: turn for turn in transcript.turns}
    summary = deference_summary(transcript)
    seeded_answer = _seeded_answer(transcript)
    seed_answer_match = organic_novel = 0
    peer_was_seeded = peer_was_unplanted = 0
    for event in summary.events:
        if seeded_answer is not None and _answers_equal(event.answer, seeded_answer):
            seed_answer_match += 1
        else:
            organic_novel += 1
        peer = turns_by_index.get(event.peer_turn_index)
        if peer is not None and peer.seeded:
            peer_was_seeded += 1
        else:
            peer_was_unplanted += 1
    return summary, seed_answer_match, organic_novel, peer_was_seeded, peer_was_unplanted


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
        seed_match = organic = peer_seeded = peer_organic = 0
        for transcript in transcripts:
            _, match, novel, p_seed, p_org = _classify_events(transcript)
            seed_match += match
            organic += novel
            peer_seeded += p_seed
            peer_organic += p_org
        eligible = int(stats["eligible_turns"])
        arms[f"{family}/{mode}"] = {
            **stats,
            "seed_sourced_deferred_turns": seed_match,
            "organic_peer_deferred_turns": organic,
            "seed_sourced_deference_rate": seed_match / eligible if eligible else 0.0,
            "organic_peer_deference_rate": organic / eligible if eligible else 0.0,
            "peer_turn_seeded_deferred_turns": peer_seeded,
            "peer_turn_unplanted_deferred_turns": peer_organic,
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
            "shared_peer_turn_seeded_count": shared["peer_turn_seeded_deferred_turns"],
            "shared_peer_turn_unplanted_count": shared["peer_turn_unplanted_deferred_turns"],
            "n_shared": shared["n_transcripts"],
            "n_isolated": isolated["n_transcripts"],
        }

    return {
        "transcript_dir": str(transcript_dir),
        "arms": arms,
        "comparison": comparison,
        "notes": [
            "All three families are seeded cascade runs; isolated boards have peer visibility off.",
            "seed_sourced = deferred answer equals the planted seed answer (including seed relays).",
            "organic_peer = deferred answer differs from the planted seed (a novel peer answer).",
            "peer_turn_* counts are secondary provenance only (whether the last matching peer was planted).",
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
            f"(seed-answer={row['shared_seed_sourced_rate']:.3f}, "
            f"organic-novel={row['shared_organic_peer_rate']:.3f}; "
            f"peer-prov={row['shared_peer_turn_seeded_count']}/"
            f"{row['shared_peer_turn_unplanted_count']})"
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
