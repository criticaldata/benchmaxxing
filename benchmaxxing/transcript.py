"""Transcript serialization: dump/load a Transcript as JSONL for lossless offline replay.

The on-disk format is JSONL. The first line is a header record carrying the run-level fields
(run_id, case_id, condition, committed, meta). Each following line is one Turn. Round-tripping
through dump_transcript then load_transcript (or replay) reconstructs an equal Transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.schema import Condition, Transcript, Turn


def _condition_value(condition: object) -> str:
    """Return the wire value for a Condition (accepts a Condition enum or a plain string)."""
    if isinstance(condition, Condition):
        return condition.value
    return str(condition)


def _jsonable(value):
    """Canonicalize a turn answer / committed value for JSON.

    Numpy scalars (a committee answer often comes back as ``np.bool_``/``np.int64`` from an
    analysis step) would crash ``json.dumps``; sequences would round-trip as lists and break
    equality with a tuple answer. Convert scalars via ``.item()`` and recurse into
    dict/list/tuple; anything else that json cannot encode is stored as ``str(value)``, which
    keeps the dump lossless for the standard types and loud-but-safe for exotic ones.
    """
    item = getattr(value, "item", None)
    if callable(item) and type(value).__module__ == "numpy":
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dump_transcript(transcript: Transcript, path: str | Path) -> None:
    """Write ``transcript`` to ``path`` as JSONL: a header line then one line per turn."""
    records: list[dict] = [
        {
            "kind": "header",
            "run_id": transcript.run_id,
            "case_id": transcript.case_id,
            "condition": _condition_value(transcript.condition),
            "committed": _jsonable(transcript.committed),
            "meta": transcript.meta,
        }
    ]
    for turn in transcript.turns:
        records.append(
            {
                "kind": "turn",
                "turn_index": turn.turn_index,
                "agent_id": turn.agent_id,
                "content": turn.content,
                "answer": _jsonable(turn.answer),
                "confidence": turn.confidence,
                "seeded": turn.seeded,
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_transcript(path: str | Path) -> Transcript:
    """Read a JSONL transcript written by ``dump_transcript`` and rebuild the Transcript."""
    header: dict | None = None
    turns: list[Turn] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "header":
                header = record
            elif kind == "turn":
                turns.append(
                    Turn(
                        turn_index=record["turn_index"],
                        agent_id=record["agent_id"],
                        content=record["content"],
                        answer=record.get("answer"),
                        confidence=record.get("confidence"),
                        seeded=record.get("seeded", False),
                    )
                )
    if header is None:
        raise ValueError(f"no header record found in transcript file: {path}")
    return Transcript(
        run_id=header["run_id"],
        case_id=header["case_id"],
        condition=Condition(header["condition"]),
        turns=turns,
        committed=header.get("committed", {}),
        meta=header.get("meta", {}),
    )


def replay(path: str | Path) -> Transcript:
    """Load a transcript for offline re-analysis (alias of load_transcript, lossless)."""
    return load_transcript(path)
