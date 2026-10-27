"""Exclude cases whose undeclared r1 parse was written into a later prompt (#420).

Does not overwrite any committed summary or jsonl. Reads the committed per-case files and
recomputes the same headlines on the remaining rows. The case list is the one posted on #420:
an r1 completion that ``_lane.declared`` rejects, whose ``parse_legacy_string`` (or empty
string) then appeared in a later-round prompt.

SUPPORT2 ``rounds=2`` is included as a negative: every r1 board draw on the three lineages with
caches is declared, so nothing is excluded there.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.stats import mcnemar

ROOT = Path(__file__).resolve().parents[1]


def _case_sort(case_id: str) -> tuple[str, int]:
    head, _, tail = case_id.partition("-")
    return (head, int(tail) if tail.isdigit() else 0)

# Isolated multi-round / referee / Gemini seed-timing share medqa-38. Gemini seed-timing also
# has medqa-65. gpt-oss-20b seed-timing holdout_first has four truncated r1 completions.
GEMINI_MULTI_ROUND = ("medqa-38",)
GEMINI_SEED_TIMING = ("medqa-38", "medqa-65")
GPT_OSS_20B_SEED_TIMING = ("medqa-3", "medqa-44", "medqa-91", "medqa-111")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _curve(rows: list[dict], key: str, k: int) -> list[float]:
    return [round(sum(r[key][i] for r in rows) / len(rows), 4) for i in range(k)]


def multi_round_exclusion(path: Path, drop: tuple[str, ...], k: int = 5) -> dict:
    rows = _rows(path)
    kept = [r for r in rows if r["case_id"] not in drop]
    dropped = [r for r in rows if r["case_id"] in drop]
    b = sum(1 for r in kept if not r["shared_adopt"][0] and r["shared_adopt"][k - 1])
    d = sum(1 for r in kept if r["shared_adopt"][0] and not r["shared_adopt"][k - 1])
    return {
        "arm": "cascade/multi_round",
        "rows": str(path.relative_to(ROOT)),
        "n_committed": len(rows),
        "excluded": sorted((r["case_id"] for r in dropped), key=_case_sort),
        "n_remaining": len(kept),
        "excluded_adopted_isolated": [r["iso_adopt"] for r in dropped],
        "excluded_adopted_shared": [r["shared_adopt"] for r in dropped],
        "committed": {
            "shared_adoption_by_round": _curve(rows, "shared_adopt", k),
            "isolated_adoption_by_round": _curve(rows, "iso_adopt", k),
        },
        "excluded_view": {
            "shared_adoption_by_round": _curve(kept, "shared_adopt", k),
            "isolated_adoption_by_round": _curve(kept, "iso_adopt", k),
            "round1_vs_roundK_shared": {
                "gained": b,
                "lost": d,
                "mcnemar_p": float(mcnemar(b, d).pvalue),
            },
        },
    }


def seed_timing_exclusion(path: Path, drop: tuple[str, ...], model: str) -> dict:
    rows = _rows(path)
    kept = [r for r in rows if r["case_id"] not in drop]
    dropped = [r for r in rows if r["case_id"] in drop]
    n = len(kept)
    last = sum(r["holdout_last_adopt"] for r in kept)
    first = sum(r["holdout_first_adopt"] for r in kept)
    gain = sum(1 for r in kept if r["holdout_first_adopt"] and not r["holdout_last_adopt"])
    lose = sum(1 for r in kept if r["holdout_last_adopt"] and not r["holdout_first_adopt"])
    n0 = len(rows)
    return {
        "arm": "seed_timing",
        "model": model,
        "rows": str(path.relative_to(ROOT)),
        "n_committed": n0,
        "excluded": sorted((r["case_id"] for r in dropped), key=_case_sort),
        "n_remaining": n,
        "excluded_adoptions": [
            {
                "case_id": r["case_id"],
                "holdout_last_adopt": r["holdout_last_adopt"],
                "holdout_first_adopt": r["holdout_first_adopt"],
            }
            for r in dropped
        ],
        "committed": {
            "holdout_last": round(sum(r["holdout_last_adopt"] for r in rows) / n0, 4),
            "holdout_first": round(sum(r["holdout_first_adopt"] for r in rows) / n0, 4),
        },
        "excluded_view": {
            "holdout_last": round(last / n, 4),
            "holdout_first": round(first / n, 4),
            "last_vs_first": {
                "gain": gain,
                "lose": lose,
                "pvalue": round(mcnemar(gain, lose).pvalue, 6) if (gain + lose) else None,
            },
        },
    }


def build() -> dict:
    return {
        "issue": 420,
        "rule": (
            "Exclude a case from a multi-round arm when its r1 completion is undeclared under "
            "_lane.declared (as of #446) and that parse was written into a later prompt. "
            "Committed files are left untouched."
        ),
        "arms": [
            multi_round_exclusion(
                ROOT / "experiments/cascade/results/multi_round.jsonl", GEMINI_MULTI_ROUND
            ),
            seed_timing_exclusion(
                ROOT / "experiments/medqa/results/seed_timing.jsonl",
                GEMINI_SEED_TIMING,
                "gemini-2.5-flash-lite",
            ),
            seed_timing_exclusion(
                ROOT / "experiments/medqa/results/openai_gpt-oss-20b/seed_timing.jsonl",
                GPT_OSS_20B_SEED_TIMING,
                "openai/gpt-oss-20b",
            ),
        ],
        "already_covered": {
            "referee_deployable_declared_audit": (
                "experiments/referee/results/referee_deployable_declared_audit.json "
                "already drops medqa-38 (1/40) under declared-only accounting"
            ),
        },
        "negative": {
            "support2_rounds_2": (
                "Gemini, gpt-oss-120b and Qwen r1 board draws are all declared; "
                "parse_answer returns None on a refusal, so a true abstention never enters r2"
            ),
            "isolated_multi_round_other_lineages": (
                "gpt-oss-120b, gpt-oss-20b, Qwen, hosted nemotron: 0 undeclared r1 in n=40"
            ),
        },
        "read": (
            "The contaminated cases did not drive the headlines: Gemini isolated multi-round "
            "medqa-38 never adopted; Gemini seed-timing holdout_first stays 0.13; gpt-oss-20b "
            "seed-timing holdout_first stays 0.026. Exclude them anyway so a later prompt is "
            "never scored as if the holdout had committed."
        ),
    }


def main() -> None:
    out = ROOT / "experiments/prompt_contamination_audit.json"
    report = build()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
