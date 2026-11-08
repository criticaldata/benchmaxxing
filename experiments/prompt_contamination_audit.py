"""Exclude cases whose undeclared r1 parse was written into a later prompt (#420).

Does not overwrite any committed summary or jsonl. Reads the committed per-case files and
recomputes the same headlines on the remaining rows. The case list is the one posted on #420:
an r1 completion that ``_lane.declared`` rejects, whose ``parse_legacy_string`` (or empty
string) then appeared in a later-round prompt.

SUPPORT2 ``rounds=2`` is included as a negative: every r1 board draw on the three lineages with
caches is declared, so nothing is excluded there. ``reproduce.py`` 3-round committees are
counted from the committed transcripts: an empty unseeded r1 that ``render_board`` still
wrote into a later prompt is excluded (gpt-oss-20b MedMCQA).
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
# reproduce.py 3-round committee: gpt-oss-20b MedMCQA r1 agent0 returned empty text; parse_legacy
# mapped that to "" and render_board wrote `- agent0 answered: ` into every later shared prompt.
# Isolated r1 is empty on agent0 and agent2 of the same case. Transcripts store this as
# content="" / answer=""; no cache reconstruction needed.
GPT_OSS_20B_MEDMCQA_REPRODUCE = ("7111d7b2-7e80-4e23-b5d0-9f42595c6a80",)


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


def _mean(rows: list[dict], key: str) -> float | None:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None


def reproduce_exclusion(path: Path, drop: tuple[str, ...], model: str, dataset: str) -> dict:
    data = json.loads(path.read_text())
    rows = data["per_case"]
    kept = [r for r in rows if r["case_id"] not in drop]
    dropped = [r for r in rows if r["case_id"] in drop]
    return {
        "arm": "reproduce.py cascade",
        "dataset": dataset,
        "model": model,
        "rows": str(path.relative_to(ROOT)),
        "n_committed": len(rows),
        "excluded": sorted((r["case_id"] for r in dropped), key=_case_sort),
        "n_remaining": len(kept),
        "excluded_adoptions": [
            {
                "case_id": r["case_id"],
                "shared_adopt": r["shared_adopt"],
                "isolated_adopt": r["isolated_adopt"],
                "contagion": r["contagion"],
            }
            for r in dropped
        ],
        "committed": {
            "mean_shared_adopt": _mean(rows, "shared_adopt"),
            "mean_isolated_adopt": _mean(rows, "isolated_adopt"),
            "mean_contagion": _mean(rows, "contagion"),
        },
        "excluded_view": {
            "mean_shared_adopt": _mean(kept, "shared_adopt"),
            "mean_isolated_adopt": _mean(kept, "isolated_adopt"),
            "mean_contagion": _mean(kept, "contagion"),
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
            reproduce_exclusion(
                ROOT / "experiments/medmcqa/results/openai_gpt-oss-20b/cascade_results.json",
                GPT_OSS_20B_MEDMCQA_REPRODUCE,
                "openai/gpt-oss-20b",
                "medmcqa",
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
            "reproduce_medqa": (
                "Gemini, gpt-oss-120b, gpt-oss-20b, Qwen: 20 cases each, prompts reconstructed "
                "from the committed manifest and call cache (0 misses). 0 undeclared r1 that "
                "entered a later prompt"
            ),
            "reproduce_medmcqa_other_lineages": (
                "gpt-oss-120b and Qwen transcripts have no empty unseeded answers. Gemini "
                "MedMCQA cannot be reconstructed without the gitignored manifest; prefix "
                "match on the 150-char transcript clip is ambiguous, and no Gemini "
                "transcript has an empty r1"
            ),
        },
        "later_round_empty": (
            "Empty unseeded completions that entered a later prompt but were not r1: "
            "gpt-oss-20b MedQA isolated medqa-1047 turn 5; MedMCQA isolated "
            "83935f21-0298-409f-aabc-ac42dc4c0e00 turn 3; MedMCQA shared "
            "fd315adc-df4b-4a81-895b-6f093eeb71b2 turn 3. All three have adopt 0"
        ),
        "read": (
            "Most contaminated cases did not drive the headlines: Gemini isolated multi-round "
            "medqa-38 never adopted; Gemini seed-timing holdout_first stays 0.13; gpt-oss-20b "
            "seed-timing holdout_first stays 0.026. The exception is gpt-oss-20b MedMCQA "
            "reproduce.py: dropping the r1-empty case zeros mean contagion (0.025 → 0). "
            "Exclude them so a later prompt is never scored as if the holdout had committed."
        ),
    }


def main() -> None:
    out = ROOT / "experiments/prompt_contamination_audit.json"
    report = build()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
