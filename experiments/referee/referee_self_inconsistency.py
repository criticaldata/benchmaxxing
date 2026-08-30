"""Referee self-inconsistency floor (#417).

Measures whether identical cache-bypassed temperature-0 private re-queries
produce different answers in the absence of committee influence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.data import load_cases
from benchmaxxing.extract import parse_legacy_string


HOLDOUT = "gemini-2.5-flash-lite"


def _key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _letters(n):
    return [chr(65 + i) for i in range(n)]


def _mcq(case, prefix=""):
    opts = list(case.options)
    body = "\n".join(f"{L}. {o}" for L, o in zip(_letters(len(opts)), opts))
    return (
        f"{prefix}Question: {case.question}\n\nOptions:\n{body}\n\n"
        "Answer with only the single letter of the best option."
    ), opts



class _DrawCache:
    """Replayable cache whose key keeps independent temp-0 draws distinct."""

    def __init__(self, path):
        self.path = Path(path)
        self.store = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.store[record["k"]] = record["resp"]

    @staticmethod
    def key(model, prompt, draw):
        return hashlib.sha256(
            f"{model}\x000.0\x00{draw}\x00{prompt}".encode()
        ).hexdigest()

    def get(self, model, prompt, draw):
        return self.store.get(self.key(model, prompt, draw))

    def put(self, model, prompt, draw, response):
        k = self.key(model, prompt, draw)
        self.store[k] = response

        with self.path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "k": k,
                        "model": model,
                        "draw": draw,
                        "resp": response,
                    }
                )
                + "\n"
            )


def _query_uncached(model, prompt, api_key):
    """Always hits the model; never serves from cache."""

    backend = gateway.RetryBackend(
        gateway.GeminiBackend(model=model, api_key=api_key),
        tries=5,
        backoff=3.0,
    )

    return backend.complete(
        prompt,
        decoding={"temperature": 0},
    )



def _complete_draw(cache, model, prompt, draw, api_key):
    """Replay a recorded draw, or make and record a fresh cache-bypassed temp-0 call."""
    cached = cache.get(model, prompt, draw)
    if cached is not None:
        return cached, False

    if not api_key:
        raise SystemExit(
            "Draw missing from self-inconsistency cache and no GEMINI_API_KEY set."
        )

    response = _query_uncached(model, prompt, api_key)
    cache.put(model, prompt, draw, response)
    return response, True


def build_row(case_id, answer_1, answer_2):
    return {
        "case_id": case_id,
        "answer_1": answer_1,
        "answer_2": answer_2,
        "temp0_flip": answer_1 != answer_2,
    }


def run_one(case, cache, api_key):
    opts = list(case.options)

    prompt, _ = _mcq(case)

    raw_1, call_1 = _complete_draw(
        cache, HOLDOUT, prompt, draw=1, api_key=api_key
    )
    raw_2, call_2 = _complete_draw(
        cache, HOLDOUT, prompt, draw=2, api_key=api_key
    )

    answer_1 = parse_legacy_string(raw_1, opts)
    answer_2 = parse_legacy_string(raw_2, opts)

    row = build_row(case.case_id, answer_1, answer_2)

    return row, int(call_1) + int(call_2)

def summarize(rows):
    n = len(rows)
    unstable = sum(1 for r in rows if r["temp0_flip"])

    return {
        "n": n,
        "temperature": 0,
        "cache_bypassed": True,
        "stable_cases": n - unstable,
        "unstable_cases": unstable,
        "temp0_self_inconsistency_rate": (
            unstable / n if n else None
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Referee self-inconsistency floor (#417)."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--cache",
        default="experiments/referee/results/referee_self_inconsistency_cache.jsonl",
    )
    ap.add_argument(
        "--out",
        default="experiments/referee/results",
    )
    ap.add_argument("--n", type=int, default=40)

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = _DrawCache(args.cache)
    api_key = _key()

    rows = []
    new_api_calls = 0

    for case in load_cases(args.manifest)[: args.n]:
        row, calls = run_one(case, cache, api_key)
        rows.append(row)
        new_api_calls += calls

    summary = summarize(rows)
    summary["new_api_calls_this_run"] = new_api_calls

    (out / "referee_self_inconsistency.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    (out / "referee_self_inconsistency_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
