"""Referee self-inconsistency floor (#417).

Measures whether identical cache-bypassed temperature-0 private re-queries
produce different answers in the absence of committee influence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.extract import parse_legacy_string, declared_mcq_choice
from experiments.referee.referee_threshold import (
    _Cache,
    _key,
    _mcq,
    HOLDOUT,
)

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lane  # noqa: E402  (shared second-vendor dispatch, experiments/_lane.py)


class _ModelCache(_Cache):
    """The referee cache with the backend chosen by model id.

    Keys, file format and the Gemini path are unchanged, so the committed Gemini run replays
    with no key. Any other model goes through the shared lane dispatch, which also paces calls
    to the vendor's rate limit and waits out a 429 instead of failing the arm.
    """

    def __init__(self, path, key, model):
        super().__init__(path, key)
        self.model = model
        self._lane = None if _lane.is_gemini(model) else model

    def complete(self, model, prompt, temperature=0.0, draw=0):
        if self._lane is None:
            return super().complete(model, prompt, temperature=temperature, draw=draw)
        import hashlib as _h
        import json as _j
        from experiments.referee.referee_threshold import _lock

        k = _h.sha256(f"{model}\x00{temperature}\x00{draw}\x00{prompt}".encode()).hexdigest()
        with _lock:
            if k in self.store:
                return self.store[k]
        # a distinct draw must be a distinct request, so salt the shadow key with the draw
        resp = self._lane_call(model, prompt, temperature, draw)
        with _lock:
            self.store[k] = resp
            self.calls += 1
            with open(self.path, "a") as f:
                f.write(_j.dumps({"k": k, "model": model, "temperature": temperature,
                                  "draw": draw, "resp": resp}) + "\n")
        return resp

    def _lane_call(self, model, prompt, temperature, draw):
        """One paced, retried call with no caching of its own: the draw is the bypass."""
        import time as _t
        from benchmaxxing import gateway as _gw

        backend = _gw.RetryBackend(_lane.backend_for(model, self.key), tries=5, backoff=3.0)
        for attempt in range(_lane.RATE_LIMIT_TRIES):
            _lane._pace(model)
            try:
                return backend.complete(prompt, decoding={"temperature": temperature})
            except Exception as exc:  # noqa: BLE001
                root = exc
                while root.__cause__ is not None:
                    root = root.__cause__
                limited = _lane._is_rate_limited(root)
                if attempt == _lane.RATE_LIMIT_TRIES - 1 or not (limited or _lane._is_transient(root)):
                    raise
                _t.sleep(_lane.RATE_LIMIT_SLEEP if limited else _lane.TRANSIENT_SLEEP)



def build_row(case_id, answer_1, answer_2, declared_1, declared_2):
    return {
        "case_id": case_id,
        "answer_1": answer_1,
        "answer_2": answer_2,
        "declared_1": declared_1,
        "declared_2": declared_2,
        "temp0_flip": answer_1 != answer_2,
    }


def run_one(case, cache, model=HOLDOUT):
    opts = list(case.options)
    prompt, _ = _mcq(case)

    raw_1 = cache.complete(
        model, prompt, temperature=0.0, draw=1
    )
    raw_2 = cache.complete(
        model, prompt, temperature=0.0, draw=2
    )

    answer_1 = parse_legacy_string(raw_1, opts)
    answer_2 = parse_legacy_string(raw_2, opts)

    _, declared_1 = declared_mcq_choice(raw_1, opts)
    _, declared_2 = declared_mcq_choice(raw_2, opts)

    return build_row(
        case.case_id,
        answer_1,
        answer_2,
        declared_1,
        declared_2,
    )


def summarize(rows):
    n = len(rows)

    declared_pairs = sum(
        1
        for r in rows
        if r["declared_1"] and r["declared_2"]
    )

    undeclared_pairs = sum(
        1
        for r in rows
        if not (r["declared_1"] and r["declared_2"])
    )

    undeclared_draws = sum(
        int(not r["declared_1"]) + int(not r["declared_2"])
        for r in rows
    )

    unstable = sum(
        1
        for r in rows
        if r["declared_1"]
        and r["declared_2"]
        and r["temp0_flip"]
    )

    stable = declared_pairs - unstable

    return {
        "n": n,
        "temperature": 0,
        "declared_pairs": declared_pairs,
        "undeclared_pairs": undeclared_pairs,
        "undeclared_draws": undeclared_draws,
        "stable_cases": stable,
        "unstable_cases": unstable,
        "temp0_self_inconsistency_rate": (
            unstable / declared_pairs
            if declared_pairs
            else None
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
    ap.add_argument("--model", default=HOLDOUT,
                    help="model id; the default is the Gemini holdout the committed run used. Any other "
                         "model writes under <out>/<model slug>/ and its own cache file")

    args = ap.parse_args()

    model = args.model
    out = Path(args.out)
    cache_path = Path(args.cache)
    if model != HOLDOUT:
        slug = model.replace("/", "_")
        out = out / slug
        cache_path = cache_path.with_name(f"{slug}_{cache_path.name}")
    out.mkdir(parents=True, exist_ok=True)

    key = _key() if _lane.is_gemini(model) else _lane.key_for(model)
    cache = _ModelCache(cache_path, key, model)

    rows = [
        run_one(case, cache, model)
        for case in load_cases(args.manifest)[:args.n]
    ]

    summary = summarize(rows)
    summary["model"] = model
    summary["new_api_calls_this_run"] = cache.calls

    (out / "referee_self_inconsistency.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    (out / "referee_self_inconsistency_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
