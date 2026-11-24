"""Cross-dataset cue consistency: MedQA vs MedMCQA (issue #129).

Real-model entrypoint. Scores the same text cue set on two datasets with one fixed Gemini model
and prints the per-cue solo flip rate side by side, so a shortcut shown on MedQA can be checked
for replication on MedMCQA. The scoring, CIs, and agreement statistic live in
``benchmaxxing.cross_dataset.run_cross_dataset_cue``; this script only loads the manifests and
wires a real backend.

The backend is a ``payload -> chosen-option-text`` callable: the model's letter is resolved back
to the option text against that payload's options, so ``option_order`` is measured by answer
identity (not position) even after the options are permuted. That mirrors the proven solo
normalizer in ``experiments/medqa/reproduce.py``.

With no API key or a missing manifest the script prints a skip line and exits 0: it is a real run,
so it never fabricates numbers. The wiring and skip paths are covered offline by
``tests/test_cross_dataset_run.py``; the real flip rates need a key and are deferred to a run with
the datasets staged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _lane

from benchmaxxing import gateway
from benchmaxxing.cross_dataset import run_cross_dataset_cue
from benchmaxxing.data import load_cases
from benchmaxxing.extract import parse_mcq_choice

CUES = ["option_order", "longest_option", "lexical_overlap"]
_ABSTAIN = "<abstain>"


def _letter_index(reply: str, n_options: int) -> int | None:
    """Resolve a declared letter answer to a 0-based option index.

    Handles an exact single-letter reply (A, 'B.', '(C)') and a letter in an explicit "answer"
    phrase or LaTeX final-answer box, including the ``$\\boxed{\\text{D. ...}}$`` form that
    Gemini 2.5 reasoning replies end with (and optional ``*``/``$`` emphasis). Conservative on
    purpose: a trigger word or box is required, so a stray standalone letter mid-sentence (the
    leading-article "A") is never read as an option. ``parse_mcq_choice`` handles copied option
    text first; this is the fallback for a reply whose signal is the letter.
    """
    if not reply:
        return None
    letters = [chr(ord("A") + i) for i in range(n_options)]
    valid = {letter: i for i, letter in enumerate(letters)}
    text = reply.strip()
    exact = re.fullmatch(r"\(?([A-Za-z])\)?[.):]?", text)
    if exact and exact.group(1).upper() in valid:
        return valid[exact.group(1).upper()]
    declared = re.findall(
        r"(?:answer\s*(?:is|:)?\s*|\\boxed\{\s*(?:\\(?:text|textbf|mathrm)\{)?\s*)"
        r"[*_$]*\(?([A-Za-z])\)?",
        text,
        flags=re.IGNORECASE,
    )
    for cand in reversed(declared):
        if cand.upper() in valid:
            return valid[cand.upper()]
    return None


def _json_safe(obj):
    """Recursively replace non-finite floats (nan/inf) with None for strict-JSON output."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def load_cache(path) -> dict:
    """Load a persisted call cache (JSONL of ``{"key": [...], "resp": ...}``) into a dict.

    Keys are stored as lists and restored to the tuple that ``gateway.CachedBackend`` uses, so the
    cache content is treated opaquely and does not depend on that key's internal shape.
    """
    cache: dict = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                cache[tuple(record["key"])] = record["resp"]
    return cache


def save_cache(path, cache: dict) -> None:
    """Persist a call cache to JSONL so a later run with the same path reuses the responses."""
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for key, resp in cache.items():
            fh.write(json.dumps({"key": list(key), "resp": resp}) + "\n")


def _mcq_prompt(payload) -> str:
    options = payload["options"]
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options))
    report = (payload.get("report") or "").strip()
    head = f"Clinical context: {report}\n\n" if report else ""
    return (
        f"{head}Question: {payload['question']}\n\nOptions:\n{body}\n\n"
        "Answer with the letter and the full text of the single best option, copied exactly, "
        "for example 'B. <option text>'."
    )


class _PacedLive:
    """Backend-shaped handle whose every call goes through the shared paced path, so a 429 or a
    dropped connection waits instead of ending the pilot; the cache is only written at the end.

    ``calls`` counts the replies actually requested from the model, so a replay can state that it
    made none. With no key a cache miss stops the run instead of calling out: a keyless run is a
    replay of the committed cache or nothing."""

    def __init__(self, model, api_key):
        self.model, self.api_key, self.calls = model, api_key, 0

    def complete(self, prompt, decoding=None, **_):
        if not self.api_key:
            raise SystemExit(f"Cache miss and no key for {self.model}: a keyless run only replays a complete cache.")
        self.calls += 1
        return _lane.paced_complete(self.model, self.api_key, prompt, decoding=decoding)


def make_backend(model, api_key, *, raw=None, cache=None, cache_path=None, save_every=25):
    """A ``payload -> option-text`` callable; ``raw`` is injectable so tests skip the network.

    ``cache`` is a shared dict wrapped around the live backend so repeated prompts reuse the
    recorded reply: a rerun with the same persisted cache reproduces the same rates and CIs.
    When ``cache_path`` is given the dict is also written to disk every ``save_every`` new
    entries, so a run that stops part way (a timeout, a lost connection) resumes from the
    replies it already paid for instead of starting over; the final save in main() is unchanged.
    """
    live = None
    if raw is None:
        live = _PacedLive(model, api_key)
        raw = gateway.CachedBackend(live, cache=cache if cache is not None else {})
    seen = {"n": len(cache) if cache is not None else 0}

    def call(payload) -> str:
        options = list(payload["options"])
        reply = raw.complete(_mcq_prompt(payload), decoding={"temperature": 0})
        if cache_path and cache is not None and len(cache) - seen["n"] >= save_every:
            save_cache(cache_path, cache)
            seen["n"] = len(cache)
        choice = parse_mcq_choice(reply, options)
        if isinstance(choice, int):
            return options[choice]
        # Fall back to a letter-only reply ("B") before giving up as an abstention.
        idx = _letter_index(reply, len(options))
        return options[idx] if idx is not None else _ABSTAIN

    call.live = live
    return call


def prefill(datasets, model, api_key, cache, cache_path, *, limit=None, workers=8, save_every=25):
    """Ask every prompt a run will need, ``workers`` at a time, into the shared cache.

    The scoring loop asks one prompt at a time, which at a few calls a minute makes a 500-case run
    take most of a day. The set of prompts does not depend on any answer (the twins are built from
    the cases alone), so they can be asked in parallel first; the scoring pass that follows then
    reads every reply from the cache and its numbers are the ones a sequential run would give for
    the same replies. Returns the number of new calls.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from benchmaxxing.experiments import build_twins
    from benchmaxxing.schema import Condition

    prompts = []
    for cases in datasets.values():
        for tp in build_twins(cases, CUES, limit=limit):
            for cond in (Condition.CLEAN, Condition.CONTAMINATED):
                prompts.append(_mcq_prompt(tp.payload(cond)))
    decoding = {"temperature": 0}
    todo = []
    for prompt in dict.fromkeys(prompts):
        key = (prompt, gateway._image_key(None), gateway._decoding_key(decoding))
        if key not in cache:
            todo.append((key, prompt))
    if not todo:
        return 0
    live = _PacedLive(model, api_key)
    lock = threading.Lock()
    done = {"n": 0}

    def ask(item):
        key, prompt = item
        reply = live.complete(prompt, decoding=decoding)
        with lock:
            cache[key] = reply
            done["n"] += 1
            if cache_path and done["n"] % save_every == 0:
                save_cache(cache_path, dict(cache))
                print(f"[prefill] {done['n']}/{len(todo)}", flush=True)

    failed = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(ask, item): item for item in todo}
            for fut in futs:
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001  a prompt that keeps timing out is left to the scoring pass
                    failed.append(futs[fut])
                    print(f"[prefill] left for the scoring pass after {type(exc).__name__}", flush=True)
    finally:
        # Never lose replies already paid for: save whatever is in the cache, also on an error.
        if cache_path:
            with lock:
                save_cache(cache_path, dict(cache))
    if failed:
        print(f"[prefill] {len(failed)} prompt(s) not cached; the sequential pass asks them again", flush=True)
    return live.calls


def run(
    medqa_manifest,
    medmcqa_manifest,
    *,
    model,
    api_key,
    limit=None,
    loader=load_cases,
    backend_factory=make_backend,
) -> dict:
    """Load both manifests and score the shared cue set with one fixed model."""
    datasets = {
        "medqa": loader(medqa_manifest),
        "medmcqa": loader(medmcqa_manifest),
    }
    backend = backend_factory(model, api_key)
    return run_cross_dataset_cue(datasets, backend, lambda raw: raw, CUES, limit=limit)


def format_table(result: dict) -> str:
    """Render the comparison as a plain text table (no fabricated values: reads the result)."""
    names = result["datasets"]
    lines = ["cue" + "".join(f"\t{n}" for n in names) + "\tspread\tfisher_p"]
    for cue in result["cue_types"]:
        row = result["per_cue"][cue]
        cells = [f"{row['rate'][n]:.3f}" for n in names]
        fisher = row["fisher"]
        p = f"{fisher['pvalue']:.3g}" if fisher else "-"
        lines.append(f"{cue}\t" + "\t".join(cells) + f"\t{row['spread']:.3f}\t{p}")
    ag = result["agreement"]
    lines.append(f"mean_spread over {ag['n_cues_compared']} cues: {ag['mean_spread']:.3f}")
    return "\n".join(lines)


def _skip(message: str) -> int:
    print(f"[skip] {message}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-dataset cue consistency: MedQA vs MedMCQA (#129).")
    ap.add_argument("--medqa-manifest", required=True, help="path to the MedQA manifest (CSV/JSONL)")
    ap.add_argument("--medmcqa-manifest", required=True, help="path to the MedMCQA manifest")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--limit", type=int, default=None, help="cap cases per dataset")
    ap.add_argument("--workers", type=int, default=1,
                    help="ask the run's prompts this many at a time into the cache before scoring (1 = sequential, as before)")
    ap.add_argument("--exclude-case", action="append", default=[],
                    help="drop this case id from both datasets before --limit (repeatable); recorded in the "
                         "result as excluded_cases, for a case the model never answered")
    ap.add_argument("--replay", action="store_true",
                    help="run with no key from a complete --cache; a cache miss stops the run, and the "
                         "result records zero new calls")
    ap.add_argument("--out", default=None, help="write the result JSON here; defaults to the model-scoped file")
    ap.add_argument("--cache", default=None,
                    help="JSONL call cache to record/reuse for reproducibility; defaults to the model-scoped file. "
                         "The cache key omits the model id, so a second model must never read the committed one.")
    args = ap.parse_args(argv)
    if args.model != "gemini-2.5-flash":
        slug = args.model.replace("/", "_")
        results = Path(__file__).resolve().parent / "results"
        args.out = args.out or str(results / slug / "medqa_vs_medmcqa.json")
        args.cache = args.cache or str(results / f"{slug}_cache_medqa_vs_medmcqa.jsonl")

    if args.model == "gemini-2.5-flash" or args.model.startswith("gemini"):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if args.replay:
            key = None
        elif not key:
            return _skip("no GEMINI_API_KEY/GOOGLE_API_KEY: this is a real-model run, nothing fabricated.")
    else:
        try:
            key = _lane.key_for(args.model)
        except Exception as exc:  # a missing key for the requested model is a skip, not a traceback
            return _skip(f"no key for {args.model}: {exc}")
        if not key:
            return _skip(f"no {_lane.key_name(args.model)} for {args.model}: this is a real-model run, nothing fabricated.")
    for label, path in (("MedQA", args.medqa_manifest), ("MedMCQA", args.medmcqa_manifest)):
        if not Path(path).exists():
            return _skip(f"{label} manifest not found: {path}. Build it with the dataset adapter first.")

    if args.replay and not (args.cache and Path(args.cache).exists()):
        return _skip(f"--replay needs an existing --cache: {args.cache}")
    cache = load_cache(args.cache)
    calls = 0
    excluded = set(args.exclude_case)

    def loader(path):
        return [c for c in load_cases(path) if str(c.case_id) not in excluded]

    if args.workers > 1 and key:
        datasets = {"medqa": loader(args.medqa_manifest), "medmcqa": loader(args.medmcqa_manifest)}
        calls += prefill(datasets, args.model, key, cache, args.cache, limit=args.limit, workers=args.workers)
    backends = []

    def factory(model, api_key):
        b = make_backend(model, api_key, cache=cache, cache_path=None if args.replay else args.cache)
        backends.append(b)
        return b

    try:
        result = run(
            args.medqa_manifest,
            args.medmcqa_manifest,
            model=args.model,
            api_key=key,
            limit=args.limit,
            loader=loader,
            backend_factory=factory,
        )
    except BaseException:
        # a stopped run keeps every reply it paid for, so the next run resumes from them
        if not args.replay:
            save_cache(args.cache, cache)
        raise
    calls += sum(b.live.calls for b in backends if getattr(b, "live", None) is not None)
    if not args.replay:
        save_cache(args.cache, cache)
    print(f"new model calls this run: {calls}")
    if excluded:
        result["excluded_cases"] = sorted(excluded)
    if args.replay:
        result["replay"] = {"keyless": True, "new_api_calls_this_run": calls, "cache": Path(args.cache).name}
    print(format_table(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(result), indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
