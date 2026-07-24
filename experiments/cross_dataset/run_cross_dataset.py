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
from pathlib import Path

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


def make_backend(model, api_key, *, raw=None, cache=None):
    """A ``payload -> option-text`` callable; ``raw`` is injectable so tests skip the network.

    ``cache`` is a shared dict wrapped around the live backend so repeated prompts reuse the
    recorded reply: a rerun with the same persisted cache reproduces the same rates and CIs.
    """
    if raw is None:
        live = gateway.RetryBackend(
            gateway.GeminiBackend(model=model, api_key=api_key), tries=5, backoff=3.0
        )
        raw = gateway.CachedBackend(live, cache=cache if cache is not None else {})

    def call(payload) -> str:
        options = list(payload["options"])
        reply = raw.complete(_mcq_prompt(payload), decoding={"temperature": 0})
        choice = parse_mcq_choice(reply, options)
        if isinstance(choice, int):
            return options[choice]
        # Fall back to a letter-only reply ("B") before giving up as an abstention.
        idx = _letter_index(reply, len(options))
        return options[idx] if idx is not None else _ABSTAIN

    return call


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
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--cache", default=None, help="JSONL call cache to record/reuse for reproducibility")
    args = ap.parse_args(argv)

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return _skip("no GEMINI_API_KEY/GOOGLE_API_KEY: this is a real-model run, nothing fabricated.")
    for label, path in (("MedQA", args.medqa_manifest), ("MedMCQA", args.medmcqa_manifest)):
        if not Path(path).exists():
            return _skip(f"{label} manifest not found: {path}. Build it with the dataset adapter first.")

    cache = load_cache(args.cache)
    result = run(
        args.medqa_manifest,
        args.medmcqa_manifest,
        model=args.model,
        api_key=key,
        limit=args.limit,
        backend_factory=lambda model, api_key: make_backend(model, api_key, cache=cache),
    )
    save_cache(args.cache, cache)
    print(format_table(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(result), indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
