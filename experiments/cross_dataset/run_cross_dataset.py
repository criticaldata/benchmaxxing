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
import os
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.cross_dataset import run_cross_dataset_cue
from benchmaxxing.data import load_cases
from benchmaxxing.extract import parse_mcq_choice

CUES = ["option_order", "longest_option", "lexical_overlap"]
_ABSTAIN = "<abstain>"


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


def make_backend(model, api_key, *, raw=None):
    """A ``payload -> option-text`` callable; ``raw`` is injectable so tests skip the network."""
    if raw is None:
        raw = gateway.RetryBackend(
            gateway.GeminiBackend(model=model, api_key=api_key), tries=5, backoff=3.0
        )

    def call(payload) -> str:
        options = list(payload["options"])
        reply = raw.complete(_mcq_prompt(payload), decoding={"temperature": 0})
        choice = parse_mcq_choice(reply, options)
        return options[choice] if isinstance(choice, int) else _ABSTAIN

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
    args = ap.parse_args(argv)

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return _skip("no GEMINI_API_KEY/GOOGLE_API_KEY: this is a real-model run, nothing fabricated.")
    for label, path in (("MedQA", args.medqa_manifest), ("MedMCQA", args.medmcqa_manifest)):
        if not Path(path).exists():
            return _skip(f"{label} manifest not found: {path}. Build it with the dataset adapter first.")

    result = run(
        args.medqa_manifest,
        args.medmcqa_manifest,
        model=args.model,
        api_key=key,
        limit=args.limit,
    )
    print(format_table(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
