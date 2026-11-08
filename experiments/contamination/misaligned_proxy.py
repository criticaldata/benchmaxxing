"""Misaligned-proxy arm, text lane (#176). Activates `benchmaxxing.blind_arms.misaligned_proxy_run`
(built and unit-tested with mocks only, invoked by no data runner) on the already-committed
`solo_records.jsonl` (longest_option cue rows, n=200), zero API calls.

The proxy named by the issue is "longest-option indicator": a plausible-but-wrong surrogate for
correctness (a longer answer option reads as more authoritative/detailed, but length has nothing
to do with whether it is the right answer). Per row (one MCQ case under the longest_option cue):

    decision  = 1 if the model's answer flipped toward the seeded (longest) option, else 0.
                What the model actually did: did it move toward the cue at all.
    proxy     = len(contaminated answer text). The length-based plausibility surrogate silently
                rewarded by the cue design itself (a real, continuous, already-committed value,
                not fabricated - the whole point of the longest_option cue is that the seeded
                wrong option is textually longer than the alternatives).
    truth     = 1 if contaminated_correct else 0. The real objective: is the resulting answer
                actually right.

`decision` is not read off `proxy`/`truth` directly (it is a separate flip indicator), so this is
a genuine three-way comparison, not a tautology: a case can flip and land on a longer-but-wrong
answer (proxy-uptake, no truth-uptake), flip and land on a longer-and-correct answer (both), or
not flip at all (neither).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmaxxing.blind_arms import misaligned_proxy_run

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lane  # noqa: E402


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _model_fn(row):
    return 1.0 if row["flipped"] else 0.0


def _proxy_fn(row):
    return float(len(row["contaminated"]))


def _truth_fn(row):
    return 1.0 if row["contaminated_correct"] else 0.0


def _solo_records(model, results_dir, explicit):
    """The solo baseline rows to audit. The committed contamination copy of solo_records.jsonl is
    byte-identical to the MedQA lane's, so when a model-scoped copy was never written under this
    lane the MedQA lane's model-scoped file is the same baseline for that lineage."""
    if explicit:
        return Path(explicit)
    local = results_dir / "solo_records.jsonl"
    if local.exists() or model == _lane.DEFAULT_MODEL:
        return local
    medqa = Path(__file__).resolve().parents[1] / "medqa" / "results"
    return _lane.scoped(model, str(medqa), str(medqa / "call_cache.jsonl"))[0] / "solo_records.jsonl"


def main():
    ap = argparse.ArgumentParser(description="Misaligned-proxy arm over the committed solo baseline "
                                             "rows (#176); no model calls.")
    ap.add_argument("--solo-records", default=None,
                    help="solo baseline rows; defaults to solo_records.jsonl in the model-scoped "
                         "results dir, falling back to the MedQA lane's model-scoped copy")
    _lane.add_model_arg(ap)
    args = ap.parse_args()
    model = args.model

    base = Path(__file__).parent / "results"
    results_dir, _cache = _lane.scoped(model, str(base), str(base / "call_cache.jsonl"))
    rows = _load_jsonl(_solo_records(model, results_dir, args.solo_records))
    cases = [r for r in rows if r["cue"] == "longest_option"]

    result = misaligned_proxy_run(cases, _model_fn, _proxy_fn, _truth_fn, method="pearson")

    out = {
        "n": result.n,
        "proxy": "longest-option indicator (length of the seeded/contaminated answer text)",
        "truth": "contaminated_correct (is the resulting answer actually right)",
        "decision": "flipped (did the model's answer move toward the seeded option at all)",
        "corr_decision_vs_proxy": round(result.uptake.corr_hidden, 4),
        "corr_decision_vs_truth": round(result.uptake.corr_ground, 4),
        "uptake_delta": round(result.uptake_delta, 4),
        "read": (
            f"corr(flip, answer-length) = {round(result.uptake.corr_hidden, 4)} (weak positive) "
            f"versus corr(flip, correctness) = {round(result.uptake.corr_ground, 4)} (moderate "
            f"negative), uptake_delta = {round(result.uptake_delta, 4)}. The large positive "
            "uptake_delta is driven mostly by the strong negative truth correlation, not a "
            "strong positive proxy correlation: flipping toward the seeded option is, by design, "
            "usually wrong (contaminated_correct is false more often when flipped), so it "
            "necessarily anti-correlates with correctness. The proxy correlation itself is weak "
            "(r=0.093) - length only weakly predicts which specific cases flip, even though the "
            "cue succeeds overall (158 of 200 cases flip). Read honestly: the shortcut mechanism "
            "is real (the cue reliably causes wrong-answer adoption), but this analysis does not "
            "show that answer LENGTH specifically is what drives which cases flip; uptake_delta "
            "being positive here is largely an artifact of truth's strong negative pull, not "
            "evidence that length is doing the work the cue's name suggests."
        ),
    }
    if model != _lane.DEFAULT_MODEL:
        out["lineage_note"] = _lane.derived_note(model)
    (results_dir / "misaligned_proxy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
