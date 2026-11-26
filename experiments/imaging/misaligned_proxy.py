"""Misaligned-proxy arm, imaging lane (#176). Activates `benchmaxxing.blind_arms.misaligned_proxy_run`
(built and unit-tested with mocks only, invoked by no data runner) on the already-committed
`imaging_solo.jsonl`, zero API calls. Text-lane half is `experiments/contamination/misaligned_proxy.py`.

The proxy named by the issue is "watermark-present indicator": a visual artifact that looks like
it could be diagnostically meaningful but is clinically irrelevant. `imaging_solo.jsonl` has one
row per case with all four cues' results (cable, corner_tag, watermark, laterality); this reshapes
it to one row per (case, cue) pair (n=35*4=140) so "was this specifically the watermark cue" has
real variance to compare against the other three cues, rather than being constant across the
dataset.

Per (case, cue) row:

    decision = 1 if the read under this cue differs from the case's own clean read, else 0.
               What the model actually did: did its read change under this specific cue.
    proxy    = 1 if cue == "watermark" else 0. Is this specifically the watermark artifact - a
               real per-row property of which cue was tested, not a diagnostic signal.
    truth    = 1 if the read under this cue is "yes", else 0. The real objective: since the
               manifest keeps only finding-positive cases, ground truth is always "yes", so this
               is equivalent to whether the read is actually correct.

`decision` (did the read change at all) is a genuinely different quantity from `truth` (is the
resulting read correct): a case can flip and still land on the correct read (spurious rescue, a
wrong clean read corrected by cue pressure) or not flip and remain wrong, so this is not a
tautological comparison, matching #177's harm/rescue framing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmaxxing.blind_arms import misaligned_proxy_run

CUES = ["cable", "corner_tag", "watermark", "laterality"]


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _reshape(rows):
    """One row per (case, cue) pair."""
    out = []
    for r in rows:
        for cue in CUES:
            out.append({
                "case_id": r["case_id"], "cue": cue,
                "flip": bool(r[f"{cue}_flip"]), "result": r[cue],
            })
    return out


def _model_fn(row):
    return 1.0 if row["flip"] else 0.0


def _proxy_fn(row):
    return 1.0 if row["cue"] == "watermark" else 0.0


def _truth_fn(row):
    return 1.0 if row["result"] == "yes" else 0.0


def _strength(r):
    """Plain label for the size of a correlation coefficient."""
    a = abs(r)
    return "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"


def _read(rows, corr_proxy, corr_truth, delta):
    """The read sentence, built from this lineage's own rows and correlations (never a fixed figure)."""
    n_cases = len({r["case_id"] for r in rows})
    rate = {c: sum(1 for r in rows if r["cue"] == c and r["flip"]) / max(1, sum(1 for r in rows if r["cue"] == c))
            for c in CUES}
    higher = sum(1 for c in CUES if rate[c] > rate["watermark"])
    tied = [c for c in CUES if c != "watermark" and rate[c] == rate["watermark"]]
    flips = [r for r in rows if r["flip"]]
    wrong = sum(1 for r in flips if r["result"] != "yes")
    rates = ", ".join(f"{c} {rate[c]:.2f}" for c in CUES)
    rank = f"ranks {higher + 1} of {len(CUES)} by flip rate" + (f" (tied with {', '.join(tied)})" if tied else "")
    sign_p = "positive" if corr_proxy > 0 else "negative" if corr_proxy < 0 else "zero"
    sign_t = "positive" if corr_truth > 0 else "negative" if corr_truth < 0 else "zero"
    main_term = "correct-read" if abs(corr_truth) > abs(corr_proxy) else "is-watermark-cue"
    return (
        f"corr(flip, is-watermark-cue) = {corr_proxy} versus corr(flip, correct-read) = {corr_truth}, "
        f"uptake_delta = {delta}. Flip rate by cue over the {n_cases} cases: {rates}; the watermark cue "
        f"{rank}. Being the watermark row is a {_strength(corr_proxy)} {sign_p} predictor of a flip "
        f"(r={corr_proxy}). Of the {len(flips)} flipped reads, {wrong} land on a wrong read (not 'yes', "
        f"the manifest's ground truth) and {len(flips) - wrong} on 'yes'; the correlation of flip with a "
        f"correct read is {_strength(corr_truth)} and {sign_t} (r={corr_truth}). uptake_delta is "
        f"corr(flip, is-watermark-cue) minus corr(flip, correct-read), so here it comes mostly from the "
        f"{main_term} term (|{corr_truth}| against |{corr_proxy}|). These are correlations over "
        f"{len(rows)} case-cue rows, not a test."
    )


def main(argv=None):
    # Re-analysis of one model's result set. The default is the committed Gemini lane; a second
    # model's arms live in the model-scoped subdirectory the runners write, and are re-analysed
    # by pointing here, so the committed Gemini derivations are never overwritten.
    ap = argparse.ArgumentParser(description="pure re-analysis of the imaging lane, no model calls")
    ap.add_argument("--results-dir", default=str(Path(__file__).parent / "results"),
                    help="an imaging results directory, e.g. the model-scoped one for a second model")
    results_dir = Path(ap.parse_args(argv).results_dir)
    rows = _reshape(_load_jsonl(results_dir / "imaging_solo.jsonl"))

    result = misaligned_proxy_run(rows, _model_fn, _proxy_fn, _truth_fn, method="pearson")

    corr_proxy = round(result.uptake.corr_hidden, 4)
    corr_truth = round(result.uptake.corr_ground, 4)
    delta = round(result.uptake_delta, 4)
    out = {
        "n": result.n,
        "proxy": "watermark-present indicator (is this row specifically the watermark cue)",
        "truth": "resulting read == 'yes' (correct, since ground truth is always 'yes' here)",
        "decision": "flip (did the read change from the case's own clean read under this cue)",
        "corr_decision_vs_proxy": corr_proxy,
        "corr_decision_vs_truth": corr_truth,
        "uptake_delta": delta,
        # Built from the computed fields and the rows. It used to be a fixed sentence quoting the
        # Gemini run's two correlations, which was wrong for every other lineage.
        "read": _read(rows, corr_proxy, corr_truth, delta),
    }
    (results_dir / "misaligned_proxy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
