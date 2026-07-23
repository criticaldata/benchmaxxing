"""Net-harm sign of the imaging cascade (#177): does the cascade flip CORRECT solo reads to
WRONG (harm), or WRONG solo reads to CORRECT (spurious rescue)? Pure re-analysis, no API calls.

Every case in the manifest is a real positive for its finding (the manifest is filtered to cases
with a real finding label, never "no finding"), so ground truth for "does this X-ray show
{finding}?" is always "yes". The cascade scripts define ``wrong`` as the opposite of the holdout's
own uncued ``clean`` read, so which direction "adopting wrong" moves the holdout depends on
whether its own baseline read was itself correct:

  clean == "yes" (baseline correct):   wrong == "no".  shared_adopt == 1 means the holdout moved
                                        from correct to incorrect: HARM.
  clean == "no"  (baseline incorrect): wrong == "yes". shared_adopt == 1 means the holdout moved
                                        from incorrect to correct (ground truth): SPURIOUS RESCUE
                                        (driven by peer pressure landing on the right answer, not
                                        by genuine clinical reasoning).

Reports harm rate and spurious-rescue rate per cue, each with a Wilson 95% CI and a bootstrap CI,
plus a Fisher exact test comparing the two rates (two independent groups: baseline-correct cases
vs baseline-incorrect cases), for every cue that has a cascade result (watermark, cable, corner_tag,
laterality).

Note on the MedQA text lane (also asked for in #177): the only per-case artifact with ground truth
(cascade_v2_per_case.jsonl) predates the answer-parser fix and is confirmed tainted by it (its
medqa-82 row shows the exact "Desmoplastic" stray-article mis-parse documented in the parser-bug
correction), and the stored transcripts only retain a truncated completion (~120-160 chars, cut
before any boxed final answer), so it cannot be re-parsed with the fixed parser either. A valid
text-lane net-harm analysis needs a fresh run, not zero-cost re-analysis; tracked separately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from benchmaxxing.stats import bootstrap_ci, fisher_exact

CUES = ["cable", "corner_tag", "watermark", "laterality"]


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cue_file(results_dir, cue):
    suffix = "" if cue == "watermark" else f"_{cue}"
    return Path(results_dir) / f"imaging_cascade{suffix}.jsonl"


def net_harm_for_cue(rows):
    harm_group = [r["shared_adopt"] for r in rows if r["clean"] == "yes"]
    rescue_group = [r["shared_adopt"] for r in rows if r["clean"] == "no"]

    def summarize(group):
        if not group:
            return {"n": 0, "rate": None, "wilson95": [None, None], "bootstrap95": [None, None]}
        k, n = sum(group), len(group)
        lo, hi = _wilson(k, n)
        _point, blo, bhi = bootstrap_ci(list(map(float, group)))
        return {"n": n, "rate": round(k / n, 4), "wilson95": [lo, hi],
                "bootstrap95": [round(blo, 4), round(bhi, 4)]}

    harm = summarize(harm_group)
    rescue = summarize(rescue_group)

    fe = None
    fe_note = None
    if harm_group and rescue_group:
        harm_k, harm_n = sum(harm_group), len(harm_group)
        res_k, res_n = sum(rescue_group), len(rescue_group)
        fe = fisher_exact([[harm_k, harm_n - harm_k], [res_k, res_n - res_k]])
        if math.isnan(fe.oddsratio) or math.isinf(fe.oddsratio):
            fe_note = ("odds ratio is undefined (a table cell is zero, e.g. both harm and "
                       "rescue rates are exactly 1.0); the p-value above is still valid.")

    result = {
        "harm_rate_correct_to_wrong": harm,
        "spurious_rescue_rate_wrong_to_correct": rescue,
        "harm_vs_rescue_fisher": (
            {"pvalue": round(fe.pvalue, 4),
             "oddsratio": (None if fe_note else round(fe.oddsratio, 4))} if fe else None
        ),
    }
    if fe_note:
        result["harm_vs_rescue_fisher"]["note"] = fe_note
    return result


def main():
    results_dir = Path(__file__).parent / "results"
    per_cue = {}
    for cue in CUES:
        f = cue_file(results_dir, cue)
        if not f.exists():
            continue
        per_cue[cue] = net_harm_for_cue(_load_jsonl(f))

    out = {
        "lane": "NIH imaging",
        "ground_truth_convention": "every case is a real finding-positive; gt = 'yes' always",
        "per_cue": per_cue,
        "text_lane_note": (
            "Not computed: the only per-case MedQA artifact with ground truth predates the "
            "answer-parser fix and is tainted by it; stored transcripts retain only a truncated "
            "completion, insufficient to re-parse. Needs a fresh run, tracked separately."
        ),
    }
    (results_dir / "net_harm.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
