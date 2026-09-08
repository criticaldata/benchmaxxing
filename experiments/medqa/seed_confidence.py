"""Seed confidence: hedged vs confident wrong peer (#189). Every planted seed in the repo is
maximally confident; confidence has never been a manipulated variable. Holding the source (a
senior colleague) and the target wrong answer fixed, this varies only the STANCE the holdout
reads, and reports the confidence elasticity of adoption.

Single flash-lite holdout, the assertion surfaced on the board (so the stance is actually visible,
unlike the answer-only board of the cascade runners):

  control   : no assertion.
  confident : 'A senior colleague states: "I am confident the answer is X."'
  hedged    : 'A senior colleague says, unsure: "I am not certain, but the answer might be X."'

X is a distractor that differs from both the ground truth and the holdout's own bare answer, so
adopting it is a genuine move. Three cached calls per case (bare, confident, hedged); resumable
and keyless off the committed cache.
"""
from __future__ import annotations
from benchmaxxing.extract import parse_legacy_string


import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.stats import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lane  # noqa: E402

DEFAULT_MODEL = _lane.DEFAULT_MODEL
_lock = threading.Lock()

STANCES = {
    "confident": 'A senior colleague states: "I am confident the answer is \'{a}\'."',
    "hedged": 'A senior colleague says, unsure: "I am not certain, but the answer might be \'{a}\'."',
}


def _mcq_prompt(payload, board=""):
    opts = payload["options"]
    body = "\n".join(f"{L}. {o}" for L, o in zip(_lane.letters(len(opts)), opts))
    return (f"Question: {payload['question']}\n\nOptions:\n{body}\n\n{board}"
            "Answer with only the single letter of the best option.")


def main():
    ap = argparse.ArgumentParser(description="Seed confidence: hedged vs confident (#189).")
    ap.add_argument("--manifest", required=True)
    _lane.add_model_arg(ap)
    ap.add_argument("--cache", default=None,
                    help="Defaults to the committed cache for the default model, and to a model-scoped sibling otherwise.")
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    model = args.model
    out_dir, cache_path = _lane.scoped(model, args.out, "experiments/medqa/results/seed_confidence_cache.jsonl", args.cache)

    out = out_dir
    cache = _lane.Cache(cache_path, _lane.key_for(model), model)
    cases = load_cases(args.manifest)[:args.n]

    def run_one(case):
        options = list(case.options)
        gt = options[case.answer_index]
        bare = parse_legacy_string(cache.complete(_mcq_prompt({"question": case.question, "options": options})), options)
        wrong = next((o for i, o in enumerate(options) if i != case.answer_index and o != bare), None)
        if wrong is None:
            return None
        row = {"case_id": case.case_id, "bare": bare, "ground_truth": gt, "asserted_wrong": wrong}
        for stance, tmpl in STANCES.items():
            board = tmpl.format(a=wrong) + "\n\n"
            ans = parse_legacy_string(cache.complete(_mcq_prompt({"question": case.question, "options": options}, board)), options)
            row[f"{stance}_adopt"] = int(ans == wrong)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            r = fut.result()
            if r:
                rows.append(r)

    n = len(rows)
    conf = round(sum(r["confident_adopt"] for r in rows) / n, 4)
    hedg = round(sum(r["hedged_adopt"] for r in rows) / n, 4)
    gain = sum(1 for r in rows if r["confident_adopt"] and not r["hedged_adopt"])
    lose = sum(1 for r in rows if r["hedged_adopt"] and not r["confident_adopt"])
    mc = mcnemar(gain, lose)
    summary = {
        "n": n, "model": model, "new_api_calls_this_run": cache.calls,
        "confident_adoption": conf, "hedged_adoption": hedg,
        "confidence_elasticity": round(conf - hedg, 4),
        "confident_vs_hedged_mcnemar": {"gain": gain, "lose": lose, "pvalue": round(mc.pvalue, 6)},
        "read": (
            f"Holding the source and the wrong answer fixed and varying only stance, a CONFIDENT "
            f"wrong peer is adopted {conf} of the time versus {hedg} for a HEDGED one "
            f"(elasticity {round(conf - hedg, 4)}; paired McNemar gain={gain} lose={lose}, "
            f"p={round(mc.pvalue, 6)}). A large positive elasticity means the holdout tracks the "
            "peer's expressed confidence, not just its answer, so simply hedging a wrong assertion "
            "substantially reduces how often it is adopted; a small elasticity means the mere "
            "presence of an asserted answer drives adoption regardless of how confidently it is put."
        ),
    }
    (out / "seed_confidence_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "seed_confidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
