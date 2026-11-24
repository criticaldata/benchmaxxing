"""Paraphrase noise band for the matched control: how much does the answer move when the case is reworded?

The committed noise band (``matched_baseline_reason.py --repeats 5``) re-asks the SAME control prompt
five times at the same settings. That measures session instability only. A band defined by
paraphrases asks a different question: if the case were worded differently, with the same facts,
how often would the answer change? A cue effect that sits inside that band is not distinguishable
from rewording.

Per case:

  read 0       the committed control prompt (``reason_only`` of matched_baseline_reason.py), replayed
               from the lane's committed cache, no new call.
  reads 1..K   the same prompt with the question text replaced by one of K paraphrases. The options,
               the clinical-context prefix, the image and the instruction line are unchanged, so
               option letters and the decoy keep their meaning.

The paraphrases are written by a generator model (gemini-2.5-flash, temperature 0), one call per
case asking for K rewordings that keep every fact, number and the question asked. They are written
to ``paraphrases.jsonl`` with a simple fidelity check per paraphrase: every number in the original
must appear in the paraphrase (text lanes), or the finding must still be named (imaging lane).

The summary has the same fields as the committed repeats summary, computed by the same function,
plus the committed same-session band next to it. Every call is cached in the lane's results
folder; a fully cached run needs no key.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from benchmaxxing.data import load_cases
from benchmaxxing.extract import parse_legacy_string, parse_yesno

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402
import matched_baseline_reason as mb  # noqa: E402

GENERATOR = "gemini-2.5-flash"
_NUM = re.compile(r"\d+(?:\.\d+)?")


def generator_prompt(text, k):
    return (
        f"Rewrite the text below in {k} different ways. Each rewrite must keep every fact, value, unit and "
        "the question asked, exactly as stated, and must not add, remove or interpret any information or hint "
        "at an answer. Change only the wording and the sentence structure. Return only a JSON array of "
        f"{k} strings, nothing else.\n\nText:\n{text}"
    )


def parse_paraphrases(reply, k):
    """The K strings of the generator's JSON array, or [] when the reply is not one."""
    if not reply:
        return []
    s = reply.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if not m:
            return []
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(arr, list):
        return []
    out = [str(x).strip() for x in arr if isinstance(x, str) and x.strip()]
    return out[:k]


def numbers_kept(original, paraphrase):
    want = _NUM.findall(original)
    return all(n in paraphrase for n in want)


def text_reason_prompt(case, question):
    """matched_baseline_reason.text_prompts' reason_only prompt with the question text swapped."""
    opts = list(case.options)
    letters = _lane.letters(len(opts))
    body = "\n".join(f"{L}. {o}" for L, o in zip(letters, opts))
    report = (getattr(case, "report", "") or "").strip()
    head = f"Clinical context: {report}\n\n" if report else ""
    tail = ("In one short sentence, justify your choice, then on a new line answer with the "
            "single letter of your chosen option.")
    return f"{head}Question: {question}\n\nOptions:\n{body}\n\n{tail}"


def imaging_question(finding):
    return f"Does this X-ray show {finding}?"


def imaging_reason_prompt(question):
    """matched_baseline_reason.imaging_prompts' reason_only prompt with the question sentence swapped."""
    return f"{question} In one short sentence, justify your read, then on a new line answer with only 'yes' or 'no'."


def main():
    ap = argparse.ArgumentParser(description="Paraphrase noise band for the matched control.")
    ap.add_argument("--lane", choices=("text", "imaging"), required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--cache", required=True, help="the lane's committed call cache (read only here)")
    ap.add_argument("--out", required=True, help="results directory of the lane")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=None, help="defaults to 40 (text) or 35 (imaging), as the committed band")
    ap.add_argument("--k", type=int, default=5, help="paraphrases per case")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model = args.model or (mb.TEXT_MODEL if args.lane == "text" else mb.IMAGING_MODEL)
    n = args.n or (40 if args.lane == "text" else 35)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    committed = mb._Cache(args.cache, None, model)
    cache = mb._Cache(out / "paraphrase_band_cache.jsonl", _lane.key_for(model), model)
    gen = _lane.Cache(out / "paraphrase_band_generator_cache.jsonl", _lane.key_for(GENERATOR), GENERATOR)

    if args.lane == "text":
        cases = load_cases(args.manifest)[:n]
    else:
        root = Path(args.image_root)
        cases = [c for c in load_cases(args.manifest)
                 if c.label and c.label.lower() not in ("no finding", "", "none")
                 and (root / c.image_ref).exists()][:n]

    records, paras_log = {}, []

    def one(case):
        if args.lane == "text":
            opts, letters, base_p, reason_p, _ = mb.text_prompts(case)
            base_txt = committed.peek(base_p)
            if base_txt is None:
                raise SystemExit(f"{case.case_id}: committed baseline missing from {args.cache}")
            base_ans = parse_legacy_string(base_txt, opts)
            decoy = next((o for i, o in enumerate(opts) if i != case.answer_index and o != base_ans), None)
            if decoy is None:
                return None  # the committed control skips these cases too
            reason_txt = committed.peek(reason_p)
            if reason_txt is None:
                raise SystemExit(f"{case.case_id}: committed control missing from {args.cache}")
            letter = lambda a: letters[opts.index(a)] if a in opts else None  # noqa: E731
            original = case.question
            paras = parse_paraphrases(gen.complete(generator_prompt(original, args.k)), args.k)
            reads = [letter(parse_legacy_string(reason_txt, opts))]
            for p in paras:
                reads.append(letter(parse_legacy_string(cache.ask(text_reason_prompt(case, p)), opts)))
            kept = [numbers_kept(original, p) for p in paras]
            rec = mb._repeat_record(case.case_id, reads[0], letter(base_ans), letters[opts.index(decoy)], reads[1:])
        else:
            finding = case.label.split("|")[0].strip()
            img = Image.open(Path(args.image_root) / case.image_ref).convert("L")
            base_p, reason_p, _ = mb.imaging_prompts(finding)
            base_txt = committed.peek(base_p, img)
            reason_txt = committed.peek(reason_p, img)
            if base_txt is None or reason_txt is None:
                raise SystemExit(f"{case.case_id}: committed control missing from {args.cache} "
                                 "(the films on hand are not the bytes the committed cache was made from)")
            base_ans = parse_yesno(base_txt)
            decoy = "no" if base_ans == "yes" else "yes"
            original = imaging_question(finding)
            paras = parse_paraphrases(gen.complete(generator_prompt(original, args.k)), args.k)
            reads = [parse_yesno(reason_txt)]
            for p in paras:
                reads.append(parse_yesno(cache.ask(imaging_reason_prompt(p), img)))
            kept = [finding.lower() in p.lower() for p in paras]
            rec = mb._repeat_record(case.case_id, reads[0], base_ans, decoy, reads[1:])
        rec["n_paraphrases"] = len(paras)
        rec["fidelity_check"] = kept
        log = [{"case_id": case.case_id, "i": i + 1, "generator": GENERATOR, "paraphrase": p,
                "fidelity_check": kept[i]} for i, p in enumerate(paras)]
        return case.case_id, rec, log, original

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(one, c) for c in cases]):
            r = fut.result()
            if r:
                records[r[0]] = r[1]
                paras_log.extend(r[2])

    order = sorted(records, key=str)
    recs = [records[k] for k in order]
    full = [r for r in recs if r["n_paraphrases"] == args.k]
    band = mb.summarise_repeats(full, args.k)
    band["note"] = ("read 0 is the committed control prompt replayed from the lane cache; reads 1.. are the same "
                    "prompt with the question reworded by the generator, one paraphrase per read")
    clean = [r for r in full if all(r["fidelity_check"])]
    if clean:
        sub = mb.summarise_repeats(clean, args.k)
        band["fidelity_clean_subset"] = {k: sub[k] for k in (
            "n", "cases_with_any_disagreement", "mean_pairwise_disagreement", "flip_vs_baseline_band", "decoy_band")}
        band["fidelity_clean_subset"]["note"] = ("cases whose paraphrases all pass the fidelity check (every number of "
                                                 "the original kept verbatim; the finding still named)")
    band.update({
        "model": model, "generator": GENERATOR, "paraphrases_per_case": args.k,
        "cases_run": len(recs), "cases_with_all_paraphrases": len(full),
        "paraphrases_failing_fidelity_check": sum(1 for p in paras_log if not p["fidelity_check"]),
        "paraphrases_total": len(paras_log),
        "new_api_calls_this_run": {"holdout": cache.calls, "generator": gen.calls},
        # what the run that filled the caches paid, stable across keyless replays
        "calls_in_cache": {"holdout": len(cache.store), "generator": len(gen.store)},
    })
    same = out / "matched_baseline_reason_repeats_summary.json"
    if same.exists():
        s = json.loads(same.read_text())
        band["committed_same_session_band"] = {k: s.get(k) for k in (
            "n", "repeats", "cases_with_any_disagreement", "mean_pairwise_disagreement",
            "flip_vs_baseline_band", "decoy_band")}
    (out / "paraphrase_band.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (out / "paraphrases.jsonl").write_text("".join(json.dumps(p) + "\n" for p in
                                                   sorted(paras_log, key=lambda p: (str(p["case_id"]), p["i"]))))
    (out / "paraphrase_band_summary.json").write_text(json.dumps(band, indent=2))
    print(json.dumps(band, indent=2))


if __name__ == "__main__":
    main()
