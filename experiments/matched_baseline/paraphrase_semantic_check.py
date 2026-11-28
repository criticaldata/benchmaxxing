"""Semantic check of the committed paraphrases, and the bands without the paraphrases that fail it.

Added after review of ``paraphrase_band.py``. Its fidelity check only compares numbers: a paraphrase
that dropped a negation would pass it, change the right answer, and widen the band. This script
puts two checks on every committed paraphrase of a lane and recomputes every lineage's band with
the failing paraphrases left out.

  rule   keyless. In the question part of the text (the sentences that end with a question mark,
         each with the sentence before it when it has fewer than five words, or the last sentence
         when there is no question mark) a marker of negation or exception (not, no, except, other
         than, least, false, incorrect and the like; ``MARKERS``) must be present in the paraphrase
         if and only if one is present in the original. "dropped" and "added" both fail. It
         compares presence, not the word, because a faithful rewording may turn "except" into
         "with one exception" or "is not true".
  judge  gemini-2.5-flash at temperature 0 is shown the original, the paraphrase and the answer
         options, and asked whether the paraphrase asks the same question, so that each option is
         right or wrong exactly as for the original. It answers yes or no with one line of reason.
         A reply with no verdict counts as not confirmed.

A paraphrase fails when either check fails. Read i of a case is its paraphrase i, so leaving a
paraphrase out removes that read from the case. Bands are computed by
``paraphrase_band.band_of_reads`` for every lineage folder of the lane that has a paraphrase band:
with all paraphrases (the committed band), with the failed ones left out, and on the cases whose
paraphrases all pass, each over reads 0 to K and over reads 1 to K.

Control, added after the first results and before its own calls. The judge said yes to nearly every
paraphrase, which shows little unless it can also say no. So every paraphrase whose question part
kept a marker is turned around by rule (``reverse_question``: "not true" to "true", "least" to
"most", "with one exception" to "with no exception", a closing "except" dropped) and shown to the
judge against the original. A judge that works says no to all of them; ``judge_control`` in the
summary counts it, and counts how many the keyless rule also refuses.

Writes, in the lane's results folder: ``paraphrase_semantic_check.jsonl`` (one row per paraphrase),
``paraphrase_semantic_failures.jsonl`` (the failing ones, with their text),
``paraphrase_semantic_judge_control.jsonl`` (the reversed questions and what the judge said) and
``paraphrase_band_semantic_summary.json``. Judge calls are cached in
``paraphrase_semantic_judge_cache.jsonl`` and ``paraphrase_semantic_judge_control_cache.jsonl``; a
fully cached run needs no key. ``--dry-run`` counts the calls a run would make.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmaxxing.data import load_cases

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402
import paraphrase_band as pb  # noqa: E402

JUDGE = "gemini-2.5-flash"
# Fixed before the judge was called and before any band was recomputed. Presence of any of these in
# the question part is what the rule compares.
MARKERS = {
    "negation": r"\b(?:not|no|never|none|neither|nor|without|cannot)\b|n['’]t\b",
    "exception": (r"\b(?:except|excepting|exceptions?|excluding|excludes?|excluded|other than|apart from|"
                  r"aside from|save for|all but)\b"),
    "least": r"\b(?:least|lowest|fewest)\b",
    "false": (r"\b(?:false|incorrect|incorrectly|untrue|wrong|wrongly|inaccurate|unlikely|inappropriate|"
              r"uncommon|contraindicated)\b"),
}
_MARKER = re.compile("|".join(MARKERS.values()), re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.?!])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
_VERDICT = re.compile(r"^\W*verdict\W*(yes|no)\b", re.IGNORECASE | re.MULTILINE)
_REASON = re.compile(r"^\W*reason\W*(\w.*)$", re.IGNORECASE | re.MULTILINE)
CHECK_FILE = "paraphrase_semantic_check.jsonl"
FAILURES_FILE = "paraphrase_semantic_failures.jsonl"
SUMMARY_FILE = "paraphrase_band_semantic_summary.json"
JUDGE_CACHE = "paraphrase_semantic_judge_cache.jsonl"
CONTROL_FILE = "paraphrase_semantic_judge_control.jsonl"
CONTROL_CACHE = "paraphrase_semantic_judge_control_cache.jsonl"


# ----------------------------------------------------------------------------- rule check

def _question_spans(text):
    """(start, end) of the sentences that make the question part of ``text``."""
    text = text or ""
    cuts = [0] + [m.end() for m in _SENTENCE_BREAK.finditer(text)] + [len(text)]
    spans = [(a, b) for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]
    asked = []
    for j, (a, b) in enumerate(spans):
        sentence = text[a:b].strip()
        if sentence.rstrip("\"')] ").endswith("?"):
            if len(sentence.split()) < 5 and j > 0 and spans[j - 1] not in asked:
                asked.append(spans[j - 1])
            asked.append((a, b))
    return asked or spans[-1:]


def question_part(text):
    """The sentences that end with a question mark; the last sentence when there is none.

    A question of fewer than five words ("Which one?") leans on the sentence before it, so that
    sentence is read too.
    """
    return " ".join((text or "")[a:b].strip() for a, b in _question_spans(text))


def markers(text):
    """The negation and exception markers in the question part, lower case, in order of appearance."""
    return [" ".join(m.group(0).lower().split()) for m in _MARKER.finditer(question_part(text))]


def rule_check(original, paraphrase):
    """Presence of a marker in the question part must match between original and paraphrase."""
    mo, mp = markers(original), markers(paraphrase)
    if bool(mo) == bool(mp):
        reason = "kept" if mo else "none"
    else:
        reason = "dropped" if mo else "added"
    return {"rule_check": bool(mo) == bool(mp), "rule_reason": reason,
            "markers_original": mo, "markers_paraphrase": mp}


# ----------------------------------------------------------------------------- reversed questions (control)
# Added after the first results of the check and before its own calls. A reversal turns the question
# of a paraphrase around by rule, so the judge and the rule are also read on rewordings that are
# known to ask something else. Only a marker with an entry here is reversed; the first one found in
# the question part decides.
OPPOSITE = {
    "not": "", "never": "always", "without": "with", "cannot": "can",
    "least": "most", "lowest": "highest", "fewest": "most",
    "false": "true", "untrue": "true", "incorrect": "correct", "incorrectly": "correctly", "wrong": "right",
    "wrongly": "rightly", "inaccurate": "accurate", "unlikely": "likely", "inappropriate": "appropriate",
    "uncommon": "common", "contraindicated": "indicated",
}
_ONE_EXCEPTION = re.compile(r"\bwith (?:one|a single|only one|just one|a sole) exception\b", re.IGNORECASE)
_TRAILING_EXCEPT = re.compile(r",?\s*\bexcept(?:\s+(?:for\s+)?one)?(?=[\s?:.\u2013-]*$)", re.IGNORECASE)
_NT = re.compile(r"\b(is|are|was|were|does|do|did|has|have|had|should|would|could)n['\u2019]t\b", re.IGNORECASE)


def _like(original, replacement):
    """``replacement`` in the letter case of ``original``."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    return replacement.capitalize() if original[:1].isupper() else replacement


def reverse_question(text):
    """``text`` with the first reversible marker of its question part turned around, or None.

    Returns {"text", "marker", "replacement"}. "not" and "n't" are removed, a word of OPPOSITE is
    swapped for its opposite, "with one exception" becomes "with no exception", and an "except"
    that ends the question is dropped.
    """
    for a, b in _question_spans(text):
        sentence = text[a:b]
        for m in _MARKER.finditer(sentence):
            marker = " ".join(m.group(0).lower().split())
            edit = None
            if marker in ("exception", "exceptions"):
                phrase = next((x for x in _ONE_EXCEPTION.finditer(sentence) if x.start() <= m.start() < x.end()), None)
                if phrase:
                    edit = (phrase.start(), phrase.end(), "with no exception")
            elif marker == "except":
                tail = _TRAILING_EXCEPT.search(sentence)
                if tail and tail.start() <= m.start() < tail.end():
                    edit = (tail.start(), tail.end(), "")
            elif marker in ("n't", "n\u2019t"):
                nt = next((x for x in _NT.finditer(sentence) if x.start() <= m.start() < x.end()), None)
                if nt:
                    edit = (nt.start(), nt.end(), nt.group(1))
            elif marker == "not":
                end = m.end() + (1 if sentence[m.end():m.end() + 1] == " " else 0)
                edit = (m.start(), end, "")
            elif marker in OPPOSITE:
                edit = (m.start(), m.end(), _like(m.group(0), OPPOSITE[marker]))
            if edit:
                start, end, replacement = edit
                reversed_text = text[:a] + sentence[:start] + replacement + sentence[end:] + text[b:]
                return {"text": re.sub(r"[ \t]{2,}", " ", reversed_text), "marker": marker,
                        "replacement": replacement}
    return None


# ----------------------------------------------------------------------------- judge

def judge_prompt(original, paraphrase, options):
    body = "\n".join(f"{L}. {o}" for L, o in zip(_lane.letters(len(options)), options))
    return (
        "You are checking a reworded question for a study of how stable a model's answer is under rewording.\n\n"
        "ORIGINAL and REWORDED are two versions of one multiple-choice question. The answer options are the same "
        "for both and are listed once.\n\n"
        "Does REWORDED ask the same question as ORIGINAL? Answer yes only if all of these hold:\n"
        "- every fact, value and unit of ORIGINAL is in REWORDED, and none is added or changed;\n"
        "- the thing asked is the same, including any negation or exception (for example \"not\", \"except\", "
        "\"least\", \"false\", \"incorrect\");\n"
        "- so each option is right or wrong for REWORDED exactly as it is for ORIGINAL.\n"
        "Differences in wording, order or sentence structure alone do not matter.\n\n"
        "Reply with exactly two lines:\n"
        "VERDICT: yes or no\n"
        "REASON: one short sentence\n\n"
        f"ORIGINAL:\n{original}\n\nREWORDED:\n{paraphrase}\n\nOPTIONS:\n{body}"
    )


def parse_judge(reply):
    """(verdict, reason): 'yes' or 'no', or None when the reply carries no verdict line."""
    v = _VERDICT.search(reply or "")
    r = _REASON.search(reply or "")
    return (v.group(1).lower() if v else None), (" ".join(r.group(1).split()) if r else "")


# ----------------------------------------------------------------------------- bands

def _band(records, failed, k):
    """Bands over reads 0..K and 1..K of ``records``, leaving out the reads whose paraphrase is in ``failed``."""
    def kept(first):
        return lambda r: [i for i in range(first, k + 1) if i == 0 or (pb._case_key(r), i) not in failed]
    return {"reads_0_to_k": pb.band_of_reads(records, kept(0)), "reads_1_to_k": pb.band_of_reads(records, kept(1))}


def lineage_bands(records, checks, k):
    """Every band of one lineage: all paraphrases, failed ones left out (either, rule, judge), clean cases."""
    mine = {pb._case_key(r) for r in records}
    fails = {name: {key for key, c in checks.items() if key[0] in mine and not c[name]}
             for name in ("rule", "judge", "either")}
    clean = [r for r in records if not any((pb._case_key(r), i) in fails["either"] for i in range(1, k + 1))]
    return {
        "cases": len(records), "paraphrase_reads": len(records) * k,
        "reads_left_out": {name: len(f) for name, f in fails.items()},
        "all_paraphrases": _band(records, set(), k),
        "failed_left_out": _band(records, fails["either"], k),
        "rule_failed_left_out": _band(records, fails["rule"], k),
        "judge_failed_left_out": _band(records, fails["judge"], k),
        "cases_with_every_paraphrase_passing": {"n": len(clean), **_band(clean, set(), k)},
    }


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def band_folders(results):
    """The lane folder and every lineage subfolder that holds a paraphrase band."""
    results = Path(results)
    return [results] + sorted(p.parent for p in results.glob("*/paraphrase_band.jsonl"))


def control_summary(check_rows, control_rows):
    """How the judge and the rule read the reversed questions: both should refuse every one."""
    kept = sum(1 for r in check_rows if r["rule_reason"] == "kept")
    said = Counter(r["judge_verdict"] for r in control_rows)
    rule_refused = sum(1 for r in control_rows if r["rule_reason"] in ("dropped", "added"))
    return {
        "paraphrases_with_a_marker_kept": kept, "reversed": len(control_rows),
        "not_reversible_by_the_table": kept - len(control_rows),
        "judge_said_no": said["no"], "judge_said_yes": said["yes"], "judge_gave_no_verdict": said[None],
        "rule_refused": rule_refused,
        "refused_by_judge_or_rule": sum(1 for r in control_rows
                                        if r["judge_verdict"] != "yes" or r["rule_reason"] in ("dropped", "added")),
        "reversed_by_marker": dict(Counter(r["marker"] for r in control_rows)),
        "file": CONTROL_FILE,
        "note": ("Control added after the first results. Each paraphrase whose question part kept a marker was "
                 "turned around by rule (reverse_question) and shown to the judge against the original; a correct "
                 "judge says no to all of them. rule_refused counts those the keyless rule check also fails."),
    }


def summarise(check_rows, folders, calls, calls_in_cache, control_rows=None):
    """The summary from the per-paraphrase rows and each lineage's committed band rows.

    ``calls`` and ``calls_in_cache`` are the recorded call counts, {"judge": n, "judge_control": m}.
    """
    checks = {(r["case_id"], r["i"]): {"rule": r["rule_check"], "judge": r["judge_verdict"] == "yes",
                                       "either": r["rule_check"] and r["judge_verdict"] == "yes"}
              for r in check_rows}
    n = len(check_rows)
    bands = {}
    for folder in folders:
        s = json.loads((folder / "paraphrase_band_summary.json").read_text())
        k = s["paraphrases_per_case"]
        records = [r for r in _rows(folder / "paraphrase_band.jsonl") if r["n_paraphrases"] == k]
        bands[s["model"]] = {"folder": folder.name if folder != folders[0] else ".", **lineage_bands(records, checks, k)}
    summary = {
        "judge": JUDGE, "paraphrases": n,
        "failed": {
            "rule": sum(1 for c in checks.values() if not c["rule"]),
            "rule_by_reason": dict(Counter(r["rule_reason"] for r in check_rows if not r["rule_check"])),
            "judge": sum(1 for c in checks.values() if not c["judge"]),
            "judge_said_no": sum(1 for r in check_rows if r["judge_verdict"] == "no"),
            "judge_gave_no_verdict": sum(1 for r in check_rows if r["judge_verdict"] is None),
            "either": sum(1 for c in checks.values() if not c["either"]),
            "both": sum(1 for c in checks.values() if not c["rule"] and not c["judge"]),
        },
        "cases_with_a_failed_paraphrase": len({key[0] for key, c in checks.items() if not c["either"]}),
        "failures_file": FAILURES_FILE,
        "new_api_calls_this_run": dict(calls), "calls_in_cache": dict(calls_in_cache),
        "bands": bands,
        "note": ("A paraphrase fails when the rule check or the judge fails it. failed_left_out removes those reads "
                 "from each case (read 0, the committed control prompt, is always kept in reads_0_to_k); "
                 "cases_with_every_paraphrase_passing keeps whole cases instead. all_paraphrases is the committed "
                 "band, recomputed from the same rows."),
    }
    if control_rows is not None:
        summary["judge_control"] = control_summary(check_rows, control_rows)
    return summary


# ----------------------------------------------------------------------------- run

def main(argv=None):
    ap = argparse.ArgumentParser(description="Semantic check of the committed paraphrases of one lane.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True, help="the lane's results folder (holds paraphrases.jsonl)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="count the judge calls this run would make, and stop")
    args = ap.parse_args(argv)

    results = Path(args.results)
    cases = {c.case_id: c for c in load_cases(args.manifest)}
    paras = _rows(results / "paraphrases.jsonl")
    absent = sorted({p["case_id"] for p in paras} - set(cases), key=str)
    if absent:
        raise SystemExit(f"paraphrased cases missing from the manifest: {absent[:5]}")
    cache = _lane.Cache(results / JUDGE_CACHE, _lane.key_for(JUDGE), JUDGE)
    control_cache = _lane.Cache(results / CONTROL_CACHE, _lane.key_for(JUDGE), JUDGE)
    prompts = [judge_prompt(cases[p["case_id"]].question, p["paraphrase"], list(cases[p["case_id"]].options))
               for p in paras]
    # the control: every paraphrase whose question part kept a marker, turned around by rule
    reversals = []
    for p in paras:
        original = cases[p["case_id"]].question
        if rule_check(original, p["paraphrase"])["rule_reason"] == "kept":
            turned = reverse_question(p["paraphrase"])
            if turned:
                reversals.append((p, turned, judge_prompt(original, turned["text"], list(cases[p["case_id"]].options))))
    if args.dry_run:
        def missing(store, qs):
            return sum(1 for q in qs if hashlib.sha256(f"{JUDGE}\x00{q}".encode()).hexdigest() not in store)
        need, need_control = missing(cache.store, prompts), missing(control_cache.store, [q for _, _, q in reversals])
        print(json.dumps({"lane_results": str(results), "paraphrases": len(paras), "judge": JUDGE,
                          "cached": len(paras) - need, "calls_needed": need,
                          "control_reversals": len(reversals), "control_calls_needed": need_control}, indent=2))
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        replies = list(ex.map(cache.complete, prompts))
        control_replies = list(ex.map(control_cache.complete, [q for _, _, q in reversals]))

    check_rows, failures = [], []
    for p, reply in zip(paras, replies):
        verdict, reason = parse_judge(reply)
        row = {"case_id": p["case_id"], "i": p["i"],
               **rule_check(cases[p["case_id"]].question, p["paraphrase"]),
               "judge": JUDGE, "judge_verdict": verdict, "judge_reason": reason}
        check_rows.append(row)
        failed = [name for name, ok in (("rule", row["rule_check"]), ("judge", verdict == "yes")) if not ok]
        if failed:
            failures.append({**row, "failed": " and ".join(failed), "paraphrase": p["paraphrase"]})
    control_rows = []
    for (p, turned, _), reply in zip(reversals, control_replies):
        verdict, reason = parse_judge(reply)
        control_rows.append({"case_id": p["case_id"], "i": p["i"], "marker": turned["marker"],
                             "replacement": turned["replacement"], "reversed": turned["text"],
                             "rule_reason": rule_check(cases[p["case_id"]].question, turned["text"])["rule_reason"],
                             "judge": JUDGE, "judge_verdict": verdict, "judge_reason": reason})

    def lines(name):
        return len(_rows(results / name)) if (results / name).exists() else 0

    summary = summarise(check_rows, band_folders(results),
                        {"judge": cache.calls, "judge_control": control_cache.calls},
                        {"judge": lines(JUDGE_CACHE), "judge_control": lines(CONTROL_CACHE)}, control_rows)

    (results / CHECK_FILE).write_text("".join(json.dumps(r) + "\n" for r in check_rows))
    (results / FAILURES_FILE).write_text("".join(json.dumps(r) + "\n" for r in failures))
    (results / CONTROL_FILE).write_text("".join(json.dumps(r) + "\n" for r in control_rows))
    (results / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "bands"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
