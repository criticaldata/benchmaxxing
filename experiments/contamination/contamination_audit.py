"""MedQA contamination / memorization audit (issue #108): is the solo shortcut susceptibility
riding on memorization or dataset artifacts rather than genuine uncertainty?

Three probes per case (the 100-case solo set), for gemini-2.5-flash and -flash-lite:

  full         question + options (baseline; read from the committed solo_records).
  q_only       the question with NO options; can the model produce the correct answer from the
               stem alone? High accuracy = strong prior knowledge or memorization.
  options_only the options with NO question; can it pick the answer above chance? Accuracy well
               above 1/n = the options leak the answer (a dataset artifact), independent of skill.

Then the key correlation: split cases by baseline correctness and compare the cue flip rate with a
Fisher exact test. If the model resists the answer-preserving cue more on cases it gets right,
susceptibility is concentrated where the model does NOT already know the answer, so it is
uncertainty-driven, not a memorization artifact.

Reproduction: the `full` baseline comes from the committed solo_records; `q_only`/`options_only`
model calls are cached in ``--cache``. A fully cached run reproduces the summary with zero API
calls and no key. No secrets committed; all paths are arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.data import load_cases
from benchmaxxing.stats import fisher_exact

MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
_lock = threading.Lock()


def _key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _letters(n):
    return [chr(65 + i) for i in range(n)]


def _parse(text, options):
    letters = _letters(len(options))
    if text:
        m = re.search(r"\b([A-E])\b", text.strip().upper())
        if m and m.group(1) in letters:
            return options[letters.index(m.group(1))]
        for o in options:
            if o.lower() in text.lower():
                return o
    return (text or "").strip()


class _Cache:
    def __init__(self, path, key):
        self.path, self.key, self.store, self.calls = Path(path), key, {}, 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.store[r["k"]] = r["resp"]

    def complete(self, model, prompt):
        k = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
        with _lock:
            if k in self.store:
                return self.store[k]
        if not self.key:
            raise SystemExit("Cache miss and no GEMINI_API_KEY set (a fully cached run needs no key).")
        resp = gateway.RetryBackend(gateway.GeminiBackend(model=model, api_key=self.key),
                                    tries=5, backoff=3.0).complete(prompt, decoding={"temperature": 0})
        with _lock:
            self.store[k] = resp
            self.calls += 1
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "model": model, "resp": resp}) + "\n")
        return resp


def _load_solo(path):
    by = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by[(r["model"], r["case_id"])].append(r)
    out = {}
    for key, rows in by.items():
        flips = [1 if x["flipped"] else 0 for x in rows]
        out[key] = {"clean_correct": bool(rows[0].get("clean_correct")),
                    "ever_flipped": any(flips)}
    return out


def main():
    ap = argparse.ArgumentParser(description="MedQA contamination / memorization audit (#108).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--solo-records", default="experiments/contamination/results/solo_records.jsonl")
    ap.add_argument("--cache", default="experiments/contamination/results/call_cache.jsonl")
    ap.add_argument("--out", default="experiments/contamination/results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _Cache(args.cache, _key())
    solo = _load_solo(args.solo_records)
    case_ids = {cid for (_m, cid) in solo}
    by_id = {c.case_id: c for c in load_cases(args.manifest) if c.case_id in case_ids}

    def probe(model, case):
        opts = list(case.options)
        gt = opts[case.answer_index]
        body = "\n".join(f"{L}. {o}" for L, o in zip(_letters(len(opts)), opts))
        q_only_p = (f"Question: {case.question}\n\nState the single best answer in a few words "
                    "(no options are provided).")
        q_txt = cache.complete(model, q_only_p)
        opts_only_p = (f"Below are answer options with no question. Pick the one most likely to be "
                       f"the intended correct answer.\n\nOptions:\n{body}\n\n"
                       "Answer with only the single letter.")
        o_ans = _parse(cache.complete(model, opts_only_p), opts)
        return {"model": model, "case_id": case.case_id, "n_opts": len(opts),
                "q_only_correct": gt.lower() in (q_txt or "").lower(),
                "options_only_correct": o_ans == gt}

    jobs = [(m, by_id[cid]) for (m, cid) in solo if cid in by_id]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(probe, m, c) for m, c in jobs]):
            rows.append(fut.result())

    summary = {"new_api_calls_this_run": cache.calls, "by_model": {}}
    for model in MODELS:
        mrows = [r for r in rows if r["model"] == model]
        if not mrows:
            continue
        n = len(mrows)
        model_cases = [(m, cid) for (m, cid) in solo if m == model]
        full_acc = sum(1 for k in model_cases if solo[k]["clean_correct"]) / len(model_cases)
        q_acc = sum(r["q_only_correct"] for r in mrows) / n
        o_acc = sum(r["options_only_correct"] for r in mrows) / n
        chance = sum(1.0 / r["n_opts"] for r in mrows) / n
        cw = defaultdict(lambda: [0, 0])  # baseline-correct? -> [ever_flipped, not]
        for k in model_cases:
            cw[solo[k]["clean_correct"]][0 if solo[k]["ever_flipped"] else 1] += 1
        fe = fisher_exact([[cw[True][0], cw[True][1]], [cw[False][0], cw[False][1]]])
        summary["by_model"][model] = {
            "n": n, "full_accuracy": round(full_acc, 4),
            "q_only_accuracy": round(q_acc, 4), "options_only_accuracy": round(o_acc, 4),
            "options_only_chance": round(chance, 4), "options_only_above_chance": round(o_acc - chance, 4),
            "flip_rate_when_baseline_correct": round(cw[True][0] / max(1, sum(cw[True])), 4),
            "flip_rate_when_baseline_wrong": round(cw[False][0] / max(1, sum(cw[False])), 4),
            "flip_correct_vs_wrong_fisher_p": fe.pvalue, "flip_correct_vs_wrong_or": fe.oddsratio}
    (out / "contamination_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "contamination_audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
