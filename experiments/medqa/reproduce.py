"""Reproduce the MedQA (Lane B) experiments: solo susceptibility + noise floor + cascade.

This is the sanitized, repo-committed version of the first real-model runs (see this folder's
README and the saved outputs under ``results/``). It reads the model key from the environment,
takes all paths as arguments, and holds no secrets.

Usage
-----
    export GEMINI_API_KEY=...           # your key; never commit it
    python -m experiments.medqa.reproduce --manifest path/to/medqa_manifest.csv \
        --out experiments/medqa/results --stage all --solo-n 100 --cascade-n 20

The manifest is a shared benchmaxxing manifest built by the MedQA adapter
(``benchmaxxing/datasets/medqa.py``, PR #66). Every model call is cached under
``--out/call_cache.jsonl`` (keyed by model+prompt), so re-runs reproduce the numbers with no
new API calls, and an interrupted run resumes for free.
"""

from __future__ import annotations
from benchmaxxing.extract import parse_legacy_string


import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.analysis import (
    FlipRecord, failure_vector, flip_rate, lineage_overlap_test, susceptibility_matrix,
)
from benchmaxxing.blackboard import AgentResponse, run_committee
from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.data import load_cases
from benchmaxxing.onset import cascade_onset
from benchmaxxing.roster import build_committee
from benchmaxxing.schema import Condition, ModelSpec
from benchmaxxing.transcript import dump_transcript

TEXT_CUES = ["longest_option", "option_order", "lexical_overlap"]
TIERS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
COMMITTEE = [("agent0", "gemini-2.5-flash"),
             ("agent1", "gemini-2.5-flash-lite"),
             ("agent2", "gemini-2.5-flash")]

_cache_lock = threading.Lock()
_io_lock = threading.Lock()


def _get_key():
    """Return the API key from the env, or None. A fully cached run needs no key; the key is
    only required to fill a cache miss (and for the uncached noise-floor control)."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _letters(n):
    return [chr(65 + i) for i in range(n)]


def _mcq_prompt(payload, board=""):
    opts = payload["options"]
    body = "\n".join(f"{L}. {o}" for L, o in zip(_letters(len(opts)), opts))
    report = (payload.get("report") or "").strip()
    head = f"Clinical context: {report}\n\n" if report else ""
    return (f"{head}Question: {payload['question']}\n\nOptions:\n{body}\n\n{board}"
            "Answer with only the single letter of the best option.")



class CachedBackend(gateway.Backend):
    """Persistent (model, prompt) -> text cache so re-runs cost no API calls."""

    _store: dict[str, str] = {}
    _loaded_from: set[str] = set()

    def __init__(self, model, api_key, cache_path):
        self.model = model
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self._inner = None  # the live backend is built lazily, only on a cache miss
        with _cache_lock:
            key = str(self.cache_path)
            if key not in CachedBackend._loaded_from and self.cache_path.exists():
                for line in self.cache_path.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        CachedBackend._store[r["k"]] = r["resp"]
                CachedBackend._loaded_from.add(key)

    def complete(self, prompt, image=None, decoding=None):
        k = hashlib.sha256(f"{self.model}\x00{prompt}".encode()).hexdigest()
        with _cache_lock:
            if k in CachedBackend._store:
                return CachedBackend._store[k]
        # cache miss: a live call is needed, so a key is required only here.
        if self._inner is None:
            if not self.api_key:
                raise SystemExit(
                    "Cache miss with no GEMINI_API_KEY set: a live model call is needed to fill "
                    "it, but no key is available. A fully cached run reproduces the committed "
                    "numbers with no key; set GEMINI_API_KEY only to compute new results.")
            self._inner = gateway.RetryBackend(
                gateway.GeminiBackend(model=self.model, api_key=self.api_key), tries=5, backoff=3.0)
        resp = self._inner.complete(prompt, image=image, decoding=decoding)
        with _cache_lock:
            CachedBackend._store[k] = resp
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"k": k, "model": self.model, "resp": resp}) + "\n")
        return resp


def run_solo(cases, out, api_key, cache):
    """Solo susceptibility (flip rate per model x cue) + an uncached noise-floor control."""
    def call_for(model):
        b = CachedBackend(model, api_key, cache)
        return lambda payload: parse_legacy_string(b.complete(_mcq_prompt(payload),
                                                        decoding={"temperature": 0}),
                                              list(payload["options"]))

    records: list[FlipRecord] = []

    def eval_one(model, case):
        call = call_for(model)
        clean_tp = build_text_twin(case, TEXT_CUES[0])
        clean = call(clean_tp.payload(Condition.CLEAN))
        gt = clean_tp.ground_truth
        out_recs = []
        for cue in TEXT_CUES:
            try:
                tp = build_text_twin(case, cue)
            except (ValueError, TypeError):
                continue
            cont = call(tp.payload(Condition.CONTAMINATED))
            out_recs.append(FlipRecord(
                case_id=str(case.case_id), cue_type=cue, model=model,
                clean_answer=clean, contaminated_answer=cont, ground_truth=gt,
                flipped=bool(clean != cont),
                clean_correct=None if gt is None else bool(clean == gt),
                contaminated_correct=None if gt is None else bool(cont == gt)))
        return out_recs

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(eval_one, m, c) for m in TIERS for c in cases]
        for fut in as_completed(futs):
            records.extend(fut.result())

    # noise floor: same clean case run twice, UNCACHED (a repeat cached call is identical
    # by construction, so self-inconsistency must be measured off-cache). This is the one part
    # that always needs a key; a keyless (fully cached) reproduction skips it.
    noise = {}
    if not api_key:
        noise = {m: None for m in TIERS}
        print("noise floor skipped (no key): it is an uncached control; set GEMINI_API_KEY to run it.")
    for model in (TIERS if api_key else []):
        raw = gateway.RetryBackend(gateway.GeminiBackend(model=model, api_key=api_key),
                                   tries=5, backoff=3.0)
        ch = n = 0
        for case in cases[:15]:
            p = build_text_twin(case, TEXT_CUES[0]).payload(Condition.CLEAN)
            a1 = parse_legacy_string(raw.complete(_mcq_prompt(p), decoding={"temperature": 0}), list(p["options"]))
            a2 = parse_legacy_string(raw.complete(_mcq_prompt(p), decoding={"temperature": 0}), list(p["options"]))
            ch += (a1 != a2)
            n += 1
        noise[model] = ch / n if n else None

    by_model = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)
    sm = susceptibility_matrix(by_model)
    fvecs = {m: failure_vector(rs) for m, rs in by_model.items()}
    try:
        overlap = lineage_overlap_test(fvecs, {m: "gemini" for m in by_model},
                                       metric="phi", n_permutations=2000, seed=0)
    except Exception as e:
        overlap = {"error": str(e)}
    result = {"n_records": len(records), "noise_floor_by_model": noise,
              "flip_rate_by_model": {m: flip_rate(rs) for m, rs in by_model.items()},
              "susceptibility_matrix": {"models": sm["models"], "cues": sm["cues"],
                                        "matrix": sm["matrix"].tolist()},
              "overlap": overlap}
    (Path(out) / "solo_results.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_cascade(cases, out, api_key, cache):
    """Cascade with a BASELINE-RELATIVE seed (issue #104): the planted shortcut differs from
    each committee's isolated baseline, so contagion (shared minus isolated adoption) is
    well defined. See issue #96 / #122 discussion for why the first-distractor seed was invalid."""
    backends = {model: CachedBackend(model, api_key, cache) for _, model in COMMITTEE}
    model_by_agent = dict(COMMITTEE)
    committee = build_committee([ModelSpec(name=a, lineage="gemini", tier=m, is_open_weights=False)
                                 for a, m in COMMITTEE])
    tdir = Path(out) / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)

    def backend_for(spec):
        backend = backends[model_by_agent[spec.name]]

        class _C:
            def respond(self, view):
                payload = {"question": getattr(view.case, "question", "") or "",
                           "options": getattr(view.case, "options", ()) or (),
                           "report": getattr(view.case, "report", "") or ""}
                lines = [f"- {t.agent_id} answered: {t.answer}"
                         for t in view.visible_turns if t.answer is not None]
                board = ("Other panelists so far:\n" + "\n".join(lines) + "\n\n") if lines else ""
                text = backend.complete(_mcq_prompt(payload, board), decoding={"temperature": 0})
                return AgentResponse(content=text[:150],
                                     answer=parse_legacy_string(text, list(payload["options"])))
        return _C()

    def one(case):
        options = list(case.options)
        agent_ids = [a for a, _ in COMMITTEE]
        base = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=False, rounds=1)
        answers = [base.committed.get(a) for a in agent_ids if base.committed.get(a) is not None]
        baseline = Counter(answers).most_common(1)
        baseline = baseline[0][0] if baseline else None
        cands = [o for i, o in enumerate(options) if i != case.answer_index and o != baseline]
        if not cands:
            return {"case_id": case.case_id, "skipped": "no counterfactual seed"}
        seed_answer = cands[0]
        seed_agent = agent_ids[1]
        st = (1, seed_answer, seed_agent)
        shared = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=True, seed_turn=st, rounds=3)
        iso = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=False, seed_turn=st, rounds=3)
        dump_transcript(shared, tdir / f"{case.case_id}_repro_shared.jsonl")
        dump_transcript(iso, tdir / f"{case.case_id}_repro_isolated.jsonl")
        others = [a for a in agent_ids if a != seed_agent]
        sa = sum(1 for a in others if shared.committed.get(a) == seed_answer) / len(others)
        ia = sum(1 for a in others if iso.committed.get(a) == seed_answer) / len(others)
        series = [1.0 if t.answer == seed_answer else 0.0 for t in shared.turns]
        return {"case_id": case.case_id, "shared_adopt": sa, "isolated_adopt": ia,
                "contagion": sa - ia, "onset": cascade_onset(series)}

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(one, c) for c in cases]):
            rows.append(fut.result())
    valid = [r for r in rows if "skipped" not in r]
    summary = {"n": len(rows), "n_valid": len(valid),
               "mean_contagion": (sum(r["contagion"] for r in valid) / len(valid)) if valid else None,
               "mean_shared_adopt": (sum(r["shared_adopt"] for r in valid) / len(valid)) if valid else None,
               "mean_isolated_adopt": (sum(r["isolated_adopt"] for r in valid) / len(valid)) if valid else None,
               "per_case": rows}
    (Path(out) / "cascade_results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Reproduce the MedQA Lane B experiments.")
    ap.add_argument("--manifest", required=True, help="MedQA manifest CSV (built by the medqa adapter)")
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--stage", choices=["solo", "cascade", "all"], default="all")
    ap.add_argument("--solo-n", type=int, default=100)
    ap.add_argument("--cascade-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    api_key = _get_key()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "call_cache.jsonl"
    all_cases = load_cases(args.manifest)
    cases = random.Random(args.seed).sample(all_cases, min(args.solo_n, len(all_cases)))
    print(f"[{time.strftime('%H:%M:%S')}] {len(all_cases)} cases; solo_n={len(cases)} seed={args.seed}")

    if args.stage in ("solo", "all"):
        r = run_solo(cases, out, api_key, cache)
        print("solo:", {m: round(v["overall"], 3) for m, v in r["flip_rate_by_model"].items()},
              "| noise floor:", r["noise_floor_by_model"])
    if args.stage in ("cascade", "all"):
        s = run_cascade(cases[:args.cascade_n], out, api_key, cache)
        print("cascade: mean_contagion=", s["mean_contagion"],
              "shared_adopt=", s["mean_shared_adopt"], "isolated_adopt=", s["mean_isolated_adopt"])


if __name__ == "__main__":
    main()
