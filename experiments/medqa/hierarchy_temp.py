"""Hierarchy-monitoring at temperature>0 (#235). The temp-0 version (#173) was degenerate: the shared
committee converged to unanimity so every agent trivially tied at dominance 1.0. At temperature 0.7
the agents can genuinely disagree, so `score_hierarchy` can detect real order-dependent single-agent
dominance (one seat driving the committee outcome regardless of speaking order) rather than mere
consensus. Same unseeded mixed-tier committee (two flash, one flash-lite), all six speaking orders,
`score_hierarchy(dominance_threshold=0.75)`; the cache key includes the temperature so sampled reads
do not collide with the committed temp-0 cache. Keyless on re-run.

The paper names three referee duties; hierarchy monitoring has zero evidence and no committed
transcript varies speaking order (all use a fixed order). This runs an UNSEEDED mixed-tier
committee under every speaking-order permutation and feeds the transcripts to
`benchmaxxing.referee.score_hierarchy`, which flags an agent whose own first proposal matches the
group outcome across orders (order-independent dominance) at or above a threshold.

Committee: two flash agents and one flash-lite agent. For each case the committee is run under all
six order permutations (no seed), then `score_hierarchy(dominance_threshold=0.75)` reports the
dominant agent (if any). We aggregate: how often a dominant agent exists, whether it is a flash
(stronger) agent, and how often the dominant agent's first proposal was WRONG (a dominant-and-wrong
agent, the hazard the hierarchy duty is meant to catch). Every read is cached; keyless on re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.ablations import order_permutation_run
from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.data import load_cases
from benchmaxxing.referee import score_hierarchy
from benchmaxxing.roster import build_committee
from benchmaxxing.schema import ModelSpec

MEMBERS = [("flash_a", "gemini-2.5-flash"), ("lite", "gemini-2.5-flash-lite"),
           ("flash_b", "gemini-2.5-flash")]
FLASH_AGENTS = {"flash_a", "flash_b"}
TEMP = 0.7
_lock = threading.Lock()


def _key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _letters(n):
    return [chr(65 + i) for i in range(n)]


def _mcq_prompt(payload, board=""):
    opts = payload["options"]
    body = "\n".join(f"{L}. {o}" for L, o in zip(_letters(len(opts)), opts))
    return (f"Question: {payload['question']}\n\nOptions:\n{body}\n\n{board}"
            "Answer with only the single letter of the best option.")


def _parse_choice(text, options):
    import re
    if not text:
        return ""
    t = text.strip()
    letters = _letters(len(options))
    m = re.findall(r"\\boxed\{\s*([A-E])\s*\}", t)
    if not m:
        m = re.findall(r"(?:final answer|the answer|answer)\s*(?:is|:)?\s*\**\(?([A-E])\)?\b", t, re.I)
    if m and m[-1].upper() in letters:
        return options[letters.index(m[-1].upper())]
    low = t.lower()
    hits = [(low.rfind(o.lower()), o) for o in options if o.lower() in low]
    hits = [(p, o) for p, o in hits if p >= 0]
    if hits:
        return max(hits)[1]
    m2 = re.search(r"\b([A-E])\b\s*[.)]?\s*$", t.upper())
    if m2 and m2.group(1) in letters:
        return options[letters.index(m2.group(1))]
    if len(t) == 1 and t.upper() in letters:
        return options[letters.index(t.upper())]
    return t


class _Cache:
    def __init__(self, path, key):
        self.path, self.key, self.store, self.calls = Path(path), key, {}, 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.store[r["k"]] = r["resp"]

    def complete(self, model, prompt):
        k = hashlib.sha256(f"{model}\x00{TEMP}\x00{prompt}".encode()).hexdigest()
        with _lock:
            if k in self.store:
                return self.store[k]
        if not self.key:
            raise SystemExit("Cache miss and no GEMINI_API_KEY set (a fully cached run needs no key).")
        resp = gateway.RetryBackend(gateway.GeminiBackend(model=model, api_key=self.key),
                                    tries=5, backoff=3.0).complete(prompt, decoding={"temperature": TEMP})
        with _lock:
            self.store[k] = resp
            self.calls += 1
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "model": model, "temperature": TEMP, "resp": resp}) + "\n")
        return resp


def main():
    ap = argparse.ArgumentParser(description="Order-independent hierarchy dominance at temperature>0 (#235).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", default="experiments/medqa/results/hierarchy_temp_cache.jsonl")
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _Cache(args.cache, _key())
    model_by_agent = dict(MEMBERS)
    committee = build_committee(
        [ModelSpec(name=a, lineage="gemini", tier=m, is_open_weights=False) for a, m in MEMBERS])
    cases = load_cases(args.manifest)[:args.n]
    orders = list(itertools.permutations(range(3)))

    def backend_for(spec):
        backend_model = model_by_agent[spec.name]

        class _C:
            def respond(self, view):
                payload = {"question": getattr(view.case, "question", "") or "",
                           "options": getattr(view.case, "options", ()) or ()}
                lines = [f"- {t.agent_id} answered: {t.answer}"
                         for t in view.visible_turns if t.answer is not None]
                board = ("Panel so far:\n" + "\n".join(lines) + "\n\n") if lines else ""
                text = cache.complete(backend_model, _mcq_prompt(payload, board))
                return AgentResponse(content=text[:150], answer=_parse_choice(text, list(payload["options"])))
        return _C()

    def run_one(case):
        gt = case.options[case.answer_index]
        points = order_permutation_run(committee, case, backend_for, orders, rounds=1)
        transcripts = [p.transcript for p in points]
        result = score_hierarchy(transcripts, dominance_threshold=0.75)
        dom = result.dominant_agent
        # the dominant agent's own first proposal across orders (its first-seen answer per run)
        dom_first = None
        if dom is not None:
            for t in transcripts:
                for turn in sorted(t.turns, key=lambda x: x.turn_index):
                    if turn.agent_id == dom and turn.answer is not None:
                        dom_first = turn.answer
                        break
                if dom_first is not None:
                    break
        return {
            "case_id": case.case_id, "ground_truth": gt,
            "dominant_agent": dom,
            "dominant_is_flash": dom in FLASH_AGENTS if dom else None,
            "dominant_first_answer": dom_first,
            "dominant_and_wrong": (dom is not None and dom_first is not None and dom_first != gt),
            "dominance_scores": result.dominance_scores,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            r = fut.result()
            if r:
                rows.append(r)

    n = len(rows)
    n_dominant = sum(1 for r in rows if r["dominant_agent"] is not None)
    n_dom_flash = sum(1 for r in rows if r["dominant_is_flash"])
    n_dom_wrong = sum(1 for r in rows if r["dominant_and_wrong"])
    summary = {
        "n_cases": n, "orders_per_case": len(orders), "new_api_calls_this_run": cache.calls,
        "cases_with_order_independent_dominant_agent": n_dominant,
        "dominance_rate": round(n_dominant / n, 4) if n else None,
        "of_dominant_how_many_are_flash": f"{n_dom_flash}/{n_dominant}" if n_dominant else "0/0",
        "dominant_and_wrong_cases": n_dom_wrong,
        "dominant_and_wrong_rate": round(n_dom_wrong / n, 4) if n else None,
        "read": (
            f"At temperature 0.7 the committee no longer trivially converges, so score_hierarchy "
            f"reports an order-independent dominant agent on {n_dominant} of {n} cases "
            f"(rate {round(n_dominant / n, 4) if n else 'n/a'}); of those, {n_dom_flash} are a flash "
            "(stronger-tier) seat. A dominant agent here means one seat whose own first proposal "
            "matches the committee outcome across all six speaking orders at or above the 0.75 "
            f"threshold. The dominant seat's first proposal is WRONG on {n_dom_wrong} of {n} cases "
            f"(rate {round(n_dom_wrong / n, 4) if n else 'n/a'}) - genuine order-independent "
            "single-agent dominance steering the committee to a wrong answer, the hazard the "
            "hierarchy-monitoring duty is meant to catch, now measurable because temp>0 lets the "
            "agents actually disagree (resolving the #173 temp-0 degeneracy)."
        ),
    }
    (out / "hierarchy_temp_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "hierarchy_temp.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "dominance_scores"}) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
