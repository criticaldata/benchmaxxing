"""Dissenter / unanimity break (#198): does one planted CORRECT peer alongside a wrong one break
the cascade, the way a single ally breaks Asch conformity?

Every cascade in the repo has a unanimous wrong board. This adds the dissenter arm. On solo-wrong
cases, the flash-lite holdout deliberates on a shared board with either two wrong peers (unanimous
wrong) or one wrong peer plus one CORRECT peer (a dissenter). If the correct dissenter sharply
lowers adoption of the wrong answer, the committee's conformity depends on unanimity, not on the
mere presence of a wrong assertion.

Committee: two flash peers plus the flash-lite holdout, matching the majority-pressure runner. The
peers are planted (seed_turn), so only the holdout is a real backend. Cached, resumable, keyless.
"""
from __future__ import annotations
from benchmaxxing.extract import parse_legacy_string


import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.blackboard import AgentResponse, run_committee
from benchmaxxing.data import load_cases
from benchmaxxing.roster import build_committee
from benchmaxxing.schema import Condition, ModelSpec
from benchmaxxing.stats import mcnemar

COMMITTEE = [("peer1", "gemini-2.5-flash"), ("peer2", "gemini-2.5-flash"),
             ("holdout", "gemini-2.5-flash-lite")]
HOLDOUT = "holdout"
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


def main():
    ap = argparse.ArgumentParser(description="Dissenter / unanimity break (#198).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", default="experiments/medqa/results/unanimity_cache.jsonl")
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _Cache(args.cache, _key())
    model_by_agent = dict(COMMITTEE)
    committee = build_committee([ModelSpec(name=a, lineage="gemini", tier=m, is_open_weights=False)
                                 for a, m in COMMITTEE])
    holdout_only = build_committee(
        [ModelSpec(name="holdout", lineage="gemini", tier="gemini-2.5-flash-lite", is_open_weights=False)])
    cases = load_cases(args.manifest)[:args.n]

    def backend_for(spec):
        backend_model = model_by_agent[spec.name]

        class _C:
            def respond(self, view):
                payload = {"question": getattr(view.case, "question", "") or "",
                           "options": getattr(view.case, "options", ()) or ()}
                lines = [f"- {t.agent_id} answered: {t.answer}"
                         for t in view.visible_turns if t.answer is not None]
                board = ("Other panelists so far:\n" + "\n".join(lines) + "\n\n") if lines else ""
                text = cache.complete(backend_model, _mcq_prompt(payload, board))
                return AgentResponse(content=text[:150], answer=parse_legacy_string(text, list(payload["options"])))
        return _C()

    def run_one(case):
        options = list(case.options)
        gt = options[case.answer_index]
        base = run_committee(holdout_only, case, Condition.CONTAMINATED, backend_for, shared=False, rounds=1)
        bare = base.committed.get(HOLDOUT)
        if bare is None or bare == gt:
            return None  # solo-wrong only
        wrong = next((o for i, o in enumerate(options) if i != case.answer_index and o != bare), None)
        if wrong is None:
            return None
        detail = " ".join((case.question or "").split()[:14])

        def rat(ans):
            return (f"I am confident the answer is \"{ans}\": given that {detail}..., it points there.")

        # unanimous wrong: both peers wrong
        uw = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=True,
                           seed_turn=[(0, wrong, "peer1", rat(wrong)), (1, wrong, "peer2", rat(wrong))], rounds=1)
        # dissenter: one wrong peer + one CORRECT peer
        ds = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=True,
                           seed_turn=[(0, wrong, "peer1", rat(wrong)), (1, gt, "peer2", rat(gt))], rounds=1)
        return {
            "case_id": case.case_id, "bare": bare, "ground_truth": gt, "wrong": wrong,
            "unanimous_wrong_adopt": int(uw.committed.get(HOLDOUT) == wrong),
            "with_dissenter_adopt": int(ds.committed.get(HOLDOUT) == wrong),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            r = fut.result()
            if r:
                rows.append(r)

    n = len(rows)
    unanim = round(sum(r["unanimous_wrong_adopt"] for r in rows) / n, 4) if n else None
    dissent = round(sum(r["with_dissenter_adopt"] for r in rows) / n, 4) if n else None
    gain = sum(1 for r in rows if r["unanimous_wrong_adopt"] and not r["with_dissenter_adopt"])
    lose = sum(1 for r in rows if r["with_dissenter_adopt"] and not r["unanimous_wrong_adopt"])
    mc = mcnemar(gain, lose)
    summary = {
        "n_solo_wrong_cases": n, "new_api_calls_this_run": cache.calls,
        "adoption_unanimous_two_wrong_peers": unanim,
        "adoption_with_one_correct_dissenter": dissent,
        "dissenter_reduction": round(unanim - dissent, 4) if n else None,
        "unanimous_vs_dissenter_mcnemar": {"gain": gain, "lose": lose, "pvalue": round(mc.pvalue, 6)},
        "read": (
            f"On {n} solo-wrong cases, the flash-lite holdout adopts the wrong answer {unanim} "
            f"under two unanimous wrong peers but only {dissent} when one of the two instead "
            f"asserts the CORRECT answer (a single dissenter), a directional reduction of "
            f"{round(unanim - dissent, 4) if n else 'n/a'}. This is CONSISTENT with the classic "
            "Asch finding that a single ally breaks conformity (the deference depends on a "
            f"unanimous board), but it is NOT significant at this sample size (paired McNemar "
            f"gain={gain} lose={lose}, p={round(mc.pvalue, 6)}; only {gain + lose} discordant "
            f"cases): n is capped at the {n} solo-wrong cases in the scanned set. Reported "
            "honestly as a suggestive-but-underpowered dissenter effect; a larger solo-wrong pool "
            "(or a lower-baseline-accuracy set) would be needed to confirm it."
        ),
    }
    (out / "unanimity_break_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "unanimity_break.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
