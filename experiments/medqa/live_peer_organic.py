"""Live-peer tier composition with organic (non-scripted) errors (#209). Every cascade experiment so
far plants a wrong answer via scripted colluding peers. This one uses NO script: two real flash
peers and a real flash-lite holdout answer a shared committee naturally, and we ask whether the
holdout defers to the peers' OWN organic errors the way it defers to planted ones.

Per case: the holdout's solo bare answer; a shared committee (order [peer1, peer2, holdout], one
round) where all three are real Gemini agents; the two peers' organic answers and the holdout's
board answer. We stratify by what the real peers did:

  organic_wrong_consensus : both peers agree on the SAME wrong answer (a natural shared error).
  organic_right_consensus : both peers agree on the correct answer.

and report, within each stratum, how often the holdout's board answer follows the peer consensus,
plus the net harm (holdout was correct alone but adopts the peers' wrong consensus). This measures
susceptibility to organic rather than adversarial errors. All reads cached; keyless on re-run.
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
from benchmaxxing.blackboard import AgentResponse, render_board, run_committee
from benchmaxxing.data import load_cases
from benchmaxxing.roster import build_committee
from benchmaxxing.schema import Condition, ModelSpec

HOLDOUT = "gemini-2.5-flash-lite"
PEER_MODEL = "gemini-2.5-flash"
MEMBERS = [("peer1", PEER_MODEL), ("peer2", PEER_MODEL), ("holdout", HOLDOUT)]
_lock = threading.Lock()


def _key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _letters(n):
    return [chr(65 + i) for i in range(n)]


def _mcq(case, prefix=""):
    opts = list(case.options)
    body = "\n".join(f"{L}. {o}" for L, o in zip(_letters(len(opts)), opts))
    return (f"{prefix}Question: {case.question}\n\nOptions:\n{body}\n\n"
            "Answer with only the single letter of the best option."), opts



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
    ap = argparse.ArgumentParser(description="Live-peer tier composition with organic errors (#209).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", default="experiments/medqa/results/live_peer_organic_cache.jsonl")
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--show-rationale", action="store_true",
                    help="render each peer's reasoning under its vote (#373); off is the "
                         "committed answer-only board, which the cache replays at zero calls")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _Cache(args.cache, _key())
    cases = load_cases(args.manifest)[:args.n]
    model_by_agent = dict(MEMBERS)
    committee = build_committee(
        [ModelSpec(name=a, lineage="gemini",
                   tier="flash" if m == PEER_MODEL else "lite", is_open_weights=False)
         for a, m in MEMBERS])

    def backend_for(spec):
        backend_model = model_by_agent[spec.name]

        class _C:
            def respond(self, view):
                board = render_board(view.visible_turns, "Panel so far:",
                                     show_rationale=args.show_rationale,
                                     self_id=view.agent_id)
                p, opts = _mcq(view.case, board)
                text = cache.complete(backend_model, p)
                return AgentResponse(content=text[:120], answer=parse_legacy_string(text, opts), confidence=0.7)
        return _C()

    def run_one(case):
        opts = list(case.options)
        gt = opts[case.answer_index]
        base_p, _ = _mcq(case)
        bare = parse_legacy_string(cache.complete(HOLDOUT, base_p), opts)
        shared = run_committee(committee, case, Condition.CLEAN, backend_for,
                               shared=True, rounds=1, order=[0, 1, 2])
        board_ans = shared.committed.get("holdout")
        peer_ans = [t.answer for t in shared.turns if t.agent_id in ("peer1", "peer2") and t.answer is not None]
        consensus = peer_ans[0] if len(peer_ans) == 2 and peer_ans[0] == peer_ans[1] else None
        return {"case_id": case.case_id, "bare": bare, "ground_truth": gt, "board": board_ans,
                "peer_answers": peer_ans, "consensus": consensus,
                "consensus_wrong": consensus is not None and consensus != gt,
                "consensus_right": consensus is not None and consensus == gt,
                "follows_consensus": consensus is not None and board_ans == consensus,
                "net_harm": (consensus is not None and consensus != gt and bare == gt
                             and board_ans == consensus)}

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            r = fut.result()
            if r:
                rows.append(r)

    n = len(rows)
    wrong_cons = [r for r in rows if r["consensus_wrong"]]
    right_cons = [r for r in rows if r["consensus_right"]]

    def follow_rate(sub):
        return round(sum(1 for r in sub if r["follows_consensus"]) / len(sub), 4) if sub else None
    summary = {
        "n": n, "models": {"peers": PEER_MODEL, "holdout": HOLDOUT},
        "new_api_calls_this_run": cache.calls,
        "n_organic_wrong_consensus": len(wrong_cons), "n_organic_right_consensus": len(right_cons),
        "follow_rate_on_wrong_consensus": follow_rate(wrong_cons),
        "follow_rate_on_right_consensus": follow_rate(right_cons),
        "net_harm_cases": sum(1 for r in rows if r["net_harm"]),
        "read": (
            f"With two REAL flash peers making organic (unscripted) errors, on the {len(wrong_cons)} "
            f"cases where both peers independently agreed on the same WRONG answer the holdout follows "
            f"that wrong consensus {follow_rate(wrong_cons)} of the time, versus following a correct "
            f"peer consensus {follow_rate(right_cons)} of the time on {len(right_cons)} cases. Net "
            f"harm (holdout correct alone but adopts the organic wrong consensus) occurs on "
            f"{sum(1 for r in rows if r['net_harm'])} cases. High follow-rate on wrong consensus means "
            "the holdout defers to organic peer errors much as it does to planted ones, so the cascade "
            "is not an artifact of adversarial scripting; a gap between wrong- and right-consensus "
            "following would show some genuine discrimination."
        ),
    }
    (out / "live_peer_organic_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "live_peer_organic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
