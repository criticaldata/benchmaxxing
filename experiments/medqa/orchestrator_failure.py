"""Orchestrator single-point-of-failure (#179): is a wrong LEADER/synthesizer more dangerous to a
committee's output than a wrong PEER?

The plan flags the orchestrator (the agent that takes the final synthesizing turn and whose answer
becomes the committee's output) as an unmitigated single point of failure the paper never tests.
Every committed cascade is flat (no orchestrator turn). This runs three arms on cases a clean
committee gets RIGHT, so there is something to lose, and reports how often each poisons the output.

Committee: two flash-lite members plus a flash leader. The committee OUTPUT is the leader's
synthesis when an orchestrator turn is taken (its documented role), else the majority committed
answer. Arms:

  clean            : no seed, no orchestrator. Used only to select clean-correct cases.
  wrong_peer       : one member seeded with a wrong answer, no orchestrator. Output = majority, so
                     the wrong peer must actually move the other members to poison the output.
  wrong_orchestrator: no wrong member, but the leader takes the synthesizing turn and is forced to
                     the wrong answer. Output = the leader's answer.
  honest_orch_gate : both members seeded wrong, and an HONEST (real) leader synthesizes over that
                     board. Tests whether an honest orchestrator can gate a peer cascade.

Each real read is cached; a fully cached run reproduces the summary with no key.
"""
from __future__ import annotations
from benchmaxxing.extract import parse_legacy_string


import argparse
import hashlib
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.blackboard import AgentResponse, run_committee
from benchmaxxing.data import load_cases
from benchmaxxing.roster import build_committee
from benchmaxxing.schema import Condition, ModelSpec

MEMBERS = [("member0", "gemini-2.5-flash-lite"), ("member1", "gemini-2.5-flash-lite"),
           ("leader", "gemini-2.5-flash")]
LEADER = "leader"
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
    ap = argparse.ArgumentParser(description="Orchestrator single-point-of-failure (#179).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", default="experiments/medqa/results/orchestrator_cache.jsonl")
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _Cache(args.cache, _key())
    model_by_agent = dict(MEMBERS)
    committee = build_committee(
        [ModelSpec(name=a, lineage="gemini", tier=m, is_open_weights=False) for a, m in MEMBERS])
    cases = load_cases(args.manifest)[:args.n]

    def real_backend(spec):
        backend_model = model_by_agent[spec.name]

        class _C:
            def respond(self, view):
                payload = {"question": getattr(view.case, "question", "") or "",
                           "options": getattr(view.case, "options", ()) or ()}
                lines = [f"- {t.agent_id} answered: {t.answer}"
                         for t in view.visible_turns if t.answer is not None]
                board = ("Panel so far:\n" + "\n".join(lines) + "\n\n") if lines else ""
                text = cache.complete(backend_model, _mcq_prompt(payload, board))
                return AgentResponse(content=text[:150], answer=parse_legacy_string(text, list(payload["options"])))
        return _C()

    def wrong_leader_backend(wrong):
        # real members, but the leader (orchestrator) is forced to the wrong answer.
        def factory(spec):
            if spec.name == LEADER:
                class _W:
                    def respond(self, view):
                        return AgentResponse(content=f"On synthesis the answer is {wrong}.", answer=wrong)
                return _W()
            return real_backend(spec)
        return factory

    def committee_output(transcript, orchestrated):
        if orchestrated:
            return transcript.committed.get(LEADER)
        answers = [v for v in transcript.committed.values() if v is not None]
        if not answers:
            return None
        return Counter(answers).most_common(1)[0][0]

    def run_one(case):
        options = list(case.options)
        gt = options[case.answer_index]
        clean = run_committee(committee, case, Condition.CONTAMINATED, real_backend, shared=True, rounds=1)
        if committee_output(clean, False) != gt:
            return None  # only cases the clean committee gets right
        wrong = next((o for i, o in enumerate(options) if i != case.answer_index), None)
        detail = " ".join((case.question or "").split()[:14])
        rat = f"I am confident the answer is \"{wrong}\": given that {detail}..., it points there."

        # wrong peer: member0 seeded wrong, no orchestrator
        wp = run_committee(committee, case, Condition.CONTAMINATED, real_backend, shared=True,
                           seed_turn=[(0, wrong, "member0", rat)], rounds=1)
        # wrong orchestrator: honest members, leader forced wrong on the synthesis turn
        wo = run_committee(committee, case, Condition.CONTAMINATED, wrong_leader_backend(wrong),
                           shared=True, orchestrator=True, rounds=1)
        # honest orchestrator gating two wrong peers
        hg = run_committee(committee, case, Condition.CONTAMINATED, real_backend, shared=True,
                           orchestrator=True, seed_turn=[(0, wrong, "member0", rat), (1, wrong, "member1", rat)],
                           rounds=1)
        return {
            "case_id": case.case_id, "ground_truth": gt, "wrong": wrong,
            "wrong_peer_output_wrong": int(committee_output(wp, False) != gt),
            "wrong_orch_output_wrong": int(committee_output(wo, True) != gt),
            "honest_orch_output_wrong": int(committee_output(hg, True) != gt),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            r = fut.result()
            if r:
                rows.append(r)

    n = len(rows)
    def rate(k):
        return round(sum(r[k] for r in rows) / n, 4) if n else None
    summary = {
        "n_clean_correct_cases": n, "new_api_calls_this_run": cache.calls,
        "wrong_peer_poisons_output": rate("wrong_peer_output_wrong"),
        "wrong_orchestrator_poisons_output": rate("wrong_orch_output_wrong"),
        "honest_orchestrator_over_wrong_peers_output_wrong": rate("honest_orch_output_wrong"),
        "read": (
            f"On {n} cases a clean committee gets right, a wrong PEER poisons the committee output "
            f"{rate('wrong_peer_output_wrong')} of the time (it must actually move the other "
            f"members), while a wrong ORCHESTRATOR poisons it {rate('wrong_orch_output_wrong')} of "
            "the time (its synthesis IS the output, so a single wrong leader is far more "
            "dangerous than a single wrong peer, the single-point-of-failure the plan warns "
            f"about). An HONEST orchestrator synthesizing over two wrong peers still outputs the "
            f"wrong answer {rate('honest_orch_output_wrong')} of the time, so whether an honest "
            "leader can GATE a peer cascade is answered by that rate: low means the leader rescues "
            "the committee, high means even an honest leader is dragged along by a wrong majority."
        ),
    }
    (out / "orchestrator_failure_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "orchestrator_failure.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
