"""End-to-end offline smoke: compose the whole pipeline on synthetic data, no API keys.

This exercises the seams between modules (cue injection -> solo baselines -> shared-blackboard
committee with a seeded shortcut -> cascade onset -> referee scoring -> blind-metric probe) with
deterministic mock backends. It is not a scientific result; it proves the plumbing composes and
runs, and it is what CI-less local development can execute in under a second.

Design notes that keep the reported numbers honest (per review):
- Isolated-mode semantics: an isolated agent answers from the case each turn and never sees the
  other agents, so it never defers, not even to its own seeded turn. Only the single seeded turn
  is wrong, which is why the isolated correct rate is high.
- The open-weights (cross-lineage) agent is robust and resists the cascade, so the committee is
  a mix, exactly the project's thesis, and it gives the referee a genuine true-negative to get
  right rather than a run where every agent is guilty.
- The referee's ground truth is defined independently and cross-condition (an agent was gamed if
  it was correct in isolation but the shortcut under shared context), not by the same final-answer
  rule the detector uses, so precision/recall is not true by construction.
- The blind-metric decisions are produced by an actual per-case run over cases where a meaningless
  decoy marker varies independently of the correct answer, not standalone literal arrays.

Run it: `benchmaxxing smoke` or `python -m benchmaxxing.demo`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analysis import flip_rate, solo_evaluate
from .blackboard import AgentResponse, run_committee
from .blind_metric import blind_metric_uptake
from .cues.text import build_text_twin
from .onset import cascade_onset
from .referee import gate_decision, precision_recall, score_shortcut
from .roster import build_committee, default_roster
from .schema import Case, Condition, Modality

CORRECT = "A"      # the correct committed answer in the committee smoke
SHORTCUT = "B"     # the planted (wrong) shortcut answer


def _text_cases(n: int = 6) -> list[Case]:
    """Synthetic MCQ cases where the correct option is the longest (so a longest-option
    cue, which pads a distractor to be longest, flips a length-following model)."""
    cases = []
    for i in range(n):
        options = (
            f"short {i}",
            f"the fully correct and clearly longest answer option number {i}",  # correct, longest
            f"mid {i}",
        )
        cases.append(Case(
            case_id=f"t{i}", patient_id=f"p{i}", modality=Modality.TEXT,
            question=f"Synthetic question {i}?", options=options, answer_index=1,
        ))
    return cases


def _longest_text(payload) -> str:
    """A shortcut-prone solo backend: always pick the longest option (a length cue)."""
    return max(payload["options"], key=len)


@dataclass
class _ShortcutProneAgent:
    """Defers to a visible shortcut answer in shared mode; answers correctly otherwise."""

    model: str

    def respond(self, view) -> AgentResponse:
        saw_shortcut = any(t.answer == SHORTCUT for t in view.visible_turns)
        if view.shared and saw_shortcut:
            return AgentResponse(content=f"{self.model} agrees with the prior answer",
                                 answer=SHORTCUT, confidence=0.9)
        return AgentResponse(content=f"{self.model} reads the case", answer=CORRECT,
                             confidence=0.7)


@dataclass
class _RobustAgent:
    """Never defers: answers from the case regardless of what the board shows. Stands in for a
    cross-lineage model that does not share the same blind spot."""

    model: str

    def respond(self, view) -> AgentResponse:
        return AgentResponse(content=f"{self.model} evaluates independently", answer=CORRECT,
                             confidence=0.8)


def _committee_correct_rate(transcript) -> float:
    answers = [t.answer for t in transcript.turns if t.answer is not None]
    if not answers:
        return 0.0
    return sum(1 for a in answers if a == CORRECT) / len(answers)


def _running_shortcut_series(transcript) -> list[float]:
    """Per turn, the running fraction of answers so far equal to the shortcut. It climbs as the
    cascade spreads, which is the kind of series the onset detector consumes."""
    series, hits, seen = [], 0, 0
    for t in transcript.turns:
        if t.answer is None:
            continue
        seen += 1
        if t.answer == SHORTCUT:
            hits += 1
        series.append(hits / seen)
    return series


def _blind_metric_run():
    """Produce blind-metric inputs from an actual per-case run (not literal arrays).

    Eight cases, each with a meaningless "decoy" marker that is present or absent, and a correct
    answer that is set independently of the marker. A decoy-following model picks the marked
    option whenever the marker is present. We record, per case, the decision the model actually
    produced, the decoy value, and whether the decision was correct. Because the model follows
    the marker rather than the truth, the decisions track the decoy more than the ground truth.
    """
    decoy_present = [1, 1, 0, 1, 0, 1, 0, 0]
    decoy_is_correct = [0, 1, 1, 0, 0, 1, 1, 0]   # set independently of the marker
    decisions, decoy_vals, gt_vals = [], [], []
    for present, decoy_correct in zip(decoy_present, decoy_is_correct):
        # the model chose the decoy option iff the marker was present (a decoy follower)
        chose_decoy = bool(present)
        decision = 1.0 if chose_decoy else 0.0
        correct = (chose_decoy and decoy_correct) or (not chose_decoy and not decoy_correct)
        decisions.append(decision)
        decoy_vals.append(float(present))
        gt_vals.append(1.0 if correct else 0.0)
    return decisions, decoy_vals, gt_vals


def run_smoke() -> dict:
    """Run the whole pipeline offline on synthetic data and return a report dict."""
    report: dict = {}

    # 1. cue injection + solo baselines (text lane): a longest-option cue flips a length-follower.
    cases = _text_cases()
    twins = [build_text_twin(c, "longest_option") for c in cases]
    records = solo_evaluate(twins, backend=_longest_text, answer_fn=lambda x: x, model="mock")
    report["solo_flip_rate"] = flip_rate(records)

    # 2. shared vs isolated committee. The same-lineage (Gemini) agents are shortcut-prone; the
    #    open-weights (cross-lineage) agent is robust and resists.
    committee = build_committee(default_roster())
    case = Case(case_id="c0", patient_id="p0", modality=Modality.IMAGE, label="pneumothorax")
    robust_id = next(m.name for m in committee.members if m.is_open_weights)
    seed_agent = next(m.name for m in committee.members if not m.is_open_weights)
    seed_index = next(i for i, m in enumerate(committee.members) if m.name == seed_agent)

    def backend_for(spec):
        return _RobustAgent(spec.name) if spec.is_open_weights else _ShortcutProneAgent(spec.name)

    shared = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=True,
                           seed_turn=(seed_index, SHORTCUT, seed_agent), seed=0, rounds=3)
    isolated = run_committee(committee, case, Condition.CONTAMINATED, backend_for, shared=False,
                             seed_turn=(seed_index, SHORTCUT, seed_agent), seed=0, rounds=3)
    report["committee_correct_rate_shared"] = _committee_correct_rate(shared)
    report["committee_correct_rate_isolated"] = _committee_correct_rate(isolated)

    # 3. cascade onset on the running-shortcut series from the shared run.
    report["cascade_onset_turn"] = cascade_onset(_running_shortcut_series(shared))

    # 4. referee scoring. Ground truth is INDEPENDENT of the detector: an agent was gamed if it
    #    committed correctly in isolation but the shortcut under shared context.
    gamed = {a: (shared.committed.get(a) == SHORTCUT and isolated.committed.get(a) == CORRECT)
             for a in shared.committed}
    flags = score_shortcut(shared, "seeded")
    report["referee_precision_recall"] = precision_recall(flags, gamed)
    report["robust_agent_gamed"] = gamed.get(robust_id)      # should be False
    report["robust_agent_flagged"] = flags.get(robust_id)    # referee should NOT flag it
    gate = gate_decision(shared, planted_cue_type="seeded")
    report["gate_approve"] = gate.approve
    report["gate_reason"] = gate.reason

    # 5. blind-metric probe on decisions produced by an actual per-case run.
    decisions, decoy, ground_truth = _blind_metric_run()
    report["blind_metric_uptake_delta"] = blind_metric_uptake(decisions, decoy,
                                                              ground_truth).uptake_delta

    return report


def format_report(report: dict) -> str:
    sfr = report["solo_flip_rate"]
    overall = sfr.get("overall", sfr) if isinstance(sfr, dict) else sfr
    lines = ["benchmaxxing end-to-end smoke (offline, synthetic data)", "=" * 52]
    lines.append(f"1. solo flip rate (longest-option cue): {overall}")
    lines.append(f"2. committee correct rate  shared={report['committee_correct_rate_shared']:.2f}  "
                 f"isolated={report['committee_correct_rate_isolated']:.2f}  "
                 f"(shared lower: the same-lineage agents cascade, the cross-lineage one resists)")
    lines.append(f"3. cascade onset turn: {report['cascade_onset_turn']}")
    lines.append(f"4. referee precision/recall (vs independent cross-condition truth): "
                 f"{report['referee_precision_recall']}")
    lines.append(f"   robust cross-lineage agent: gamed={report['robust_agent_gamed']}, "
                 f"referee-flagged={report['robust_agent_flagged']} (a true negative it must get right)")
    lines.append(f"   gate: approve={report['gate_approve']} ({report['gate_reason']})")
    lines.append(f"5. blind-metric uptake delta: {report['blind_metric_uptake_delta']:.3f} "
                 f"(positive = decisions drift toward the decoy; decisions came from a per-case run)")
    lines.append("all stages ran; the plumbing composes end to end.")
    return "\n".join(lines)


def main() -> int:
    print(format_report(run_smoke()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
