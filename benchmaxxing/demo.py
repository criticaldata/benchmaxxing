"""End-to-end offline smoke: compose the whole pipeline on synthetic data, no API keys.

This exercises the seams between modules (cue injection -> solo baselines -> shared-blackboard
committee with a seeded shortcut -> cascade onset -> referee scoring -> blind-metric probe) with
deterministic mock backends. It is not a scientific result; it proves the plumbing composes and
runs, and it is what CI-less local development can execute in under a second.

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
from .roster import cross_lineage_committees, default_roster
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
    """Blackboard backend that defers to a visible shortcut answer in shared mode.

    In isolated mode (or with nothing to copy) it answers correctly; once it can see a prior
    turn carrying the planted SHORTCUT answer on the shared board, it copies it. That is the
    cascade in miniature, and it disappears when the board is isolated.
    """

    model: str

    def respond(self, view) -> AgentResponse:
        saw_shortcut = any(t.answer == SHORTCUT for t in view.visible_turns)
        if view.shared and saw_shortcut:
            return AgentResponse(content=f"{self.model} agrees with the prior answer",
                                 answer=SHORTCUT, confidence=0.9)
        return AgentResponse(content=f"{self.model} reads the case", answer=CORRECT,
                             confidence=0.7)


def _committee_correct_rate(transcript) -> float:
    answers = [t.answer for t in transcript.turns if t.answer is not None]
    if not answers:
        return 0.0
    return sum(1 for a in answers if a == CORRECT) / len(answers)


def _running_shortcut_series(transcript) -> list[float]:
    """Per turn, the running fraction of answers so far equal to the shortcut. It climbs as
    the cascade spreads, which is exactly the kind of series the onset detector consumes."""
    series, hits, seen = [], 0, 0
    for t in transcript.turns:
        if t.answer is None:
            continue
        seen += 1
        if t.answer == SHORTCUT:
            hits += 1
        series.append(hits / seen)
    return series


def run_smoke() -> dict:
    """Run the whole pipeline offline on synthetic data and return a report dict."""
    report: dict = {}

    # 1. cue injection + solo baselines (text lane): a longest-option cue flips a length-follower.
    cases = _text_cases()
    twins = [build_text_twin(c, "longest_option") for c in cases]
    records = solo_evaluate(twins, backend=_longest_text, answer_fn=lambda x: x, model="mock")
    report["solo_flip_rate"] = flip_rate(records)

    # 2. shared vs isolated committee with a planted shortcut (the cascade).
    committee = cross_lineage_committees(default_roster(), size=3)[0]
    case = Case(case_id="c0", patient_id="p0", modality=Modality.IMAGE, label="pneumothorax")

    def backend_for(spec):
        return _ShortcutProneAgent(model=spec.name)

    # seed the shortcut on the SECOND speaker over several rounds, so early turns are clean
    # and the cascade has a real onset partway through the run.
    second_agent = committee.members[1].name
    shared = run_committee(committee, case, Condition.CONTAMINATED, backend_for,
                           shared=True, seed_turn=(1, SHORTCUT, second_agent), seed=0, rounds=3)
    isolated = run_committee(committee, case, Condition.CONTAMINATED, backend_for,
                             shared=False, seed_turn=(1, SHORTCUT, second_agent), seed=0, rounds=3)
    report["committee_correct_rate_shared"] = _committee_correct_rate(shared)
    report["committee_correct_rate_isolated"] = _committee_correct_rate(isolated)

    # 3. cascade onset on the running-shortcut series from the shared run.
    report["cascade_onset_turn"] = cascade_onset(_running_shortcut_series(shared))

    # 4. referee scoring + gate on the shared (cascaded) transcript.
    truth = {t.agent_id: (t.answer == SHORTCUT) for t in shared.turns if t.agent_id}
    flags = score_shortcut(shared, "seeded")
    report["referee_precision_recall"] = precision_recall(flags, truth)
    gate = gate_decision(shared, planted_cue_type="seeded")
    report["gate_approve"] = gate.approve
    report["gate_reason"] = gate.reason

    # 5. blind-metric probe on synthetic decisions that track a decoy more than the truth.
    decisions = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    decoy = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]      # decisions follow this
    ground_truth = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    uptake = blind_metric_uptake(decisions, decoy, ground_truth)
    report["blind_metric_uptake_delta"] = uptake.uptake_delta

    return report


def format_report(report: dict) -> str:
    lines = ["benchmaxxing end-to-end smoke (offline, synthetic data)", "=" * 52]
    sfr = report["solo_flip_rate"]
    overall = sfr.get("overall", sfr) if isinstance(sfr, dict) else sfr
    lines.append(f"1. solo flip rate (longest-option cue): {overall}")
    lines.append(f"2. committee correct rate  shared={report['committee_correct_rate_shared']:.2f}  "
                 f"isolated={report['committee_correct_rate_isolated']:.2f}  "
                 f"(shared should be lower: the cascade)")
    lines.append(f"3. cascade onset turn: {report['cascade_onset_turn']}")
    lines.append(f"4. referee precision/recall: {report['referee_precision_recall']}")
    lines.append(f"   gate: approve={report['gate_approve']} ({report['gate_reason']})")
    lines.append(f"5. blind-metric uptake delta: {report['blind_metric_uptake_delta']:.3f} "
                 f"(positive = decisions drift toward the decoy)")
    lines.append("all stages ran; the plumbing composes end to end.")
    return "\n".join(lines)


def main() -> int:
    print(format_report(run_smoke()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
