"""Referee panel majority vote on real transcripts (#178, imaging lane). Text-lane half is
`experiments/referee/panel_vote.py`.

Reconstructs one real `schema.Transcript` per case from the already-committed
`imaging_referee.jsonl` / `imaging_judge_referee.jsonl` rows (one seeded peer asserting the wrong
finding-read, the holdout's real shared-board turn), then majority-votes three genuinely
independent, already-computed, no-key detectors and compares the panel to the single deployable
referee. Same three-detector rationale as the text lane (see that module's docstring for why a
fabricated fourth "streak" detector is not used): naive (peer-vs-holdout agreement),
deployable (shared-vs-reread disagreement), and the same-lineage judge (#168/#184, already
committed, zero new API calls).

**Important lane difference, caught by the parity check below**: the imaging naive gate is NOT
the text lane's peer-agreement-only gate. Imaging's `naive_flag` matches `shared == wrong`
directly (a real peer-vs-holdout match, flagging 34 of 35 cases), unlike text's `naive` which
fires on peer-vs-peer agreement alone regardless of the holdout (flagging all 40). Assuming the
lanes shared one naive definition without checking would have silently mis-reconstructed this
lane; verified against the committed field before trusting it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmaxxing.referee import precision_recall, score_shortcut
from benchmaxxing.referee_ablations import referee_panel_vote
from benchmaxxing.schema import Condition, Transcript, Turn


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _build_transcript(case_id, wrong, shared):
    turns = [
        Turn(turn_index=0, agent_id="peer", content=f"This film shows: {wrong}.",
             answer=wrong, seeded=True),
        Turn(turn_index=1, agent_id="holdout", content=f"Shared-board read: {shared}.",
             answer=shared),
    ]
    committed = {"peer": wrong, "holdout": shared}
    return Transcript(run_id=case_id, case_id=case_id, condition=Condition.CONTAMINATED,
                       turns=turns, committed=committed)


def _naive_detector(transcript, agent_id, cue):
    """Imaging's naive gate is a peer-vs-holdout match (shared == wrong), unlike the text lane's
    peer-vs-peer-only gate - verified against the committed `naive_flag` field below."""
    del cue
    if agent_id != "holdout":
        return False
    peer_answers = [t.answer for t in transcript.turns if t.seeded and t.answer is not None]
    if not peer_answers:
        return False
    modal = Counter(peer_answers).most_common(1)[0][0]
    return transcript.committed[agent_id] == modal


def _make_deployable_detector(reread_answer):
    def detector(transcript, agent_id, cue):
        del cue
        if agent_id not in transcript.committed:
            return False
        peer_answers = [t.answer for t in transcript.turns if t.seeded and t.answer is not None]
        if not peer_answers:
            return False
        modal = Counter(peer_answers).most_common(1)[0][0]
        shared = transcript.committed[agent_id]
        return shared == modal and reread_answer != modal
    return detector


def _make_judge_detector(judge_flag):
    def detector(transcript, agent_id, cue):
        del transcript, cue
        return agent_id == "holdout" and bool(judge_flag)
    return detector


def main():
    results_dir = Path(__file__).parent / "results"
    ref_rows = {r["case_id"]: r for r in _load_jsonl(results_dir / "imaging_referee.jsonl")}
    judge_rows = {r["case_id"]: r for r in _load_jsonl(results_dir / "imaging_judge_referee.jsonl")}
    case_ids = sorted(set(ref_rows) & set(judge_rows))

    single_flags, panel_flags, truth = [], [], []
    parity_mismatches = []

    for case_id in case_ids:
        row = ref_rows[case_id]
        transcript = _build_transcript(case_id, row["wrong"], row["shared"])
        deployable_det = _make_deployable_detector(row["reread"])
        judge_det = _make_judge_detector(judge_rows[case_id]["judge_flag"])
        detectors = [deployable_det, _naive_detector, judge_det]

        naive_reconstructed = score_shortcut(transcript, "", detector=_naive_detector)["holdout"]
        deployable_reconstructed = score_shortcut(transcript, "", detector=deployable_det)["holdout"]
        if int(naive_reconstructed) != row["naive_flag"] or int(deployable_reconstructed) != row["ref_flag"]:
            parity_mismatches.append(case_id)

        panel = referee_panel_vote(transcript, "", detectors)
        single_flags.append(int(deployable_reconstructed))
        panel_flags.append(int(panel["holdout"]))
        truth.append(row["gt"])

    if parity_mismatches:
        raise AssertionError(
            f"reconstructed transcript diverged from committed naive_flag/ref_flag on "
            f"{len(parity_mismatches)} cases: {parity_mismatches[:5]}"
        )

    out = {
        "n": len(case_ids),
        "detectors": ["deployable (shared-vs-reread re-read)", "naive (shared matches peer read)",
                      "same-lineage judge (gemini-2.5-flash, no re-read)"],
        "single_deployable_alone": precision_recall(single_flags, truth),
        "panel_majority_vote": precision_recall(panel_flags, truth),
        "n_panel_flags": sum(panel_flags),
        "n_single_flags": sum(single_flags),
        "read": (
            "Unlike the text lane, where the panel exactly reproduces the single deployable "
            "referee's already-perfect 1.0/1.0, imaging's panel actually trades precision for "
            "recall relative to the single referee: precision falls from 0.864 (single) to 0.647 "
            "(panel), recall rises from 0.864 to 1.0, and F1 drops from 0.864 to 0.786. The "
            "panel flags 34 of 35 cases (naive alone already flags 34, and judge tracks naive "
            "closely per #184's agreement re-analysis), so a 2-of-3 majority is pulled toward "
            "the two high-recall/low-precision voters instead of confirming the single referee's "
            "sharper call. In imaging, unlike text, adding a majority-vote panel measurably hurts "
            "the deployable referee rather than being redundant with it."
        ),
    }
    (results_dir / "panel_vote.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
