"""Parity of a second lineage with the sibling lineages, as a test rather than an audit.

The owner's rule for a lineage PR is apples to apples: every text-lane result file any other
lineage committed must exist for this lineage too, at the same path, so the cross-lineage
comparisons are over the same arms. Imaging lanes are excluded for a text-only model. One naming
divergence is declared rather than hidden: the Qwen lineage keeps its 600-case MIMIC solo as
results/<model>/solo_results_600.json while Gemini and gpt-oss keep it under results_n600/<model>/.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = "nvidia_nemotron-3-super-120b-a12b"
TEXT_ONLY = {LINEAGE}
IMAGING_LANES = {"imaging", "imaging_chexpert", "chexpert", "mimic_cxr_image"}
# sibling path -> the path this lane uses for the same artifact
ALIASES = {
    "results/{slug}/solo_results_600.json": "results_n600/{slug}/solo_results.json",
}
# Files a lineage cannot produce from the repository, declared with the reason so the gap is visible
# rather than hidden by an alias. Key: (lineage slug, lane, results dir, file name). An entry must say
# what is missing and what would close it; the test below checks that the reason is a real sentence.
DECLARED_GAPS = {
    ("nvidia_nemotron-3-super-120b-a12b", "referee", "results", "referee_deployable_declared_audit.json"):
        "the nemotron referee call cache was never committed (the arm ran on a machine whose caches are "
        "lost), so the declared-only rescore of the deployable referee cannot be replayed from the "
        "repository; closing it means re-running that arm, which replaces committed numbers and needs "
        "the owner's decision. The plausible-distractor audit for the same lineage is committed.",
    ("nvidia_nemotron-3-super-120b-a12b", "blind_metric", "results", "counterfactual_referee.json"):
        "the derived analysis experiments/blind_metric/counterfactual_referee.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "cascade", "results", "positional_regression.json"):
        "the derived analysis experiments/cascade/positional_regression.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "contamination", "results", "misaligned_proxy.json"):
        "the derived analysis experiments/contamination/misaligned_proxy.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "deference_reanalysis.json"):
        "the derived analysis experiments/medqa/deference_reanalysis.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "onset_distribution.json"):
        "the derived analysis experiments/medqa/onset_distribution.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "stats_reconciliation_summary.json"):
        "the derived analysis experiments/medqa/stats_reconciliation.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "referee", "results", "net_harm.json"):
        "the derived analysis experiments/referee/net_harm.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "referee", "results", "panel_vote.json"):
        "the derived analysis experiments/referee/panel_vote.py gained a --model flag on the gpt-oss branches "
        "after this lineage merged; it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b "
        "on this lineage's committed rows closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "empty_content_by_condition.json"):
        "the keyless replay experiments/medqa/empty_content_by_condition.py classifies a model's empty completions "
        "by prompt condition from its committed caches; it was written for the gpt-oss lineages, whose completions come "
        "back empty when the hidden-reasoning budget is exhausted, and whether this lineage produces empty completions "
        "has not been checked, so running it belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "empty_content_tally.json"):
        "the per-arm tally of empty completions is written by the gpt-oss-20b lineage because that model exhausts "
        "its hidden-reasoning budget on some calls and returns no answer; whether this lineage ever returns an "
        "empty completion has not been checked, so producing the tally belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "reasoning_effort_sensitivity_summary.json"):
        "experiments/medqa/reasoning_effort_sensitivity.py sweeps the reasoning_effort field of the chat API, which "
        "this lineage's serving stack does not expose; closing it needs a served nemotron with an equivalent knob "
        "and belongs to that lineage's pass.",
}
PAT = re.compile(r"experiments/([a-z_0-9]+)/(results[^/]*)/([^/]+)/([^/]+\.json)$")


def _result_files():
    out = {}
    for p in ROOT.glob("experiments/*/results*/*/*.json"):
        m = PAT.match(p.relative_to(ROOT).as_posix())
        if not m or "cache" in m.group(4):
            continue
        lane, res, slug, name = m.groups()
        out.setdefault(slug, set()).add((lane, res, name))
    return out


def _siblings(files):
    return {s for s in files if s != LINEAGE and s not in ("n100",)}


@pytest.mark.parametrize("lineage", sorted(TEXT_ONLY))
def test_the_lineage_has_every_text_lane_file_a_sibling_has(lineage):
    files = _result_files()
    mine = files.get(lineage, set())
    missing = []
    for sib in _siblings(files):
        for lane, res, name in files[sib]:
            if lane in IMAGING_LANES:
                continue
            want = (lane, res, name)
            alias = ALIASES.get(f"{res}/{{slug}}/{name}")
            if alias:
                ares, aname = alias.format(slug=lineage).split("/")[0], alias.split("/")[-1]
                want = (lane, ares, aname)
            if want not in mine and (lineage, lane, res, name) not in DECLARED_GAPS:
                missing.append(f"{lane}/{res}/{sib}/{name}")
    assert not missing, "sibling result files with no counterpart for " + lineage + ":\n  " + "\n  ".join(sorted(missing))


def test_every_declared_gap_carries_a_reason_and_names_a_real_lineage():
    for (lineage, lane, res, name), why in DECLARED_GAPS.items():
        assert lineage in TEXT_ONLY, lineage
        assert len(why.split()) >= 25, f"{lineage}/{lane}/{res}/{name}: the reason has to say what is missing and what would close it"
