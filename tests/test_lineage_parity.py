"""Parity of a second lineage with the sibling lineages, as a test rather than an audit.

The owner's rule for a lineage PR is apples to apples: every text-lane result file any other
lineage committed must exist for this lineage too, at the same path, so the cross-lineage
comparisons are over the same arms. Imaging lanes are excluded for a text-only model. One naming
divergence is declared rather than hidden: the Qwen lineage keeps its 600-case MIMIC solo as
results/<model>/solo_results_600.json while Gemini and gpt-oss keep it under results_n600/<model>/.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = "nvidia_nemotron-3-super-120b-a12b"
OLLAMA_Q4_LINEAGE = "ollama_llama-3.2-11b-vision-instruct-q4_k_m"
HOSTED_LLAMA_RESULT_SLUG = "meta_llama-3.2-11b-vision-instruct"
PARITY_LINEAGES = {LINEAGE, OLLAMA_Q4_LINEAGE}
Q4_EXCLUDED_LANES = {"mimic_cxr_image", "mimic_cxr_text"}
Q4_ALIASES = {
    ("imaging", "results", "imaging_cascade_summary.json"):
        ("imaging", "results", "imaging_cascade_none_summary.json"),
}
Q4_GAP_LEDGER_PATH = ROOT / "tests/ollama_q4_declared_gaps.json"
Q4_METADATA_PATH = ROOT / "experiments/model_metadata/ollama_llama-3.2-11b-vision-instruct-q4_k_m.json"
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
    # hidden-note arms run self-hosted under the fp8-local lineage
    ("nvidia_nemotron-3-super-120b-a12b", "blind_metric", "results", "matched_baseline_reason_repeats_summary.json"):
        "The hosted endpoint for this model id has been withdrawn, so this hidden-note arm cannot be run under the hosted id. It is run on the same FP8 weights served locally and committed under the separate lineage nemotron-3-super-120b-a12b-fp8-local, so hosted and self-hosted answers keep separate ids and caches; this entry closes if the hosted id is served again.",
    ("nvidia_nemotron-3-super-120b-a12b", "medmcqa", "results", "blind_metric_summary.json"):
        "The hosted endpoint for this model id has been withdrawn, so this hidden-note arm cannot be run under the hosted id. It is run on the same FP8 weights served locally and committed under the separate lineage nemotron-3-super-120b-a12b-fp8-local, so hosted and self-hosted answers keep separate ids and caches; this entry closes if the hosted id is served again.",
    ("nvidia_nemotron-3-super-120b-a12b", "support2", "results", "blind_metric_summary.json"):
        "The hosted endpoint for this model id has been withdrawn, so this hidden-note arm cannot be run under the hosted id. It is run on the same FP8 weights served locally and committed under the separate lineage nemotron-3-super-120b-a12b-fp8-local, so hosted and self-hosted answers keep separate ids and caches; this entry closes if the hosted id is served again.",
    ("nvidia_nemotron-3-super-120b-a12b", "support2", "results", "matched_baseline_reason_repeats_summary.json"):
        "The hosted endpoint for this model id has been withdrawn, so this hidden-note arm cannot be run under the hosted id. It is run on the same FP8 weights served locally and committed under the separate lineage nemotron-3-super-120b-a12b-fp8-local, so hosted and self-hosted answers keep separate ids and caches; this entry closes if the hosted id is served again.",
    ("nvidia_nemotron-3-super-120b-a12b", "support2", "results", "matched_baseline_reason_summary.json"):
        "The hosted endpoint for this model id has been withdrawn, so this hidden-note arm cannot be run under the hosted id. It is run on the same FP8 weights served locally and committed under the separate lineage nemotron-3-super-120b-a12b-fp8-local, so hosted and self-hosted answers keep separate ids and caches; this entry closes if the hosted id is served again.",
    ("nvidia_nemotron-3-super-120b-a12b", "medmcqa", "results", "matched_baseline_reason_repeats_summary.json"):
        "The hosted endpoint for this model id has been withdrawn, so this hidden-note arm cannot be run under the hosted id. It is run on the same FP8 weights served locally and committed under the separate lineage nemotron-3-super-120b-a12b-fp8-local, so hosted and self-hosted answers keep separate ids and caches; this entry closes if the hosted id is served again.",
    ("nvidia_nemotron-3-super-120b-a12b", "medmcqa", "results", "matched_baseline_reason_summary.json"):
        "The hosted endpoint for this model id has been withdrawn, so this hidden-note arm cannot be run under the hosted id. It is run on the same FP8 weights served locally and committed under the separate lineage nemotron-3-super-120b-a12b-fp8-local, so hosted and self-hosted answers keep separate ids and caches; this entry closes if the hosted id is served again.",
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
    ("nvidia_nemotron-3-super-120b-a12b", "blind_metric", "results", "blind_metric_ci.json"):
        "the derived analysis experiments/blind_metric/reanalysis.py gained a --model flag on the gpt-oss branches after this lineage merged; "
        "it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b on this lineage's committed rows "
        "closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "referee", "results", "referee_agreement.json"):
        "the derived analysis experiments/referee/agreement.py gained a --model flag on the gpt-oss branches after this lineage merged; "
        "it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b on this lineage's committed rows "
        "closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medmcqa", "results", "onset_distribution.json"):
        "the derived analysis experiments/medqa/onset_distribution.py --results-dir experiments/medmcqa/results --label MedMCQA gained a --model flag on the gpt-oss branches after this lineage merged; "
        "it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b on this lineage's committed rows "
        "closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medmcqa", "results", "stats_reconciliation_summary.json"):
        "the derived analysis experiments/medqa/stats_reconciliation.py --results-dir experiments/medmcqa/results --out experiments/medmcqa/results gained a --model flag on the gpt-oss branches after this lineage merged; "
        "it makes no model call, so one run with --model nvidia/nemotron-3-super-120b-a12b on this lineage's committed rows "
        "closes the gap, which belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "empty_content_tally.json"):
        "the per-arm tally of empty completions is written by the gpt-oss-20b lineage because that model exhausts "
        "its hidden-reasoning budget on some calls and returns no answer; whether this lineage ever returns an "
        "empty completion has not been checked, so producing the tally belongs to that lineage's own pass.",
    ("nvidia_nemotron-3-super-120b-a12b", "medqa", "results", "reasoning_effort_sensitivity_summary.json"):
        "experiments/medqa/reasoning_effort_sensitivity.py sweeps the reasoning_effort field of the chat API, which "
        "this lineage's serving stack does not expose; closing it needs a served nemotron with an equivalent knob "
        "and belongs to that lineage's pass.",
}

DECLARED_GAPS.update({('nvidia_nemotron-3-super-120b-a12b', 'blind_metric', 'results', 'run_manifest.json'): 'This is a '
                                                                                        'model-specific '
                                                                                        'execution manifest '
                                                                                        'from the hosted '
                                                                                        'Llama lineage, and '
                                                                                        'the Nemotron lane '
                                                                                        'has no counterpart '
                                                                                        'file. Close the gap '
                                                                                        'by recording and '
                                                                                        'committing a '
                                                                                        'Nemotron '
                                                                                        'blind-metric run '
                                                                                        'manifest; do not '
                                                                                        'reuse the Llama '
                                                                                        'manifest.',
 ('nvidia_nemotron-3-super-120b-a12b', 'contamination', 'results', 'run_manifest_contamination_audit.json'): 'This '
                                                                                                             'is '
                                                                                                             'the '
                                                                                                             'hosted '
                                                                                                             'Llama '
                                                                                                             'contamination-audit '
                                                                                                             'run '
                                                                                                             'manifest; '
                                                                                                             'the '
                                                                                                             'Nemotron '
                                                                                                             'result '
                                                                                                             'directory '
                                                                                                             'has '
                                                                                                             'no '
                                                                                                             'corresponding '
                                                                                                             'run '
                                                                                                             'record. '
                                                                                                             'Close '
                                                                                                             'the '
                                                                                                             'gap '
                                                                                                             'by '
                                                                                                             'recording '
                                                                                                             'and '
                                                                                                             'committing '
                                                                                                             'a '
                                                                                                             'Nemotron-specific '
                                                                                                             'manifest '
                                                                                                             'for '
                                                                                                             'the '
                                                                                                             'same '
                                                                                                             'audit '
                                                                                                             'protocol.',
 ('nvidia_nemotron-3-super-120b-a12b', 'medqa', 'results', 'majority_pressure_summary_pre_validation.json'): 'This '
                                                                                                             'pre-validation '
                                                                                                             'majority-pressure '
                                                                                                             'summary '
                                                                                                             'exists '
                                                                                                             'only '
                                                                                                             'for '
                                                                                                             'the '
                                                                                                             'Llama '
                                                                                                             'run, '
                                                                                                             'with '
                                                                                                             'no '
                                                                                                             'Nemotron '
                                                                                                             'result '
                                                                                                             'counterpart. '
                                                                                                             'Close '
                                                                                                             'the '
                                                                                                             'gap '
                                                                                                             'by '
                                                                                                             'running '
                                                                                                             'the '
                                                                                                             'same '
                                                                                                             'pre-validation '
                                                                                                             'arm '
                                                                                                             'on '
                                                                                                             'Nemotron '
                                                                                                             'or '
                                                                                                             'recording '
                                                                                                             'that '
                                                                                                             'the '
                                                                                                             'protocol '
                                                                                                             'is '
                                                                                                             'not '
                                                                                                             'applicable.',
 ('nvidia_nemotron-3-super-120b-a12b', 'medqa', 'results', 'run_manifest_live_peer_organic.json'): 'This '
                                                                                                   'execution '
                                                                                                   'manifest '
                                                                                                   'documents '
                                                                                                   'the '
                                                                                                   'Llama '
                                                                                                   'live-peer-organic '
                                                                                                   'arm and '
                                                                                                   'has no '
                                                                                                   'Nemotron '
                                                                                                   'counterpart. '
                                                                                                   'Close '
                                                                                                   'the gap '
                                                                                                   'by '
                                                                                                   'running '
                                                                                                   'that arm '
                                                                                                   'for '
                                                                                                   'Nemotron '
                                                                                                   'and '
                                                                                                   'committing '
                                                                                                   'a '
                                                                                                   'model-specific '
                                                                                                   'manifest '
                                                                                                   'with its '
                                                                                                   'cohort '
                                                                                                   'IDs.',
 ('nvidia_nemotron-3-super-120b-a12b', 'medqa', 'results', 'run_manifest_majority_pressure.json'): 'This '
                                                                                                   'execution '
                                                                                                   'manifest '
                                                                                                   'documents '
                                                                                                   'the '
                                                                                                   'Llama '
                                                                                                   'majority-pressure '
                                                                                                   'arm and '
                                                                                                   'has no '
                                                                                                   'Nemotron '
                                                                                                   'counterpart. '
                                                                                                   'Close '
                                                                                                   'the gap '
                                                                                                   'by '
                                                                                                   'running '
                                                                                                   'that arm '
                                                                                                   'for '
                                                                                                   'Nemotron '
                                                                                                   'and '
                                                                                                   'committing '
                                                                                                   'a '
                                                                                                   'model-specific '
                                                                                                   'manifest '
                                                                                                   'with its '
                                                                                                   'cohort '
                                                                                                   'IDs.',
 ('nvidia_nemotron-3-super-120b-a12b', 'medqa', 'results', 'run_manifest_reproduce_solo.json'): 'This '
                                                                                                'execution '
                                                                                                'manifest '
                                                                                                'documents '
                                                                                                'the Llama '
                                                                                                'solo-reproduction '
                                                                                                'arm and has '
                                                                                                'no Nemotron '
                                                                                                'counterpart. '
                                                                                                'Close the '
                                                                                                'gap by '
                                                                                                'running the '
                                                                                                'same solo '
                                                                                                'protocol '
                                                                                                'for '
                                                                                                'Nemotron '
                                                                                                'and '
                                                                                                'committing '
                                                                                                'its case-ID '
                                                                                                'manifest.',
 ('nvidia_nemotron-3-super-120b-a12b', 'support2', 'results', 'queue_status.json'): 'This queue status file '
                                                                                    'records the hosted '
                                                                                    'Llama SUPPORT2 '
                                                                                    'execution and is not '
                                                                                    'present for Nemotron. '
                                                                                    'Close the gap by '
                                                                                    'recording a Nemotron '
                                                                                    'queue status if that '
                                                                                    'arm is run; the Llama '
                                                                                    'status cannot stand in '
                                                                                    'for it.'})

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


# results subdirectories that are cohorts or arms, not model slugs: n100 (the 100-case blind cohort)
# and matched (the matched-conditions arm, whose lineage runs sit one level below it)
NOT_A_LINEAGE = ("n100", "matched")


def _siblings(files):
    return {s for s in files if s != LINEAGE and s not in NOT_A_LINEAGE}


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

def _q4_missing_hosted_paths(files):
    mine = files.get(OLLAMA_Q4_LINEAGE, set())
    hosted = files.get(HOSTED_LLAMA_RESULT_SLUG, set())
    missing = set()
    for lane, res, name in hosted:
        if lane in Q4_EXCLUDED_LANES:
            continue
        want = (lane, res, name)
        want = Q4_ALIASES.get(want, want)
        if want not in mine:
            missing.add((OLLAMA_Q4_LINEAGE, lane, res, name))
    return missing


def test_ollama_q4_is_a_distinct_lineage_with_hosted_parity():
    identity = json.loads(Q4_METADATA_PATH.read_text(encoding="utf-8"))
    assert OLLAMA_Q4_LINEAGE in PARITY_LINEAGES
    assert OLLAMA_Q4_LINEAGE != HOSTED_LLAMA_RESULT_SLUG
    assert identity["lineage_id"] == OLLAMA_Q4_LINEAGE
    assert identity["results_slug"] == OLLAMA_Q4_LINEAGE
    assert identity["hosted_results_slug"] == HOSTED_LLAMA_RESULT_SLUG
    assert identity["mimic_cxr_in_scope"] is False

    files = _result_files()
    assert OLLAMA_Q4_LINEAGE in files and HOSTED_LLAMA_RESULT_SLUG in files
    assert not any(lane in Q4_EXCLUDED_LANES for lane, _, _ in files[OLLAMA_Q4_LINEAGE])
    ledger = json.loads(Q4_GAP_LEDGER_PATH.read_text(encoding="utf-8"))
    declared = {
        (item["lineage_id"], item["lane"], item["results_dir"], item["file"])
        for item in ledger
    }
    assert declared == _q4_missing_hosted_paths(files)
    assert not declared, "all hosted sibling artifact paths should have Q4 counterparts"


def test_q4_closure_manifest_accounts_for_all_30_gaps():
    path = ROOT / "experiments/model_metadata/q4_gap_closure_manifest.json"
    closure = json.loads(path.read_text(encoding="utf-8"))
    assert closure["lineage_id"] == OLLAMA_Q4_LINEAGE
    assert closure["total_gaps"] == closure["closed_artifact_paths"] == 30
    assert closure["unresolved_artifact_paths"] == 0
    assert closure["new_model_calls"] == 0
    assert len(closure["entries"]) == 30
    for entry in closure["entries"]:
        assert entry["status"] == "closed_artifact_path"
        assert entry["lineage_id"] == OLLAMA_Q4_LINEAGE
        assert len(entry["reason"].split()) >= 25
        assert entry["artifact_path"].startswith(f"experiments/{entry['lane']}/")
        artifact = ROOT / entry["artifact_path"]
        assert artifact.is_file(), artifact
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["artifact_sha256"]


def test_q4_chexpert_image_provenance_is_complete():
    path = ROOT / "experiments/imaging_chexpert/results" / OLLAMA_Q4_LINEAGE / "manifest_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    coverage = provenance["provenance_coverage"]
    assert provenance["lineage_id"] == OLLAMA_Q4_LINEAGE
    assert provenance["source_archive"] == "data/chexpert/CheXpert-v1.0-small.zip"
    assert coverage["unique_cohort_images"] == coverage["hashed_images"] == 244
    assert coverage["missing_hash_count"] == 0
    assert coverage["complete"] is True
    assert coverage["missing_case_ids"] == []
    assert len(provenance["images"]) == 244
    assert all(len(row["sha256"]) == 64 for row in provenance["images"].values())


def test_q4_imaging_results_report_baseline_accuracy():
    baseline = json.loads((ROOT / "experiments/model_metadata/q4_image_baseline_accuracy.json").read_text())
    assert baseline["lineage_id"] == OLLAMA_Q4_LINEAGE
    assert len(baseline["arms"]) == 30
    nih = next(row for row in baseline["arms"] if row["case_set_key"] == "imaging/imaging_solo")
    assert (nih["baseline_correct"], nih["n"], nih["display"]) == (1, 35, "1/35 (2.9%)")
    hidden_nih = next(row for row in baseline["supplemental_hidden_note_arms"]
                      if row["dataset"] == "NIH ChestX-ray14" and row["arm"] == "imaging_blind_metric")
    assert hidden_nih["baseline_source_file"].endswith("/imaging_solo.jsonl")
    assert hidden_nih["baseline_field"] == "clean"

    for lane in ("imaging", "imaging_chexpert", "chexpert"):
        directory = ROOT / "experiments" / lane / "results" / OLLAMA_Q4_LINEAGE
        for path in directory.glob("*summary*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            assert report.get("lineage_id") == OLLAMA_Q4_LINEAGE, path
            if path.name == "selection_summary.json":
                assert report["cohorts"]
                assert all("baseline_accuracy" in item["observed_q4_cascade"]
                           for item in report["cohorts"].values())
            else:
                assert "baseline_accuracy" in report, path
                base = report["baseline_accuracy"]
                assert 0 <= base["correct"] <= base["n"]
                assert base["display"] == f"{base['correct']}/{base['n']} ({base['accuracy']*100:.1f}%)"
                assert base.get("field") != "decoy", path

    derived = ROOT / "experiments/imaging/results" / OLLAMA_Q4_LINEAGE
    for name in ("claim4_quantification.json", "counterfactual_referee.json",
                 "effect_sizes_imaging.json", "finding_subgroup.json",
                 "imaging_blind_metric_ci.json", "imaging_referee_agreement.json",
                 "misaligned_proxy.json", "net_harm.json", "onset_battery.json",
                 "panel_vote.json", "positional_regression.json"):
        report = json.loads((derived / name).read_text(encoding="utf-8"))
        assert "baseline_accuracy" in report, name

    docs = (ROOT / "docs/ollama_q4_imaging_results.md").read_text(encoding="utf-8")
    assert "1 of 35" in docs and "1/35" in docs
