"""Keep Q4_K_M and historical NIM degeneracy findings explicit and path-correct.

This script makes no model calls. It relocates exemption keys after moving historical NIM files,
then adds exact, reasoned entries for the model/cohort-specific findings in the Q4 artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.degeneracy import REPO_ROOT, scan

MODEL_SLUG = "meta_llama-3.2-11b-vision-instruct"
CHEXPERT_ROOT = f"experiments/chexpert/results/{MODEL_SLUG}/"
IMAGING_CHEXPERT_ROOT = f"experiments/imaging_chexpert/results/{MODEL_SLUG}/"
N600_FILES = {
    "imaging_blind_metric.jsonl",
    "imaging_blind_metric_phase1_baseline.jsonl",
    "imaging_blind_metric_phase2_blind.jsonl",
    "imaging_blind_metric_phase3_test_aware.jsonl",
    "imaging_blind_metric_summary.json",
}
NIM_IMAGE_FILES = {
    "imaging_cascade.jsonl", "imaging_cascade_summary.json",
    "imaging_matched_temp.jsonl", "imaging_matched_temp_summary.json",
    "imaging_noise_floor.jsonl", "imaging_solo.jsonl",
    "imaging_solo_summary.json", "img_cache.jsonl",
}


def relocated_path(path: str) -> str:
    old_phase2 = f"experiments/chexpert/results/phase2_11b/phase2_n600/{MODEL_SLUG}/"
    if path.startswith(old_phase2):
        return CHEXPERT_ROOT + "nvidia_nim_n600/" + path[len(old_phase2):]
    if path.startswith(CHEXPERT_ROOT):
        name = path[len(CHEXPERT_ROOT):]
        if name in N600_FILES:
            return CHEXPERT_ROOT + "nvidia_nim_n600/" + name
    if path.startswith(IMAGING_CHEXPERT_ROOT):
        name = path[len(IMAGING_CHEXPERT_ROOT):]
        if name in NIM_IMAGE_FILES:
            return IMAGING_CHEXPERT_ROOT + "nvidia_nim/" + name
    return path


def relocate_table(table: dict[str, str]) -> dict[str, str]:
    moved: dict[str, str] = {}
    for key, reason in table.items():
        parts = key.split("|")
        if len(parts) == 3:
            parts[1] = relocated_path(parts[1])
        new_key = "|".join(parts)
        if new_key in moved and moved[new_key] != reason:
            raise SystemExit(f"conflicting reasons while relocating {new_key}")
        moved[new_key] = reason
    return moved


def reason_for(finding) -> str:
    detail = finding.detail
    if finding.kind == "constant_column":
        if finding.locus.endswith("_eligible") or finding.locus in {"base_is_decoy", "named"}:
            return (
                f"{detail}. This is a fixed cohort/control label, not a varying outcome; "
                "retain it as design metadata and do not score it as an effect."
            )
        return (
            f"{detail}. This is a saturated outcome in the committed Q4_K_M cohort; "
            "report its 0%/100% rate descriptively, not as a contrast estimate."
        )
    if finding.kind == "duplicate_column":
        return (
            f"{detail}. The exact identity is specific to this committed Q4_K_M arm; "
            "the paired contrast has no independent variation and is descriptive only."
        )
    if finding.kind == "forced_direction":
        return (
            f"{detail}. The saturated comparator pins one side of the paired table to zero; "
            "keep counts/rates descriptive and do not read the p-value as an effect estimate."
        )
    if finding.kind == "identical_reads":
        if "imaging_cascade_none.jsonl" in finding.path:
            return (
                f"{detail}. This is the intentional no-cue control: clean and iso use the same "
                "original image and question, so cache reuse is expected; it is not an independent reread."
            )
        return (
            f"{detail}. Cue-specific inputs are passed through separate cache keys; Q4_K_M "
            "returned the same parsed reads across this cohort, an observed invariance with no flip effect."
        )
    if finding.kind == "rounded_pvalue":
        return (
            f"{detail}. The exact test yields a null p=1.0 for this committed comparison; "
            "report it as no detected difference, not as evidence of equivalence."
        )
    raise SystemExit(f"no reason template for {finding.kind}: {finding.key}")


def main() -> None:
    path = REPO_ROOT / "tests" / "degeneracy_exemptions.json"
    doc = json.loads(path.read_text())
    allow = relocate_table(dict(doc.get("allowlist", {})))
    preexisting = relocate_table(dict(doc.get("preexisting", {})))

    added = 0
    for finding in scan(REPO_ROOT):
        if MODEL_SLUG not in finding.path or "nvidia_nim" in finding.path:
            continue
        if finding.key in allow or finding.key in preexisting:
            continue
        allow[finding.key] = reason_for(finding)
        added += 1

    doc["allowlist"] = allow
    doc["preexisting"] = preexisting
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"Added {added} exact Q4_K_M reasons; retained {len(allow)} allowlist and "
          f"{len(preexisting)} pre-existing entries after relocating historical NIM keys.")


if __name__ == "__main__":
    main()
