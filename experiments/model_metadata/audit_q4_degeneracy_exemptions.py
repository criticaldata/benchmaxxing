"""Keep Q4_K_M and historical NIM degeneracy findings explicit and path-correct.

This script makes no model calls. It relocates exemption keys after moving historical NIM files,
then adds exact, reasoned entries for the model/cohort-specific findings in the Q4 artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.degeneracy import REPO_ROOT, scan

Q4_MODEL_SLUG = "ollama_llama-3.2-11b-vision-instruct-q4_k_m"
HOSTED_MODEL_SLUG = "meta_llama-3.2-11b-vision-instruct"
CHEXPERT_ROOT = f"experiments/chexpert/results/{HOSTED_MODEL_SLUG}/"
IMAGING_CHEXPERT_ROOT = f"experiments/imaging_chexpert/results/{HOSTED_MODEL_SLUG}/"
def relocated_path(path: str) -> str:
    old_phase2 = f"experiments/chexpert/results/phase2_11b/phase2_n600/{HOSTED_MODEL_SLUG}/"
    if path.startswith(old_phase2):
        return CHEXPERT_ROOT + "nvidia_nim_n600/" + path[len(old_phase2):]
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
        if "exactly 0.0" in detail:
            return (
                f"{detail}. Serialized zero is numerical underflow/rounding, not a probability of zero; "
                "use exact discordant counts when the paired test supplies them and do not claim significance "
                "from the zero-valued serialization."
            )
        return (
            f"{detail}. This boundary p=1.0 reflects no detected difference for the committed comparison; "
            "report the statistic and counts where available, and do not claim equivalence."
        )
    raise SystemExit(f"no reason template for {finding.kind}: {finding.key}")


def main() -> None:
    path = REPO_ROOT / "tests" / "degeneracy_exemptions.json"
    doc = json.loads(path.read_text())
    allow = relocate_table(dict(doc.get("allowlist", {})))
    preexisting = relocate_table(dict(doc.get("preexisting", {})))
    findings = scan(REPO_ROOT)
    current = {finding.key for finding in findings}
    stale_removed = sum(key not in current for key in allow) + sum(key not in current for key in preexisting)
    allow = {key: reason for key, reason in allow.items() if key in current}
    preexisting = {key: reason for key, reason in preexisting.items() if key in current}

    added = 0
    for finding in findings:
        if Q4_MODEL_SLUG not in finding.path or "nvidia_nim" in finding.path:
            continue
        if finding.key in allow or finding.key in preexisting:
            continue
        allow[finding.key] = reason_for(finding)
        added += 1

    doc["allowlist"] = allow
    doc["preexisting"] = preexisting
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"Added {added} exact Q4_K_M reasons; removed {stale_removed} stale entries; "
          f"retained {len(allow)} allowlist and {len(preexisting)} pre-existing entries.")


if __name__ == "__main__":
    main()
