"""No credentialed-dataset identifier may be committed.

MIMIC-CXR is PhysioNet-restricted. Its per-case rows used to be committed with `case_id` values of
the form ``mimic-cxr-text-<subject_id>-<study_id>``, which are the real MIMIC subject and study
identifiers: 633 distinct subjects across nine files. The imaging lane had already decided this the
other way, gitignoring its transcripts because they key MIMIC dicom_ids to reads and committing
`results/deid/*.csv` on a `case_index` instead (#372). The text lane never got the same treatment.

The rows themselves are worth keeping, because every published number from that lane is computed
from the outcome flags and none of it from `case_id`, so swapping in a `case_index` costs no
reproducibility and removes only the linkage back to credentialed records.

The runners still emit raw ids, so without this test the next run silently reintroduces them. That
is the whole reason it exists: the guard, not the cleanup, is what makes the property hold.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# subject_id and study_id as MIMIC-CXR issues them: 8 digits each.
MIMIC_CASE_ID = re.compile(r"mimic-cxr-text-\d{8}-\d{8}")
# p10000032 / s50414267 as they appear in MIMIC directory layout.
MIMIC_DIR_ID = re.compile(r"\bp1\d{7}\b|\bs5\d{7}\b")

# The adapter and its tests must name the id shape to parse and to test it; that is code, not data.
ALLOWED = {
    "benchmaxxing/datasets/mimic_cxr.py",
    "tests/test_mimic_cxr_adapter.py",
    "tests/test_no_credentialed_identifiers.py",
    "experiments/mimic_cxr_image/.gitignore",
    "experiments/mimic_cxr_image/results/deid/README.md",
    # Synthetic fixtures, not real cases. Verified against the pre-migration solo file: neither
    # 10000032-50000001 nor 10009999-50000009 appears among the 600 real ids.
    "tests/test_mimic_cxr_text_adapter.py",
    # Same: the index-map tests must name the id shape they rank. 10000000-50000000 and
    # 10000001-50000001 were checked against the full 633-id pre-migration union, not just the
    # 600-row solo sample, and appear in neither.
    "tests/test_mimic_cxr_text_case_index.py",
}

# Every id-shaped string in the ALLOWED files above. Each was checked against the full 633-id
# pre-migration union (not just the 600-row solo sample) and appears in none of it. Listed
# explicitly so the history scan can subtract them: a blob-level scan cannot tell which file a
# blob belongs to, so path-based allowlisting does not reach it.
SYNTHETIC_FIXTURE_IDS = {
    "mimic-cxr-text-10000000-50000000",
    "mimic-cxr-text-10000001-50000001",
    "mimic-cxr-text-10000032-50000001",
    "mimic-cxr-text-10009999-50000009",
}


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    return [p for p in out.stdout.splitlines() if p]


def test_no_mimic_case_ids_in_tracked_files():
    offenders: list[str] = []
    for rel in _tracked():
        if rel in ALLOWED:
            continue
        path = REPO_ROOT / rel
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        hits = len(MIMIC_CASE_ID.findall(text))
        if hits:
            offenders.append(f"{rel}: {hits} MIMIC subject/study id(s)")
    assert not offenders, (
        "MIMIC-CXR identifiers are committed. Replace `case_id` with the stable `case_index` the "
        "rest of the lane uses; the numbers are computed from the outcome flags and do not need "
        "the id:\n" + "\n".join(f"  {o}" for o in offenders)
    )


def test_no_mimic_directory_identifiers_in_results():
    offenders: list[str] = []
    for rel in _tracked():
        if rel in ALLOWED or "/results/" not in rel:
            continue
        try:
            text = (REPO_ROOT / rel).read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if MIMIC_DIR_ID.search(text):
            offenders.append(rel)
    assert not offenders, (
        "MIMIC directory identifiers (pXXXXXXXX / sXXXXXXXX) in committed results:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


@pytest.mark.history
def test_no_mimic_case_ids_anywhere_in_history():
    """The tip being clean is not the property that matters; the repository being clean is.

    De-identifying the working tree leaves every pre-migration blob fetchable, and because
    `case_index` is a deterministic rank over the sorted union of those ids, history access
    re-identifies every committed row. This scans every blob reachable from every ref.

    Deselected by default (`-m "not history"` in pyproject) because it walks all objects and is
    expected to fail until `scripts/rewrite_history_before_public.sh` has run. It is the gate on
    the visibility flip, so run it explicitly:

        python -m pytest tests/test_no_credentialed_identifiers.py -m history
    """
    objects = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    shas = [line.split()[0] for line in objects if line.strip()]

    check = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=REPO_ROOT, input="\n".join(shas), capture_output=True, text=True, check=True,
    )
    blobs = [ln.split()[0] for ln in check.stdout.splitlines() if ln.endswith(" blob")]

    dump = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT, input="\n".join(blobs).encode(), capture_output=True, check=True,
    )
    found = set(MIMIC_CASE_ID.findall(dump.stdout.decode("utf-8", errors="ignore")))
    # The synthetic fixtures in the ALLOWED test files are themselves blobs, so a blob-level scan
    # sees them. They are id-shaped but not real cases, each verified absent from the full 633-id
    # pre-migration union, so the rewrite is not expected to remove them.
    hits = sorted(found - SYNTHETIC_FIXTURE_IDS)
    assert not hits, (
        f"{len(hits)} distinct MIMIC identifier(s) remain in git history. The tip is clean but "
        "the repository is not: pre-migration blobs are still fetchable and the case_index map is "
        "a rank over these same ids, so every de-identified row is re-linkable. Run "
        "scripts/rewrite_history_before_public.sh, then GitHub Support GC, before flipping public."
    )
