"""The MIMIC-CXR text cohort rule, pinned on synthetic cases.

Offline: no key, no network, no credentialed data. The real checksums can only be checked with
a PhysioNet copy of the dataset, which `experiments/mimic_cxr_text/build_manifests.py` does.
"""
from __future__ import annotations

import csv
import random

from benchmaxxing.data import load_cases, write_manifest
from benchmaxxing.datasets.mimic_cxr_text import FINDING_COLUMNS, build_manifest
from benchmaxxing.schema import Case, Modality
from experiments.mimic_cxr_text.build_manifests import (
    CASCADE_N,
    CASCADE_PREFIX,
    PREFIX_ARMS_N,
    SOLO_N,
    SOLO_PREFIX,
    checksum,
    describe_inputs,
    select_index_space,
)


def _cases(n=2000):
    return [
        Case(
            case_id=f"synthetic-{i:05d}",
            patient_id=str(i),
            modality=Modality.TEXT,
            label="edema",
            report=f"   REPORT {i}\n finding text \n",
            question="What is the primary finding described in this report?",
            options=("Atelectasis", "Edema", "Fracture", "Pneumonia"),
            answer_index=1,
            meta={},
        )
        for i in range(n)
    ]


def test_index_space_is_the_union_of_the_three_draws_in_manifest_order():
    cases = _cases()
    picked = select_index_space(cases)
    ids = [c.case_id for c in picked]
    order = {c.case_id: i for i, c in enumerate(cases)}
    assert [order[i] for i in ids] == sorted(order[i] for i in ids)
    solo = {c.case_id for c in random.Random(0).sample(cases[:SOLO_PREFIX], SOLO_N)}
    cascade = {c.case_id for c in random.Random(0).sample(cases[:CASCADE_PREFIX], CASCADE_N)}
    first = {c.case_id for c in cases[:PREFIX_ARMS_N]}
    assert set(ids) == solo | cascade | first
    assert ids[:PREFIX_ARMS_N] == [c.case_id for c in cases[:PREFIX_ARMS_N]]
    assert all(order[i] < SOLO_PREFIX for i in ids)


def test_reloading_strips_report_whitespace_so_the_checksum_depends_on_it(tmp_path):
    cases = _cases(50)
    direct = tmp_path / "direct.csv"
    reloaded = tmp_path / "reloaded.csv"
    write_manifest(cases, direct)
    write_manifest(load_cases(direct), reloaded)
    assert direct.read_bytes() != reloaded.read_bytes()
    assert all(c.report == c.report.strip() for c in load_cases(reloaded))
    again = tmp_path / "again.csv"
    write_manifest(load_cases(reloaded), again)
    assert again.read_bytes() == reloaded.read_bytes()


def _dataset(root, line_ending: bytes, n=12):
    """A synthetic report tree and label file; reports end their lines with `line_ending`."""
    reports = root / "files"
    labels = root / "mimic-cxr-2.0.0-chexpert.csv"
    columns = ["subject_id", "study_id", *FINDING_COLUMNS, "No Finding"]
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i in range(n):
            subject, study = str(10_000_000 + i), str(50_000_000 + i)
            row = dict.fromkeys(columns, "")
            row.update(subject_id=subject, study_id=study, Edema="1.0", Atelectasis="0.0",
                       Fracture="0.0", Pneumonia="0.0")
            writer.writerow(row)
            path = reports / f"p{subject[:2]}" / f"p{subject}" / f"s{study}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            body = [b"                 FINAL REPORT", b" EXAMINATION: chest film " + str(i).encode(),
                    b"", b" IMPRESSION: mild edema. ", b""]
            path.write_bytes(line_ending.join(body))
    return reports, labels


def test_a_report_tree_with_windows_line_endings_builds_the_same_bytes(tmp_path):
    """A tree extracted on Windows with CRLF reports must not change any checksum."""
    sums = []
    for name, ending in (("lf", b"\n"), ("crlf", b"\r\n")):
        root = tmp_path / name
        root.mkdir()
        reports, labels = _dataset(root, ending)
        full = root / "full.csv"
        build_manifest(reports, labels, full)
        final = root / "final.csv"
        write_manifest(load_cases(full), final)
        raw = final.read_bytes()
        assert raw.count(b"\r\n") == 12 + 1  # one per row plus the header, on every platform
        sums.append((checksum(full), checksum(final)))
    assert sums[0] == sums[1]


def test_describe_inputs_reports_what_was_read(tmp_path):
    _, labels = _dataset(tmp_path, b"\n", n=3)
    text = describe_inputs(labels, 3)
    assert "label rows      3 " in text and "full manifest   3 cases" in text
