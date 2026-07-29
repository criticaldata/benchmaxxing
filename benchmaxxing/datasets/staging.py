"""Dataset acquisition and staging with a provenance record (issue 103).

An adapter turns a raw release into a manifest. Staging is the step around it: where the raw
release came from, whether it needed credentials, what the resulting manifest contains, and
whether it validated. Without that, a manifest on disk months later is an anonymous CSV and no
result built on it can be traced back to a source.

One call does the whole thing::

    benchmaxxing datasets stage nih_cxr14 --raw-root /data/nih --check-images

which runs the registered adapter, validates the manifest it produced (reusing
``benchmaxxing.validate``), checksums it, and writes ``provenance.json`` plus a short
``SOURCE.txt`` next to it. Raw data and credentials never enter the repo; only code, docs and
checksums do.

Raw releases live under a dataset root, ``$BENCHMAXXING_DATASET_ROOT`` (default ``data/``, which
is gitignored), one directory per dataset.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import registry
from benchmaxxing.validate import manifest_checksum, validate_manifest

__all__ = [
    "SOURCES",
    "Source",
    "dataset_root",
    "raw_dir",
    "stage_dataset",
    "provenance_text",
]

DATASET_ROOT_ENV = "BENCHMAXXING_DATASET_ROOT"
DEFAULT_DATASET_ROOT = "data"


@dataclass(frozen=True)
class Source:
    """Where one dataset comes from and what it costs to get it.

    ``access`` is the gate a contributor has to clear: ``open`` (just download), ``registration``
    (an account and a signed licence), or ``credentialed`` (identity verification and a training
    course, i.e. PhysioNet). It is the field that decides whether an arm is blocked on data.
    """

    name: str
    url: str
    access: str
    license: str
    layout: str
    notes: str = ""


SOURCES: dict[str, Source] = {
    "medqa": Source(
        name="MedQA-USMLE",
        url="https://github.com/jind11/MedQA",
        access="open",
        license="MIT (see the release repository)",
        layout="data_clean/questions/US/{train,dev,test}.jsonl",
        notes="Staged and verified: train 10178 / dev 1272 / test 1273.",
    ),
    "medmcqa": Source(
        name="MedMCQA",
        url="https://medmcqa.github.io/",
        access="open",
        license="MIT",
        layout="{train,dev,test}.json (one JSON object per line)",
    ),
    "pubmedqa": Source(
        name="PubMedQA",
        url="https://pubmedqa.github.io/",
        access="open",
        license="MIT",
        layout="ori_pqal.json (labelled subset)",
    ),
    "nih_cxr14": Source(
        name="NIH ChestX-ray14",
        url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        access="open",
        license="NIH Clinical Center open access, cite Wang et al. 2017",
        layout="images_XXX/images/*.png plus Data_Entry_2017.csv",
        notes="Downloads in 2-4 GB batches; one batch is enough to unblock Lane A locally.",
    ),
    "chexpert": Source(
        name="CheXpert-small",
        url="https://stanfordmlgroup.github.io/competitions/chexpert/",
        access="registration",
        license="Stanford University research use agreement",
        layout="CheXpert-v1.0-small/{train,valid}.csv plus the patient image tree",
        notes="Needed for the natural Support-Devices cue arm (#12, #94). About 11 GB.",
    ),
    "mimic_cxr": Source(
        name="MIMIC-CXR-JPG",
        url="https://physionet.org/content/mimic-cxr-jpg/",
        access="credentialed",
        license="PhysioNet credentialed health data licence",
        layout="mimic-cxr-2.0.0-metadata.csv plus files/pXX/pXXXXXXXX/sYYYYYYYY/*.jpg",
        notes="Credentialing plus CITI training. Stage a small subset before any full run (#92).",
    ),
    "ehr": Source(
        name="MIMIC-IV derived resource table",
        url="https://physionet.org/content/mimiciv/",
        access="credentialed",
        license="PhysioNet credentialed health data licence",
        layout="a CSV of resource-constraint contexts (bed occupancy, staffing, budget pressure)",
        notes="Feeds the stage-5 scrutiny panel, not a case manifest.",
    ),
    "mimic_cxr_text": Source(
        name="MIMIC-CXR report text + CheXpert labels",
        url="https://physionet.org/content/mimic-cxr/",
        access="credentialed",
        license="PhysioNet credentialed health data licence",
        layout=(
            "free-text reports (pXX/pYYYY/sZZZZ.txt) plus mimic-cxr-2.0.0-chexpert.csv[.gz] "
            "(from the separate mimic-cxr-jpg release's label file) -- two roots, not one, "
            "see build_manifest's docstring"
        ),
        notes="Staged and verified: full #296 battery (#316-#321) run against the real API.",
    ),
}


def dataset_root(root=None) -> Path:
    """The directory raw releases are staged under: the argument, ``$BENCHMAXXING_DATASET_ROOT``,
    or ``data/``."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get(DATASET_ROOT_ENV) or DEFAULT_DATASET_ROOT)


def raw_dir(name: str, root=None) -> Path:
    """Where one dataset's raw release is expected to live."""
    return dataset_root(root) / name


def _case_counts(manifest_path) -> dict:
    """Row counts by modality plus the label distribution, for the provenance record."""
    cases = load_cases(manifest_path)
    return {
        "n_cases": len(cases),
        "by_modality": dict(Counter(case.modality.value for case in cases)),
        "by_label": dict(Counter(case.label for case in cases if case.label).most_common(20)),
        "n_with_meta": sum(1 for case in cases if case.meta),
    }


def stage_dataset(name: str, raw_root=None, out=None, *, limit: int | None = None,
                  root=None, check_images: bool = False) -> dict:
    """Build, validate and record a manifest for one registered dataset.

    Runs the adapter's ``build_manifest``, validates the result with
    :func:`benchmaxxing.validate.validate_manifest`, checksums it, and writes the provenance
    record next to the manifest. Returns the provenance dict.

    Raises ``KeyError`` for an unknown dataset, ``FileNotFoundError`` when the raw release is not
    where it was expected, and ``ValueError`` when the manifest the adapter produced does not
    validate, because a manifest that fails validation should not become the input to a run.
    """
    module = registry.get(name)
    source = SOURCES.get(name)
    raw = Path(raw_root) if raw_root is not None else raw_dir(name, root)
    if not raw.exists():
        hint = f" Expected the raw release at {raw}."
        if source:
            hint += f" Source: {source.url} (access: {source.access})."
        raise FileNotFoundError(f"No raw data for {name!r}.{hint}")

    manifest_path = Path(out) if out is not None else raw.parent / f"{name}_manifest.csv"
    module.build_manifest(raw, manifest_path, limit=limit)

    report = validate_manifest(manifest_path, check_images=check_images, root=raw)
    if not report.is_clean:
        detail = "; ".join(str(problem) for problem in report.problems[:5])
        raise ValueError(
            f"the manifest built for {name!r} did not validate ({len(report.problems)} "
            f"problem(s)): {detail}"
        )

    import benchmaxxing
    from benchmaxxing.manifest import git_sha

    provenance = {
        "dataset": name,
        "source": asdict(source) if source else None,
        "staged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_root": str(raw),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_checksum(manifest_path),
        "limit": limit,
        "adapter": f"{module.__name__} (SPEC {module.SPEC.name})",
        "modality": module.SPEC.modality.value,
        "benchmaxxing": benchmaxxing.__version__,
        "git_sha": git_sha(),
        "validation": {
            "clean": report.is_clean,
            "n_cases": report.n_cases,
            "n_images_checked": report.n_images_checked,
            "n_missing_images": report.n_missing_images,
            "images_checked": check_images,
        },
        "counts": _case_counts(manifest_path),
    }

    out_dir = manifest_path.parent
    (out_dir / f"{name}_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / f"{name}_SOURCE.txt").write_text(provenance_text(provenance), encoding="utf-8")
    return provenance


def provenance_text(provenance: dict) -> str:
    """The human-readable stanza written as SOURCE.txt beside a staged manifest."""
    source = provenance.get("source") or {}
    counts = provenance.get("counts", {})
    lines = [
        f"dataset:   {provenance['dataset']}",
        f"source:    {source.get('name', 'unknown')} <{source.get('url', 'unknown')}>",
        f"access:    {source.get('access', 'unknown')}",
        f"license:   {source.get('license', 'unknown')}",
        f"staged:    {provenance['staged_at']} by benchmaxxing {provenance['benchmaxxing']} "
        f"(git {provenance['git_sha']})",
        f"raw root:  {provenance['raw_root']}",
        f"manifest:  {provenance['manifest']}",
        f"sha256:    {provenance['manifest_sha256']}",
        f"rows:      {counts.get('n_cases', 0)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(counts.get('by_modality', {}).items()))})",
    ]
    if provenance.get("limit") is not None:
        lines.append(f"limit:     {provenance['limit']} (a subset, not the full release)")
    if source.get("notes"):
        lines.append(f"notes:     {source['notes']}")
    lines.append("")
    lines.append("Raw data is not committed to this repository. Re-stage it from the source above.")
    return "\n".join(lines) + "\n"
