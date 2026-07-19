"""Validate a produced case manifest and stamp it with a content checksum.

A produced manifest is the flat case table that :func:`benchmaxxing.data.load_cases` parses into
``schema.Case`` objects (see :mod:`benchmaxxing.data`). Before a run consumes such a manifest it is
worth confirming that the table is internally consistent: case ids are unique, the fields the
benchmark needs are actually populated, and (optionally) every referenced image file is on disk.

:func:`validate_manifest` performs those checks and *collects* every problem it finds into a report
instead of raising on the first one, so a caller gets the full picture in one pass. A clean manifest
yields an empty problem list. :func:`manifest_checksum` returns a stable sha256 of the raw manifest
bytes, suitable for stamping into a ``RunManifest`` so a run records exactly which manifest it ran
against.

Nothing here imports a model backend or touches the network; it is pure file/consistency checking.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.schema import Case, Modality

# Fields that must be present and non-empty for a case to be usable, keyed by modality. These go a
# step beyond what ``load_cases`` structurally requires: in particular an imaging case must carry a
# ground-truth ``label`` (the diagnosis being scored), which ``load_cases`` tolerates as blank.
_REQUIRED_FIELDS: dict[Modality, tuple[str, ...]] = {
    Modality.IMAGE: ("case_id", "patient_id", "image_ref", "label"),
    Modality.TEXT: ("case_id", "patient_id", "question", "options", "answer_index"),
}

_CHUNK = 1 << 20  # 1 MiB streaming read for the checksum


@dataclass(frozen=True)
class Problem:
    """A single validation issue. ``case_id`` is None when the problem is manifest-wide."""

    kind: str          # "load_error" | "duplicate_case_id" | "missing_field" | "missing_image"
    detail: str
    case_id: str | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        where = f" [{self.case_id}]" if self.case_id is not None else ""
        return f"{self.kind}{where}: {self.detail}"


@dataclass
class ValidationReport:
    """Outcome of validating one manifest: counts plus every problem found (empty if clean)."""

    n_cases: int = 0
    n_unique_case_ids: int = 0
    n_duplicate_case_ids: int = 0
    n_images_checked: int = 0
    n_missing_images: int = 0
    n_missing_fields: int = 0
    problems: list[Problem] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when no problems were collected."""
        return not self.problems

    def problems_of(self, kind: str) -> list[Problem]:
        """Return the collected problems whose ``kind`` matches (handy in tests/callers)."""
        return [p for p in self.problems if p.kind == kind]


def _is_missing(value: object) -> bool:
    """A required value counts as missing when it is None or an empty string/sequence.

    ``answer_index`` of ``0`` is a valid answer, so only ``None`` (not falsiness) marks a numeric
    field missing; empty strings and empty option tuples are treated as missing.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (tuple, list)):
        return len(value) == 0
    return False


def _check_required_fields(case: Case, report: ValidationReport) -> None:
    for name in _REQUIRED_FIELDS.get(case.modality, ()):
        if _is_missing(getattr(case, name, None)):
            report.n_missing_fields += 1
            report.problems.append(
                Problem(
                    kind="missing_field",
                    detail=f"required field {name!r} is empty for a {case.modality.value} case",
                    case_id=case.case_id,
                )
            )


def _check_duplicates(cases: list[Case], report: ValidationReport) -> None:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            duplicated.add(case.case_id)
            report.n_duplicate_case_ids += 1
            report.problems.append(
                Problem(
                    kind="duplicate_case_id",
                    detail=f"case_id {case.case_id!r} appears more than once",
                    case_id=case.case_id,
                )
            )
        else:
            seen.add(case.case_id)
    report.n_unique_case_ids = len(seen)


def _check_images(cases: list[Case], base: Path, report: ValidationReport) -> None:
    for case in cases:
        if case.modality is not Modality.IMAGE or not case.image_ref:
            continue
        report.n_images_checked += 1
        candidate = base / case.image_ref
        if not candidate.is_file():
            report.n_missing_images += 1
            report.problems.append(
                Problem(
                    kind="missing_image",
                    detail=f"image_ref {case.image_ref!r} not found at {candidate}",
                    case_id=case.case_id,
                )
            )


def validate_manifest(
    path: str | Path,
    *,
    check_images: bool = False,
    root: str | Path | None = None,
) -> ValidationReport:
    """Validate the manifest at ``path`` and return a :class:`ValidationReport`.

    Checks performed:

    * the manifest parses via :func:`benchmaxxing.data.load_cases`;
    * every ``case_id`` is unique;
    * the required fields for each modality are present and non-empty (imaging cases must carry a
      ground-truth ``label``);
    * when ``check_images`` is set, each imaging ``image_ref`` resolves to a file. Relative refs are
      resolved against ``root`` if given, otherwise against the manifest's own directory.

    Problems are collected rather than raised, so a malformed or missing manifest still yields a
    report (with a single ``load_error`` problem) instead of an exception. A valid manifest yields a
    report whose ``problems`` list is empty.
    """
    manifest_path = Path(path)
    report = ValidationReport()

    try:
        cases = load_cases(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        report.problems.append(Problem(kind="load_error", detail=str(exc)))
        return report

    report.n_cases = len(cases)
    _check_duplicates(cases, report)
    for case in cases:
        _check_required_fields(case, report)

    if check_images:
        base = Path(root) if root is not None else manifest_path.parent
        _check_images(cases, base, report)

    return report


def manifest_checksum(path: str | Path) -> str:
    """Return a stable sha256 hex digest of the raw bytes of the manifest at ``path``.

    The digest depends only on the file contents, so the same manifest always hashes to the same
    value. Stamp it into a ``RunManifest`` to pin exactly which manifest a run consumed. Raises
    ``FileNotFoundError`` if the path does not exist.
    """
    manifest_path = Path(path)
    digest = hashlib.sha256()
    with manifest_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
