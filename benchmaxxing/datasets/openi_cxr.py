"""Open-i / Indiana University chest X-ray adapter (imaging, Lane A).

Openly downloadable, no credentialing and no signed license, which is why it is the second
imaging source next to the credentialed MIMIC-CXR and CheXpert lanes (issue 119). The NLM
publishes it as two tarballs::

    https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz        # 7470 png images
    https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz    # 3955 ecgen-radiology xml

License: the collection is public domain / open access for research use (NLM Open-i terms).
Cite Demner-Fushman et al., JAMIA 2016.

One report describes one study and points at its images (usually a frontal and a lateral) via
``parentImage`` elements, so one Case is emitted per image and every image of a study shares the
report id as its ``patient_id``. That is what keeps the same-patient swap available to the
pairing step.

Label policy (multi-label source, documented because the choice matters): the label is built
from the MeSH ``major`` terms, which look like ``normal`` or ``Cardiomegaly/mild``. The severity
qualifier after the slash is dropped, the finding is lower-cased, duplicates are removed, and the
remaining findings are joined with ``|`` (``cardiomegaly|pulmonary congestion``), matching the
convention the NIH ChestX-ray14 adapter already uses. The healthy MeSH class ``normal`` is emitted
as ``no finding``, the negative-class sentinel the imaging experiments and the other CXR adapters
share, so it is filtered out of finding runs rather than scored as pathology. The untouched major
and automatic terms stay in ``meta`` so a caller can apply a different policy without re-parsing.

The known wart in that policy: a minority of major terms lead with the anatomy rather than the
finding (``Lung/hypoinflation``), so their label collapses to the anatomy word. Filtering on a
finding is still safe (an ``Opacity/lung/base`` case labels as ``opacity``), but a caller that
cares about those cases should read ``meta["mesh_major"]``, which is verbatim.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

REPORTS_DIR = "ecgen-radiology"
IMAGE_SUFFIX = ".png"

SPEC = DatasetSpec(
    name="openi_cxr",
    raw_hint=(
        "Open-i / Indiana University Chest X-ray Collection (NLM, open access, no credentialing): "
        "'NLMCXR_reports.tgz' unpacks to an 'ecgen-radiology/' folder of one XML per study "
        "(MeSH major/minor terms, AbstractText sections COMPARISON/INDICATION/FINDINGS/IMPRESSION, "
        "and one 'parentImage' element per image), and 'NLMCXR_png.tgz' unpacks to the matching "
        "'<parentImage id>.png' files. Point raw_root at the reports folder or its parent."
    ),
    modality=Modality.IMAGE,
    notes=(
        "One Case per image, patient_id=report id so both views of a study pair together. "
        "label=MeSH major findings, severity qualifier dropped, joined with '|'. report=FINDINGS "
        "plus IMPRESSION. Images live flat in NLMCXR_png, so image_ref is '<image id>.png' and "
        "--image-root is that folder."
    ),
)


def _resolve_reports_dir(raw_root) -> Path:
    """Return the ecgen-radiology folder, given it directly or a directory containing it."""
    root = Path(raw_root)
    candidates = [root, root / REPORTS_DIR]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.xml")):
            return candidate
    raise FileNotFoundError(
        f"{SPEC.name}: no XML reports under {root!r}. Point raw_root at the "
        f"{REPORTS_DIR!r} folder unpacked from NLMCXR_reports.tgz, or at its parent."
    )


def _section(root: ET.Element, label: str) -> str | None:
    """Text of the AbstractText section with ``Label`` (FINDINGS, IMPRESSION, ...), if any."""
    node = root.find(f".//AbstractText[@Label='{label}']")
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None


def _mesh_terms(root: ET.Element, tag: str) -> list[str]:
    """All MeSH ``tag`` child texts (``major``, ``minor``, ``automatic``) in document order."""
    mesh = root.find("MeSH")
    if mesh is None:
        return []
    return [node.text.strip() for node in mesh.findall(tag) if node.text and node.text.strip()]


def _label_from_major(major_terms: list[str]) -> str | None:
    """Apply the documented label policy to the MeSH major terms."""
    findings: list[str] = []
    for term in major_terms:
        finding = term.split("/")[0].strip().lower()
        # the healthy MeSH class is spelled "normal"; the imaging finding filter and the sibling
        # CXR adapters use "no finding" as the negative sentinel, so normalise onto that
        if finding == "normal":
            finding = "no finding"
        if finding and finding not in findings:
            findings.append(finding)
    return "|".join(findings) or None


def _cases_from_report(path: Path) -> list[Case]:
    """Turn one ecgen-radiology XML into one Case per parentImage (empty if it has none)."""
    root = ET.parse(path).getroot()
    study_id = path.stem
    major = _mesh_terms(root, "major")
    label = _label_from_major(major)
    report = "\n".join(
        part for part in (_section(root, "FINDINGS"), _section(root, "IMPRESSION")) if part
    ) or None

    cases = []
    for image in root.findall("parentImage"):
        image_id = (image.get("id") or "").strip()
        if not image_id:
            continue
        caption = image.findtext("caption") or ""
        cases.append(
            Case(
                case_id=image_id,
                patient_id=study_id,
                modality=Modality.IMAGE,
                label=label,
                image_ref=image_id + IMAGE_SUFFIX,
                report=report,
                meta={
                    "mesh_major": major,
                    "mesh_automatic": _mesh_terms(root, "automatic"),
                    "indication": _section(root, "INDICATION"),
                    "view": caption.strip(),
                    "study_id": study_id,
                },
            )
        )
    return cases


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the unpacked Open-i release.

    ``raw_root`` is the ``ecgen-radiology`` folder or a directory containing it. Reports are read
    in filename order and each one contributes a Case per ``parentImage``; a report with no
    images (the collection has a few) contributes nothing rather than aborting the build.
    ``limit`` caps the number of cases, not the number of reports, and only ever cuts at a study
    boundary so a study's views are never split across the limit.
    """
    reports_dir = _resolve_reports_dir(raw_root)
    cases: list[Case] = []
    for path in sorted(reports_dir.glob("*.xml")):
        if limit is not None and len(cases) >= limit:
            break
        cases.extend(_cases_from_report(path))
    return finalize(cases, out)
