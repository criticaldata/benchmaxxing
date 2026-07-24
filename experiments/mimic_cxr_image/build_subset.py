"""Select a deterministic, nested MIMIC-CXR study subset and emit its download list (#92, #295).

The full MIMIC-CXR-JPG release is ~600 GB, but the imaging battery only needs a few hundred
studies per arm, and the arm sizes nest (blind 100 < cascade 150 < referee 300 < solo/cross-modal
600, with a 35-study set nested inside the 600 for an apples-to-apples comparison against the NIH
ChestX-ray14 run). So we never download the whole release: we build the manifest from the small
metadata CSVs (no images required on disk), rank *studies* by a stable hash of the seed, take the
top N for each arm, and write one ``download_list.txt`` of just the images in the 600-study
superset. The union of every arm is exactly that superset, so a single download covers all arms.

Determinism and nesting both come from ``benchmaxxing.budget.subsample_cases`` (hash-ranked, so
the selection depends only on study identity, not on the order the pool arrives in). Same seed
gives the identical list byte for byte, which is what makes the download reproducible: the list
plus the per-arm manifests plus the ``verify`` provenance checksums are the reproducibility
artifacts, exactly as the NIH imaging manifest pins its cases and sha256s its pixels.

Two modes:

    select   build/reuse the full manifest, rank studies, write per-arm manifests + download_list.txt
    verify   after the image download: confirm every referenced image resolves and write provenance.json

Selection needs only the metadata CSVs + reports (what ``mimic_cxr.build_manifest`` reads); it does
not need any image on disk. ``verify`` is the step that needs the downloaded images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from benchmaxxing.budget import RunBudget, subsample_cases
from benchmaxxing.data import load_cases
from benchmaxxing.datasets import mimic_cxr
from benchmaxxing.datasets.base import finalize

# Arm -> number of studies. solo and cross-modal share the 600 superset; every smaller arm is a
# strict subset of it, so the download only ever needs the 600. nih_match is the 35 nested inside
# the 600 for the matched NIH comparison.
DEFAULT_SIZES = {
    "solo": 600,
    "cross_modal": 600,
    "referee": 300,
    "cascade": 150,
    "blind_metric": 100,
    "nih_match": 35,
}

DEFAULT_URL_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"


def _study_key(case) -> str:
    """Stable per-study identity: ``patient_id/study_id``. study_id alone is globally unique in
    MIMIC, but pairing it with the subject id is defensive and self-documenting."""
    return f"{case.patient_id}/{case.meta.get('study_id')}"


def _is_finding(case) -> bool:
    """True when the case carries a real finding label (the adapter emits 'pneumothorax' vs
    'no finding'). Mirrors the eligibility filter the NIH imaging pipeline uses."""
    return bool(case.label) and case.label.strip().lower() not in ("no finding", "", "none")


def select_arms(cases, seed: int, sizes: dict[str, int], require_finding: bool = True) -> dict[str, list]:
    """Group cases into studies, rank studies deterministically by ``seed``, and return the nested
    per-arm case lists.

    Each arm gets *all* images of its top-N studies. Because every arm ranks the same study pool
    with the same seed, the smaller arms are strict subsets of the larger ones: the top-100 studies
    are a prefix of the top-300, which are a prefix of the top-600 (the prefix property of
    :func:`subsample_cases`). Returns ``{arm_name: [Case, ...]}``.
    """
    eligible = [c for c in cases if not require_finding or _is_finding(c)]
    studies: dict[str, list] = {}
    for case in eligible:
        studies.setdefault(_study_key(case), []).append(case)
    study_keys = list(studies)

    arms: dict[str, list] = {}
    for arm, n in sizes.items():
        picked = subsample_cases(study_keys, RunBudget(max_cases=n, seed=seed))
        arms[arm] = [case for key in picked for case in studies[key]]
    return arms


def _download_lines(cases, url_base: str) -> list[str]:
    """One URL per unique image in ``cases``, in a deterministic order, for ``wget -i``."""
    base = url_base.rstrip("/") + "/"
    seen: dict[str, None] = {}
    for case in cases:
        if case.image_ref and case.image_ref not in seen:
            seen[case.image_ref] = None
    return [base + ref for ref in seen]


def _build_full_cases(raw_root: str, full_manifest: str | None):
    """Return every case in the release. Reuses ``--full-manifest`` when given (fast reruns),
    otherwise builds it once into a temp file (discarded) via the registered adapter."""
    if full_manifest and Path(full_manifest).is_file():
        return load_cases(full_manifest)
    with tempfile.TemporaryDirectory() as tmp:
        return mimic_cxr.build_manifest(raw_root, Path(tmp) / "full_manifest.csv")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def do_select(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = dict(DEFAULT_SIZES)

    all_cases = _build_full_cases(args.raw_root, args.full_manifest)
    arms = select_arms(all_cases, seed=args.seed, sizes=sizes, require_finding=not args.include_no_finding)

    written = {}
    for arm, cases in arms.items():
        path = out_dir / f"{arm}_{sizes[arm]}.csv"
        finalize(cases, path)
        n_studies = len({_study_key(c) for c in cases})
        written[arm] = {"manifest": str(path), "studies": n_studies, "images": len(cases)}

    # The download superset is the largest arm; every other arm is nested inside it.
    superset = max(arms.values(), key=len)
    lines = _download_lines(superset, args.url_base)
    (out_dir / "download_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "seed": args.seed,
        "require_finding": not args.include_no_finding,
        "url_base": args.url_base,
        "arms": written,
        "download_list": str(out_dir / "download_list.txt"),
        "download_images": len(lines),
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def do_verify(args) -> None:
    image_root = Path(args.image_root)
    cases = load_cases(args.manifest)
    missing = [c.image_ref for c in cases if not (image_root / c.image_ref).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(cases)} images not found under {image_root} "
            f"(first few: {missing[:5]}). Run the download first."
        )
    provenance = {
        "manifest": str(args.manifest),
        "image_root": str(image_root),
        "n_cases": len(cases),
        "cases": [
            {
                "case_id": c.case_id,
                "image_ref": c.image_ref,
                "label": c.label,
                "sha256": _sha256(image_root / c.image_ref),
                "bytes": (image_root / c.image_ref).stat().st_size,
            }
            for c in cases
        ],
    }
    Path(args.provenance).write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"n_cases": len(cases), "provenance": str(args.provenance), "all_present": True}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    sel = sub.add_parser("select", help="rank studies and write per-arm manifests + download_list.txt")
    sel.add_argument("--raw-root", required=True, help="dir with the metadata/chexpert CSVs and unpacked reports")
    sel.add_argument("--out-dir", required=True, help="output dir for per-arm manifests + download list")
    sel.add_argument("--seed", type=int, default=0, help="selection seed (fixes the nested study set)")
    sel.add_argument("--url-base", default=DEFAULT_URL_BASE, help="prefix for image_ref paths in download_list.txt")
    sel.add_argument("--full-manifest", default=None, help="reuse a prebuilt full manifest instead of rebuilding")
    sel.add_argument("--include-no-finding", action="store_true", help="keep 'no finding' studies too (default: findings only)")
    sel.set_defaults(func=do_select)

    ver = sub.add_parser("verify", help="check images resolve on disk and write provenance.json")
    ver.add_argument("--manifest", required=True, help="a per-arm manifest (usually the 600-study superset)")
    ver.add_argument("--image-root", required=True, help="dir holding the downloaded images")
    ver.add_argument("--provenance", required=True, help="output provenance JSON path (sha256 per image)")
    ver.set_defaults(func=do_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
