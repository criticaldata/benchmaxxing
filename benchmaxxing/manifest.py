"""RunManifest persistence and a library-version stamp for reproducibility.

write_manifest / read_manifest round-trip a RunManifest through JSON. library_versions collects
the installed versions of the core numeric stack plus any optional benchmaxxing dependencies that
are present, so a caller can stamp them into a RunManifest before a run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmaxxing.schema import RunManifest

# Distribution names to probe. The core three are always expected; the rest are optional extras
# that are only reported when actually installed.
_PROBE_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "ruptures",
    "pillow",
    "opencv-python-headless",
    "opencv-python",
    "google-genai",
    "litellm",
    "transformers",
    "torch",
    "pyyaml",
)


def git_sha() -> str:
    """Short git commit SHA of the working tree, or 'unknown' outside a checkout.

    Part of what makes an artifact traceable: the library versions say what was installed, this
    says which revision of the pipeline produced it.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - degrade gracefully, this is diagnostic output
        return "unknown"


def library_versions() -> dict:
    """Return name -> version for installed numeric/optional deps (for RunManifest stamping)."""
    versions: dict[str, str] = {}
    for dist in _PROBE_DISTRIBUTIONS:
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            continue
    return versions


def write_manifest(manifest: RunManifest, path: str | Path) -> None:
    """Write ``manifest`` to ``path`` as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_manifest(path: str | Path) -> RunManifest:
    """Read a JSON manifest written by ``write_manifest`` and rebuild the RunManifest."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    known = {f.name for f in fields(RunManifest)}
    kwargs = {key: value for key, value in data.items() if key in known}
    return RunManifest(**kwargs)
