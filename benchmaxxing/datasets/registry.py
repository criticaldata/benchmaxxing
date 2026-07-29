"""Registry of dataset adapters, keyed by their SPEC.name.

Each value is the adapter module itself, which exposes ``SPEC`` (a DatasetSpec) and
``build_manifest(raw_root, out, limit=None)``.
"""

from __future__ import annotations

from types import ModuleType

from benchmaxxing.datasets import (
    chexpert,
    ehr,
    medmcqa,
    medqa,
    mimic_cxr,
    mimic_cxr_text,
    nih_cxr14,
    probmed,
    pubmedqa,
    support2,
)

REGISTRY: dict[str, ModuleType] = {
    mimic_cxr.SPEC.name: mimic_cxr,
    mimic_cxr_text.SPEC.name: mimic_cxr_text,
    chexpert.SPEC.name: chexpert,
    nih_cxr14.SPEC.name: nih_cxr14,
    medqa.SPEC.name: medqa,
    medmcqa.SPEC.name: medmcqa,
    pubmedqa.SPEC.name: pubmedqa,
    probmed.SPEC.name: probmed,
    ehr.SPEC.name: ehr,
    support2.SPEC.name: support2,
}


def names() -> list[str]:
    """Return the sorted list of registered dataset names."""
    return sorted(REGISTRY)


def get(name: str) -> ModuleType:
    """Return the adapter module registered under ``name``.

    Raises KeyError with the available names if ``name`` is unknown.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {names()}") from None
