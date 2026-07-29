"""Guard the NIH cohort selection contract (#337 review).

#337 changed the default selection from release order to seeded round-robin stratification and deleted the
docstring sentence promising that a rebuild reproduces the cohort every published number was computed on.
Stratification is useful for building a NEW cohort with better finding coverage, but as a default it
silently changes which cases a rebuild picks, which breaks the audit trail. These tests pin the contract:
release order is the default, stratification is opt-in, and the committed cohort stays reproducible.
"""
import inspect

from experiments.imaging import build_manifest


def test_stratify_is_opt_in_not_the_default():
    src = inspect.getsource(build_manifest.main)
    assert '"--stratify"' in src, "the stratify flag is gone"
    assert "action=\"store_true\"" in src, "stratify must be a flag, not a value with a default"
    assert "elif not args.stratify:" in src, "release order must be the default branch"


def test_release_order_branch_still_exists():
    """The default path must be a plain prefix of the eligible list, not a resampling."""
    src = inspect.getsource(build_manifest.main)
    assert "eligible[: args.n]" in src, "the release-order slice was removed again"


def test_docstring_states_the_reproducibility_guarantee():
    """This sentence is the audit trail for the paper's NIH cohort. It was deleted once already."""
    doc = build_manifest.__doc__ or ""
    assert "case-ids-file" in doc
    assert "reproduces the identical" in doc, "the cohort reproducibility guarantee is missing"


def test_label_format_is_only_requested_when_stratifying():
    """Asking the adapter for stratified labels unconditionally changes the default cohort too."""
    src = inspect.getsource(build_manifest.main)
    assert "args.stratify and" in src, "label_format is still requested unconditionally"
