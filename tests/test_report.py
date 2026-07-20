"""Tests for benchmaxxing.report (self-contained HTML run reports)."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

from benchmaxxing.report import render_html_report, write_html_report

# Tags that HTML does not require to be closed; the balance checker ignores them.
_VOID_TAGS = {"meta", "br", "img", "input", "hr", "link", "col", "source", "area"}


class _BalanceChecker(HTMLParser):
    """Verify every non-void start tag is matched by a properly nested end tag."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        # Explicit self-closing tag, e.g. <br/>: opens and closes at once.
        return

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"end tag </{tag}> with empty stack")
        elif self.stack[-1] != tag:
            self.errors.append(f"end tag </{tag}> does not match open <{self.stack[-1]}>")
            self.stack.pop()
        else:
            self.stack.pop()


def _assert_well_formed(doc: str) -> None:
    lower = doc.lstrip().lower()
    assert lower.startswith("<!doctype html>") or lower.startswith("<html"), (
        "document must begin with a doctype or <html>"
    )
    checker = _BalanceChecker()
    checker.feed(doc)
    checker.close()
    assert not checker.errors, f"unbalanced markup: {checker.errors}"
    assert not checker.stack, f"unclosed tags remain open: {checker.stack}"


def _synthetic_report() -> dict:
    """A report dict shaped like a real benchmaxxing run summary."""
    return {
        "run_id": "run-2026-07-18-abc",
        "flip_rate": {
            "overall": 0.4231,
            "per_cue": {"cable": 0.5, "option_order": 0.34, "longest_option": 0.2917},
            "n": 120,
        },
        "shortcut_reliance_index": {
            "overall": 0.18,
            "clean_accuracy": 0.9,
            "contaminated_accuracy": 0.72,
            "per_cue": {"cable": 0.2},
            "n": 100,
        },
        "cascade_onset": {"predicted_turn": 3, "true_onset": 2, "detected": True, "gap": None},
        "referee_precision_recall": {"precision": 0.8, "recall": 0.6667, "f1": 0.7273},
        "blind_metric_uptake": {
            "method": "spearman",
            "uptake": 0.55,
            "adopted": ["blind_a", "blind_b"],
        },
        "lineage_overlap_test": {
            "metric": "phi",
            "within_mean": 0.61,
            "cross_mean": 0.22,
            "observed_diff": 0.39,
            "p_value": 0.0015,
            "n_models": 6,
        },
        "notes": ["synthetic fixture", "runs offline"],
        "config": {"seed": 0, "cue_set_version": "v1", "empty_section": {}},
    }


def test_render_returns_well_formed_document():
    doc = render_html_report(_synthetic_report(), title="run summary")
    assert isinstance(doc, str)
    _assert_well_formed(doc)
    assert "<title>run summary</title>" in doc
    assert "run summary" in doc  # also shown as the <h1>


def test_render_contains_section_names_and_values():
    report = _synthetic_report()
    doc = render_html_report(report)

    # Section names appear as headings.
    for name in report:
        assert name in doc

    # Scalar and nested values appear (floats formatted with %.6g).
    assert "run-2026-07-18-abc" in doc
    assert "0.4231" in doc
    assert "cable" in doc
    assert "option_order" in doc
    assert "spearman" in doc
    assert "blind_a" in doc and "blind_b" in doc
    assert "0.0015" in doc
    assert "synthetic fixture" in doc
    # Booleans and None get friendly labels.
    assert "true" in doc
    assert "null" in doc
    # An empty nested dict renders the empty marker rather than crashing.
    assert "(empty)" in doc


def test_values_are_html_escaped():
    report = {"danger": {"payload": "<script>alert('x')</script> & <b>bold</b>"}}
    doc = render_html_report(report)
    assert "<script>alert" not in doc
    assert "&lt;script&gt;" in doc
    assert "&amp;" in doc


def test_nested_structures_render_gracefully():
    report = {
        "deep": {"a": {"b": {"c": [1, 2, {"d": "leaf"}]}}},
        "list_of_dicts": [{"x": 1}, {"y": 2}],
        "tuple_section": ("t0", "t1"),
        "set_section": {"s_b", "s_a"},
    }
    doc = render_html_report(report)
    _assert_well_formed(doc)
    assert "leaf" in doc
    assert "t0" in doc and "t1" in doc
    assert "s_a" in doc and "s_b" in doc


def test_empty_report_is_valid():
    doc = render_html_report({})
    _assert_well_formed(doc)
    assert "(no sections)" in doc


def test_default_title_used():
    doc = render_html_report({"k": "v"})
    assert "benchmaxxing run" in doc


def test_non_mapping_report_raises():
    with pytest.raises(TypeError):
        render_html_report(["not", "a", "mapping"])


def test_float_edge_cases_render():
    report = {"edges": {"nan": float("nan"), "pos_inf": float("inf"), "neg_inf": float("-inf")}}
    doc = render_html_report(report)
    _assert_well_formed(doc)
    assert "nan" in doc
    assert "inf" in doc
    assert "-inf" in doc


def test_write_html_report_writes_file(tmp_path):
    report = _synthetic_report()
    target = tmp_path / "report.html"
    returned = write_html_report(report, target, title="written run")

    assert isinstance(returned, Path)
    assert returned == target
    assert target.exists()

    contents = target.read_text(encoding="utf-8")
    assert contents == render_html_report(report, title="written run")
    _assert_well_formed(contents)
    assert "written run" in contents


def test_write_accepts_string_path(tmp_path):
    out = write_html_report({"k": "v"}, str(tmp_path / "r.html"))
    assert out.exists()
    assert out.read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype html>")
