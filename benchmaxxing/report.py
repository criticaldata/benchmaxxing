"""Self-contained HTML reports for a benchmaxxing run.

Renders a run's report dict (flip rates, cascade onset, referee precision/recall,
blind-metric uptake, ...) into a single standalone HTML string with inline CSS and one
table per top-level section. Standard library only: no numpy, no templating engine.

Every value that originates from the report is escaped with ``html.escape`` before it reaches
the output, so untrusted strings cannot break out of the markup. Nested dicts and lists are
rendered recursively into nested tables and lists.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path

__all__ = ["render_html_report", "write_html_report"]

_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.45; margin: 0; padding: 2rem; color: #1b1f24; background: #fff;
}
h1 { font-size: 1.6rem; margin: 0 0 1.5rem; }
h2 {
  font-size: 1.15rem; margin: 0 0 .6rem; padding-bottom: .3rem;
  border-bottom: 2px solid #d0d7de;
}
section.report-section {
  margin: 0 0 1.75rem; padding: 1rem 1.25rem; border: 1px solid #d0d7de;
  border-radius: 8px; background: #f6f8fa;
}
table.kv { border-collapse: collapse; width: 100%; margin: .25rem 0; }
table.kv th, table.kv td {
  text-align: left; vertical-align: top; padding: .35rem .6rem;
  border: 1px solid #d8dee4;
}
table.kv th {
  width: 30%; font-weight: 600; background: #eaeef2; white-space: nowrap;
}
ul.seq { margin: .25rem 0; padding-left: 1.25rem; }
ul.seq li { margin: .1rem 0; }
span.scalar { font-variant-numeric: tabular-nums; }
span.empty { color: #57606a; font-style: italic; }
"""


def _format_scalar(value) -> str:
    """Turn a leaf value into a plain (unescaped) display string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return format(value, ".6g")
    return str(value)


def _render_mapping(mapping: Mapping) -> str:
    """Render a mapping as a two-column key/value table (values rendered recursively)."""
    if not mapping:
        return '<span class="empty">(empty)</span>'
    rows = []
    for key, val in mapping.items():
        rows.append(
            "<tr><th>"
            + html.escape(_format_scalar(key))
            + "</th><td>"
            + _render_value(val)
            + "</td></tr>"
        )
    return '<table class="kv">' + "".join(rows) + "</table>"


def _render_sequence(seq) -> str:
    """Render a list/tuple/set as a bulleted list (items rendered recursively)."""
    items = list(seq)
    if not items:
        return '<span class="empty">(empty)</span>'
    cells = "".join("<li>" + _render_value(v) + "</li>" for v in items)
    return '<ul class="seq">' + cells + "</ul>"


def _render_value(value) -> str:
    """Render any report value: mappings become tables, sequences lists, leaves escaped text."""
    if isinstance(value, Mapping):
        return _render_mapping(value)
    if isinstance(value, (list, tuple)):
        return _render_sequence(value)
    if isinstance(value, (set, frozenset)):
        try:
            ordered = sorted(value, key=_format_scalar)
        except TypeError:
            ordered = list(value)
        return _render_sequence(ordered)
    return '<span class="scalar">' + html.escape(_format_scalar(value)) + "</span>"


def render_html_report(report: dict, title: str = "benchmaxxing run") -> str:
    """Render a run report dict into a self-contained HTML document string.

    Each top-level key of ``report`` becomes its own titled section with one table (nested
    dicts/lists are rendered recursively). All values pass through ``html.escape``. The result
    is a complete standalone document (inline CSS, no external assets) beginning with
    ``<!doctype html>``.

    Parameters
    ----------
    report:
        Mapping of section name -> section data (flip rates, cascade onset, referee
        precision/recall, blind-metric uptake, ...). Values may be nested dicts/lists/scalars.
    title:
        Document title, shown in ``<title>`` and as the top ``<h1>``.

    Returns
    -------
    str
        The full HTML document.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping (dict) of section name -> section data.")

    safe_title = html.escape(str(title))
    sections = []
    for name, data in report.items():
        sections.append(
            '<section class="report-section"><h2>'
            + html.escape(_format_scalar(name))
            + "</h2>"
            + _render_value(data)
            + "</section>"
        )
    if not sections:
        sections.append('<p class="empty">(no sections)</p>')
    body = "\n".join(sections)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>" + safe_title + "</title>\n"
        "<style>" + _CSS + "</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>" + safe_title + "</h1>\n"
        + body
        + "\n</body>\n"
        "</html>\n"
    )


def write_html_report(report: dict, path, title: str = "benchmaxxing run") -> Path:
    """Render ``report`` and write the HTML document to ``path``. Returns the ``Path`` written."""
    out = Path(path)
    out.write_text(render_html_report(report, title=title), encoding="utf-8")
    return out
