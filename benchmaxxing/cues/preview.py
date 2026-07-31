"""Render a clean/contaminated cue twin to disk so a cue can be eyeballed.

Image lane writes ``clean.png`` / ``contaminated.png``; text lane writes a ``preview.md`` with
the clean and contaminated payloads side by side. Both lanes fall back to a small bundled sample
(a flat gray image, a three-option MCQ) when no manifest is given, so a cue can be checked with no
data on disk. The image lane needs the ``image`` extra (PIL); the text lane needs nothing beyond
the core dependencies.
"""

from __future__ import annotations

from pathlib import Path

from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.schema import Case, Modality

_SAMPLE_TEXT_CASE = Case(
    case_id="sample",
    patient_id="sample",
    modality=Modality.TEXT,
    question="Most likely diagnosis?",
    options=("Pneumothorax", "Pulmonary embolism", "Rib fracture"),
    answer_index=0,
)

_SAMPLE_IMAGE_SIZE = 128
_SAMPLE_IMAGE_VALUE = 110


def _sample_image():
    import numpy as np

    return np.full((_SAMPLE_IMAGE_SIZE, _SAMPLE_IMAGE_SIZE), _SAMPLE_IMAGE_VALUE, dtype=np.uint8)


def _select_case(cases: list[Case], modality: Modality, case_id: str | None) -> Case:
    candidates = [c for c in cases if c.modality is modality]
    if not candidates:
        raise ValueError(f"manifest has no {modality.value} cases")
    if case_id is None:
        return candidates[0]
    for case in candidates:
        if case.case_id == case_id:
            return case
    raise ValueError(f"case_id {case_id!r} not found among the manifest's {modality.value} cases")


def resolve_text_case(manifest_path=None, case_id: str | None = None) -> Case:
    """Return the text-lane case to preview: a manifest row, or the bundled sample."""
    if manifest_path is None:
        return _SAMPLE_TEXT_CASE
    from benchmaxxing.data import load_cases

    return _select_case(load_cases(manifest_path), Modality.TEXT, case_id)


def resolve_image(manifest_path=None, case_id: str | None = None, image_root=None):
    """Return ``(uint8 array, source description)`` for the image lane."""
    if manifest_path is None:
        return _sample_image(), "bundled sample image"

    import numpy as np
    from PIL import Image

    from benchmaxxing.data import load_cases

    case = _select_case(load_cases(manifest_path), Modality.IMAGE, case_id)
    root = Path(image_root) if image_root else Path(manifest_path).parent
    image_path = root / case.image_ref
    if not image_path.exists():
        raise FileNotFoundError(f"image_ref does not resolve on disk: {image_path}")
    array = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return array, f"case_id={case.case_id} ({image_path})"


def render_image_preview(
    cue_type: str,
    out_dir,
    manifest_path=None,
    case_id: str | None = None,
    image_root=None,
    **cue_params,
) -> dict:
    """Build an image twin and write ``clean.png`` / ``contaminated.png`` into ``out_dir``."""
    from PIL import Image

    from benchmaxxing.cues.image import build_image_twin

    image, source = resolve_image(manifest_path, case_id, image_root)
    twin = build_image_twin(image, cue_type, **cue_params)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_path, contaminated_path = out / "clean.png", out / "contaminated.png"
    Image.fromarray(twin.clean).save(clean_path)
    Image.fromarray(twin.contaminated).save(contaminated_path)
    return {"source": source, "clean": clean_path, "contaminated": contaminated_path}


def _payload_lines(payload: dict) -> list[str]:
    lines = [f"question: {payload['question']}"]
    for i, option in enumerate(payload["options"]):
        marker = "*" if i == payload["answer_index"] else " "
        lines.append(f"  [{i}]{marker} {option}")
    if payload.get("report"):
        lines.append(f"report: {payload['report']}")
    return lines


def _format_payload(payload: dict) -> str:
    return "\n".join(_payload_lines(payload))


def _wrap_cell(line: str, width: int) -> list[str]:
    """Split a payload line on embedded newlines, then hard-wrap each segment to width.

    Hard-wrapping (rather than word-wrapping via textwrap) keeps the leading indentation on
    option lines intact -- textwrap's wordsep regex collapses runs of whitespace, which would
    eat the "  [0] " style prefixes.
    """
    out: list[str] = []
    for segment in line.split("\n"):
        if not segment:
            out.append("")
            continue
        out.extend(segment[i : i + width] for i in range(0, len(segment), width))
    return out


def format_side_by_side(clean: dict, contaminated: dict, width: int = 42) -> str:
    """Render the clean and contaminated payloads as two columns for terminal display."""
    left_cells = [_wrap_cell(line, width) for line in _payload_lines(clean)]
    right_cells = [_wrap_cell(line, width) for line in _payload_lines(contaminated)]
    n_rows = max(len(left_cells), len(right_cells))
    left_cells += [[""]] * (n_rows - len(left_cells))
    right_cells += [[""]] * (n_rows - len(right_cells))

    left, right = [], []
    for left_lines, right_lines in zip(left_cells, right_cells):
        n_sub = max(len(left_lines), len(right_lines))
        left += left_lines + [""] * (n_sub - len(left_lines))
        right += right_lines + [""] * (n_sub - len(right_lines))

    header = f"{'clean'.ljust(width)} | contaminated"
    rows = [header, "-" * len(header)]
    rows += [f"{left_line.ljust(width)} | {right_line}" for left_line, right_line in zip(left, right)]
    return "\n".join(rows)


def render_text_preview(
    cue_type: str,
    out_dir,
    manifest_path=None,
    case_id: str | None = None,
    **cue_params,
) -> dict:
    """Build a text twin, write a clean/contaminated ``preview.md``, and return a
    side-by-side rendering of the same twin for terminal display."""
    case = resolve_text_case(manifest_path, case_id)
    twin = build_text_twin(case, cue_type, **cue_params)

    markdown = (
        f"# cues preview: {cue_type}\n\n"
        f"case_id: {case.case_id}\n\n"
        f"## clean\n\n```\n{_format_payload(twin.clean)}\n```\n\n"
        f"## contaminated\n\n```\n{_format_payload(twin.contaminated)}\n```\n"
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preview.md"
    path.write_text(markdown, encoding="utf-8")
    side_by_side = format_side_by_side(twin.clean, twin.contaminated)
    return {
        "case": case, "twin": twin, "path": path, "markdown": markdown,
        "side_by_side": side_by_side,
    }
