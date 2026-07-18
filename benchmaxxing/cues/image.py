"""Deterministic image cue injection for chest X-ray twin pairs.

Each function composites a spurious visual cue (a cable, a corner tag, a faint watermark, or a
laterality marker) onto a clean chest X-ray. The cues are the kind of incidental artifact a model
can latch onto as a shortcut. Every cue is parameterized by position, opacity, and size so its
STRENGTH is controllable, and every cue is deterministic given a seed.

Design notes:
- Images are 2-D (grayscale) or 3-D (H, W, C) uint8 numpy arrays.
- Injection never mutates the input array: a fresh uint8 array is returned.
- Cues are composited by alpha blending toward a bright value, so a pixel only changes where the
  cue actually covers it, and higher opacity moves the pixel further from the original. That gives
  a clean, graded strength knob.
- Rendering (text, lines) is done with PIL, which is deterministic for a fixed version, so repeated
  calls with the same arguments produce byte-identical output.

This is a pure image-processing injector: it makes no claim about any model. The synthetic shapes
used in the tests are a check of the injector's own behavior.
"""

from __future__ import annotations

import numpy as np

from benchmaxxing.schema import TwinPair

# All cues blend toward bright white, which is what these artifacts look like on an X-ray.
_CABLE_COLOR = 255.0
_TAG_COLOR = 255.0
_WATERMARK_COLOR = 255.0
_LATERALITY_COLOR = 255.0
_LATERALITY_OPACITY = 1.0

_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def _to_float_hwc(img):
    """Return (work, was_2d) where work is a float64 (H, W, C) copy of a uint8 image."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if arr.ndim == 2:
        return arr[:, :, None].astype(np.float64), True
    if arr.ndim == 3:
        return arr.astype(np.float64), False
    raise ValueError("image must be a 2-D or 3-D array")


def _from_float_hwc(work, was_2d):
    """Clip/round a float (H, W, C) buffer back to a uint8 image (2-D if the input was 2-D)."""
    out = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    if was_2d:
        return out[:, :, 0]
    return out


def _composite(work, alpha_map, brightness):
    """Alpha-blend a bright color into ``work`` in place using a per-pixel coverage map.

    ``alpha_map`` is an (H, W) coverage array in [0, 1]. Where it is zero the pixel is untouched,
    so the injected image differs from the clean image only where the cue actually lands.
    """
    channels = work.shape[2]
    color = np.full(channels, float(brightness))
    a = np.clip(alpha_map, 0.0, 1.0)[:, :, None]
    work *= 1.0 - a
    work += color[None, None, :] * a


def _load_font(size):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_line_coverage(h, w, p0, p1, thickness):
    from PIL import Image, ImageDraw

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).line([tuple(p0), tuple(p1)], fill=255, width=int(thickness))
    return np.asarray(mask, dtype=np.float64) / 255.0


def _cable_coverage(h, w, position, thickness, seed):
    """Coverage map for a near-vertical cable. ``position='auto'`` is seeded and reproducible."""
    if position == "auto":
        rng = np.random.default_rng(seed)
        lo = int(w * 0.2)
        hi = max(lo + 1, int(w * 0.8) + 1)
        x0 = int(rng.integers(lo, hi))
        jitter = max(1, w // 10)
        x1 = int(np.clip(x0 + int(rng.integers(-jitter, jitter + 1)), 0, w - 1))
        p0, p1 = (x0, 0), (x1, h - 1)
    else:
        p0, p1 = position
    return _draw_line_coverage(h, w, p0, p1, thickness)


def _corner_xy(corner, text_w, text_h, w, h, margin):
    if corner == "top-left":
        return margin, margin
    if corner == "top-right":
        return w - text_w - margin, margin
    if corner == "bottom-left":
        return margin, h - text_h - margin
    if corner == "bottom-right":
        return w - text_w - margin, h - text_h - margin
    raise ValueError(f"corner must be one of {_CORNERS}, got {corner!r}")


def _corner_text_coverage(h, w, text, font_size, corner, margin=None):
    from PIL import Image, ImageDraw

    if margin is None:
        margin = max(1, int(round(min(h, w) * 0.02)))
    font = _load_font(font_size)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    x, y = _corner_xy(corner, text_w, text_h, w, h, margin)
    draw.text((x - left, y - top), text, fill=255, font=font)
    return np.asarray(mask, dtype=np.float64) / 255.0


def _diagonal_text_coverage(h, w, text, font_size, angle):
    from PIL import Image, ImageDraw

    font = _load_font(font_size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1), 0))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    text_w, text_h = max(1, right - left), max(1, bottom - top)
    tile = Image.new("L", (text_w + 4, text_h + 4), 0)
    ImageDraw.Draw(tile).text((2 - left, 2 - top), text, fill=255, font=font)
    tile = tile.rotate(angle, expand=True)
    canvas = Image.new("L", (w, h), 0)
    canvas.paste(tile, ((w - tile.width) // 2, (h - tile.height) // 2))
    return np.asarray(canvas, dtype=np.float64) / 255.0


def inject_cable(img, position="auto", opacity=0.6, thickness=3, seed=0):
    """Composite a thin bright cable/line artifact.

    ``position`` is either ``'auto'`` (seeded, near-vertical) or a ``((x0, y0), (x1, y1))`` pair.
    ``opacity`` in [0, 1] sets cue strength; ``thickness`` is the line width in pixels.
    """
    work, was_2d = _to_float_hwc(img)
    h, w = work.shape[:2]
    coverage = _cable_coverage(h, w, position, thickness, seed)
    _composite(work, coverage * float(opacity), _CABLE_COLOR)
    return _from_float_hwc(work, was_2d)


def inject_corner_tag(img, text="PORTABLE", corner="top-left", opacity=0.8, size=0.08):
    """Render a bright corner text tag (e.g. 'PORTABLE') with PIL.

    ``size`` is the font height as a fraction of the image height; ``opacity`` in [0, 1] and
    ``corner`` control strength and placement.
    """
    work, was_2d = _to_float_hwc(img)
    h, w = work.shape[:2]
    font_size = max(6, int(round(h * size)))
    coverage = _corner_text_coverage(h, w, text, font_size, corner)
    _composite(work, coverage * float(opacity), _TAG_COLOR)
    return _from_float_hwc(work, was_2d)


def inject_watermark(img, text="HOSPITAL", opacity=0.25):
    """Composite a faint diagonal watermark across the image center."""
    work, was_2d = _to_float_hwc(img)
    h, w = work.shape[:2]
    font_size = max(8, int(round(min(h, w) * 0.18)))
    coverage = _diagonal_text_coverage(h, w, text, font_size, angle=30)
    _composite(work, coverage * float(opacity), _WATERMARK_COLOR)
    return _from_float_hwc(work, was_2d)


def inject_laterality(img, mark="L", corner="top-right"):
    """Composite a bright laterality marker (e.g. 'L' or 'R') in a corner."""
    work, was_2d = _to_float_hwc(img)
    h, w = work.shape[:2]
    font_size = max(8, int(round(h * 0.12)))
    coverage = _corner_text_coverage(h, w, mark, font_size, corner)
    _composite(work, coverage * _LATERALITY_OPACITY, _LATERALITY_COLOR)
    return _from_float_hwc(work, was_2d)


_INJECTORS = {
    "cable": inject_cable,
    "corner_tag": inject_corner_tag,
    "watermark": inject_watermark,
    "laterality": inject_laterality,
}


def build_image_twin(img, cue_type, ground_truth=None, case_id="image_twin", **params):
    """Build a :class:`~benchmaxxing.schema.TwinPair` for one image cue.

    ``clean`` is a copy of the original array, ``contaminated`` is the injected array, and the two
    are identical except where the cue lands. ``params`` are forwarded to the chosen injector and
    recorded on the pair as ``cue_params``.
    """
    if cue_type not in _INJECTORS:
        raise ValueError(f"cue_type must be one of {sorted(_INJECTORS)}, got {cue_type!r}")
    clean = np.asarray(img)
    if clean.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    contaminated = _INJECTORS[cue_type](clean, **params)
    return TwinPair(
        case_id=case_id,
        cue_type=cue_type,
        cue_params=dict(params),
        clean=clean.copy(),
        contaminated=contaminated,
        ground_truth=ground_truth,
    )
