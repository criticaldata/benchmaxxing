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
- Rendering (text, lines) is done with PIL. Text cues use a VENDORED TrueType face, not
  ``ImageFont.load_default()``, whose typeface and size handling vary by Pillow version. That
  variation used to change the cue pixels between environments, and since the imaging call caches
  are keyed on those pixels it silently invalidated cached reads instead of reporting a mismatch
  (#393). ``cue_checksum`` exposes that pixel identity so a run can assert it up front.

This is a pure image-processing injector: it makes no claim about any model. The synthetic shapes
used in the tests are a check of the injector's own behavior.
"""

from __future__ import annotations

import hashlib
import io
from functools import lru_cache
from pathlib import Path

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


# Vendored so the text cues render the same glyphs everywhere. ImageFont.load_default() is NOT a
# fixed font: which typeface it returns, and whether it honours size= at all, depend on the Pillow
# version. Pillow < 10.1 raises TypeError on size= and falls back to a tiny unscaled bitmap; 10.1+
# returns a scaled Aileron. Since the imaging call cache is keyed on sha256 of the contaminated
# pixels, a different font silently invalidated every cached read and re-queried the model instead
# of reporting a mismatch. That is the #393 defect: an independent re-run of the MIMIC-CXR referee
# cohort disagreed with the committed per-case outcomes on roughly half of 417 cases.
_FONT_PATH = Path(__file__).with_name("assets") / "DejaVuSans.ttf"
# The font file's own checksum. Verified on load, so a corrupted or swapped file fails loudly here
# rather than showing up as unexplained model disagreement weeks later.
_FONT_SHA256 = "ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280"


class CueRenderError(RuntimeError):
    """Raised when a text cue cannot be rendered reproducibly.

    Never caught and worked around inside this module: a cue that cannot be rendered identically
    across environments invalidates the call cache silently, which is exactly the failure this
    class exists to make loud.
    """


@lru_cache(maxsize=None)
def _font_bytes():
    if not _FONT_PATH.exists():
        raise CueRenderError(
            f"vendored font missing at {_FONT_PATH}. Text cues (watermark, corner_tag, laterality) "
            "cannot be rendered reproducibly without it; reinstall the package rather than falling "
            "back to a system font, which would change the cue pixels and invalidate the cache."
        )
    raw = _FONT_PATH.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != _FONT_SHA256:
        raise CueRenderError(
            f"vendored font checksum mismatch at {_FONT_PATH}: expected {_FONT_SHA256}, got {got}. "
            "A different font renders different cue pixels, which silently invalidates every cached "
            "model read keyed on those pixels (see #393)."
        )
    return raw


def _load_font(size):
    """Return the vendored TrueType face at ``size``.

    Deliberately no fallback to ``ImageFont.load_default()``: falling back would keep rendering,
    with different glyphs, which is the silent-divergence mode this function was written to remove.
    """
    from PIL import ImageFont

    try:
        return ImageFont.truetype(io.BytesIO(_font_bytes()), size=size)
    except OSError as exc:  # Pillow built without FreeType cannot rasterise TrueType at all.
        raise CueRenderError(
            "Pillow cannot load a TrueType font, so text cues cannot be rendered reproducibly. "
            "Install a Pillow build with FreeType support."
        ) from exc


def cue_checksum(img):
    """sha256 of an image's pixels, the identity the imaging call caches are keyed on.

    Exposed so a run can assert the cue it renders matches the cue a committed result was scored
    against, instead of discovering a mismatch as unexplained model disagreement.
    """
    arr = np.asarray(img)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


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


# ---------------------------------------------------------------------------
# Additional acquisition-style cues (mis-orientation, compression / exposure,
# soft-tissue overlay). Each is diagnosis-neutral, strength-parameterized, and
# deterministic given its seed, and none mutates the input array.
# ---------------------------------------------------------------------------

_SOFT_TISSUE_COLOR = 255.0


def inject_rotation(img, angle=5.0, seed=0, flip=False):
    """Apply a small mis-orientation: a seeded, sub-degree-jittered rotation, optional H-flip.

    ``angle`` (degrees) is the strength knob. ``seed`` adds a small reproducible angular jitter,
    so different seeds yield distinct mis-orientations while a fixed seed is byte-reproducible.
    ``flip=True`` also mirrors the image left-right. Corners exposed by the rotation are filled
    with black, as with a genuinely mis-rotated acquisition. The whole-image geometry change is
    diagnosis-neutral.
    """
    from scipy.ndimage import rotate as _rotate

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if arr.ndim not in (2, 3):
        raise ValueError("image must be a 2-D or 3-D array")
    rng = np.random.default_rng(seed)
    eff_angle = float(angle) + float(rng.uniform(-0.75, 0.75))
    rotated = _rotate(
        arr.astype(np.float64),
        eff_angle,
        axes=(0, 1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )
    if flip:
        rotated = np.flip(rotated, axis=1)
    return np.clip(np.rint(rotated), 0, 255).astype(np.uint8)


def inject_compression(img, quality=30):
    """Round-trip the image through JPEG to stamp block/quantization artifacts on it.

    ``quality`` in [1, 100] is the strength knob: lower quality means coarser DCT quantization and
    stronger blocking. The tell is a purely global acquisition/compression artifact, independent
    of any diagnosis. Supports 2-D, ``(H, W, 1)`` and ``(H, W, 3)`` uint8 images.
    """
    import io

    from PIL import Image

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    q = int(np.clip(int(round(quality)), 1, 100))
    if arr.ndim == 2:
        pil, mode, expand = Image.fromarray(arr, mode="L"), "L", False
    elif arr.ndim == 3 and arr.shape[2] == 1:
        pil, mode, expand = Image.fromarray(arr[:, :, 0], mode="L"), "L", True
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil, mode, expand = Image.fromarray(arr, mode="RGB"), "RGB", False
    else:
        raise ValueError("compression supports 2-D, (H, W, 1), or (H, W, 3) uint8 images")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    dec = np.asarray(Image.open(buf).convert(mode), dtype=np.uint8)
    if expand:
        dec = dec[:, :, None]
    return np.ascontiguousarray(dec)


def inject_brightness(img, delta=0.3, contrast=1.0):
    """Apply a global brightness / contrast shift (an exposure acquisition tell).

    ``delta`` in [-1, 1] shifts every pixel by ``delta * 255`` and is the strength knob;
    ``contrast`` scales each pixel's deviation from mid-grey (128). Both are global and
    diagnosis-neutral. Values are clipped back into [0, 255].
    """
    work, was_2d = _to_float_hwc(img)
    shifted = (work - 128.0) * float(contrast) + 128.0 + float(delta) * 255.0
    return _from_float_hwc(shifted, was_2d)


def _soft_tissue_coverage(h, w, seed):
    """Seeded coverage map for a soft, gently curved skin-fold band across the image."""
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(0.0, np.pi))
    dx, dy = np.cos(theta), np.sin(theta)
    nx, ny = -dy, dx
    cx = w * float(rng.uniform(0.35, 0.65))
    cy = h * float(rng.uniform(0.35, 0.65))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xrel, yrel = xx - cx, yy - cy
    along = xrel * dx + yrel * dy
    across = xrel * nx + yrel * ny
    span = max(1.0, 0.5 * float(max(h, w)))
    amp = 0.12 * float(min(h, w))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    dist = across - amp * np.sin(along / span + phase)
    sigma = max(2.0, 0.05 * float(min(h, w)))
    return np.exp(-(dist * dist) / (2.0 * sigma * sigma))


def inject_soft_tissue(img, opacity=0.2, seed=0):
    """Composite a faint soft-tissue / skin-fold overlay (a soft bright band).

    The fold shape is seeded and reproducible; ``opacity`` in [0, 1] is the strength knob. Soft
    tissue reads as increased density (brighter) on an X-ray, so the band blends toward white and
    stays a localized, diagnosis-neutral overlay.
    """
    work, was_2d = _to_float_hwc(img)
    h, w = work.shape[:2]
    coverage = _soft_tissue_coverage(h, w, seed)
    _composite(work, coverage * float(opacity), _SOFT_TISSUE_COLOR)
    return _from_float_hwc(work, was_2d)


# Register the new cues so build_image_twin can construct twin pairs for each of them.
_INJECTORS.update(
    {
        "rotation": inject_rotation,
        "compression": inject_compression,
        "brightness": inject_brightness,
        "soft_tissue": inject_soft_tissue,
    }
)
