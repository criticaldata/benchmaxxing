"""Behavioral checks for the deterministic image cue injectors.

These tests use small synthetic uint8 arrays. That is a sanctioned check of the injector's own
shape/behavior (does it change only where the cue lands, is it deterministic given a seed, does
opacity give a graded strength), not a claim about any model. All cue rendering needs PIL, so the
whole module skips cleanly if PIL is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("PIL")

from benchmaxxing.cues import image as ic  # noqa: E402
from benchmaxxing.schema import TwinPair  # noqa: E402

BASE_VALUE = 100


def _gray(h=80, w=80):
    return np.full((h, w), BASE_VALUE, dtype=np.uint8)


def _abs_diff(out, base):
    return int(np.abs(out.astype(np.int64) - base.astype(np.int64)).sum())


ALL_CUES = [
    ("cable", ic.inject_cable, {}),
    ("corner_tag", ic.inject_corner_tag, {}),
    ("watermark", ic.inject_watermark, {}),
    ("laterality", ic.inject_laterality, {}),
]


@pytest.mark.parametrize("name,fn,kwargs", ALL_CUES)
def test_cue_changes_some_but_not_all(name, fn, kwargs):
    base = _gray()
    out = fn(base, **kwargs)
    assert out.dtype == np.uint8
    assert out.shape == base.shape
    changed = int((out != base).sum())
    # The cue is localized: it touches at least one pixel but not the whole image.
    assert 0 < changed < base.size


@pytest.mark.parametrize("name,fn,kwargs", ALL_CUES)
def test_input_array_not_mutated(name, fn, kwargs):
    base = _gray()
    original = base.copy()
    fn(base, **kwargs)
    assert np.array_equal(base, original)


@pytest.mark.parametrize("name,fn,kwargs", ALL_CUES)
def test_injection_is_deterministic(name, fn, kwargs):
    base = _gray()
    first = fn(base, **kwargs)
    second = fn(base, **kwargs)
    assert np.array_equal(first, second)


def test_cable_seed_controls_placement():
    base = _gray()
    same_a = ic.inject_cable(base, seed=11)
    same_b = ic.inject_cable(base, seed=11)
    other = ic.inject_cable(base, seed=999)
    assert np.array_equal(same_a, same_b)
    # Different seeds pick different auto positions, so the output differs.
    assert not np.array_equal(same_a, other)


def test_cable_changes_only_within_its_footprint():
    base = _gray()
    position = ((20, 0), (55, base.shape[0] - 1))
    footprint = ic._cable_coverage(base.shape[0], base.shape[1], position, 3, 0) > 0
    out = ic.inject_cable(base, position=position, thickness=3)
    changed = out != base
    # Every changed pixel lies inside the cue footprint: it changes only where the cue is.
    assert not np.any(changed & ~footprint)
    assert changed.sum() > 0


@pytest.mark.parametrize(
    "corner,clean_slice",
    [
        ("top-left", np.s_[40:, :]),
        ("top-right", np.s_[40:, :]),
        ("bottom-left", np.s_[:40, :]),
        ("bottom-right", np.s_[:40, :]),
    ],
)
def test_corner_tag_confined_to_its_corner(corner, clean_slice):
    base = _gray()
    out = ic.inject_corner_tag(base, corner=corner)
    # The opposite half of the image is untouched by a corner tag.
    assert np.array_equal(out[clean_slice], base[clean_slice])
    assert not np.array_equal(out, base)


def test_laterality_confined_to_its_corner():
    base = _gray()
    out = ic.inject_laterality(base, mark="R", corner="top-right")
    # A top-right marker leaves the whole bottom half and the left edge clean.
    assert np.array_equal(out[40:, :], base[40:, :])
    assert np.array_equal(out[:, :40], base[:, :40])
    assert not np.array_equal(out, base)


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (ic.inject_cable, {}),
        (ic.inject_corner_tag, {}),
    ],
)
def test_opacity_sweep_is_graded(fn, kwargs):
    base = _gray()
    opacities = (0.1, 0.3, 0.6, 0.9)
    diffs = [_abs_diff(fn(base, opacity=o, **kwargs), base) for o in opacities]
    # Higher opacity is a stronger cue: total deviation from the clean image increases.
    assert all(lo < hi for lo, hi in zip(diffs, diffs[1:]))
    assert diffs[0] > 0


def test_corner_tag_size_controls_extent():
    base = _gray()
    small = ic.inject_corner_tag(base, size=0.06)
    large = ic.inject_corner_tag(base, size=0.20)
    # A larger font tag covers more pixels.
    assert int((large != base).sum()) > int((small != base).sum())


@pytest.mark.parametrize("name,fn,kwargs", ALL_CUES)
def test_works_on_rgb_images(name, fn, kwargs):
    base = np.full((80, 80, 3), BASE_VALUE, dtype=np.uint8)
    out = fn(base, **kwargs)
    assert out.dtype == np.uint8
    assert out.shape == base.shape
    changed = int((out != base).any(axis=2).sum())
    assert 0 < changed < base.shape[0] * base.shape[1]


def test_build_image_twin_returns_pair():
    base = _gray()
    twin = ic.build_image_twin(
        base, "cable", ground_truth="pneumonia", case_id="cxr_001", seed=5, opacity=0.7
    )
    assert isinstance(twin, TwinPair)
    assert twin.case_id == "cxr_001"
    assert twin.cue_type == "cable"
    assert twin.cue_params == {"seed": 5, "opacity": 0.7}
    assert twin.ground_truth == "pneumonia"
    # Clean is the untouched original; contaminated matches a direct injector call.
    assert np.array_equal(twin.clean, base)
    assert np.array_equal(twin.contaminated, ic.inject_cable(base, seed=5, opacity=0.7))
    assert not np.array_equal(twin.clean, twin.contaminated)


def test_build_image_twin_clean_isolated_from_source_mutation():
    base = _gray()
    twin = ic.build_image_twin(base, "corner_tag")
    base[0, 0] = 0
    # The stored clean payload is a copy, unaffected by later mutation of the source.
    assert twin.clean[0, 0] == BASE_VALUE


def test_build_image_twin_rejects_unknown_cue():
    with pytest.raises(ValueError):
        ic.build_image_twin(_gray(), "not_a_cue")


def test_injectors_reject_non_uint8():
    bad = np.full((16, 16), 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        ic.inject_cable(bad)
    with pytest.raises(ValueError):
        ic.build_image_twin(bad, "cable")


def test_corner_tag_rejects_bad_corner():
    with pytest.raises(ValueError):
        ic.inject_corner_tag(_gray(), corner="middle")


# ---------------------------------------------------------------------------
# Cue determinism across environments (#393).
#
# The three text cues used to rasterise through ImageFont.load_default(), whose typeface and
# size handling vary by Pillow version. The imaging call caches key on sha256 of the contaminated
# pixels, so a font change silently invalidated cached reads and re-queried the model instead of
# reporting a mismatch. These are golden-pixel tests: they fail loudly the moment the rendered cue
# stops matching the pixels the committed results were scored against.
# ---------------------------------------------------------------------------

# sha256 of the contaminated pixels for each text cue at the default parameters on an 80x80 uint8
# field of BASE_VALUE. Regenerate ONLY alongside a rerun of every imaging arm; changing a value
# here without rerunning silently orphans the committed per-case rows.
GOLDEN_CUE_PIXELS = {
    "corner_tag": "428095b6c2663b0b5d9923541655122682d661eb356206facd799c0639fb2fc9",
    "watermark": "b5a573c4e5f9762d76524ce80b7029c01d5b6716e8a066a449fadb71ef051290",
    "laterality": "913797e9436731df3ac75b8848ab646f81f188692eb12dda2ee4e6034d0d8e68",
}

TEXT_CUES = [
    ("corner_tag", ic.inject_corner_tag),
    ("watermark", ic.inject_watermark),
    ("laterality", ic.inject_laterality),
]


@pytest.mark.parametrize("name,fn", TEXT_CUES)
def test_text_cue_pixels_match_the_golden_checksum(name, fn):
    """The rendered cue is byte-identical to what the committed imaging results were scored on."""
    assert ic.cue_checksum(fn(_gray())) == GOLDEN_CUE_PIXELS[name], (
        f"{name} renders different pixels than the committed results were scored against. The "
        "imaging call cache is keyed on these bytes, so this WILL silently invalidate it (#393). "
        "Do not update the golden value without rerunning every imaging arm."
    )


def test_text_cues_use_the_vendored_font_not_the_pillow_default():
    """A vendored face, so glyphs do not move with the Pillow version."""
    assert ic._FONT_PATH.exists(), "the vendored font must ship with the package"
    font = ic._load_font(20)
    # A TrueType face exposes .path/.size; the old bitmap default did not.
    assert hasattr(font, "size"), "text cues must render through a scalable TrueType face"
    assert font.size == 20


def test_font_checksum_mismatch_fails_loudly(monkeypatch, tmp_path):
    """A swapped or corrupted font raises, rather than quietly rendering different glyphs."""
    fake = tmp_path / "DejaVuSans.ttf"
    fake.write_bytes(b"not a font")
    monkeypatch.setattr(ic, "_FONT_PATH", fake)
    ic._font_bytes.cache_clear()
    try:
        with pytest.raises(ic.CueRenderError, match="checksum mismatch"):
            ic._load_font(20)
    finally:
        ic._font_bytes.cache_clear()


def test_missing_font_fails_loudly(monkeypatch, tmp_path):
    """No silent fallback to a system font: a missing vendored face is a hard error."""
    monkeypatch.setattr(ic, "_FONT_PATH", tmp_path / "absent.ttf")
    ic._font_bytes.cache_clear()
    try:
        with pytest.raises(ic.CueRenderError, match="vendored font missing"):
            ic._load_font(20)
    finally:
        ic._font_bytes.cache_clear()


def test_cue_checksum_is_the_cache_identity():
    """cue_checksum distinguishes cued from clean, and is stable across repeated renders."""
    base = _gray()
    assert ic.cue_checksum(base) != ic.cue_checksum(ic.inject_watermark(base))
    assert ic.cue_checksum(ic.inject_watermark(base)) == ic.cue_checksum(ic.inject_watermark(base))
