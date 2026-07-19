"""Behavioral checks for the added acquisition-style image cues.

Covers the mis-orientation cue (rotation + horizontal flip, issue 67), the compression / exposure
cues (JPEG blocking and brightness shift, issue 68), and the soft-tissue overlay (issue 69). As
with the sibling suite these use small synthetic uint8 arrays: a check of each injector's own
behavior (determinism, non-mutation, a change vs the clean image, and a graded strength knob), not
a claim about any model. All rendering needs PIL/scipy, so the module skips cleanly without them.
"""

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("scipy")

from benchmaxxing.cues import image as ic  # noqa: E402
from benchmaxxing.schema import TwinPair  # noqa: E402

BASE_VALUE = 100


def _gray(h=80, w=80):
    return np.full((h, w), BASE_VALUE, dtype=np.uint8)


def _pattern(h=96, w=96):
    """A deterministic, left-right-asymmetric image with edges (good for JPEG/flip/rotation)."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx * 2 + yy) % 256).astype(np.uint8)
    base[10:30, 10:40] = 240  # off-centre bright block for asymmetry
    return base


def _abs_diff(out, base):
    return int(np.abs(out.astype(np.int64) - base.astype(np.int64)).sum())


# (name, fn, kwargs, base-factory) covering every new injector.
NEW_CUES = [
    ("rotation", ic.inject_rotation, {"angle": 8.0, "seed": 0}, _gray),
    ("compression", ic.inject_compression, {"quality": 20}, _pattern),
    ("brightness", ic.inject_brightness, {"delta": 0.3}, _gray),
    ("soft_tissue", ic.inject_soft_tissue, {"opacity": 0.3, "seed": 0}, _gray),
]


@pytest.mark.parametrize("name,fn,kwargs,factory", NEW_CUES)
def test_output_is_uint8_same_shape_and_differs(name, fn, kwargs, factory):
    base = factory()
    out = fn(base, **kwargs)
    assert out.dtype == np.uint8
    assert out.shape == base.shape
    assert not np.array_equal(out, base)


@pytest.mark.parametrize("name,fn,kwargs,factory", NEW_CUES)
def test_input_array_not_mutated(name, fn, kwargs, factory):
    base = factory()
    original = base.copy()
    fn(base, **kwargs)
    assert np.array_equal(base, original)


@pytest.mark.parametrize("name,fn,kwargs,factory", NEW_CUES)
def test_injection_is_deterministic(name, fn, kwargs, factory):
    base = factory()
    first = fn(base, **kwargs)
    second = fn(base, **kwargs)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name,fn,kwargs,factory", NEW_CUES)
def test_works_on_rgb_images(name, fn, kwargs, factory):
    plane = factory()
    base = np.stack([plane, plane, plane], axis=2)
    out = fn(base, **kwargs)
    assert out.dtype == np.uint8
    assert out.shape == base.shape
    assert int((out != base).any(axis=2).sum()) > 0


@pytest.mark.parametrize("name,fn,kwargs,factory", NEW_CUES)
def test_injectors_reject_non_uint8(name, fn, kwargs, factory):
    bad = np.full((16, 16), 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        fn(bad, **kwargs)


# --- rotation / mis-orientation (issue 67) ------------------------------------------------


def test_rotation_angle_is_graded():
    base = _gray()
    angles = (2.0, 5.0, 10.0, 20.0)
    diffs = [_abs_diff(ic.inject_rotation(base, angle=a, seed=3), base) for a in angles]
    # A larger tilt exposes more black corner, so total deviation grows with the angle.
    assert all(lo < hi for lo, hi in zip(diffs, diffs[1:]))
    assert diffs[0] > 0


def test_rotation_is_localized_on_uniform_image():
    base = _gray()
    out = ic.inject_rotation(base, angle=8.0, seed=0)
    changed = int((out != base).sum())
    # A small rotation of a flat image only disturbs the exposed corners, not the whole frame.
    assert 0 < changed < base.size


def test_rotation_seed_controls_orientation():
    base = _pattern()
    same_a = ic.inject_rotation(base, angle=5.0, seed=7)
    same_b = ic.inject_rotation(base, angle=5.0, seed=7)
    other = ic.inject_rotation(base, angle=5.0, seed=8)
    assert np.array_equal(same_a, same_b)
    # Different seeds jitter the angle differently, so the mis-orientation is distinct.
    assert not np.array_equal(same_a, other)


def test_rotation_horizontal_flip_variant():
    base = _pattern()
    no_flip = ic.inject_rotation(base, angle=0.0, seed=0, flip=False)
    flipped = ic.inject_rotation(base, angle=0.0, seed=0, flip=True)
    # The flip variant is deterministic and mirrors the frame, so it differs from the un-flipped.
    assert np.array_equal(flipped, ic.inject_rotation(base, angle=0.0, seed=0, flip=True))
    assert not np.array_equal(flipped, no_flip)


# --- compression / brightness acquisition tell (issue 68) ---------------------------------


def test_compression_quality_is_graded():
    base = _pattern()
    qualities = (90, 60, 30, 10)
    diffs = [_abs_diff(ic.inject_compression(base, quality=q), base) for q in qualities]
    # Lower JPEG quality quantizes more coarsely, so deviation grows as quality drops.
    assert all(lo < hi for lo, hi in zip(diffs, diffs[1:]))
    assert diffs[0] > 0


def test_compression_rejects_unsupported_channel_count():
    bad = np.zeros((8, 8, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        ic.inject_compression(bad)


def test_brightness_delta_is_graded():
    base = _gray()
    deltas = (0.1, 0.2, 0.4, 0.6)
    diffs = [_abs_diff(ic.inject_brightness(base, delta=d), base) for d in deltas]
    # A larger brightness shift moves every pixel further from the clean value.
    assert all(lo < hi for lo, hi in zip(diffs, diffs[1:]))
    assert diffs[0] > 0


def test_brightness_shift_is_global():
    base = _gray()
    out = ic.inject_brightness(base, delta=0.3)
    # An exposure shift is global: every pixel changes, unlike a localized overlay.
    assert int((out != base).sum()) == base.size


def test_brightness_sign_moves_in_expected_direction():
    base = _gray()
    brighter = ic.inject_brightness(base, delta=0.3)
    darker = ic.inject_brightness(base, delta=-0.3)
    assert int(brighter.mean()) > BASE_VALUE
    assert int(darker.mean()) < BASE_VALUE


# --- soft-tissue / skin-fold overlay (issue 69) -------------------------------------------


def test_soft_tissue_opacity_is_graded():
    base = _gray()
    opacities = (0.1, 0.2, 0.4, 0.7)
    diffs = [_abs_diff(ic.inject_soft_tissue(base, opacity=o, seed=0), base) for o in opacities]
    # Higher opacity blends the fold more strongly toward white.
    assert all(lo < hi for lo, hi in zip(diffs, diffs[1:]))
    assert diffs[0] > 0


def test_soft_tissue_is_localized():
    base = _gray()
    out = ic.inject_soft_tissue(base, opacity=0.3, seed=0)
    changed = int((out != base).sum())
    # The fold is a band: it touches many pixels but leaves most of the frame clean.
    assert 0 < changed < base.size


def test_soft_tissue_seed_controls_fold():
    base = _gray()
    same_a = ic.inject_soft_tissue(base, opacity=0.4, seed=1)
    same_b = ic.inject_soft_tissue(base, opacity=0.4, seed=1)
    other = ic.inject_soft_tissue(base, opacity=0.4, seed=2)
    assert np.array_equal(same_a, same_b)
    # Different seeds place the fold differently.
    assert not np.array_equal(same_a, other)


# --- build_image_twin entries for the new cues --------------------------------------------


def test_build_image_twin_soft_tissue():
    base = _gray()
    twin = ic.build_image_twin(
        base, "soft_tissue", ground_truth="effusion", case_id="cxr_soft", seed=3, opacity=0.4
    )
    assert isinstance(twin, TwinPair)
    assert twin.case_id == "cxr_soft"
    assert twin.cue_type == "soft_tissue"
    assert twin.cue_params == {"seed": 3, "opacity": 0.4}
    assert twin.ground_truth == "effusion"
    assert np.array_equal(twin.clean, base)
    assert np.array_equal(twin.contaminated, ic.inject_soft_tissue(base, seed=3, opacity=0.4))
    assert not np.array_equal(twin.clean, twin.contaminated)


@pytest.mark.parametrize(
    "cue_type,fn,params",
    [
        ("rotation", ic.inject_rotation, {"angle": 6.0, "seed": 2}),
        ("compression", ic.inject_compression, {"quality": 25}),
        ("brightness", ic.inject_brightness, {"delta": 0.5}),
    ],
)
def test_build_image_twin_registers_each_new_cue(cue_type, fn, params):
    base = _pattern()
    twin = ic.build_image_twin(base, cue_type, **params)
    assert isinstance(twin, TwinPair)
    assert twin.cue_type == cue_type
    assert twin.cue_params == params
    assert np.array_equal(twin.clean, base)
    assert np.array_equal(twin.contaminated, fn(base, **params))
    assert not np.array_equal(twin.clean, twin.contaminated)
