# Your first dataset adapter in 20 minutes

Everything downstream of a manifest is dataset-agnostic and already built, so an adapter is the
only code a new dataset needs. This walkthrough rebuilds the MedQA adapter from scratch. MedQA is
already in the repo (`benchmaxxing/datasets/medqa.py`), so you can diff your version against the
real one at any point, and the same five steps work for whatever dataset you are adding.

Start from a working checkout:

```bash
pip install -e ".[dev]"
pytest -q
```

## 1. Know what you are producing

An adapter turns a raw release into a list of `benchmaxxing.schema.Case` and writes it as a
manifest. That is the whole job. A `Case` is frozen and small:

| Field | Text / MCQ lane | Imaging lane |
| --- | --- | --- |
| `case_id` | required, unique across the manifest | required, unique |
| `patient_id` | optional, defaults to `case_id` on load | required, and it must group images of the same patient |
| `modality` | `Modality.TEXT` | `Modality.IMAGE` |
| `question` | required, the stem | unused |
| `options` | required, ordered tuple | unused |
| `answer_index` | required, index into `options` | unused |
| `image_ref` | unused | required, path resolved against `--image-root` |
| `report` | optional context | optional free text |
| `label` | optional | the finding, your documented multi-label policy |
| `meta` | anything else worth keeping | anything else worth keeping |

Two rules that bite later if you get them wrong: `case_id` has to be unique (`finalize` raises on
duplicates), and imaging cases have to be keyed by patient, because the pairing step picks a
same-patient swap image.

## 2. Read the raw release first

Look at one record before writing any code. MedQA-USMLE ships JSONL splits where each line is:

```json
{"question": "A 23-year-old pregnant woman at 22 weeks gestation presents with ...",
 "options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin"},
 "answer_idx": "D",
 "meta_info": "step2&3"}
```

So the mapping is: `question` to the stem, the `options` map ordered by its sorted keys (A, B, C,
D, E) to the `options` tuple, and the position of `answer_idx` in that order to `answer_index`.
Write that mapping down in prose before you write it in Python. It goes in the module docstring
and in `SPEC.notes`, and it is what a reviewer checks.

## 3. Write the module

One file per dataset, in `benchmaxxing/datasets/<name>.py`. It publishes a `SPEC` (so the
registry and `benchmaxxing datasets` can describe it) and a `build_manifest(raw_root, out,
limit=None)` entry point:

```python
"""MedQA adapter (text / MCQ, Lane B)."""

from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="medqa",
    raw_hint=(
        "MedQA-USMLE (Jin et al. 2020): JSONL splits (train/dev/test) where each line has "
        "'question', an 'options' map (A..E) and 'answer_idx'/'answer'."
    ),
    modality=Modality.TEXT,
    notes=(
        "One Case per question: case_id=stable question id, question=stem, options=ordered "
        "A..E values, answer_index=position of answer_idx in that order."
    ),
)


def _case_from_obj(obj: dict, index: int) -> Case:
    options_map = obj["options"]
    keys = sorted(options_map)
    answer_letter = obj["answer_idx"]
    if answer_letter not in keys:
        raise ValueError(
            f"Row {index}: answer_idx {answer_letter!r} is not among option keys {keys}."
        )
    return Case(
        case_id=str(obj.get("id") or f"medqa-{index}"),
        patient_id="",
        modality=Modality.TEXT,
        question=str(obj["question"]),
        options=tuple(str(options_map[key]) for key in keys),
        answer_index=keys.index(answer_letter),
    )


def build_manifest(raw_root, out, limit=None):
    path = Path(raw_root)
    if path.is_dir():
        path = path / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"MedQA JSONL not found: {path}")
    cases = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if limit is not None and len(cases) >= limit:
            break
        cases.append(_case_from_obj(json.loads(line), index))
    return finalize(cases, out)
```

Three things to copy from that shape, whatever your dataset looks like:

- Accept both a file and the directory that contains it as `raw_root`. Contributors will pass
  either one, and guessing wrong costs them five minutes of confusion.
- Raise `FileNotFoundError` with the path you actually looked for. "MedQA JSONL not found:
  /data/medqa" is a fixable error; a `KeyError` from three frames down is not.
- Let `finalize(cases, out)` do the writing. It enforces unique `case_id` values and delegates to
  `data.write_manifest`, so every adapter emits the same columns and everything downstream
  (`load_cases`, the cue builders, `benchmaxxing datasets stats`) just works.

Imaging adapters differ in one place only: fill `image_ref` and a real `patient_id` instead of the
MCQ fields, and put the extra findings and the view in `meta`. `benchmaxxing/datasets/nih_cxr14.py`
is the shortest imaging example.

## 4. Register it

Add the module to `benchmaxxing/datasets/registry.py`:

```python
from benchmaxxing.datasets import chexpert, ehr, medmcqa, medqa, mimic_cxr, nih_cxr14, pubmedqa

REGISTRY: dict[str, ModuleType] = {
    ...
    medqa.SPEC.name: medqa,
}
```

The key is `SPEC.name`, never a hand-typed string, so the registry cannot drift from the spec.

## 5. Test it two ways

The testing policy is two tiers, and an adapter needs both. Put them in
`tests/test_<name>_adapter.py`:

```python
import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import medqa
from benchmaxxing.schema import Modality

ROW = (
    '{"question": "Best next step?", "options": {"A": "Aspirin", "B": "Heparin"}, '
    '"answer_idx": "B", "id": "q1"}'
)


def test_build_manifest_roundtrip(tmp_path):
    (tmp_path / "test.jsonl").write_text(ROW + "\n", encoding="utf-8")
    out = tmp_path / "manifest.csv"
    medqa.build_manifest(tmp_path, out)

    cases = load_cases(out)
    assert len(cases) == 1
    assert cases[0].modality is Modality.TEXT
    assert cases[0].options == ("Aspirin", "Heparin")
    assert cases[0].answer_index == 1


def test_missing_raw_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="MedQA"):
        medqa.build_manifest(tmp_path, tmp_path / "manifest.csv")
```

The first is a shape test on a tiny synthetic raw layout: it pins the mapping, and it runs
anywhere with no data and no key. The second pins the error path, which is the one a contributor
with a half-downloaded dataset will hit. Behavioral claims about what the data means still belong
on real data, never on fabricated rows.

One gotcha: the manifest round trip preserves `meta` as a JSON column, but if you assert on
anything unusual, assert on the `Case` your parser returns rather than on the reloaded manifest.
That keeps the test about your mapping instead of about manifest I/O.

## 6. Verify end to end

```bash
# your adapter shows up in the registry
benchmaxxing datasets

# build a small manifest from the real release
python -c "from benchmaxxing.datasets import medqa; medqa.build_manifest('/data/medqa', 'medqa.csv', limit=20)"

# sanity-check what you produced (row counts, MCQ shape, label distribution)
benchmaxxing datasets stats medqa.csv

# imaging adapters: confirm the image paths resolve
benchmaxxing datasets stats cxr.csv --image-root /data/images
```

Even a one-row real manifest counts. It proves the mapping survives contact with the actual
release, which a synthetic fixture cannot.

## 7. Before you open the PR

```bash
ruff check . && pytest -q
```

Then check the four things review will ask about:

- License and access are stated in the module docstring and in `SPEC.raw_hint`, including whether
  the source needs credentialing.
- The multi-label policy is documented, if the source has one label per finding.
- Imaging cases are keyed by patient.
- `benchmaxxing datasets stats` on your manifest reports the row counts you expect.

Open the PR against `main` with "Closes #<issue>" in the description. Each dataset has one owner,
so pick up the open issue for your dataset first and assign it to yourself.
