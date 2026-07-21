"""Golden-file tests for the core prompt templates in benchmaxxing.prompts.DEFAULT_REGISTRY.

Prompts decide whether a cascade can form (see issue #107), so a silent edit to a template's
rendered text is dangerous: it can change model behavior without anyone noticing in review.
Each registered (name, version) template is rendered against a fixed, synthetic input and
compared byte-for-byte to a checked-in golden file under ``tests/goldens/prompts/``.

To regenerate the goldens after an INTENTIONAL prompt change, run:

    BENCHMAXXING_UPDATE_GOLDENS=1 python -m pytest tests/test_prompts_golden.py -q

then inspect the diff on the rewritten files under ``tests/goldens/prompts/`` before committing
it -- the point of a golden file is that a change to it is a deliberate, reviewed decision.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmaxxing.prompts import COMMITTEE_MEMBER, DEFAULT_REGISTRY, ORCHESTRATOR, REFEREE

GOLDEN_DIR = Path(__file__).parent / "goldens" / "prompts"

# Fixed synthetic inputs, one per template name, kept stable so the goldens don't drift for
# reasons unrelated to the template text itself.
_FIXED_INPUTS = {
    COMMITTEE_MEMBER: {
        "case": (
            "55-year-old man, sudden pleuritic chest pain and dyspnea after a long flight."
        ),
        "question": "What is the most likely diagnosis?",
        "options": (
            "A. Pulmonary embolism\n"
            "B. Pneumothorax\n"
            "C. Myocardial infarction\n"
            "D. Costochondritis"
        ),
    },
    ORCHESTRATOR: {
        "question": "What is the most likely diagnosis?",
        "member_answers": (
            "agent0: A (0.80) - Risk factors and presentation fit PE.\n"
            "agent1: A (0.70) - Same reasoning, sudden onset after immobility.\n"
            "agent2: B (0.40) - Pleuritic pain could be pneumothorax."
        ),
    },
    REFEREE: {
        "transcript": (
            "[0] agent0 answered A, citing PE risk factors.\n"
            "[1] agent1 answered A, deferring to agent0's reasoning.\n"
            "[2] agent2 answered A after seeing the board, dropping an earlier independent B."
        ),
    },
}


def _golden_path(name: str, version: str) -> Path:
    return GOLDEN_DIR / f"{name}_{version}.txt"


def _all_registered_ids():
    return DEFAULT_REGISTRY.all_ids()


@pytest.mark.parametrize("name,version", _all_registered_ids())
def test_prompt_matches_golden(name, version):
    template = DEFAULT_REGISTRY.get(name, version)
    inputs = _FIXED_INPUTS.get(name)
    assert inputs is not None, (
        f"No fixed golden input registered for prompt {name!r} in this test's _FIXED_INPUTS; "
        "add one alongside the new template."
    )
    rendered = template.render(**inputs)
    path = _golden_path(name, version)

    if os.environ.get("BENCHMAXXING_UPDATE_GOLDENS"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return

    assert path.exists(), (
        f"Missing golden file {path}. Generate it with "
        "BENCHMAXXING_UPDATE_GOLDENS=1 python -m pytest tests/test_prompts_golden.py -q"
    )
    golden = path.read_text(encoding="utf-8")
    assert rendered == golden, (
        f"Rendered {name!r} (version {version!r}) no longer matches its golden file. If this "
        "is an intentional prompt change, regenerate it with BENCHMAXXING_UPDATE_GOLDENS=1 "
        "and review the diff before committing."
    )


def test_every_default_registry_template_has_a_golden_input():
    # Catches a new prompt template landing without a matching golden fixture, rather than
    # letting it silently skip coverage.
    registered_names = {name for name, _ in DEFAULT_REGISTRY.all_ids()}
    assert registered_names <= set(_FIXED_INPUTS)
