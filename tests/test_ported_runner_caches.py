"""Two runners ported to the shared text-lane dispatch could not run for any second model.

live_peer_organic.py passed (model, prompt) to a cache whose signature is complete(prompt, model=None),
so the question text went out as the model id, and its --model never reached the committee, whose
holdout was bound to the Gemini constant. temperature_sensitivity.py called the shared cache with a
temperature and sample index it does not take. These tests pin the repaired behaviour and, for the
sweep, that the cache key is still the one the committed Gemini sweep was written with, so that arm
replays with no calls.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "medqa"))
import _lane  # noqa: E402
import temperature_sensitivity as ts  # noqa: E402

MODEL = "openai/gpt-oss-120b"


class _Backend:
    def __init__(self):
        self.seen = []

    def complete(self, prompt, decoding=None):
        self.seen.append((prompt, dict(decoding or {})))
        return "B"


def test_the_sweep_cache_keys_on_temperature_and_sample_and_reads_the_committed_key_format(tmp_path):
    cache = ts._DrawCache(tmp_path / "c.jsonl", None, MODEL)
    k = hashlib.sha256(f"{MODEL}\x000.7\x002\x00Q".encode()).hexdigest()
    cache.store[k] = "C"
    # A hit on the committed key format needs no key and no backend.
    assert cache.complete("Q", 0.7, 2) == "C"
    # A different draw of the same prompt is a different key, so sampled draws never collide.
    with pytest.raises(SystemExit):
        cache.complete("Q", 0.7, 3)


def test_the_sweep_passes_the_temperature_to_the_backend_and_records_the_draw(tmp_path, monkeypatch):
    backend = _Backend()
    monkeypatch.setattr(_lane, "backend_for", lambda model, key, client=None: backend)
    monkeypatch.setattr(_lane.gateway, "RetryBackend", lambda b, tries, backoff: b)
    monkeypatch.setattr(_lane, "_pace", lambda model: None)
    cache = ts._DrawCache(tmp_path / "c.jsonl", "k", MODEL)
    assert cache.complete("Q", 1.0, 1) == "B"
    assert backend.seen == [("Q", {"temperature": 1.0})]
    row = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])
    assert (row["model"], row["temperature"], row["sample"]) == (MODEL, 1.0, 1)
    assert cache.calls == 1


def test_live_peer_organic_sends_the_prompt_as_the_prompt_and_the_model_as_the_model():
    src = (Path(__file__).resolve().parents[1] / "experiments" / "medqa" / "live_peer_organic.py").read_text()
    # The shared Cache takes (prompt, model=None); every call in this runner must lead with the prompt.
    assert "cache.complete(p, backend_model)" in src
    assert "cache.complete(base_p, model)" in src
    assert "cache.complete(backend_model, p)" not in src
    assert "cache.complete(HOLDOUT, base_p)" not in src


def test_live_peer_organic_binds_the_holdout_to_the_requested_model():
    src = (Path(__file__).resolve().parents[1] / "experiments" / "medqa" / "live_peer_organic.py").read_text()
    assert 'members = [(a, model if a == "holdout" else m) for a, m in MEMBERS]' in src
    assert '"models": {"peers": PEER_MODEL, "holdout": model}' in src
