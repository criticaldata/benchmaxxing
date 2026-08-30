"""Referee self-inconsistency floor (#417).

Measures whether identical cache-bypassed temperature-0 private re-queries
produce different answers in the absence of committee influence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path



class _DrawCache:
    """Replayable cache whose key keeps independent temp-0 draws distinct."""

    def __init__(self, path):
        self.path = Path(path)
        self.store = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.store[record["k"]] = record["resp"]

    @staticmethod
    def key(model, prompt, draw):
        return hashlib.sha256(
            f"{model}\x000.0\x00{draw}\x00{prompt}".encode()
        ).hexdigest()


def summarize(rows):
    n = len(rows)
    unstable = sum(1 for r in rows if r["temp0_flip"])

    return {
        "n": n,
        "temperature": 0,
        "cache_bypassed": True,
        "stable_cases": n - unstable,
        "unstable_cases": unstable,
        "temp0_self_inconsistency_rate": (
            unstable / n if n else None
        ),
    }


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
