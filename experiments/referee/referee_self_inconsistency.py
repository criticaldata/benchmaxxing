"""Referee self-inconsistency floor (#417).

Measures whether identical cache-bypassed temperature-0 private re-queries
produce different answers in the absence of committee influence.
"""

from __future__ import annotations


def summarize(rows):
    n = len(rows)
    unstable = sum(1 for r in rows if r["temp0_flip"])

    return {
        "n": n,
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
