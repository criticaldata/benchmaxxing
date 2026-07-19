"""On-disk run store for resumable benchmaxxing runs (issue 89).

A long committee run produces one :class:`~benchmaxxing.schema.Transcript` per unit of work
(typically a case + condition + committee). If the process dies partway through we do not want to
redo the units that already finished. :class:`RunStore` persists each finished transcript to its
own JSONL file (reusing :func:`benchmaxxing.transcript.dump_transcript`, so the round-trip stays
lossless) under a caller-chosen string key. On restart, :meth:`RunStore.iter_pending` walks the
planned keys and yields only the ones with no file on disk, so a run resumes exactly where it
stopped.

Keys are arbitrary strings (for example ``"case-abc::contaminated"``). Each is percent-encoded into
a single filesystem-safe filename and decoded back when listing, so :meth:`RunStore.keys` returns
the exact set of saved keys with no sidecar index to keep in sync. Writes are atomic (dump to a
temp file, then :func:`os.replace`) so :meth:`RunStore.has` is never true for a half-written
transcript left behind by a crash.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote, unquote

from benchmaxxing.schema import Transcript
from benchmaxxing.transcript import dump_transcript, load_transcript

_SUFFIX = ".jsonl"


class RunStore:
    """A directory of saved transcripts, one file per key, for resumable runs."""

    def __init__(self, dir: str | Path) -> None:
        """Open (creating if needed) the store rooted at ``dir``."""
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        """Map a key to its on-disk transcript path (percent-encoded, so any key is safe)."""
        return self.dir / (quote(str(key), safe="") + _SUFFIX)

    @staticmethod
    def _key_for(path: Path) -> str:
        """Recover the original key from a saved transcript path (inverse of ``_path_for``)."""
        return unquote(path.name[: -len(_SUFFIX)])

    def has(self, key: str) -> bool:
        """Return True if a transcript for ``key`` has already been saved."""
        return self._path_for(key).is_file()

    def save_transcript(self, key: str, transcript: Transcript) -> Path:
        """Persist ``transcript`` under ``key`` losslessly and return its path.

        Overwrites any existing entry for the key. The write is atomic: the transcript is dumped
        to a temporary file in the same directory and then renamed into place, so an interrupted
        save never leaves a partial file that :meth:`has` or :meth:`load` would pick up.
        """
        final = self._path_for(key)
        tmp = final.with_name(f"{final.name}.tmp-{os.getpid()}")
        try:
            dump_transcript(transcript, tmp)
            os.replace(tmp, final)
        finally:
            if tmp.exists():
                tmp.unlink()
        return final

    def load(self, key: str) -> Transcript:
        """Read back the transcript saved under ``key`` (raises ``KeyError`` if absent)."""
        path = self._path_for(key)
        if not path.is_file():
            raise KeyError(f"no saved transcript for key: {key!r}")
        return load_transcript(path)

    def keys(self) -> list[str]:
        """Return the sorted list of keys that have a saved transcript."""
        found = [
            self._key_for(p)
            for p in self.dir.iterdir()
            if p.is_file() and p.name.endswith(_SUFFIX)
        ]
        return sorted(found)

    def iter_pending(self, all_keys: Iterable[str]) -> Iterator[str]:
        """Yield keys in ``all_keys`` that are not yet saved, preserving first-seen order.

        This is the resume primitive: pass the full plan of keys for the run and iterate the
        result to process only the outstanding work. Duplicate keys in the input are yielded at
        most once, so a plan with repeats still maps to each unit of work exactly once.
        """
        seen: set[str] = set()
        for key in all_keys:
            if key in seen:
                continue
            seen.add(key)
            if not self.has(key):
                yield key
