"""Run configuration for benchmaxxing.

A tiny, dependency-light config layer. A JSON file is parsed with the standard library and a
YAML file when pyyaml is importable (the optional ``config`` extra); a plain dict is also
accepted directly so the core stays usable with no extra installed. Everything resolves to a
:class:`Config` dataclass with ``from_dict`` / ``to_dict`` for round-tripping.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Same-lineage Gemini-only default roster (see the model-backend note in the README).
DEFAULT_MODELS = ("gemini-2.5-flash",)


@dataclass
class Config:
    """Resolved run configuration.

    Attributes:
        models: model ids in the roster for this run.
        roster: optional per-model declarations (``id``, ``lineage``, ``tier``,
            ``open_weights``), parallel to ``models``. Empty when the config lists bare ids,
            in which case lineage is inferred from the id.
        dataset: dataset adapter name (see ``benchmaxxing.datasets``).
        cue_set: cue-set version to inject.
        seed: master random seed for reproducibility.
        out_dir: directory where run artifacts are written.
    """

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    roster: list[dict] = field(default_factory=list)
    dataset: str = "worldmedqa_v2"
    cue_set: str = "v1"
    seed: int = 0
    out_dir: str = "runs"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Config":
        """Build a Config from a mapping, ignoring unknown keys.

        A ``models`` string is promoted to a one-element list and ``seed`` is coerced to
        int, so hand-written or YAML-parsed configs stay forgiving. A ``models`` entry may
        also be a mapping that declares the model's lineage instead of leaving it to be
        guessed from the id::

            models:
              - gemini-2.5-flash                       # lineage inferred
              - {id: my-tuned-cxr, lineage: llama, tier: 8b, open_weights: true}

        Declared entries are split into ``models`` (the ids, so every existing caller is
        unchanged) and ``roster`` (the declarations).
        """
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError(
                f"Config.from_dict expects a mapping, got {type(data).__name__}."
            )
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        models = kwargs.get("models")
        if models is not None:
            if isinstance(models, str):
                models = [models]
            ids, roster = _split_roster(models)
            kwargs["models"] = ids
            # An explicit roster key wins over one recovered from the model entries, so a
            # to_dict/from_dict round trip does not double up.
            kwargs.setdefault("roster", roster)
        if kwargs.get("roster") is not None:
            kwargs["roster"] = [dict(entry) for entry in kwargs["roster"]]
        if kwargs.get("seed") is not None:
            kwargs["seed"] = int(kwargs["seed"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dict of the config fields."""
        return asdict(self)


def _split_roster(models) -> tuple[list[str], list[dict]]:
    """Split a mixed list of ids and declaration mappings into ids plus declarations.

    A bare id contributes no declaration, so the roster stays empty for the common case and
    a caller can tell a declared lineage from an inferred one.
    """
    ids: list[str] = []
    roster: list[dict] = []
    for entry in models:
        if isinstance(entry, Mapping):
            model_id = entry.get("id") or entry.get("name")
            if not model_id:
                raise ValueError(
                    f"a model entry must carry an 'id' (or 'name'); got {dict(entry)!r}"
                )
            ids.append(str(model_id))
            declaration = {k: v for k, v in entry.items() if k != "name"}
            declaration["id"] = str(model_id)
            roster.append(declaration)
        else:
            ids.append(str(entry))
    return ids, roster


def _parse_yaml(text: str) -> dict[str, Any]:
    """Parse YAML text into a dict, guarding the optional pyyaml import."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reading a YAML config file requires pyyaml. Install it with "
            "'pip install \"benchmaxxing[config]\"', or pass a plain dict to load_config."
        ) from exc
    data = yaml.safe_load(text)
    return data if data is not None else {}


def load_config(path: str | Path | Mapping[str, Any] | Config | None = None) -> Config:
    """Resolve a :class:`Config` from a YAML file, a plain dict, or the defaults.

    - ``path is None``: return the default config.
    - ``path`` is a Config: return it unchanged.
    - ``path`` is a mapping: build directly from it (no YAML parser needed).
    - ``path`` is a ``.json`` file: parsed with the standard library, so a run needs no extra.
    - ``path`` is any other file: parsed as YAML. This needs pyyaml (the ``config`` extra);
      a clear ImportError is raised if pyyaml is not importable.
    """
    if path is None:
        return Config()
    if isinstance(path, Config):
        return path
    if isinstance(path, Mapping):
        return Config.from_dict(path)
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if file.suffix.lower() == ".json":
        return Config.from_dict(json.loads(text))
    return Config.from_dict(_parse_yaml(text))
