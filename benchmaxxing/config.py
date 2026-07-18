"""Run configuration for benchmaxxing.

A tiny, dependency-light config layer. A YAML file is parsed when pyyaml is importable
(the optional ``config`` extra); otherwise a plain dict is accepted directly so the core
stays usable with no extra installed. Everything resolves to a :class:`Config` dataclass
with ``from_dict`` / ``to_dict`` for round-tripping.
"""

from __future__ import annotations

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
        dataset: dataset adapter name (see ``benchmaxxing.datasets``).
        cue_set: cue-set version to inject.
        seed: master random seed for reproducibility.
        out_dir: directory where run artifacts are written.
    """

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    dataset: str = "worldmedqa_v2"
    cue_set: str = "v1"
    seed: int = 0
    out_dir: str = "runs"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Config":
        """Build a Config from a mapping, ignoring unknown keys.

        A ``models`` string is promoted to a one-element list and ``seed`` is coerced to
        int, so hand-written or YAML-parsed configs stay forgiving.
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
            kwargs["models"] = list(models)
        if kwargs.get("seed") is not None:
            kwargs["seed"] = int(kwargs["seed"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dict of the config fields."""
        return asdict(self)


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
    - ``path`` is a file path: parse it as YAML. This needs pyyaml (the ``config`` extra);
      a clear ImportError is raised if pyyaml is not importable.
    """
    if path is None:
        return Config()
    if isinstance(path, Config):
        return path
    if isinstance(path, Mapping):
        return Config.from_dict(path)
    text = Path(path).read_text(encoding="utf-8")
    return Config.from_dict(_parse_yaml(text))
