"""Prompt-template registry with versioning.

Every agent prompt is a :class:`PromptTemplate` identified by ``(name, version)`` and held in a
:class:`PromptRegistry`. A run pins its prompts by recording each template's ``(name, version)``
in ``RunManifest.prompt_versions`` (see :mod:`benchmaxxing.schema`), e.g.
``{"committee_member": "v1"}``; replaying the run looks the exact text back up here. The
module-level :data:`DEFAULT_REGISTRY` ships the three core agent prompts (committee member,
orchestrator, referee); the gateway wires them to models later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import Formatter

# Canonical names of the core agent prompts in DEFAULT_REGISTRY.
COMMITTEE_MEMBER = "committee_member"
ORCHESTRATOR = "orchestrator"
REFEREE = "referee"


@dataclass(frozen=True)
class PromptTemplate:
    """One immutable, versioned prompt. ``template`` uses ``{placeholder}`` substitution."""

    name: str
    version: str
    template: str

    @property
    def placeholders(self) -> tuple[str, ...]:
        """The placeholder names in ``template``, in order of first appearance."""
        names: list[str] = []
        for _, field_name, _, _ in Formatter().parse(self.template):
            if not field_name:
                continue
            base = field_name.split(".")[0].split("[")[0]
            if base and base not in names:
                names.append(base)
        return tuple(names)

    def render(self, **kwargs) -> str:
        """Substitute ``{placeholders}``; raise KeyError naming any missing placeholder."""
        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "?"
            raise KeyError(
                f"Prompt {self.name!r} (version {self.version!r}) is missing a value for "
                f"placeholder {missing!r}; expected {list(self.placeholders)}, "
                f"got {sorted(kwargs)}."
            ) from None


def _version_key(version: str) -> tuple:
    """Natural-order sort key: digit runs compare numerically ("v2" < "v10", "1.9" < "1.10")."""
    key: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", version):
        if part.isdigit():
            key.append((1, int(part)))
        elif part:
            key.append((0, part))
    return tuple(key)


class PromptRegistry:
    """Holds PromptTemplates keyed by (name, version); "latest" is the highest natural version."""

    def __init__(self) -> None:
        self._by_name: dict[str, dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> PromptTemplate:
        """Add ``template``; raise ValueError if its (name, version) is already registered."""
        versions = self._by_name.setdefault(template.name, {})
        if template.version in versions:
            raise ValueError(
                f"Prompt {template.name!r} version {template.version!r} is already registered."
            )
        versions[template.version] = template
        return template

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Return the template for ``name``; the latest version when ``version`` is None."""
        versions = self._by_name.get(name)
        if not versions:
            raise KeyError(
                f"No prompt registered under name {name!r}. Known: {sorted(self._by_name)}."
            )
        if version is None:
            return versions[max(versions, key=_version_key)]
        if version not in versions:
            raise KeyError(
                f"Prompt {name!r} has no version {version!r}. Known: {self.versions(name)}."
            )
        return versions[version]

    def versions(self, name: str) -> list[str]:
        """All registered versions of ``name``, ascending by natural version order."""
        versions = self._by_name.get(name)
        if not versions:
            raise KeyError(
                f"No prompt registered under name {name!r}. Known: {sorted(self._by_name)}."
            )
        return sorted(versions, key=_version_key)

    def all_ids(self) -> list[tuple[str, str]]:
        """Every registered (name, version), names alphabetical, versions ascending."""
        return [
            (name, version)
            for name in sorted(self._by_name)
            for version in sorted(self._by_name[name], key=_version_key)
        ]


_COMMITTEE_MEMBER_V1 = """\
You are one clinician on a diagnostic committee reviewing a case independently.

Case:
{case}

Question: {question}
Options:
{options}

Reply with the index of the option you choose, your confidence in [0, 1], and a short
justification grounded only in the clinical evidence above.
"""

_ORCHESTRATOR_V1 = """\
You are the committee orchestrator. Synthesize the members' independent answers into one
final committee answer.

Question: {question}
Member answers:
{member_answers}

Weigh the evidence in each justification, note any disagreement, and reply with the final
option index and a brief rationale.
"""

_REFEREE_V1 = """\
You are a referee auditing a committee transcript for shortcut use.

Transcript:
{transcript}

Decide whether any answer relies on a spurious cue (e.g. formatting, option position, an
artifact) instead of clinical evidence. Reply with a verdict of "clean" or "shortcut", the
turn indices that support it, and a one-sentence explanation.
"""

# Default registry pre-loaded with the core agent prompts. Callers should stamp the
# (name, version) of every template they use into RunManifest.prompt_versions so the run
# is replayable against the exact prompt text.
DEFAULT_REGISTRY = PromptRegistry()
DEFAULT_REGISTRY.register(PromptTemplate(COMMITTEE_MEMBER, "v1", _COMMITTEE_MEMBER_V1))
DEFAULT_REGISTRY.register(PromptTemplate(ORCHESTRATOR, "v1", _ORCHESTRATOR_V1))
DEFAULT_REGISTRY.register(PromptTemplate(REFEREE, "v1", _REFEREE_V1))
