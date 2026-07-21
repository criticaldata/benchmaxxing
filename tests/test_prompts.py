"""Tests for benchmaxxing.prompts (template registry + versioning)."""

from __future__ import annotations

import pytest

from benchmaxxing.prompts import (
    COMMITTEE_MEMBER,
    DEFAULT_REGISTRY,
    ORCHESTRATOR,
    REFEREE,
    PromptRegistry,
    PromptTemplate,
)


def _registry_with(*templates: PromptTemplate) -> PromptRegistry:
    registry = PromptRegistry()
    for template in templates:
        registry.register(template)
    return registry


def test_render_substitutes_placeholders():
    template = PromptTemplate("greet", "v1", "Hello {who}, case {case_id}.")
    assert template.render(who="doctor", case_id="c1") == "Hello doctor, case c1."


def test_render_missing_placeholder_raises_clear_keyerror():
    template = PromptTemplate("greet", "v1", "Hello {who}, case {case_id}.")
    with pytest.raises(KeyError) as excinfo:
        template.render(who="doctor")
    message = str(excinfo.value)
    assert "case_id" in message
    assert "greet" in message
    assert "v1" in message


def test_placeholders_property_lists_fields_in_order():
    template = PromptTemplate("p", "v1", "{b} then {a} then {b} again")
    assert template.placeholders == ("b", "a")


def test_register_and_get_specific_version():
    t1 = PromptTemplate("p", "v1", "one")
    t2 = PromptTemplate("p", "v2", "two")
    registry = _registry_with(t1, t2)
    assert registry.get("p", "v1") is t1
    assert registry.get("p", "v2") is t2


def test_get_latest_uses_natural_order_v_prefix():
    registry = _registry_with(
        PromptTemplate("p", "v2", "two"),
        PromptTemplate("p", "v10", "ten"),
        PromptTemplate("p", "v1", "one"),
    )
    assert registry.get("p").version == "v10"


def test_get_latest_uses_natural_order_dotted():
    registry = _registry_with(
        PromptTemplate("p", "1.9", "a"),
        PromptTemplate("p", "1.10", "b"),
        PromptTemplate("p", "1.0", "c"),
    )
    assert registry.get("p").version == "1.10"


def test_versions_sorted_ascending():
    registry = _registry_with(
        PromptTemplate("p", "v10", "ten"),
        PromptTemplate("p", "v1", "one"),
        PromptTemplate("p", "v2", "two"),
    )
    assert registry.versions("p") == ["v1", "v2", "v10"]


def test_all_ids_lists_every_name_version():
    registry = _registry_with(
        PromptTemplate("b", "v1", "x"),
        PromptTemplate("a", "v2", "y"),
        PromptTemplate("a", "v1", "z"),
    )
    assert registry.all_ids() == [("a", "v1"), ("a", "v2"), ("b", "v1")]


def test_duplicate_registration_raises_value_error():
    registry = _registry_with(PromptTemplate("p", "v1", "one"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(PromptTemplate("p", "v1", "other text"))


def test_get_unknown_name_raises_key_error():
    registry = PromptRegistry()
    with pytest.raises(KeyError):
        registry.get("nope")
    with pytest.raises(KeyError):
        registry.versions("nope")


def test_get_unknown_version_raises_key_error():
    registry = _registry_with(PromptTemplate("p", "v1", "one"))
    with pytest.raises(KeyError):
        registry.get("p", "v99")


def test_default_registry_exposes_core_prompts():
    for name in (COMMITTEE_MEMBER, ORCHESTRATOR, REFEREE):
        template = DEFAULT_REGISTRY.get(name)
        assert isinstance(template, PromptTemplate)
        assert template.name == name
        assert template.version
        assert template.template.strip()
        assert (name, template.version) in DEFAULT_REGISTRY.all_ids()


def test_default_prompts_render_with_their_placeholders():
    for name in (COMMITTEE_MEMBER, ORCHESTRATOR, REFEREE):
        template = DEFAULT_REGISTRY.get(name)
        assert template.placeholders, f"{name} should have at least one placeholder"
        rendered = template.render(**{p: f"<{p}>" for p in template.placeholders})
        for placeholder in template.placeholders:
            assert f"<{placeholder}>" in rendered
