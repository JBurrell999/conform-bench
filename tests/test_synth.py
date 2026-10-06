"""Tests for the schema synthesizer, including against every real schema
under testcases/ -- this doubles as a regression test that catches new test
cases with unsatisfiable schemas or checks pointing at nonexistent paths."""

from __future__ import annotations

import jsonschema
import pytest

from llm_conform.synth import hints_from_checks, synthesize_value
from llm_conform.testcase_loader import load_structured_tests, load_tool_tests
from llm_conform.validators import run_field_checks


@pytest.mark.parametrize("test", load_structured_tests(), ids=lambda t: t.id)
def test_synthesized_value_is_schema_valid_and_matches_checks(test):
    hints = hints_from_checks(test.checks)
    value = synthesize_value(test.schema, hints)
    jsonschema.validate(instance=value, schema=test.schema)
    assert run_field_checks(value, test.checks) == []


@pytest.mark.parametrize("test", load_tool_tests(), ids=lambda t: t.id)
def test_synthesized_tool_arguments_are_schema_valid(test):
    target_name = test.expect_tool_name or (test.expect_any_of[0] if test.expect_any_of else None)
    for tool in test.tools:
        hints = hints_from_checks(test.checks) if tool.name == target_name else {}
        args = synthesize_value(tool.parameters, hints)
        jsonschema.validate(instance=args, schema=tool.parameters)
        if tool.name == target_name:
            assert run_field_checks(args, test.checks) == []


class TestSynthesizeValuePrimitives:
    def test_object_includes_only_required_properties_by_default(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }
        value = synthesize_value(schema)
        assert value == {"a": "example"}

    def test_enum_picks_first_value(self):
        assert synthesize_value({"type": "string", "enum": ["x", "y"]}) == "x"

    def test_const_is_honored(self):
        assert synthesize_value({"const": 42}) == 42

    def test_hint_overrides_enum(self):
        schema = {"type": "string", "enum": ["x", "y"]}
        assert synthesize_value(schema, hints={"": "y"}) == "y"

    def test_array_items_get_distinct_indexed_paths(self):
        schema = {
            "type": "array",
            "minItems": 2,
            "items": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        }
        value = synthesize_value(schema, hints={"0.n": 1, "1.n": 2})
        assert value == [{"n": 1}, {"n": 2}]

    def test_local_ref_is_resolved(self):
        schema = {
            "$defs": {"leaf": {"type": "string"}},
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/leaf"}},
            "required": ["x"],
        }
        assert synthesize_value(schema) == {"x": "example"}

    def test_recursive_ref_terminates_via_hint(self):
        schema = {
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                    },
                    "required": ["name", "children"],
                }
            },
            "$ref": "#/$defs/node",
        }
        value = synthesize_value(schema, hints={"children.0.children": []})
        assert value == {"name": "example", "children": [{"name": "example", "children": []}]}

    def test_nullable_type_list_with_hint(self):
        schema = {"type": ["string", "null"]}
        assert synthesize_value(schema, hints={"": None}) is None
