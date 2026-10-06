"""Unit tests for conform_bench.validators: given a hand-built RawResponse,
does the verdict match what a human would call correct?
"""

from __future__ import annotations

import json

import pytest

from conform_bench.models import (
    FieldCheck,
    Outcome,
    RawResponse,
    StructuredTestCase,
    ToolSpec,
    ToolTestCase,
)
from conform_bench.validators import strip_markdown_fence, validate_structured_response, validate_tool_response

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def _structured_test(**overrides) -> StructuredTestCase:
    defaults = dict(
        id="t1",
        description="d",
        schema=SCHEMA,
        prompt="p",
        checks=[FieldCheck(path="name", equals="Priya")],
    )
    defaults.update(overrides)
    return StructuredTestCase(**defaults)


def _raw(content: str | None = None, **overrides) -> RawResponse:
    defaults = dict(ok=True, status_code=200, elapsed_ms=1.0, content=content)
    defaults.update(overrides)
    return RawResponse(**defaults)


class TestStructuredResponse:
    def test_pass_on_exact_conforming_json(self):
        raw = _raw(json.dumps({"name": "Priya", "age": 29}))
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.PASS

    def test_pass_but_notes_markdown_fence(self):
        raw = _raw("```json\n" + json.dumps({"name": "Priya", "age": 29}) + "\n```")
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.PASS
        assert "fence" in verdict.reason

    def test_fail_on_missing_required_field(self):
        raw = _raw(json.dumps({"name": "Priya"}))
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.FAIL

    def test_fail_on_extra_property_when_additional_properties_false(self):
        raw = _raw(json.dumps({"name": "Priya", "age": 29, "extra": "nope"}))
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.FAIL

    def test_fail_on_wrong_type(self):
        raw = _raw(json.dumps({"name": "Priya", "age": "twenty-nine"}))
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.FAIL

    def test_fail_on_malformed_json(self):
        raw = _raw('{"name": "Priya", "age": 29')  # truncated
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.FAIL
        assert "not valid JSON" in verdict.reason

    def test_fail_on_prose_instead_of_json(self):
        raw = _raw("Sure! Her name is Priya and she is 29 years old.")
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.FAIL

    def test_fail_on_field_check_mismatch_even_if_schema_valid(self):
        raw = _raw(json.dumps({"name": "SomeoneElse", "age": 29}))
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.FAIL
        assert "name" in verdict.reason

    def test_unsupported_on_http_400(self):
        raw = _raw(None, ok=False, status_code=400, http_error_body="bad request")
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.UNSUPPORTED

    def test_unsupported_on_http_422(self):
        raw = _raw(None, ok=False, status_code=422, http_error_body="unprocessable")
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.UNSUPPORTED

    def test_error_on_http_500(self):
        raw = _raw(None, ok=False, status_code=500, http_error_body="boom")
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.ERROR

    def test_error_on_network_failure(self):
        raw = RawResponse(ok=False, status_code=None, elapsed_ms=1.0, error="timeout")
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.ERROR

    def test_fail_on_none_content(self):
        raw = _raw(None)
        verdict = validate_structured_response(raw, _structured_test())
        assert verdict.outcome == Outcome.FAIL


class TestStripMarkdownFence:
    def test_strips_json_fence(self):
        assert strip_markdown_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_bare_fence(self):
        assert strip_markdown_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_leaves_unfenced_text_alone(self):
        assert strip_markdown_fence('{"a": 1}') == '{"a": 1}'


def _tool_test(**overrides) -> ToolTestCase:
    defaults = dict(
        id="t1",
        description="d",
        tools=[
            ToolSpec(
                name="get_weather",
                description="get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
        ],
        messages=[{"role": "user", "content": "weather in Rome?"}],
        expect_call=True,
        expect_tool_name="get_weather",
        checks=[FieldCheck(path="city", equals="Rome")],
    )
    defaults.update(overrides)
    return ToolTestCase(**defaults)


class TestToolResponse:
    def test_pass_on_correct_call(self):
        raw = RawResponse(
            ok=True,
            status_code=200,
            elapsed_ms=1.0,
            tool_calls=[{"name": "get_weather", "arguments": json.dumps({"city": "Rome"})}],
        )
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.PASS

    def test_pass_when_arguments_already_a_dict(self):
        """Ollama's native API hands back decoded dicts, not JSON strings."""
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Rome"}}],
        )
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.PASS

    def test_fail_on_no_call_when_expected(self):
        raw = RawResponse(ok=True, status_code=200, elapsed_ms=1.0, content="It's sunny in Rome.")
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.FAIL
        assert "expected a tool call" in verdict.reason

    def test_pass_on_no_call_when_not_expected(self):
        raw = RawResponse(ok=True, status_code=200, elapsed_ms=1.0, content="four")
        verdict = validate_tool_response(raw, _tool_test(expect_call=False, expect_tool_name=None, checks=[]))
        assert verdict.outcome == Outcome.PASS

    def test_fail_on_call_when_not_expected(self):
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[{"name": "get_weather", "arguments": "{}"}],
        )
        verdict = validate_tool_response(raw, _tool_test(expect_call=False, expect_tool_name=None, checks=[]))
        assert verdict.outcome == Outcome.FAIL

    def test_fail_on_wrong_tool_called(self):
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[{"name": "unexpected_tool", "arguments": "{}"}],
        )
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.FAIL
        assert "unknown tool name" in verdict.reason

    def test_fail_on_malformed_argument_json(self):
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[{"name": "get_weather", "arguments": '{"city": "Rome"'}],
        )
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.FAIL
        assert "valid JSON" in verdict.reason

    def test_fail_on_arguments_violating_parameter_schema(self):
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[{"name": "get_weather", "arguments": json.dumps({"city": 123})}],
        )
        verdict = validate_tool_response(raw, _tool_test(checks=[]))
        assert verdict.outcome == Outcome.FAIL
        assert "do not conform" in verdict.reason

    def test_fail_on_argument_value_mismatch(self):
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[{"name": "get_weather", "arguments": json.dumps({"city": "Berlin"})}],
        )
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.FAIL

    def test_fail_on_too_many_calls(self):
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[
                {"name": "get_weather", "arguments": json.dumps({"city": "Rome"})},
                {"name": "get_weather", "arguments": json.dumps({"city": "Berlin"})},
            ],
        )
        verdict = validate_tool_response(raw, _tool_test(max_calls=1))
        assert verdict.outcome == Outcome.FAIL
        assert "<=" in verdict.reason

    def test_pass_on_expected_multiple_calls(self):
        raw = RawResponse(
            ok=True, status_code=200, elapsed_ms=1.0,
            tool_calls=[
                {"name": "get_weather", "arguments": json.dumps({"city": "Rome"})},
                {"name": "get_weather", "arguments": json.dumps({"city": "Berlin"})},
            ],
        )
        test = _tool_test(min_calls=2, max_calls=2, checks=[], expect_tool_name=None, expect_any_of=["get_weather"])
        verdict = validate_tool_response(raw, test)
        assert verdict.outcome == Outcome.PASS

    def test_unsupported_on_http_400(self):
        raw = RawResponse(ok=False, status_code=400, elapsed_ms=1.0, http_error_body="tools not supported")
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.UNSUPPORTED

    def test_error_on_http_500(self):
        raw = RawResponse(ok=False, status_code=500, elapsed_ms=1.0, http_error_body="boom")
        verdict = validate_tool_response(raw, _tool_test())
        assert verdict.outcome == Outcome.ERROR
