"""The most important regression test in this repo: for every real test
case under testcases/, synthesize the response a perfectly-conformant
engine *would* give (using conform_bench.synth, which reads the test's own
schema/tools/checks -- not an LLM), feed it through the real adapter and
validator code paths via httpx.MockTransport, and assert the result is
PASS.

If this test ever fails, it means a test case itself is broken (an
unsatisfiable schema, a `checks` path that can't exist, a `tool_name` typo)
-- not that some real engine misbehaved. Anyone adding a new YAML test case
should run this before anything else.
"""

from __future__ import annotations

import json

import httpx
import pytest

from conform_bench.engines.ollama import OllamaAdapter
from conform_bench.engines.openai_compat import OpenAICompatAdapter
from conform_bench.models import EngineConfig, Outcome
from conform_bench.synth import hints_from_checks, synthesize_value
from conform_bench.testcase_loader import load_structured_tests, load_tool_tests
from conform_bench.validators import validate_structured_response, validate_tool_response

STRUCTURED_TESTS = load_structured_tests()
TOOL_TESTS = load_tool_tests()
STRUCTURED_BY_ID = {t.id: t for t in STRUCTURED_TESTS}
STRUCTURED_BY_SCHEMA_JSON = {json.dumps(t.schema, sort_keys=True): t for t in STRUCTURED_TESTS}
# Keyed by the literal `messages` payload, not the set of tool names: several
# test cases deliberately offer the *same* single tool (no_tool_needed,
# single_tool_basic, parallel_tool_calls all offer just {get_weather}) to
# test different behaviors around it, so tool names alone aren't a unique key.
TOOL_BY_MESSAGES = {json.dumps(t.messages, sort_keys=True): t for t in TOOL_TESTS}
assert len(TOOL_BY_MESSAGES) == len(TOOL_TESTS), "two tool test cases have identical `messages` -- give them distinct prompts"


def _ideal_tool_calls(test) -> list[dict] | None:
    """Builds the exact set of tool calls that should earn a PASS for this
    test, per its own `expect_*` fields -- not by reading the prompt."""
    if not test.expect_call:
        return None
    name = test.expect_tool_name or (test.expect_any_of[0] if test.expect_any_of else test.tools[0].name)
    tool = next(t for t in test.tools if t.name == name)
    hints = hints_from_checks(test.checks)
    args = synthesize_value(tool.parameters, hints)
    count = test.max_calls if test.max_calls is not None else test.min_calls
    return [{"name": name, "arguments": args} for _ in range(max(count, test.min_calls))]


def _openai_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if "response_format" in body:
        js = body["response_format"]["json_schema"]
        test = STRUCTURED_BY_ID[js["name"]]
        value = synthesize_value(js["schema"], hints_from_checks(test.checks))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps(value)}}]})

    test = TOOL_BY_MESSAGES[json.dumps(body["messages"], sort_keys=True)]
    calls = _ideal_tool_calls(test)
    if calls is None:
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Here you go.", "tool_calls": None}}]})
    raw_calls = [
        {"id": f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
        for i, c in enumerate(calls)
    ]
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": raw_calls}}]})


def _ollama_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if "format" in body:
        test = STRUCTURED_BY_SCHEMA_JSON[json.dumps(body["format"], sort_keys=True)]
        value = synthesize_value(body["format"], hints_from_checks(test.checks))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(value), "tool_calls": None}})

    test = TOOL_BY_MESSAGES[json.dumps(body["messages"], sort_keys=True)]
    calls = _ideal_tool_calls(test)
    if calls is None:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Here you go.", "tool_calls": None}})
    raw_calls = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
    return httpx.Response(200, json={"message": {"role": "assistant", "content": None, "tool_calls": raw_calls}})


@pytest.mark.parametrize("test", STRUCTURED_TESTS, ids=lambda t: t.id)
def test_ideal_response_passes_structured_via_openai_compat(test):
    config = EngineConfig(name="mock", type="openai_compat", base_url="http://mock", model="m")
    client = httpx.Client(transport=httpx.MockTransport(_openai_handler))
    with OpenAICompatAdapter(config, client=client) as adapter:
        raw = adapter.complete_structured(test)
    verdict = validate_structured_response(raw, test)
    assert verdict.outcome == Outcome.PASS, verdict.reason


@pytest.mark.parametrize("test", TOOL_TESTS, ids=lambda t: t.id)
def test_ideal_response_passes_tools_via_openai_compat(test):
    config = EngineConfig(name="mock", type="openai_compat", base_url="http://mock", model="m")
    client = httpx.Client(transport=httpx.MockTransport(_openai_handler))
    with OpenAICompatAdapter(config, client=client) as adapter:
        raw = adapter.complete_tools(test)
    verdict = validate_tool_response(raw, test)
    assert verdict.outcome == Outcome.PASS, verdict.reason


@pytest.mark.parametrize("test", STRUCTURED_TESTS, ids=lambda t: t.id)
def test_ideal_response_passes_structured_via_ollama(test):
    config = EngineConfig(name="mock", type="ollama", base_url="http://mock", model="m")
    client = httpx.Client(transport=httpx.MockTransport(_ollama_handler))
    with OllamaAdapter(config, client=client) as adapter:
        raw = adapter.complete_structured(test)
    verdict = validate_structured_response(raw, test)
    assert verdict.outcome == Outcome.PASS, verdict.reason


@pytest.mark.parametrize("test", TOOL_TESTS, ids=lambda t: t.id)
def test_ideal_response_passes_tools_via_ollama(test):
    config = EngineConfig(name="mock", type="ollama", base_url="http://mock", model="m")
    client = httpx.Client(transport=httpx.MockTransport(_ollama_handler))
    with OllamaAdapter(config, client=client) as adapter:
        raw = adapter.complete_tools(test)
    verdict = validate_tool_response(raw, test)
    assert verdict.outcome == Outcome.PASS, verdict.reason
