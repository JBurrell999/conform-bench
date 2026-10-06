"""Tests for llm_conform.mock_server: run it for real (bound to an ephemeral
TCP port) and drive it through the OpenAI-compat and Ollama adapters,
checking that each `--behavior` preset produces the Outcome it's supposed to
simulate. These are deliberately simple fixture test cases (no `checks`)
since the mock server only knows how to be "generically schema-valid," not
"factually correct" -- see llm_conform/synth.py's docstring.
"""

from __future__ import annotations

from llm_conform.engines.ollama import OllamaAdapter
from llm_conform.engines.openai_compat import OpenAICompatAdapter
from llm_conform.mock_server import MockEngineServer
from llm_conform.models import EngineConfig, Outcome, StructuredTestCase, ToolSpec, ToolTestCase
from llm_conform.validators import validate_structured_response, validate_tool_response

SIMPLE_STRUCTURED = StructuredTestCase(
    id="simple",
    description="d",
    schema={
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    },
    prompt="describe someone",
)

SIMPLE_TOOL = ToolTestCase(
    id="simple_tool",
    description="d",
    tools=[
        ToolSpec(
            name="get_weather",
            description="get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        ),
        ToolSpec(
            name="get_time",
            description="get time",
            parameters={"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]},
        ),
    ],
    messages=[{"role": "user", "content": "weather in Rome?"}],
    expect_tool_name="get_weather",
)


def _config(base_url: str, engine_type: str = "openai_compat") -> EngineConfig:
    return EngineConfig(name="mock", type=engine_type, base_url=base_url, model="mock-model")


class TestOpenAICompatBehaviors:
    def test_conformant_passes_structured(self):
        with MockEngineServer(behavior="conformant") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_structured(SIMPLE_STRUCTURED)
        verdict = validate_structured_response(raw, SIMPLE_STRUCTURED)
        assert verdict.outcome == Outcome.PASS

    def test_markdown_fence_still_passes_but_is_noted(self):
        with MockEngineServer(behavior="markdown_fence") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_structured(SIMPLE_STRUCTURED)
        verdict = validate_structured_response(raw, SIMPLE_STRUCTURED)
        assert verdict.outcome == Outcome.PASS
        assert "fence" in verdict.reason

    def test_schema_violator_fails(self):
        with MockEngineServer(behavior="schema_violator") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_structured(SIMPLE_STRUCTURED)
        verdict = validate_structured_response(raw, SIMPLE_STRUCTURED)
        assert verdict.outcome == Outcome.FAIL

    def test_prose_instead_of_json_fails(self):
        with MockEngineServer(behavior="prose_instead_of_json") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_structured(SIMPLE_STRUCTURED)
        verdict = validate_structured_response(raw, SIMPLE_STRUCTURED)
        assert verdict.outcome == Outcome.FAIL

    def test_http_400_is_unsupported(self):
        with MockEngineServer(behavior="http_400") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_structured(SIMPLE_STRUCTURED)
        verdict = validate_structured_response(raw, SIMPLE_STRUCTURED)
        assert verdict.outcome == Outcome.UNSUPPORTED

    def test_conformant_passes_tool_call(self):
        with MockEngineServer(behavior="conformant") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_tools(SIMPLE_TOOL)
        verdict = validate_tool_response(raw, SIMPLE_TOOL)
        assert verdict.outcome == Outcome.PASS

    def test_malformed_tool_args_fails(self):
        with MockEngineServer(behavior="malformed_tool_args") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_tools(SIMPLE_TOOL)
        verdict = validate_tool_response(raw, SIMPLE_TOOL)
        assert verdict.outcome == Outcome.FAIL

    def test_no_tool_call_fails_when_a_call_was_expected(self):
        with MockEngineServer(behavior="no_tool_call") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_tools(SIMPLE_TOOL)
        verdict = validate_tool_response(raw, SIMPLE_TOOL)
        assert verdict.outcome == Outcome.FAIL

    def test_wrong_tool_fails(self):
        with MockEngineServer(behavior="wrong_tool") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_tools(SIMPLE_TOOL)
        verdict = validate_tool_response(raw, SIMPLE_TOOL)
        assert verdict.outcome == Outcome.FAIL

    def test_http_400_tools_is_unsupported(self):
        with MockEngineServer(behavior="http_400") as server:
            with OpenAICompatAdapter(_config(server.base_url)) as adapter:
                raw = adapter.complete_tools(SIMPLE_TOOL)
        verdict = validate_tool_response(raw, SIMPLE_TOOL)
        assert verdict.outcome == Outcome.UNSUPPORTED


class TestOllamaNativeBehaviors:
    def test_conformant_passes_structured(self):
        with MockEngineServer(behavior="conformant") as server:
            with OllamaAdapter(_config(server.base_url, "ollama")) as adapter:
                raw = adapter.complete_structured(SIMPLE_STRUCTURED)
        verdict = validate_structured_response(raw, SIMPLE_STRUCTURED)
        assert verdict.outcome == Outcome.PASS

    def test_conformant_passes_tool_call_with_dict_arguments(self):
        with MockEngineServer(behavior="conformant") as server:
            with OllamaAdapter(_config(server.base_url, "ollama")) as adapter:
                raw = adapter.complete_tools(SIMPLE_TOOL)
        assert isinstance(raw.tool_calls[0]["arguments"], dict)
        verdict = validate_tool_response(raw, SIMPLE_TOOL)
        assert verdict.outcome == Outcome.PASS

    def test_schema_violator_fails(self):
        with MockEngineServer(behavior="schema_violator") as server:
            with OllamaAdapter(_config(server.base_url, "ollama")) as adapter:
                raw = adapter.complete_structured(SIMPLE_STRUCTURED)
        verdict = validate_structured_response(raw, SIMPLE_STRUCTURED)
        assert verdict.outcome == Outcome.FAIL


def test_version_endpoints_respond():
    import httpx

    with MockEngineServer(behavior="conformant") as server:
        assert httpx.get(f"{server.base_url}/version").json()["version"] == "mock-1.0.0"
        assert httpx.get(f"{server.base_url}/api/version").json()["version"] == "mock-1.0.0"
        assert httpx.get(f"{server.base_url}/props").json()["build_info"] == "mock-build"
        assert httpx.get(f"{server.base_url}/get_server_info").json()["version"] == "mock-1.0.0"


def test_unknown_path_is_404():
    import httpx

    with MockEngineServer(behavior="conformant") as server:
        resp = httpx.get(f"{server.base_url}/nope")
        assert resp.status_code == 404
