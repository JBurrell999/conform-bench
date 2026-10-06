"""Tests for the per-engine adapters: do they build the request payload
each wire format actually expects, and parse the response back correctly?
Uses httpx.MockTransport so no real server or network is involved.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_conform.engines.llamacpp import LlamaCppAdapter
from llm_conform.engines.ollama import OllamaAdapter
from llm_conform.engines.openai_compat import OpenAICompatAdapter
from llm_conform.engines.sglang import SGLangAdapter
from llm_conform.engines.vllm import VLLMAdapter
from llm_conform.models import EngineConfig, FieldCheck, Outcome, StructuredTestCase, ToolSpec, ToolTestCase

STRUCTURED_TEST = StructuredTestCase(
    id="my_test",
    description="d",
    schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    prompt="say x",
    system_prompt="be terse",
)

TOOL_TEST = ToolTestCase(
    id="tool_test",
    description="d",
    tools=[
        ToolSpec(
            name="get_weather",
            description="get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        )
    ],
    messages=[{"role": "user", "content": "weather in Rome?"}],
)


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestOpenAICompatStructured:
    def test_sends_json_schema_response_format_with_system_and_user_messages(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": '{"x": "y"}'}}]},
            )

        config = EngineConfig(name="e", type="openai_compat", base_url="http://mock", model="m")
        adapter = OpenAICompatAdapter(config, client=_client_with(handler))
        raw = adapter.complete_structured(STRUCTURED_TEST)

        body = captured["body"]
        assert body["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "say x"},
        ]
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "my_test"
        assert body["response_format"]["json_schema"]["schema"] == STRUCTURED_TEST.schema
        assert body["response_format"]["json_schema"]["strict"] is True
        assert raw.ok and raw.content == '{"x": "y"}'

    def test_posts_to_v1_chat_completions(self):
        urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            urls.append(str(request.url))
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        config = EngineConfig(name="e", type="openai_compat", base_url="http://mock:1234/", model="m")
        adapter = OpenAICompatAdapter(config, client=_client_with(handler))
        adapter.complete_structured(STRUCTURED_TEST)
        assert urls == ["http://mock:1234/v1/chat/completions"]

    def test_http_error_is_surfaced_without_raising(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="no structured output support")

        config = EngineConfig(name="e", type="openai_compat", base_url="http://mock", model="m")
        adapter = OpenAICompatAdapter(config, client=_client_with(handler))
        raw = adapter.complete_structured(STRUCTURED_TEST)
        assert not raw.ok
        assert raw.status_code == 400
        assert "no structured output support" in raw.http_error_body

    def test_malformed_json_body_is_surfaced_as_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        config = EngineConfig(name="e", type="openai_compat", base_url="http://mock", model="m")
        adapter = OpenAICompatAdapter(config, client=_client_with(handler))
        raw = adapter.complete_structured(STRUCTURED_TEST)
        assert not raw.ok
        assert raw.error

    def test_unexpected_response_shape_is_surfaced_as_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        config = EngineConfig(name="e", type="openai_compat", base_url="http://mock", model="m")
        adapter = OpenAICompatAdapter(config, client=_client_with(handler))
        raw = adapter.complete_structured(STRUCTURED_TEST)
        assert not raw.ok


class TestOpenAICompatTools:
    def test_sends_tools_and_parses_tool_calls_back(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        config = EngineConfig(name="e", type="openai_compat", base_url="http://mock", model="m")
        adapter = OpenAICompatAdapter(config, client=_client_with(handler))
        raw = adapter.complete_tools(TOOL_TEST)

        body = captured["body"]
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert body["tool_choice"] == "auto"
        assert raw.tool_calls == [{"name": "get_weather", "arguments": '{"city": "Rome"}'}]

    def test_forced_tool_choice_dict_is_normalized_to_openai_shape(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": []}}]})

        test = ToolTestCase(**{**TOOL_TEST.__dict__, "tool_choice": {"name": "get_weather"}})
        config = EngineConfig(name="e", type="openai_compat", base_url="http://mock", model="m")
        adapter = OpenAICompatAdapter(config, client=_client_with(handler))
        adapter.complete_tools(test)
        assert captured["body"]["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


class TestOllamaAdapter:
    def test_uses_native_api_chat_with_format_field_and_no_tool_choice(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": '{"x": "y"}', "tool_calls": None}}
            )

        config = EngineConfig(name="e", type="ollama", base_url="http://mock:11434", model="m")
        adapter = OllamaAdapter(config, client=_client_with(handler))
        raw = adapter.complete_structured(STRUCTURED_TEST)

        assert captured["url"] == "http://mock:11434/api/chat"
        assert captured["body"]["format"] == STRUCTURED_TEST.schema
        assert "response_format" not in captured["body"]
        assert raw.content == '{"x": "y"}'

    def test_tools_payload_has_no_tool_choice_key_at_all(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"message": {"content": None, "tool_calls": []}})

        config = EngineConfig(name="e", type="ollama", base_url="http://mock", model="m")
        adapter = OllamaAdapter(config, client=_client_with(handler))
        adapter.complete_tools(TOOL_TEST)
        assert "tool_choice" not in captured["body"]

    def test_accepts_dict_shaped_tool_call_arguments(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}],
                    }
                },
            )

        config = EngineConfig(name="e", type="ollama", base_url="http://mock", model="m")
        adapter = OllamaAdapter(config, client=_client_with(handler))
        raw = adapter.complete_tools(TOOL_TEST)
        assert raw.tool_calls == [{"name": "get_weather", "arguments": {"city": "Rome"}}]

    def test_get_version_hits_api_version(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://mock/api/version"
            return httpx.Response(200, json={"version": "0.35.1"})

        config = EngineConfig(name="e", type="ollama", base_url="http://mock", model="m")
        adapter = OllamaAdapter(config, client=_client_with(handler))
        assert adapter.get_version() == "0.35.1"

    def test_get_version_returns_none_on_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        config = EngineConfig(name="e", type="ollama", base_url="http://mock", model="m")
        adapter = OllamaAdapter(config, client=_client_with(handler))
        assert adapter.get_version() is None


@pytest.mark.parametrize("adapter_cls", [VLLMAdapter, SGLangAdapter, LlamaCppAdapter])
class TestOpenAICompatSubclassesShareBaseBehavior(object):
    def test_still_posts_to_v1_chat_completions(self, adapter_cls):
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://mock/v1/chat/completions"
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        config = EngineConfig(name="e", type=adapter_cls.engine_type, base_url="http://mock", model="m")
        adapter = adapter_cls(config, client=_client_with(handler))
        raw = adapter.complete_structured(STRUCTURED_TEST)
        assert raw.ok


def test_vllm_get_version():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://mock/version"
        return httpx.Response(200, json={"version": "0.8.0"})

    config = EngineConfig(name="e", type="vllm", base_url="http://mock", model="m")
    adapter = VLLMAdapter(config, client=_client_with(handler))
    assert adapter.get_version() == "0.8.0"


def test_sglang_get_version():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://mock/get_server_info"
        return httpx.Response(200, json={"version": "0.4.1"})

    config = EngineConfig(name="e", type="sglang", base_url="http://mock", model="m")
    adapter = SGLangAdapter(config, client=_client_with(handler))
    assert adapter.get_version() == "0.4.1"


def test_llamacpp_get_version():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://mock/props"
        return httpx.Response(200, json={"build_info": "b1234"})

    config = EngineConfig(name="e", type="llamacpp", base_url="http://mock", model="m")
    adapter = LlamaCppAdapter(config, client=_client_with(handler))
    assert adapter.get_version() == "b1234"
