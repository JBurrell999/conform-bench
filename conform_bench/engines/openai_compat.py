"""Adapter for the OpenAI `/v1/chat/completions` contract, which vLLM,
SGLang, and llama.cpp's `llama-server` all implement (with varying degrees
of fidelity -- that variance is exactly what this project measures).

Subclasses override `_structured_payload_extra` / `_tools_payload_extra` for
engine-specific quirks instead of duplicating the whole request path.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from conform_bench.engines.base import EngineAdapter
from conform_bench.models import RawResponse, StructuredTestCase, ToolTestCase


class OpenAICompatAdapter(EngineAdapter):
    engine_type = "openai_compat"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> RawResponse:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        start = time.perf_counter()
        try:
            resp = self._client.post(url, json=payload, headers=self._headers())
        except httpx.TimeoutException:
            return RawResponse(
                ok=False, status_code=None, elapsed_ms=self._elapsed_ms(start), error="timeout"
            )
        except httpx.RequestError as exc:
            return RawResponse(
                ok=False, status_code=None, elapsed_ms=self._elapsed_ms(start), error=str(exc)
            )

        elapsed = self._elapsed_ms(start)
        if resp.status_code >= 400:
            return RawResponse(
                ok=False,
                status_code=resp.status_code,
                elapsed_ms=elapsed,
                http_error_body=resp.text[:2000],
            )

        try:
            body = resp.json()
        except ValueError:
            return RawResponse(
                ok=False,
                status_code=resp.status_code,
                elapsed_ms=elapsed,
                error="response was not valid JSON",
                http_error_body=resp.text[:2000],
            )
        return RawResponse(ok=True, status_code=resp.status_code, elapsed_ms=elapsed, raw=body)

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    # -- structured output -------------------------------------------------

    def _structured_payload_extra(self, test: StructuredTestCase) -> dict[str, Any]:
        """Hook for subclasses to adjust the structured-output payload."""
        return {}

    def complete_structured(self, test: StructuredTestCase) -> RawResponse:
        messages = []
        if test.system_prompt:
            messages.append({"role": "system", "content": test.system_prompt})
        messages.append({"role": "user", "content": test.prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": test.id,
                    "schema": test.schema,
                    "strict": True,
                },
            },
            "temperature": 0,
        }
        payload.update(self._structured_payload_extra(test))

        response = self._post("/v1/chat/completions", payload)
        if not response.ok:
            return response
        return self._extract_chat_message(response)

    # -- tool calling --------------------------------------------------

    def _tools_payload_extra(self, test: ToolTestCase) -> dict[str, Any]:
        """Hook for subclasses to adjust the tool-calling payload."""
        return {}

    def complete_tools(self, test: ToolTestCase) -> RawResponse:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in test.tools
        ]
        tool_choice = test.tool_choice
        if isinstance(tool_choice, dict) and "name" in tool_choice and "type" not in tool_choice:
            tool_choice = {"type": "function", "function": {"name": tool_choice["name"]}}

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": test.messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0,
        }
        payload.update(self._tools_payload_extra(test))

        response = self._post("/v1/chat/completions", payload)
        if not response.ok:
            return response
        return self._extract_chat_message(response)

    # -- shared response parsing -----------------------------------------

    @staticmethod
    def _extract_chat_message(response: RawResponse) -> RawResponse:
        try:
            choice = response.raw["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            response.ok = False
            response.error = "response JSON did not have the expected choices[0].message shape"
            return response

        response.content = message.get("content")
        raw_tool_calls = message.get("tool_calls") or []
        normalized = []
        for call in raw_tool_calls:
            fn = call.get("function", {})
            normalized.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        response.tool_calls = normalized
        return response

    def get_version(self) -> str | None:
        return None
