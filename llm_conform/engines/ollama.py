"""Ollama adapter, using the native `/api/chat` endpoint (not the newer
OpenAI-compat shim at `/v1/chat/completions`) since the native API is what
most Ollama deployments actually front and is the first-class surface for
`format` (JSON-schema-constrained output).

Known gap this project exists to catch: the native API has no `tool_choice`
equivalent -- there is no way to force a call, force a specific tool, or
disable tool use. We never fake that support: if a test case needs forced
tool choice, we simply don't send anything for it, and a resulting FAIL
("expected a call, model just answered") is an accurate conformance report,
not a harness bug.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from llm_conform.engines.base import EngineAdapter
from llm_conform.models import RawResponse, StructuredTestCase, ToolTestCase


class OllamaAdapter(EngineAdapter):
    engine_type = "ollama"

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
            return RawResponse(ok=False, status_code=None, elapsed_ms=self._elapsed(start), error="timeout")
        except httpx.RequestError as exc:
            return RawResponse(ok=False, status_code=None, elapsed_ms=self._elapsed(start), error=str(exc))

        elapsed = self._elapsed(start)
        if resp.status_code >= 400:
            return RawResponse(
                ok=False, status_code=resp.status_code, elapsed_ms=elapsed, http_error_body=resp.text[:2000]
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
    def _elapsed(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    def complete_structured(self, test: StructuredTestCase) -> RawResponse:
        messages = []
        if test.system_prompt:
            messages.append({"role": "system", "content": test.system_prompt})
        messages.append({"role": "user", "content": test.prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "format": test.schema,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = self._post("/api/chat", payload)
        if not response.ok:
            return response
        return self._extract_message(response)

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
        payload = {
            "model": self.config.model,
            "messages": test.messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0},
        }
        # Deliberately no `tool_choice`: the native API has nothing to put
        # it in. See module docstring.
        response = self._post("/api/chat", payload)
        if not response.ok:
            return response
        return self._extract_message(response)

    @staticmethod
    def _extract_message(response: RawResponse) -> RawResponse:
        try:
            message = response.raw["message"]
        except (KeyError, TypeError):
            response.ok = False
            response.error = "response JSON did not have the expected message shape"
            return response

        response.content = message.get("content")
        raw_tool_calls = message.get("tool_calls") or []
        normalized = []
        for call in raw_tool_calls:
            fn = call.get("function", {})
            # Ollama's native API typically hands back `arguments` already
            # decoded as an object, not a JSON-encoded string like OpenAI's
            # wire format -- the validator accepts either.
            normalized.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        response.tool_calls = normalized
        return response

    def get_version(self) -> str | None:
        url = f"{self.config.base_url.rstrip('/')}/api/version"
        try:
            resp = self._client.get(url, headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("version")
        except Exception:
            pass
        return None
