"""vLLM adapter.

vLLM's OpenAI-compatible server implements `response_format: json_schema`
directly (backed by outlines/xgrammar/lm-format-enforcer depending on
version and `--guided-decoding-backend`).

Known gotcha this project exists to catch: tool calling only works if the
server was launched with `--enable-auto-tool-choice --tool-call-parser
<parser>` matching the model family. Without those flags, `tool_choice`
is silently ignored and the model just talks -- which our validator
correctly reports as a FAIL (expected a tool call, got none), not an
engine bug. See README's matrix notes for per-model parser names.
"""

from __future__ import annotations

from conform_bench.engines.openai_compat import OpenAICompatAdapter


class VLLMAdapter(OpenAICompatAdapter):
    engine_type = "vllm"

    def get_version(self) -> str | None:
        url = f"{self.config.base_url.rstrip('/')}/version"
        try:
            resp = self._client.get(url, headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("version")
        except Exception:
            pass
        return None
