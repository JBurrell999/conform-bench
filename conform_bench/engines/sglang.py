"""SGLang adapter.

SGLang's OpenAI-compatible frontend (`sglang.launch_server`) supports
`response_format: json_schema` via its own grammar backend (xgrammar by
default on recent versions, outlines on older ones). Tool calling requires
`--tool-call-parser <name>` at server launch for most open-weight models,
mirroring vLLM's gotcha -- omit it and `tool_choice` is a no-op.
"""

from __future__ import annotations

from conform_bench.engines.openai_compat import OpenAICompatAdapter


class SGLangAdapter(OpenAICompatAdapter):
    engine_type = "sglang"

    def get_version(self) -> str | None:
        url = f"{self.config.base_url.rstrip('/')}/get_server_info"
        try:
            resp = self._client.get(url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json()
                return data.get("version") or data.get("sglang_version")
        except Exception:
            pass
        return None
