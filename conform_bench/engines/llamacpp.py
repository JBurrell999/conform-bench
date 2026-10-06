"""llama.cpp (`llama-server`) adapter.

This is the engine where conformance tends to be shakiest in practice:
`response_format: json_schema` support (schema -> GBNF grammar translation)
and tool-call parsing both depend heavily on (a) the llama.cpp build version
and (b) whether the GGUF's embedded chat template -- or a `--chat-template`
override -- actually emits tool-call-shaped output the server's parser
recognizes. A 400 here usually means the build is too old for
`response_format`; a 200 with no tool call despite `tool_choice` usually
means the chat template doesn't support tools for this model.
"""

from __future__ import annotations

from conform_bench.engines.openai_compat import OpenAICompatAdapter


class LlamaCppAdapter(OpenAICompatAdapter):
    engine_type = "llamacpp"

    def get_version(self) -> str | None:
        url = f"{self.config.base_url.rstrip('/')}/props"
        try:
            resp = self._client.get(url, headers=self._headers())
            if resp.status_code == 200:
                data = resp.json()
                return (
                    data.get("build_info")
                    or data.get("version")
                    or (data.get("default_generation_settings") or {}).get("model")
                )
        except Exception:
            pass
        return None
