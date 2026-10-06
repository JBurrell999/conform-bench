"""Engine adapters: one per serving engine, all implementing the same
`EngineAdapter` interface so the runner never has to special-case a vendor.
"""

from llm_conform.engines.base import EngineAdapter
from llm_conform.engines.llamacpp import LlamaCppAdapter
from llm_conform.engines.ollama import OllamaAdapter
from llm_conform.engines.openai_compat import OpenAICompatAdapter
from llm_conform.engines.sglang import SGLangAdapter
from llm_conform.engines.vllm import VLLMAdapter

ADAPTERS: dict[str, type[EngineAdapter]] = {
    "vllm": VLLMAdapter,
    "sglang": SGLangAdapter,
    "llamacpp": LlamaCppAdapter,
    "ollama": OllamaAdapter,
    "openai_compat": OpenAICompatAdapter,
}


def get_adapter_class(engine_type: str) -> type[EngineAdapter]:
    try:
        return ADAPTERS[engine_type]
    except KeyError as exc:
        raise ValueError(
            f"unknown engine type '{engine_type}'; supported: {sorted(ADAPTERS)}"
        ) from exc


__all__ = [
    "EngineAdapter",
    "OpenAICompatAdapter",
    "VLLMAdapter",
    "SGLangAdapter",
    "LlamaCppAdapter",
    "OllamaAdapter",
    "ADAPTERS",
    "get_adapter_class",
]
