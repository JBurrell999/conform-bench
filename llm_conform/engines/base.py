"""Base interface every engine adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from llm_conform.models import EngineConfig, RawResponse, StructuredTestCase, ToolTestCase


class EngineAdapter(ABC):
    engine_type: str = "base"

    def __init__(self, config: EngineConfig, client: httpx.Client | None = None):
        self.config = config
        # Allowing an injected client is what makes the adapters unit-testable
        # without a real server: tests pass an httpx.Client built on
        # httpx.MockTransport. See tests/test_engines.py.
        self._client = client or httpx.Client(timeout=config.timeout_s)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "EngineAdapter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @abstractmethod
    def complete_structured(self, test: StructuredTestCase) -> RawResponse:
        """Ask the engine to produce output constrained to `test.schema`."""

    @abstractmethod
    def complete_tools(self, test: ToolTestCase) -> RawResponse:
        """Ask the engine to respond to `test.messages` given `test.tools`."""

    def get_version(self) -> str | None:
        """Best-effort version/build probe, used only for report metadata.
        Must never raise -- a version string is a nice-to-have, not something
        worth failing a whole run over."""
        return None
