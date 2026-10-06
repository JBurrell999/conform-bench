"""Core data types shared by the loader, engine adapters, runner, and report.

Keeping these as plain dataclasses (rather than, say, pydantic models) keeps
the harness's own dependency footprint tiny -- the whole point of this
project is to probe *other* people's JSON handling, so ours should be boring
and easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Category(str, Enum):
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"


@dataclass
class FieldCheck:
    """A pointer into a JSON value (dotted path, e.g. ``user.address.city``)
    and the exact value expected there. Used for semantic spot-checks on top
    of plain schema validation, e.g. "did it actually use the name I gave
    it" rather than just "is this a syntactically valid object"."""

    path: str
    equals: Any


@dataclass
class StructuredTestCase:
    id: str
    description: str
    schema: dict[str, Any]
    prompt: str
    difficulty: str = "basic"
    system_prompt: str | None = None
    checks: list[FieldCheck] = field(default_factory=list)
    category: Category = Category.STRUCTURED_OUTPUT


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolTestCase:
    id: str
    description: str
    tools: list[ToolSpec]
    messages: list[dict[str, str]]
    difficulty: str = "basic"
    tool_choice: Any = "auto"  # "auto" | "required" | "none" | {"name": "..."}
    expect_call: bool = True
    expect_tool_name: str | None = None
    expect_any_of: list[str] | None = None  # for "pick the right tool" tests
    min_calls: int = 1
    max_calls: int | None = 1  # None means unbounded (parallel-call tests)
    checks: list[FieldCheck] = field(default_factory=list)
    category: Category = Category.TOOL_CALLING


@dataclass
class EngineConfig:
    name: str
    type: str  # "vllm" | "sglang" | "llamacpp" | "ollama" | "openai_compat"
    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    notes: str | None = None


@dataclass
class RawResponse:
    """Normalized view of whatever the engine sent back, independent of the
    transport quirks of any particular API shape."""

    ok: bool
    status_code: int | None
    elapsed_ms: float
    content: str | None = None          # assistant message text, if any
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None                      # full decoded JSON body, for debugging
    error: str | None = None             # network / transport-level error
    http_error_body: str | None = None   # body of a non-2xx response


@dataclass
class TestResult:
    engine: str
    engine_type: str
    model: str
    test_id: str
    category: Category
    difficulty: str
    outcome: Outcome
    reason: str
    elapsed_ms: float
    raw_excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["category"] = self.category.value
        d["outcome"] = self.outcome.value
        return d


@dataclass
class RunMetadata:
    engine: str
    engine_type: str
    model: str
    base_url: str
    version: str | None
    timestamp: str
    harness_version: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RunReport:
    metadata: RunMetadata
    results: list[TestResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
