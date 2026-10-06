"""Orchestrates running the test suite against one or more engines."""

from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from conform_bench import __version__
from conform_bench.engines import get_adapter_class
from conform_bench.models import (
    Category,
    EngineConfig,
    RawResponse,
    RunMetadata,
    RunReport,
    StructuredTestCase,
    TestResult,
    ToolTestCase,
)
from conform_bench.testcase_loader import load_structured_tests, load_tool_tests
from conform_bench.validators import Verdict, validate_structured_response, validate_tool_response


def run_engine(
    config: EngineConfig,
    structured_tests: list[StructuredTestCase] | None = None,
    tool_tests: list[ToolTestCase] | None = None,
    categories: Iterable[str] | None = None,
) -> RunReport:
    """Run the full test suite (or a subset of categories) against one engine."""
    structured_tests = structured_tests if structured_tests is not None else load_structured_tests()
    tool_tests = tool_tests if tool_tests is not None else load_tool_tests()
    wanted = set(categories) if categories is not None else None

    adapter_cls = get_adapter_class(config.type)
    results: list[TestResult] = []
    with adapter_cls(config) as adapter:
        version = adapter.get_version()

        if wanted is None or Category.STRUCTURED_OUTPUT.value in wanted:
            for test in structured_tests:
                raw = adapter.complete_structured(test)
                verdict = validate_structured_response(raw, test)
                results.append(
                    _to_result(config, test.id, Category.STRUCTURED_OUTPUT, test.difficulty, verdict, raw)
                )

        if wanted is None or Category.TOOL_CALLING.value in wanted:
            for test in tool_tests:
                raw = adapter.complete_tools(test)
                verdict = validate_tool_response(raw, test)
                results.append(
                    _to_result(config, test.id, Category.TOOL_CALLING, test.difficulty, verdict, raw)
                )

    metadata = RunMetadata(
        engine=config.name,
        engine_type=config.type,
        model=config.model,
        base_url=config.base_url,
        version=version,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        harness_version=__version__,
        notes=config.notes,
    )
    return RunReport(metadata=metadata, results=results)


def run_many(
    configs: list[EngineConfig],
    structured_tests: list[StructuredTestCase] | None = None,
    tool_tests: list[ToolTestCase] | None = None,
    categories: Iterable[str] | None = None,
    parallel: bool = False,
) -> list[RunReport]:
    """Run the suite against several engines. `parallel=True` runs each
    engine's suite on its own thread (safe: engines are independent servers
    and each gets its own httpx.Client); test cases within one engine's run
    stay sequential so timing numbers for a single engine remain meaningful."""
    structured_tests = structured_tests if structured_tests is not None else load_structured_tests()
    tool_tests = tool_tests if tool_tests is not None else load_tool_tests()

    if not parallel or len(configs) <= 1:
        return [run_engine(c, structured_tests, tool_tests, categories) for c in configs]

    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = [
            pool.submit(run_engine, c, structured_tests, tool_tests, categories) for c in configs
        ]
        return [f.result() for f in futures]


def _to_result(
    config: EngineConfig,
    test_id: str,
    category: Category,
    difficulty: str,
    verdict: Verdict,
    raw: RawResponse,
) -> TestResult:
    excerpt = None
    if raw.content:
        excerpt = raw.content[:300]
    elif raw.http_error_body:
        excerpt = raw.http_error_body[:300]
    return TestResult(
        engine=config.name,
        engine_type=config.type,
        model=config.model,
        test_id=test_id,
        category=category,
        difficulty=difficulty,
        outcome=verdict.outcome,
        reason=verdict.reason,
        elapsed_ms=raw.elapsed_ms,
        raw_excerpt=excerpt,
    )
