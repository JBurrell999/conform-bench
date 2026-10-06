"""End-to-end test of `run_engine`/`run_many` against the real mock server
(not MockTransport) -- this exercises the full stack including real HTTP,
which catches issues the in-process MockTransport tests can't (header
handling, real (de)serialization, thread-based parallel execution).

Uses simple, `checks`-free fixture test cases rather than the real catalog
under testcases/: the mock server's generic synthesis only knows how to be
schema-valid, not factually correct (see conform_bench/synth.py), so it can't
satisfy the real catalog's exact-value checks. That full-catalog
self-consistency check already lives in test_runner_integration.py.
"""

from __future__ import annotations

from conform_bench.mock_server import MockEngineServer
from conform_bench.models import Category, EngineConfig, Outcome, StructuredTestCase, ToolSpec, ToolTestCase
from conform_bench.runner import run_engine, run_many

STRUCTURED = [
    StructuredTestCase(
        id="s1",
        description="d",
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        prompt="describe someone",
    )
]

TOOLS = [
    ToolTestCase(
        id="t1",
        description="d",
        tools=[
            ToolSpec(
                name="get_weather",
                description="get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            )
        ],
        messages=[{"role": "user", "content": "weather in Rome?"}],
        expect_tool_name="get_weather",
    )
]


def _run(server, categories=None):
    config = EngineConfig(name="mock", type="vllm", base_url=server.base_url, model="m")
    return run_engine(config, structured_tests=STRUCTURED, tool_tests=TOOLS, categories=categories)


def test_run_engine_against_mock_server_conformant_behavior_passes():
    with MockEngineServer(behavior="conformant") as server:
        report = _run(server)

    assert report.metadata.engine == "mock"
    assert report.metadata.version == "mock-1.0.0"  # vllm adapter probes GET /version
    categories = {r.category for r in report.results}
    assert categories == {Category.STRUCTURED_OUTPUT, Category.TOOL_CALLING}
    assert all(r.outcome == Outcome.PASS for r in report.results)


def test_run_engine_respects_category_filter():
    with MockEngineServer(behavior="conformant") as server:
        report = _run(server, categories=["tool_calling"])

    assert all(r.category == Category.TOOL_CALLING for r in report.results)
    assert len(report.results) == len(TOOLS)


def test_run_engine_http_400_behavior_marks_everything_unsupported():
    with MockEngineServer(behavior="http_400") as server:
        report = _run(server)

    assert all(r.outcome == Outcome.UNSUPPORTED for r in report.results)


def test_run_many_sequential_and_parallel_give_same_results():
    def make_configs(server_a, server_b):
        return [
            EngineConfig(name="a", type="vllm", base_url=server_a.base_url, model="m"),
            EngineConfig(name="b", type="vllm", base_url=server_b.base_url, model="m"),
        ]

    with MockEngineServer(behavior="conformant") as server_a, MockEngineServer(behavior="schema_violator") as server_b:
        configs = make_configs(server_a, server_b)
        sequential = run_many(configs, structured_tests=STRUCTURED, tool_tests=TOOLS, parallel=False)
        parallel = run_many(configs, structured_tests=STRUCTURED, tool_tests=TOOLS, parallel=True)

    seq_outcomes = {(r.engine, r.test_id): r.outcome for rep in sequential for r in rep.results}
    par_outcomes = {(r.engine, r.test_id): r.outcome for rep in parallel for r in rep.results}
    assert seq_outcomes == par_outcomes
    assert seq_outcomes[("a", "s1")] == Outcome.PASS
    assert seq_outcomes[("b", "s1")] == Outcome.FAIL
