"""Turns one or more saved run-result JSON files into a markdown
compatibility matrix, and can inject that matrix into a README between
marker comments so the doc never drifts from the actual results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_conform.testcase_loader import load_structured_tests, load_tool_tests

OUTCOME_SYMBOL = {
    "pass": "✅",
    "fail": "❌",
    "error": "⚠️",
    "unsupported": "➖",
    "timeout": "⏱️",
}

LEGEND = (
    "✅ pass &nbsp;·&nbsp; ❌ fail (ran, output didn't conform) &nbsp;·&nbsp; "
    "⚠️ error (request/transport failure) &nbsp;·&nbsp; "
    "➖ unsupported (engine rejected the request, e.g. HTTP 400) &nbsp;·&nbsp; "
    "⏱️ timeout &nbsp;·&nbsp; *(blank)* not run"
)

MATRIX_START = "<!-- LLM-CONFORM:MATRIX:START -->"
MATRIX_END = "<!-- LLM-CONFORM:MATRIX:END -->"


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


def dedupe_latest_per_engine(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """If several result files exist for the same engine name, keep only the
    one with the most recent timestamp -- a stale result shouldn't shadow
    a fresh one just because it happened to load first."""
    latest: dict[str, dict[str, Any]] = {}
    for r in reports:
        name = r["metadata"]["engine"]
        existing = latest.get(name)
        if existing is None or r["metadata"]["timestamp"] > existing["metadata"]["timestamp"]:
            latest[name] = r
    return list(latest.values())


def _pass_rate(results: list[dict[str, Any]]) -> str:
    if not results:
        return "n/a"
    passed = sum(1 for r in results if r["outcome"] == "pass")
    return f"{passed}/{len(results)} ({round(100 * passed / len(results))}%)"


def build_summary_table(reports: list[dict[str, Any]]) -> str:
    lines = [
        "| Engine | Version | Model | Structured Output | Tool Calling | Tested |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(reports, key=lambda r: r["metadata"]["engine"]):
        meta = r["metadata"]
        results = r["results"]
        structured = [x for x in results if x["category"] == "structured_output"]
        tools = [x for x in results if x["category"] == "tool_calling"]
        tested_date = meta["timestamp"].split("T")[0]
        lines.append(
            f"| **{meta['engine']}** | {meta.get('version') or 'unknown'} | `{meta['model']}` "
            f"| {_pass_rate(structured)} | {_pass_rate(tools)} | {tested_date} |"
        )
    return "\n".join(lines)


def _result_lookup(reports: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """engine -> test_id -> result dict"""
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for r in reports:
        engine = r["metadata"]["engine"]
        lookup[engine] = {res["test_id"]: res for res in r["results"]}
    return lookup


def build_category_matrix(
    reports: list[dict[str, Any]],
    category: str,
    test_ids_in_order: list[str],
    descriptions: dict[str, str],
) -> str:
    engines = sorted(r["metadata"]["engine"] for r in reports)
    lookup = _result_lookup(reports)

    header = "| Test | " + " | ".join(engines) + " |"
    sep = "|---|" + "|".join(["---"] * len(engines)) + "|"
    lines = [header, sep]

    for test_id in test_ids_in_order:
        desc = descriptions.get(test_id, test_id)
        cells = []
        for engine in engines:
            result = lookup.get(engine, {}).get(test_id)
            if result is None:
                cells.append(" ")
            else:
                symbol = OUTCOME_SYMBOL.get(result["outcome"], "?")
                cells.append(symbol)
        lines.append(f"| `{test_id}` ({desc}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_failure_appendix(reports: list[dict[str, Any]]) -> str:
    """Collapsible per-engine list of every non-pass result and *why*."""
    sections = []
    for r in sorted(reports, key=lambda r: r["metadata"]["engine"]):
        engine = r["metadata"]["engine"]
        non_passing = [x for x in r["results"] if x["outcome"] != "pass"]
        if not non_passing:
            continue
        body = "\n".join(
            f"- `{x['test_id']}` ({x['category']}): **{x['outcome']}** -- {x['reason']}"
            for x in non_passing
        )
        sections.append(
            f"<details>\n<summary>{engine}: {len(non_passing)} non-passing test(s)</summary>\n\n{body}\n\n</details>"
        )
    return "\n\n".join(sections) if sections else "_No failures recorded._"


def build_full_matrix_markdown(reports: list[dict[str, Any]]) -> str:
    reports = dedupe_latest_per_engine(reports)
    structured_tests = load_structured_tests()
    tool_tests = load_tool_tests()

    descriptions = {t.id: t.description for t in structured_tests}
    descriptions.update({t.id: t.description for t in tool_tests})

    parts = [
        MATRIX_START,
        "### Summary",
        "",
        build_summary_table(reports),
        "",
        "### Structured output (JSON-schema-constrained generation)",
        "",
        build_category_matrix(
            reports, "structured_output", [t.id for t in structured_tests], descriptions
        ),
        "",
        "### Tool calling",
        "",
        build_category_matrix(reports, "tool_calling", [t.id for t in tool_tests], descriptions),
        "",
        f"**Legend:** {LEGEND}",
        "",
        "<details>",
        "<summary>Why each non-passing cell failed</summary>",
        "",
        build_failure_appendix(reports),
        "",
        "</details>",
        MATRIX_END,
    ]
    return "\n".join(parts)


def update_readme(readme_path: Path, matrix_markdown: str) -> str:
    text = readme_path.read_text(encoding="utf-8")
    if MATRIX_START not in text or MATRIX_END not in text:
        raise ValueError(
            f"{readme_path} is missing the {MATRIX_START} / {MATRIX_END} markers; "
            "can't safely inject the matrix. Add them around the section to replace."
        )
    before, _, rest = text.partition(MATRIX_START)
    _, _, after = rest.partition(MATRIX_END)
    new_text = before + matrix_markdown + after
    readme_path.write_text(new_text, encoding="utf-8")
    return new_text
