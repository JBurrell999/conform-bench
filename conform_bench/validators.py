"""Validation logic: did the engine's response actually conform?

This module is deliberately strict. Engines get partial credit for nothing:
"close enough" JSON (trailing commentary, markdown fences, truncated output,
schema violations) is a FAIL, because a downstream program parsing this
output with ``json.loads`` would also fail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import jsonschema

from conform_bench.models import FieldCheck, Outcome, RawResponse, StructuredTestCase, ToolTestCase

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


@dataclass
class Verdict:
    outcome: Outcome
    reason: str


def _resolve_path(value: Any, path: str) -> tuple[bool, Any]:
    """Walk a dotted path (``a.b.0.c``) through dicts/lists. Returns
    ``(found, value)``; ``found`` is False if any segment is missing."""
    current = value
    for part in path.split("."):
        if isinstance(current, list):
            if not part.lstrip("-").isdigit():
                return False, None
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return False, None
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        else:
            return False, None
    return True, current


def run_field_checks(value: Any, checks: list[FieldCheck]) -> list[str]:
    """Returns a list of human-readable failure messages (empty = all passed)."""
    failures = []
    for check in checks:
        found, actual = _resolve_path(value, check.path)
        if not found:
            failures.append(f"path '{check.path}' not present in output")
        elif actual != check.equals:
            failures.append(
                f"path '{check.path}' expected {check.equals!r}, got {actual!r}"
            )
    return failures


def strip_markdown_fence(text: str) -> str:
    """Engines frequently wrap JSON in ```json fences even when asked for
    raw JSON. We strip that defensively before attempting to parse, but a
    test is only marked PASS if we had to do so when it wasn't supposed
    to be necessary is tracked separately by the caller via `was_fenced`.
    """
    match = _FENCE_RE.match(text.strip())
    if match:
        return match.group(1)
    return text


def validate_structured_response(
    response: RawResponse, test: StructuredTestCase
) -> Verdict:
    if not response.ok:
        if response.status_code == 400 or response.status_code == 422:
            return Verdict(
                Outcome.UNSUPPORTED,
                f"HTTP {response.status_code}: engine rejected the request "
                f"(likely no support for structured output) -- {response.http_error_body or response.error}",
            )
        return Verdict(
            Outcome.ERROR,
            f"request failed: HTTP {response.status_code} {response.error or response.http_error_body or ''}".strip(),
        )

    if response.content is None:
        return Verdict(Outcome.FAIL, "response had no text content to parse as JSON")

    raw_text = response.content
    stripped = strip_markdown_fence(raw_text)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        preview = raw_text[:200].replace("\n", "\\n")
        return Verdict(Outcome.FAIL, f"output is not valid JSON ({exc}); got: {preview!r}")

    try:
        jsonschema.validate(instance=parsed, schema=test.schema)
    except jsonschema.ValidationError as exc:
        return Verdict(Outcome.FAIL, f"JSON does not conform to schema: {exc.message}")
    except jsonschema.SchemaError as exc:  # pragma: no cover - our own schemas are checked in tests
        return Verdict(Outcome.ERROR, f"test case schema itself is invalid: {exc.message}")

    failures = run_field_checks(parsed, test.checks)
    if failures:
        return Verdict(Outcome.FAIL, "; ".join(failures))

    if raw_text.strip() != stripped.strip():
        return Verdict(
            Outcome.PASS,
            "valid JSON conforming to schema (note: wrapped in a markdown code fence)",
        )
    return Verdict(Outcome.PASS, "valid JSON conforming to schema")


def validate_tool_response(response: RawResponse, test: ToolTestCase) -> Verdict:
    if not response.ok:
        if response.status_code in (400, 422):
            return Verdict(
                Outcome.UNSUPPORTED,
                f"HTTP {response.status_code}: engine rejected the tools request "
                f"-- {response.http_error_body or response.error}",
            )
        return Verdict(
            Outcome.ERROR,
            f"request failed: HTTP {response.status_code} {response.error or response.http_error_body or ''}".strip(),
        )

    calls = response.tool_calls

    if not test.expect_call:
        if calls:
            names = ", ".join(c.get("name", "?") for c in calls)
            return Verdict(
                Outcome.FAIL,
                f"expected no tool call (this prompt doesn't need one), but got: {names}",
            )
        return Verdict(Outcome.PASS, "correctly answered without calling a tool")

    if not calls:
        return Verdict(
            Outcome.FAIL,
            f"expected a tool call but got none; content was: {(response.content or '')[:200]!r}",
        )

    if len(calls) < test.min_calls:
        return Verdict(Outcome.FAIL, f"expected >= {test.min_calls} tool call(s), got {len(calls)}")
    if test.max_calls is not None and len(calls) > test.max_calls:
        return Verdict(Outcome.FAIL, f"expected <= {test.max_calls} tool call(s), got {len(calls)}")

    valid_names = {t.name for t in test.tools}
    for call in calls:
        name = call.get("name")
        if name not in valid_names:
            return Verdict(Outcome.FAIL, f"tool call used unknown tool name {name!r}")

    if test.expect_tool_name is not None:
        used = {c.get("name") for c in calls}
        if test.expect_tool_name not in used:
            return Verdict(
                Outcome.FAIL,
                f"expected a call to '{test.expect_tool_name}', but tool(s) called were: {sorted(used)}",
            )

    if test.expect_any_of is not None:
        used = {c.get("name") for c in calls}
        if not used & set(test.expect_any_of):
            return Verdict(
                Outcome.FAIL,
                f"expected a call to one of {test.expect_any_of}, got {sorted(used)}",
            )

    # Validate each call's arguments both as parseable JSON and against that
    # tool's own parameter schema.
    tools_by_name = {t.name: t for t in test.tools}
    for call in calls:
        name = call.get("name")
        args_raw = call.get("arguments")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError as exc:
                return Verdict(
                    Outcome.FAIL,
                    f"tool call to '{name}' has arguments that aren't valid JSON: {exc}; got {args_raw[:200]!r}",
                )
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            return Verdict(
                Outcome.FAIL,
                f"tool call to '{name}' has arguments of unexpected type {type(args_raw).__name__}",
            )

        tool = tools_by_name.get(name)
        if tool is not None:
            try:
                jsonschema.validate(instance=args, schema=tool.parameters)
            except jsonschema.ValidationError as exc:
                return Verdict(
                    Outcome.FAIL,
                    f"arguments for tool '{name}' do not conform to its parameter schema: {exc.message}",
                )

        if name == test.expect_tool_name or (test.expect_any_of and name in test.expect_any_of):
            failures = run_field_checks(args, test.checks)
            if failures:
                return Verdict(Outcome.FAIL, "; ".join(failures))

    return Verdict(Outcome.PASS, f"called expected tool(s) with valid, schema-conforming arguments")
