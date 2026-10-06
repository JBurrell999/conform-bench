"""Loads test case YAML files from `testcases/` into typed dataclasses.

File layout:

    testcases/
      structured_output/*.yaml   -> StructuredTestCase
      tool_calling/*.yaml        -> ToolTestCase

See CONTRIBUTING.md for the field reference and how to add a new case.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from conform_bench.models import FieldCheck, StructuredTestCase, ToolSpec, ToolTestCase

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTCASES_DIR = REPO_ROOT / "testcases"


def _parse_checks(raw: list[dict[str, Any]] | None) -> list[FieldCheck]:
    return [FieldCheck(path=c["path"], equals=c["equals"]) for c in (raw or [])]


def _load_yaml_files(directory: Path) -> list[dict[str, Any]]:
    docs = []
    for path in sorted(directory.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if doc is None:
            continue
        doc["_source_file"] = os.path.relpath(path, REPO_ROOT)
        docs.append(doc)
    return docs


def load_structured_tests(directory: Path | None = None) -> list[StructuredTestCase]:
    directory = directory or (DEFAULT_TESTCASES_DIR / "structured_output")
    tests = []
    for doc in _load_yaml_files(directory):
        tests.append(
            StructuredTestCase(
                id=doc["id"],
                description=doc["description"].strip(),
                schema=doc["schema"],
                prompt=doc["prompt"],
                difficulty=doc.get("difficulty", "basic"),
                system_prompt=doc.get("system_prompt"),
                checks=_parse_checks(doc.get("checks")),
            )
        )
    _assert_unique_ids(tests)
    return tests


def load_tool_tests(directory: Path | None = None) -> list[ToolTestCase]:
    directory = directory or (DEFAULT_TESTCASES_DIR / "tool_calling")
    tests = []
    for doc in _load_yaml_files(directory):
        tools = [
            ToolSpec(name=t["name"], description=t["description"], parameters=t["parameters"])
            for t in doc["tools"]
        ]
        expect = doc.get("expect", {})
        tests.append(
            ToolTestCase(
                id=doc["id"],
                description=doc["description"].strip(),
                tools=tools,
                messages=doc["messages"],
                difficulty=doc.get("difficulty", "basic"),
                tool_choice=doc.get("tool_choice", "auto"),
                expect_call=expect.get("calls_tool", True),
                expect_tool_name=expect.get("tool_name"),
                expect_any_of=expect.get("tool_name_any_of"),
                min_calls=expect.get("min_calls", 1),
                max_calls=expect.get("max_calls", 1),
                checks=_parse_checks(expect.get("args_checks")),
            )
        )
    _assert_unique_ids(tests)
    return tests


def _assert_unique_ids(tests: list[Any]) -> None:
    seen: dict[str, str] = {}
    for t in tests:
        src = getattr(t, "_source_file", t.id)
        if t.id in seen:
            raise ValueError(f"duplicate test id '{t.id}' in {src}")
        seen[t.id] = src
