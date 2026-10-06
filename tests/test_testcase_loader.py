from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_conform.testcase_loader import load_structured_tests, load_tool_tests


def test_loads_all_structured_fixtures():
    tests = load_structured_tests()
    assert len(tests) >= 8
    ids = [t.id for t in tests]
    assert len(ids) == len(set(ids)), "duplicate ids among structured tests"


def test_loads_all_tool_fixtures():
    tests = load_tool_tests()
    assert len(tests) >= 6
    ids = [t.id for t in tests]
    assert len(ids) == len(set(ids)), "duplicate ids among tool tests"


def test_descriptions_have_no_trailing_whitespace():
    """YAML `>` folded block scalars keep a trailing newline by default,
    which used to split markdown table rows in the generated matrix."""
    for test in load_structured_tests() + load_tool_tests():
        assert test.description == test.description.strip()
        assert "\n" not in test.description


def test_tool_test_tools_are_parsed_as_toolspec():
    tests = load_tool_tests()
    single = next(t for t in tests if t.id == "single_tool_basic")
    assert single.tools[0].name == "get_weather"
    assert "city" in single.tools[0].parameters["properties"]


def test_duplicate_id_raises(tmp_path: Path):
    d = tmp_path / "structured"
    d.mkdir()
    doc = {
        "id": "dup",
        "description": "x",
        "schema": {"type": "object"},
        "prompt": "p",
    }
    (d / "a.yaml").write_text(yaml.dump(doc))
    (d / "b.yaml").write_text(yaml.dump(doc))
    with pytest.raises(ValueError, match="duplicate test id"):
        load_structured_tests(d)


def test_missing_optional_fields_default_sensibly(tmp_path: Path):
    d = tmp_path / "structured"
    d.mkdir()
    (d / "a.yaml").write_text(
        yaml.dump({"id": "a", "description": "x", "schema": {"type": "object"}, "prompt": "p"})
    )
    [test] = load_structured_tests(d)
    assert test.difficulty == "basic"
    assert test.checks == []
    assert test.system_prompt is None
