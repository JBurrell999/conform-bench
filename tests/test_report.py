from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_conform.report import (
    MATRIX_END,
    MATRIX_START,
    build_full_matrix_markdown,
    build_summary_table,
    dedupe_latest_per_engine,
    update_readme,
)

STRUCTURED_ID = "flat_object_basic"  # a real id from testcases/structured_output/


def _fake_report(engine: str, timestamp: str, outcome: str = "pass") -> dict:
    return {
        "metadata": {
            "engine": engine,
            "engine_type": "ollama",
            "model": "m",
            "base_url": "http://x",
            "version": "1.0",
            "timestamp": timestamp,
            "harness_version": "0.1.0",
        },
        "results": [
            {
                "engine": engine,
                "engine_type": "ollama",
                "model": "m",
                "test_id": STRUCTURED_ID,
                "category": "structured_output",
                "difficulty": "basic",
                "outcome": outcome,
                "reason": "ok" if outcome == "pass" else "it broke",
                "elapsed_ms": 12.3,
                "raw_excerpt": None,
            }
        ],
    }


def test_summary_table_includes_pass_rate():
    report = _fake_report("ollama", "2026-01-01T00:00:00+00:00")
    table = build_summary_table([report])
    assert "ollama" in table
    assert "1/1 (100%)" in table


def test_dedupe_keeps_latest_timestamp():
    old = _fake_report("ollama", "2025-01-01T00:00:00+00:00", outcome="fail")
    new = _fake_report("ollama", "2026-01-01T00:00:00+00:00", outcome="pass")
    [kept] = dedupe_latest_per_engine([old, new])
    assert kept["results"][0]["outcome"] == "pass"


def test_full_matrix_has_markers_and_legend():
    report = _fake_report("ollama", "2026-01-01T00:00:00+00:00")
    markdown = build_full_matrix_markdown([report])
    assert markdown.startswith(MATRIX_START)
    assert markdown.endswith(MATRIX_END)
    assert "Legend" in markdown
    assert STRUCTURED_ID in markdown


def test_full_matrix_includes_failure_appendix_entry():
    report = _fake_report("ollama", "2026-01-01T00:00:00+00:00", outcome="fail")
    markdown = build_full_matrix_markdown([report])
    assert "it broke" in markdown


def test_update_readme_replaces_only_marked_region(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\nIntro text.\n\n{MATRIX_START}\nold matrix\n{MATRIX_END}\n\nFooter.\n")
    new_text = update_readme(readme, f"{MATRIX_START}\nNEW MATRIX\n{MATRIX_END}")
    assert "Intro text." in new_text
    assert "Footer." in new_text
    assert "NEW MATRIX" in new_text
    assert "old matrix" not in new_text


def test_update_readme_requires_markers(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nno markers here\n")
    with pytest.raises(ValueError, match="missing"):
        update_readme(readme, "whatever")
