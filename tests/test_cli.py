"""Smoke tests for the `llm-conform` CLI subcommands, driven in-process via
`main(argv)` rather than subprocess -- fast, and still exercises the real
argparse wiring."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from llm_conform.cli import main
from llm_conform.mock_server import MockEngineServer


def test_list_tests_runs_cleanly(capsys):
    assert main(["list-tests"]) == 0
    out = capsys.readouterr().out
    assert "structured_output" in out
    assert "tool_calling" in out
    assert "flat_object_basic" in out


def test_run_writes_results_and_latest_pointer(tmp_path: Path, capsys):
    with MockEngineServer(behavior="http_400") as server:
        config_path = tmp_path / "engines.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "engines": [
                        {"name": "mock", "type": "openai_compat", "base_url": server.base_url, "model": "m"}
                    ]
                }
            )
        )
        out_dir = tmp_path / "results"
        rc = main(
            [
                "run",
                "--config",
                str(config_path),
                "--output-dir",
                str(out_dir),
                "--latest",
            ]
        )

    assert rc == 0
    latest = out_dir.parent / "mock-latest.json"
    assert latest.exists()
    data = json.loads(latest.read_text())
    assert data["metadata"]["engine"] == "mock"
    assert all(r["outcome"] == "unsupported" for r in data["results"])
    assert "unsupported" in capsys.readouterr().out


def test_run_with_unknown_engine_name_returns_error(tmp_path: Path, capsys):
    config_path = tmp_path / "engines.yaml"
    config_path.write_text(
        yaml.dump({"engines": [{"name": "real", "type": "openai_compat", "base_url": "http://x", "model": "m"}]})
    )
    rc = main(["run", "--config", str(config_path), "--engine", "nope"])
    assert rc == 2
    assert "no engines matching" in capsys.readouterr().err


def test_report_prints_matrix_to_stdout(tmp_path: Path, capsys):
    result = {
        "metadata": {
            "engine": "mock",
            "engine_type": "ollama",
            "model": "m",
            "base_url": "http://x",
            "version": "1.0",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "harness_version": "0.1.0",
        },
        "results": [
            {
                "engine": "mock",
                "engine_type": "ollama",
                "model": "m",
                "test_id": "flat_object_basic",
                "category": "structured_output",
                "difficulty": "basic",
                "outcome": "pass",
                "reason": "ok",
                "elapsed_ms": 1.0,
                "raw_excerpt": None,
            }
        ],
    }
    result_path = tmp_path / "mock-latest.json"
    result_path.write_text(json.dumps(result))

    rc = main(["report", str(result_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Summary" in out
    assert "flat_object_basic" in out


def test_report_update_readme(tmp_path: Path):
    from llm_conform.report import MATRIX_END, MATRIX_START

    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{MATRIX_START}\nold\n{MATRIX_END}\n")

    result = {
        "metadata": {
            "engine": "mock",
            "engine_type": "ollama",
            "model": "m",
            "base_url": "http://x",
            "version": "1.0",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "harness_version": "0.1.0",
        },
        "results": [],
    }
    result_path = tmp_path / "mock-latest.json"
    result_path.write_text(json.dumps(result))

    rc = main(["report", str(result_path), "--update-readme", str(readme)])
    assert rc == 0
    assert "old" not in readme.read_text()
    assert MATRIX_START in readme.read_text()
