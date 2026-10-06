from __future__ import annotations

from pathlib import Path

from conform_bench.config import load_engine_configs

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_loads_example_config():
    configs = load_engine_configs(REPO_ROOT / "engines.example.yaml")
    assert len(configs) == 4
    by_name = {c.name: c for c in configs}
    assert by_name["ollama"].type == "ollama"
    assert by_name["vllm"].base_url == "http://localhost:8000"
    assert by_name["vllm"].timeout_s == 120.0


def test_defaults_are_applied(tmp_path: Path):
    path = tmp_path / "engines.yaml"
    path.write_text(
        """
engines:
  - name: e1
    type: openai_compat
    base_url: http://x
    model: m
"""
    )
    [config] = load_engine_configs(path)
    assert config.api_key is None
    assert config.timeout_s == 120.0
    assert config.extra_headers == {}
