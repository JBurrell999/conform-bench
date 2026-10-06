"""Loads engine configs (base URL, model, type) from a YAML file.

See `engines.example.yaml` for the format.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from conform_bench.models import EngineConfig


def load_engine_configs(path: str | Path) -> list[EngineConfig]:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    entries = doc.get("engines", [])
    configs = []
    for entry in entries:
        configs.append(
            EngineConfig(
                name=entry["name"],
                type=entry["type"],
                base_url=entry["base_url"],
                model=entry["model"],
                api_key=entry.get("api_key"),
                timeout_s=float(entry.get("timeout_s", 120.0)),
                extra_headers=entry.get("extra_headers", {}),
                notes=entry.get("notes"),
            )
        )
    return configs
