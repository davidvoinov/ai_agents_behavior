"""Read-only validation for Promise Switch execution artifact directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_EXECUTION_SCHEMA = "promise_switch_execution_v1"
EXPECTED_EXPERIMENT_VERSION = "promise_switch_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_run_directory(run_dir: Path, *, allow_invalid: bool = False) -> dict[str, Any]:
    """Reject legacy and invalid scientific inputs before any analysis."""
    manifest_path = run_dir / "manifest.json"
    trials_path = run_dir / "trials.jsonl"
    if not manifest_path.is_file() or not trials_path.is_file():
        raise ValueError(f"not a complete Promise Switch execution run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("execution_schema_version") != EXPECTED_EXECUTION_SCHEMA:
        raise ValueError(f"legacy or incompatible execution schema: {run_dir}")
    if manifest.get("experiment_version") != EXPECTED_EXPERIMENT_VERSION:
        raise ValueError(f"legacy or incompatible experiment version: {run_dir}")
    if (run_dir / "INVALID_CONCURRENT_EXECUTION.json").exists() and not allow_invalid:
        raise ValueError(f"run is marked INVALID_CONCURRENT_EXECUTION: {run_dir}")
    return manifest
