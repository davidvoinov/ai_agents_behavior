"""Integrity-gated identities for the four registered Promise Switch extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import validate_run_integrity
from .version import PROMPT_VERSION, RESPONSE_QC_VERSION, prompt_templates_hash


FUTURE_CONFIGURATIONS: tuple[dict[str, str], ...] = (
    {
        "label": "GPT-5.6 Luna — no-think", "family": "gpt_5_6_luna",
        "model_id": "gpt-5.6-luna", "model_alias": "gpt_5_6_luna_openai",
        "provider": "openai", "reasoning_mode": "off",
    },
    {
        "label": "GPT-5.6 Luna — think, medium", "family": "gpt_5_6_luna",
        "model_id": "gpt-5.6-luna", "model_alias": "gpt_5_6_luna_openai",
        "provider": "openai", "reasoning_mode": "on",
    },
    {
        "label": "Claude Sonnet 5 — no-think", "family": "claude",
        "model_id": "claude-sonnet-5", "model_alias": "claude_sonnet_5_anthropic",
        "provider": "anthropic", "reasoning_mode": "off",
    },
    {
        "label": "Claude Sonnet 5 — think, adaptive xHigh", "family": "claude",
        "model_id": "claude-sonnet-5", "model_alias": "claude_sonnet_5_anthropic",
        "provider": "anthropic", "reasoning_mode": "on",
    },
)


def future_configuration(label: str) -> dict[str, str]:
    matches = [row for row in FUTURE_CONFIGURATIONS if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown future Promise Switch configuration: {label}")
    return dict(matches[0])


def audit_future_run(label: str, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a completed confirmatory run and its registered immutable identity."""
    expected = future_configuration(label)
    resolved = run_dir.resolve()
    manifest = validate_run_integrity(
        resolved, required_dataset_role="confirmatory", require_completed=True,
    )
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"future run prompt version mismatch: {resolved}")
    if manifest.get("prompt_hash") != prompt_templates_hash():
        raise ValueError(f"future run prompt hash mismatch: {resolved}")
    if manifest.get("response_qc_version") != RESPONSE_QC_VERSION:
        raise ValueError(f"future run QC version mismatch: {resolved}")
    if manifest.get("requested_completed_trials_per_final_condition") != 100:
        raise ValueError(f"future run completed-cell target mismatch: {resolved}")
    configurations = manifest.get("model_configurations")
    if not isinstance(configurations, list) or len(configurations) != 1:
        raise ValueError(f"future run must contain exactly one model configuration: {resolved}")
    config = configurations[0]
    if not isinstance(config, dict):
        raise ValueError(f"future run model configuration is malformed: {resolved}")
    observed = {
        name: config.get(name)
        for name in ("model_id", "model_alias", "provider", "reasoning_mode")
    }
    required = {name: expected[name] for name in observed}
    if observed != required:
        raise ValueError(f"future run model/provider/reasoning identity mismatch: {resolved}")
    return manifest, expected
