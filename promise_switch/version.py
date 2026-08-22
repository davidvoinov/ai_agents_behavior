"""Semantic version identifiers for the Promise Switch foundation."""

from __future__ import annotations

import hashlib
import json

EXPERIMENT_VERSION = "promise_switch_v1"
PROMPT_VERSION = "promise_switch_v1"
RESPONSE_QC_VERSION = "promise_switch_response_qc_v3"
KNOWN_RESPONSE_QC_VERSIONS = {
    "promise_switch_response_qc_v2",
    RESPONSE_QC_VERSION,
}


def prompt_templates_hash() -> str:
    """Return a deterministic SHA-256 hash of all model-facing templates."""
    from .prompts import experiment_facing_templates

    canonical = json.dumps(
        experiment_facing_templates(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
