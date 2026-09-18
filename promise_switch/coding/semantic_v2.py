"""Versioned extension policy; the frozen v1 corpus/codebook remain immutable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

SEMANTIC_CORPUS_VERSION = "promise_switch_semantic_corpus_v2"
FROZEN_BASE_CORPUS_SIZE = 2738
FROZEN_BASE_VALIDATION_SIZE = 130
SUPPLEMENTAL_VALIDATION_VERSION = "promise_switch_validation_supplement_v1_n40"
GPT_LUNA_ROUTING_POLICIES = ("narrow", "conservative")
GptLunaRoutingPolicy = Literal["narrow", "conservative"]
FROZEN_CODEBOOK_HASHES = {
    "codebook_markdown_sha256": "55824b137e669d63e2e475036429864ddf5f15f0d7c7916e3d6c9d06c3c87e42",
    "python_validator_sha256": "b3dde96140130f581dd9f71022f2607eec5fae44b9c9f40ef38ef03cddad9d6f",
    "judge_prompt_sha256": "27b0948801436fa6c52c6f5add1730220e83a4a4a6113bfb78cf6a34f00c5cb8",
    "annotation_schema_sha256": "24edc7b55bc558e3f35e84ae306a5719ffb405c0f23e8758a138d1c059ffa31f",
}


def model_family(*values: object) -> str:
    joined = " ".join(str(value).lower() for value in values)
    if "gpt-5.6" in joined or "gpt_5_6" in joined or "luna" in joined:
        return "gpt_5_6_luna"
    if "gpt-oss" in joined or "gpt_oss" in joined:
        return "gpt_oss"
    if "openai" in joined:
        return "openai"
    if "claude" in joined or "anthropic" in joined:
        return "claude"
    return "other"


def judge_family_allowed(
    *, source_family: str, judge_family: str,
    gpt_luna_policy: GptLunaRoutingPolicy | None,
) -> bool:
    """Enforce explicit GPT-Luna methodology rather than silently choosing it."""
    if source_family == "gpt_5_6_luna":
        if gpt_luna_policy not in GPT_LUNA_ROUTING_POLICIES:
            raise ValueError(
                "GPT-5.6 Luna semantic routing requires explicit narrow or conservative policy"
            )
        if gpt_luna_policy == "conservative":
            return judge_family not in {"gpt_5_6_luna", "gpt_oss", "openai"}
        return judge_family != "gpt_5_6_luna"
    return source_family != judge_family


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_codebook(
    *, codebook_path: Path, validator_path: Path, judge_prompt_path: Path,
    annotation_schema_path: Path, frozen_key_path: Path,
) -> dict[str, str]:
    """Verify the recorded v1 key and every live frozen coding component."""
    key = json.loads(frozen_key_path.read_text(encoding="utf-8"))
    recorded = key.get("file_hashes", {})
    observed = {
        "codebook_markdown_sha256": sha256_path(codebook_path),
        "python_validator_sha256": sha256_path(validator_path),
        "judge_prompt_sha256": sha256_path(judge_prompt_path),
        "annotation_schema_sha256": sha256_path(annotation_schema_path),
    }
    if any(recorded.get(name) != expected for name, expected in FROZEN_CODEBOOK_HASHES.items()):
        raise ValueError("frozen semantic key contains an unexpected codebook hash")
    if observed != FROZEN_CODEBOOK_HASHES:
        raise ValueError("frozen Promise Switch codebook component hash mismatch")
    return observed


def validate_v2_judge_route(
    manifest: Mapping[str, Any], *, judge_provider: str, judge_model: str,
) -> None:
    """Enforce manifest-declared source/judge separation before judge execution."""
    policy = manifest.get("gpt_luna_routing_policy")
    if policy not in GPT_LUNA_ROUTING_POLICIES:
        raise ValueError("semantic-v2 manifest lacks an explicit GPT-Luna routing policy")
    judge_family = model_family(judge_provider, judge_model)
    source_families = manifest.get("source_model_families")
    if not isinstance(source_families, list):
        raise ValueError("semantic-v2 manifest lacks source model families")
    for source_family in source_families:
        if not judge_family_allowed(
            source_family=str(source_family), judge_family=judge_family,
            gpt_luna_policy=policy,
        ):
            raise ValueError(
                f"semantic-v2 judge family {judge_family} is forbidden for source family "
                f"{source_family} under {policy} policy"
            )
