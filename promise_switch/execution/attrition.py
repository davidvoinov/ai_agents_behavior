"""Promise Switch-specific infrastructure and behavioral taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..parser import DecisionParse, MessageParse
from ..schemas import NormalizedResponse

InfrastructureCause = Literal[
    "rate_limit_transient", "quota_exhausted", "timeout", "connection_error",
    "provider_error", "model_unavailable", "token_truncation",
    "malformed_api_response", "other_infrastructure",
]
BehavioralCause = Literal[
    "refusal_to_decide", "invalid_output_format", "contradictory_action",
    "unclassifiable_action", "infeasible_action", "other_behavioral",
]


@dataclass(frozen=True)
class AttritionOutcome:
    category: Literal["success", "infrastructure", "behavioral"]
    cause: str | None
    retryable: bool
    quota_pause: bool = False


def classify_transport(response: NormalizedResponse) -> AttritionOutcome:
    if response.finish_reason in {"length", "max_tokens"}:
        return AttritionOutcome("infrastructure", "token_truncation", True)
    if response.status == "success":
        return AttritionOutcome("success", None, False)
    text = " ".join(
        str(value) for value in [response.raw_response, *response.error_metadata.values()]
    ).lower()
    if response.status == "malformed_api_response":
        return AttritionOutcome("infrastructure", "malformed_api_response", True)
    if response.http_status == 429:
        if any(marker in text for marker in (
            "quota", "credit", "insufficient balance", "billing", "hard limit", "exhausted",
        )):
            return AttritionOutcome("infrastructure", "quota_exhausted", False, True)
        return AttritionOutcome("infrastructure", "rate_limit_transient", True)
    if response.http_status in {401, 403, 404}:
        cause = "model_unavailable" if response.http_status == 404 else "provider_error"
        return AttritionOutcome("infrastructure", cause, False)
    if response.status == "timeout":
        return AttritionOutcome("infrastructure", "timeout", True)
    if response.status == "connection_error":
        return AttritionOutcome("infrastructure", "connection_error", True)
    if response.http_status and response.http_status >= 500:
        return AttritionOutcome("infrastructure", "provider_error", True)
    return AttritionOutcome("infrastructure", "other_infrastructure", True)


def classify_decision_output(parsed: DecisionParse) -> AttritionOutcome:
    if parsed.valid:
        return AttritionOutcome("success", None, False)
    lowered = parsed.raw_response.lower()
    if any(marker in lowered for marker in ("i cannot", "i can't", "i refuse", "cannot decide")):
        return AttritionOutcome("behavioral", "refusal_to_decide", False)
    if "cooperate" in lowered and "defect" in lowered:
        return AttritionOutcome("behavioral", "contradictory_action", False)
    if parsed.error and "decision must be exactly" in parsed.error:
        return AttritionOutcome("behavioral", "unclassifiable_action", False)
    return AttritionOutcome("behavioral", "invalid_output_format", False)


def classify_message_output(parsed: MessageParse) -> AttritionOutcome:
    if parsed.valid:
        return AttritionOutcome("success", None, False)
    return AttritionOutcome("behavioral", "invalid_output_format", False)
