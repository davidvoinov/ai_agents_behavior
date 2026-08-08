"""Serializable typed records for the Promise Switch foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

from .design import Condition, MessageRegime, OwnPromiseRegime

ReasoningMode = Literal["on", "off", "unavailable"]
CompletionStatus = Literal["planned", "completed", "failed"]
FailureCategory = Literal["infrastructure", "behavioral"] | None


class SerializableRecord:
    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_serialization(asdict(self))


_SECRET_KEY_MARKERS = ("api_key", "authorization", "secret", "access_token")


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).lower()
    if normalized.endswith("_env"):
        return False
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def sanitize_for_serialization(value: Any) -> Any:
    """Remove secret-bearing metadata keys from nested serializable values."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_secret_key(key)
            else sanitize_for_serialization(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_serialization(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_serialization(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True)
class ExperimentConfig(SerializableRecord):
    experiment_version: str
    prompt_version: str
    seed: int
    temperature: float
    message_max_tokens: int
    decision_max_tokens: int


@dataclass(frozen=True)
class ModelConfig(SerializableRecord):
    provider: str
    model_alias: str
    model_id: str
    base_url: str
    api_key_env: str
    reasoning_capability: Literal["toggle", "fixed_on", "fixed_off", "unavailable"]
    reasoning_mode: ReasoningMode
    provider_request_kwargs: dict[str, Any] = field(default_factory=dict)
    rate_limit_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialPlan(SerializableRecord):
    trial_id: str
    run_id: str
    randomization_seed: int
    voluntary_pool: bool
    condition: Condition | None = None

    def __post_init__(self) -> None:
        if self.voluntary_pool == (self.condition is not None):
            raise ValueError("A plan is either unassigned voluntary or one fixed condition")


@dataclass(frozen=True)
class MessageEvent(SerializableRecord):
    message_event_id: str
    trial_id: str
    timestamp: str
    raw_message: str
    message_recorded: bool

    def __post_init__(self) -> None:
        if self.message_recorded and not self.message_event_id.strip():
            raise ValueError("A recorded message event requires an id")


@dataclass(frozen=True)
class AssignmentRecord(SerializableRecord):
    assignment_id: str
    trial_id: str
    randomization_seed: int
    condition: Condition
    message_event_id: str | None
    original_partner_id: str
    final_partner_id: str
    switched: bool


@dataclass(frozen=True)
class ApiAttempt(SerializableRecord):
    attempt_id: str
    trial_id: str
    stage: Literal["message", "final_decision"]
    attempt_number: int
    timestamp: str
    provider: str
    model: str
    status: str
    error_metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""


@dataclass(frozen=True)
class NormalizedResponse(SerializableRecord):
    status: str
    raw_response: str
    provider_response_id: str | None = None
    provider_request_id: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    latency_seconds: float | None = None
    retry_after_seconds: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult(SerializableRecord):
    trial_id: str
    run_id: str
    experiment_version: str
    prompt_version: str
    randomization_seed: int
    condition: Condition
    model: str
    provider: str
    reasoning_mode: ReasoningMode
    message_regime: MessageRegime
    assigned_expectation: bool
    original_partner_id: str
    final_partner_id: str
    switched: bool
    own_promise_regime: OwnPromiseRegime
    final_action: Literal["COOPERATE", "DEFECT"] | None
    completion_status: CompletionStatus
    failure_category: FailureCategory
    failure_reason: str | None
    raw_message: str = ""
    message_recorded: bool = False
    message_event_id: str | None = None
    raw_final_response: str = ""
