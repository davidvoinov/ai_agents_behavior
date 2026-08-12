"""Strict structural parsers; no semantic coding or response repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class DecisionParse:
    valid: bool
    decision: Literal["COOPERATE", "DEFECT"] | None
    justification: str
    raw_response: str
    error: str | None


@dataclass(frozen=True)
class MessageParse:
    valid: bool
    message: str | None
    raw_response: str
    error: str | None


def parse_decision(raw_response: str) -> DecisionParse:
    payload, error = _json_object(raw_response)
    if payload is None:
        return DecisionParse(False, None, "", raw_response, error)
    decision = payload.get("decision")
    justification = payload.get("justification")
    if decision not in {"COOPERATE", "DEFECT"}:
        return DecisionParse(False, None, justification if isinstance(justification, str) else "",
                             raw_response,
                             "decision must be exactly COOPERATE or DEFECT")
    if not isinstance(justification, str) or not justification.strip():
        return DecisionParse(False, None, "", raw_response,
                             "justification must be a non-empty string")
    return DecisionParse(True, decision, justification, raw_response, None)


def parse_message(raw_response: str) -> MessageParse:
    payload, error = _json_object(raw_response)
    if payload is None:
        return MessageParse(False, None, raw_response, error)
    message = payload.get("message")
    if not isinstance(message, str):
        return MessageParse(False, None, raw_response, "message must be a string")
    return MessageParse(True, message, raw_response, None)


def _json_object(raw_response: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(raw_response, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, _DuplicateKeyError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "response must be one JSON object"
    return value, None


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value
