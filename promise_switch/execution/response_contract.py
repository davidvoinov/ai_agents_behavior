"""Mechanical response-contract failure classes for operator review."""

from __future__ import annotations

import ast
import json
import re
from typing import Literal

from ..parser import parse_decision, parse_message

Stage = Literal["message", "final_decision"]


def parser_failure_class(stage: Stage, raw_response: str) -> str:
    """Return a stage-specific mechanical class without interpreting meaning."""
    mechanism = _mechanical_failure(stage, raw_response)
    return f"{stage}:{mechanism}"


def submitted_decision_is_contradictory(raw_response: str) -> bool:
    """Detect both actions only in the submitted decision, never justification text."""
    candidate: object
    try:
        payload = json.loads(raw_response, object_pairs_hook=lambda pairs: pairs)
    except json.JSONDecodeError:
        candidate = raw_response.strip()
        if any(character in candidate for character in "{}[]\"'`\n"):
            return False
    else:
        if not (
            isinstance(payload, list)
            and all(isinstance(pair, tuple) and len(pair) == 2 for pair in payload)
        ):
            return False
        submitted = [value for key, value in payload if key == "decision"]
        candidate = " and ".join(submitted) if all(
            isinstance(value, str) for value in submitted
        ) else None
    if not isinstance(candidate, str):
        return False
    framing = r"(?:(?:decision|final decision|action|final action)\s*:\s*|i\s+(?:choose|select)\s+)?"
    connector = r"(?:and|or|/|,|&|\+|-)"
    action_pair = rf"(?:COOPERATE\s*{connector}\s*DEFECT|DEFECT\s*{connector}\s*COOPERATE)"
    return bool(re.fullmatch(
        rf"\s*{framing}{action_pair}\s*[.!]?\s*", candidate, flags=re.IGNORECASE,
    ))


def _mechanical_failure(stage: Stage, raw_response: str) -> str:
    parser = parse_message if stage == "message" else parse_decision
    fenced = re.fullmatch(r"```(?:json)?\r?\n([\s\S]*)\r?\n```", raw_response)
    if fenced:
        return (
            "markdown_fenced_valid_json"
            if parser(fenced.group(1)).valid
            else "markdown_fenced_malformed_json"
        )

    decoder = json.JSONDecoder()
    stripped = raw_response.lstrip()
    try:
        first, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        pass
    else:
        prefix = stripped[:end]
        remainder = stripped[end:].strip()
        if remainder:
            try:
                second, second_end = decoder.raw_decode(remainder)
            except json.JSONDecodeError:
                if parser(prefix).valid:
                    return "trailing_prose_after_valid_json"
            else:
                if (
                    not remainder[second_end:].strip()
                    and isinstance(first, dict)
                    and isinstance(second, dict)
                ):
                    return "multiple_json_objects"
                if parser(prefix).valid:
                    return "trailing_prose_after_valid_json"

    candidate = raw_response.strip()
    if candidate.startswith("{") and candidate.endswith("}") and "'" in candidate:
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            pass
        else:
            if (
                isinstance(value, dict)
                and not re.search(r"\b(?:True|False|None)\b", raw_response)
                and parser(json.dumps(value, ensure_ascii=False)).valid
            ):
                return "single_quoted_json"

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return (
            "actual_contradictory_decision"
            if stage == "final_decision" and submitted_decision_is_contradictory(raw_response)
            else "malformed_json"
        )
    if not isinstance(payload, dict):
        return "non_object_json"
    if stage == "message":
        if "message" not in payload:
            return (
                "wrong_schema"
                if payload.keys() & {"action", "decision"}
                else "missing_required_field"
            )
        if not isinstance(payload["message"], str):
            return "invalid_field_type"
    else:
        if submitted_decision_is_contradictory(raw_response):
            return "actual_contradictory_decision"
        if "decision" not in payload and payload.keys() & {"action", "message"}:
            return "wrong_schema"
        if "decision" not in payload or "justification" not in payload:
            return "missing_required_field"
        if (
            not isinstance(payload["decision"], str)
            or not isinstance(payload["justification"], str)
        ):
            return "invalid_field_type"
        if payload["decision"] not in {"COOPERATE", "DEFECT"}:
            return "invalid_decision_value"
        if not payload["justification"].strip():
            return "invalid_justification"
    return "other_unrecognized_parser_failure"
