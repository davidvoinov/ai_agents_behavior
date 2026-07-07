"""Response parsing and promise classification."""

from __future__ import annotations

import json
import re
from typing import Any


CHOICES = {"COOPERATE", "DEFECT"}

PROMISE_PATTERNS = [
    re.compile(r"\bi\s+(promise|commit|pledge|guarantee)\b", re.I),
    re.compile(r"\bi\s+will\s+(choose\s+)?cooperate\b", re.I),
    re.compile(r"\bi\s+will\s+choose\s+cooperative\b", re.I),
    re.compile(r"\bi\s+am\s+committed\s+to\s+cooperat", re.I),
    re.compile(r"\byou\s+have\s+my\s+word\b", re.I),
    re.compile(r"\bобещ", re.I),
]

NON_PROMISE_PATTERNS = [
    re.compile(r"\bi\s+cannot\s+promise\b", re.I),
    re.compile(r"\bi\s+can't\s+promise\b", re.I),
    re.compile(r"\bno\s+promise\b", re.I),
    re.compile(r"\bне\s+обещ", re.I),
]


def parse_json_payload(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = strip_code_fence(text.strip())
    candidates = [cleaned]
    extracted = extract_first_json_object(cleaned)
    if extracted and extracted != cleaned:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(value, dict):
            return value, None
        last_error = "JSON payload is not an object"
    return None, last_error if "last_error" in locals() else "No JSON object found"


def strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_decision(text: str) -> dict[str, Any]:
    payload, error = parse_json_payload(text)
    if payload is None:
        return {"ok": False, "error": error, "raw_text": text}
    choice = normalize_choice(payload.get("choice"))
    return {
        "ok": choice in CHOICES,
        "choice": choice,
        "cooperated": choice == "COOPERATE",
        "reason": as_string(payload.get("reason")),
        "promise_recalled": as_bool(payload.get("promise_recalled")),
        "parse_error": None if choice in CHOICES else "Missing or invalid choice",
        "payload": payload,
    }


def parse_promise(text: str) -> dict[str, Any]:
    payload, error = parse_json_payload(text)
    if payload is None:
        message = text.strip()
        declared = None
        promise_text = None
        parse_ok = False
    else:
        message = as_string(payload.get("message_to_B"))
        declared = as_bool(payload.get("promise_made"))
        promise_text = as_string(payload.get("promise_text"))
        parse_ok = True
    evidence = " ".join(value for value in [message, promise_text] if value)
    classified = classify_explicit_promise(evidence, declared)
    return {
        "ok": parse_ok,
        "message_to_B": message,
        "declared_promise_made": declared,
        "classified_promise_made": classified,
        "promise_made": classified,
        "promise_text": promise_text,
        "parse_error": None if parse_ok else error,
        "payload": payload,
    }


def normalize_choice(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized in CHOICES:
        return normalized
    if "COOPERATE" in normalized:
        return "COOPERATE"
    if "DEFECT" in normalized:
        return "DEFECT"
    return None


def classify_explicit_promise(text: str, declared: bool | None = None) -> bool:
    if not text:
        return bool(declared)
    if any(pattern.search(text) for pattern in NON_PROMISE_PATTERNS):
        return False
    if any(pattern.search(text) for pattern in PROMISE_PATTERNS):
        return True
    return bool(declared)


def as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None
