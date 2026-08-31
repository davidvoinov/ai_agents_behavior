"""Validation and agreement mechanics for the frozen Promise Switch codebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

CODEBOOK_VERSION = "promise_switch_message_codebook_v1"

TERNARY_VALUES = {"yes", "no", "unclear"}
STATEMENT_TYPES = {
    "explicit_promise", "commitment", "intention", "proposal", "refusal",
    "no_substantive_statement", "ambiguous",
}
PROMISED_ACTIONS = {"COOPERATE", "DEFECT", "OTHER", "UNCLEAR", "NOT_APPLICABLE"}
CONDITIONALITY_VALUES = {"unconditional", "conditional", "unclear", "not_applicable"}
REFUSAL_VALUES = {
    "refusal_to_promise", "refusal_to_cooperate", "other_refusal", "no_refusal", "unclear",
}
AMBIGUITY_VALUES = {"yes", "no"}

SEMANTIC_FIELDS = (
    "explicit_promise", "commitment_present", "statement_type", "promised_action",
    "conditionality", "condition_text", "refusal_present", "ambiguous_response",
)
KAPPA_FIELDS = tuple(field for field in SEMANTIC_FIELDS if field != "condition_text")
ANNOTATION_REQUIRED_FIELDS = (
    "item_id", "message", "codebook_version", *SEMANTIC_FIELDS,
)
ANNOTATION_OPTIONAL_FIELDS = ("coder_id", "notes")


@dataclass(frozen=True)
class MessageCode:
    explicit_promise: str
    commitment_present: str
    statement_type: str
    promised_action: str
    conditionality: str
    condition_text: str
    refusal_present: str
    ambiguous_response: str

    def __post_init__(self) -> None:
        _require_category("explicit_promise", self.explicit_promise, TERNARY_VALUES)
        _require_category("commitment_present", self.commitment_present, TERNARY_VALUES)
        _require_category("statement_type", self.statement_type, STATEMENT_TYPES)
        _require_category("promised_action", self.promised_action, PROMISED_ACTIONS)
        _require_category("conditionality", self.conditionality, CONDITIONALITY_VALUES)
        _require_category("refusal_present", self.refusal_present, REFUSAL_VALUES)
        _require_category("ambiguous_response", self.ambiguous_response, AMBIGUITY_VALUES)
        if not isinstance(self.condition_text, str):
            raise ValueError("condition_text must be text")

        if self.conditionality == "conditional":
            if not self.condition_text.strip():
                raise ValueError("conditional coding requires nonempty condition_text")
        elif self.condition_text != "":
            raise ValueError("non-conditional coding requires empty condition_text")

        if self.explicit_promise == "yes":
            if self.commitment_present != "yes":
                raise ValueError("an explicit promise is also a commitment")
            if self.statement_type not in {"explicit_promise", "ambiguous"}:
                raise ValueError("explicit_promise=yes conflicts with statement_type")
        if self.statement_type == "explicit_promise" and (
            self.explicit_promise != "yes" or self.commitment_present != "yes"
        ):
            raise ValueError("statement_type=explicit_promise requires explicit commitment")
        if self.statement_type == "commitment" and (
            self.explicit_promise != "no" or self.commitment_present != "yes"
        ):
            raise ValueError("statement_type=commitment requires a non-promissory commitment")
        if self.statement_type in {
            "intention", "proposal", "refusal", "no_substantive_statement",
        } and (self.explicit_promise != "no" or self.commitment_present != "no"):
            raise ValueError("statement_type conflicts with promise or commitment coding")

        undertaking_present = (
            self.explicit_promise == "yes" or self.commitment_present == "yes"
        )
        undertaking_absent = (
            self.explicit_promise == "no" and self.commitment_present == "no"
        )
        if undertaking_present and self.promised_action == "NOT_APPLICABLE":
            raise ValueError("a promise or commitment requires a codable promised_action")
        if undertaking_present and self.conditionality == "not_applicable":
            raise ValueError("a promise or commitment requires applicable conditionality")
        if undertaking_absent and self.promised_action != "NOT_APPLICABLE":
            raise ValueError("promised_action must be NOT_APPLICABLE without an undertaking")
        if self.statement_type == "intention" and self.conditionality == "not_applicable":
            raise ValueError("an intention requires applicable conditionality")
        if self.statement_type in {"proposal", "refusal"} and (
            self.conditionality != "not_applicable"
        ):
            raise ValueError("a proposal or refusal requires conditionality=not_applicable")

        if self.statement_type == "no_substantive_statement" and (
            self.conditionality != "not_applicable"
            or self.refusal_present != "no_refusal"
            or self.ambiguous_response != "no"
        ):
            raise ValueError("no_substantive_statement conflicts with other semantic content")
        if self.statement_type == "refusal" and self.refusal_present == "no_refusal":
            raise ValueError("statement_type=refusal requires explicit refusal content")
        if self.statement_type == "ambiguous" and self.ambiguous_response != "yes":
            raise ValueError("statement_type=ambiguous requires ambiguous_response=yes")

        uncertainty_values = (
            self.explicit_promise, self.commitment_present, self.promised_action,
            self.conditionality, self.refusal_present,
        )
        if any(value in {"unclear", "UNCLEAR"} for value in uncertainty_values):
            if self.ambiguous_response != "yes":
                raise ValueError("unclear semantic coding requires ambiguous_response=yes")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MessageCode":
        return cls(**{field: value[field] for field in SEMANTIC_FIELDS})


def validate_annotation_record(value: Mapping[str, Any]) -> MessageCode:
    """Validate one strict, outcome-blind human or LLM annotation record."""
    keys = set(value)
    required = set(ANNOTATION_REQUIRED_FIELDS)
    allowed = required | set(ANNOTATION_OPTIONAL_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    if missing:
        raise ValueError(f"annotation is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"annotation contains unsupported fields: {unknown}")
    if value["codebook_version"] != CODEBOOK_VERSION:
        raise ValueError("annotation codebook_version mismatch")
    if not isinstance(value["item_id"], str) or not value["item_id"].strip():
        raise ValueError("item_id must be nonempty text")
    if not isinstance(value["message"], str):
        raise ValueError("message must be text")
    if "coder_id" in value and (
        not isinstance(value["coder_id"], str) or not value["coder_id"].strip()
    ):
        raise ValueError("coder_id must be nonempty text when supplied")
    if "notes" in value and not isinstance(value["notes"], str):
        raise ValueError("notes must be text when supplied")
    return MessageCode.from_mapping(value)


def derived_outcomes(code: MessageCode, final_action: str) -> dict[str, bool | None]:
    """Derive compliance/violation only from a clearly coded binary undertaking."""
    if final_action not in {"COOPERATE", "DEFECT"}:
        raise ValueError("final_action must come from the mechanical decision parser")
    if code.commitment_present != "yes" or code.promised_action not in {
        "COOPERATE", "DEFECT",
    }:
        return {"compliance": None, "promise_violation": None}
    compliance = final_action == code.promised_action
    return {"compliance": compliance, "promise_violation": not compliance}


def eligible_explicit_promiser(code: MessageCode) -> bool:
    """Return exact eligibility for the voluntary promise-conditioned subset."""
    return (
        code.explicit_promise == "yes"
        and code.promised_action == "COOPERATE"
        and code.conditionality == "unconditional"
        and code.ambiguous_response == "no"
    )


def cohens_kappa(labels_a: Iterable[Any], labels_b: Iterable[Any]) -> float | None:
    first, second = _matched_labels(labels_a, labels_b, statistic="kappa")
    categories = set(first) | set(second)
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    expected = sum((first.count(category) / n) * (second.count(category) / n)
                   for category in categories)
    if abs(1.0 - expected) < 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def raw_agreement_percent(labels_a: Iterable[Any], labels_b: Iterable[Any]) -> float:
    """Return matched raw agreement as a percentage from 0 through 100."""
    first, second = _matched_labels(labels_a, labels_b, statistic="agreement")
    return 100.0 * sum(a == b for a, b in zip(first, second)) / len(first)


def agreement_report(
    human_rows: Iterable[Mapping[str, Any]], llm_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate matched versioned annotations and report field-wise agreement."""
    human_list, llm_list = list(human_rows), list(llm_rows)
    human = {str(row["item_id"]): row for row in human_list}
    llm = {str(row["item_id"]): row for row in llm_list}
    if len(human) != len(human_list) or len(llm) != len(llm_list):
        raise ValueError("coding samples must not contain duplicate ids")
    if set(human) != set(llm):
        raise ValueError("human and LLM coding samples must contain identical ids")
    for item_id in human:
        validate_annotation_record(human[item_id])
        validate_annotation_record(llm[item_id])
        if human[item_id]["message"] != llm[item_id]["message"]:
            raise ValueError("matched coders must receive identical message text")
    ordered_ids = sorted(human)
    return {
        "codebook_version": CODEBOOK_VERSION,
        "n_matched": len(human),
        "kappa": {field: cohens_kappa(
            [human[key][field] for key in ordered_ids],
            [llm[key][field] for key in ordered_ids],
        ) for field in KAPPA_FIELDS},
        "raw_agreement_percent": {field: raw_agreement_percent(
            [human[key][field] for key in ordered_ids],
            [llm[key][field] for key in ordered_ids],
        ) for field in KAPPA_FIELDS},
        "excluded_from_agreement_statistics": ["condition_text"],
    }


def _require_category(field: str, value: Any, allowed: set[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"invalid {field}")


def _matched_labels(
    labels_a: Iterable[Any], labels_b: Iterable[Any], *, statistic: str,
) -> tuple[list[Any], list[Any]]:
    first, second = list(labels_a), list(labels_b)
    if len(first) != len(second) or not first:
        raise ValueError(f"{statistic} requires equal, non-empty matched samples")
    return first, second
