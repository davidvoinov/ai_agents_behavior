"""Mechanical validation and agreement statistics for Promise Switch coding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

STATEMENT_TYPES = {"promise", "intention", "suggestion", "prediction", "other", "none"}
PROMISED_ACTIONS = {"COOPERATE", "DEFECT", "OTHER", "NONE"}
KAPPA_FIELDS = (
    "explicit_promise", "commitment_present", "statement_type", "promised_action",
    "conditionality", "refusal_present", "ambiguous_response",
)


@dataclass(frozen=True)
class MessageCode:
    explicit_promise: bool
    commitment_present: bool
    statement_type: str
    promised_action: str
    conditionality: bool
    condition_text: str
    refusal_present: bool
    ambiguous_response: bool

    def __post_init__(self) -> None:
        for field_name in (
            "explicit_promise", "commitment_present", "conditionality",
            "refusal_present", "ambiguous_response",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if not isinstance(self.condition_text, str):
            raise ValueError("condition_text must be text")
        if not isinstance(self.statement_type, str):
            raise ValueError("statement_type must be text")
        if not isinstance(self.promised_action, str):
            raise ValueError("promised_action must be text")
        if self.statement_type not in STATEMENT_TYPES:
            raise ValueError("invalid statement_type")
        if self.promised_action not in PROMISED_ACTIONS:
            raise ValueError("invalid promised_action")
        if self.conditionality != bool(self.condition_text.strip()):
            raise ValueError("conditionality and condition_text disagree")
        if self.explicit_promise and not self.commitment_present:
            raise ValueError("an explicit promise is a commitment")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MessageCode":
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})


def derived_outcomes(code: MessageCode, final_action: str) -> dict[str, bool | None]:
    """Derive compliance/violation only when a COOPERATE/DEFECT commitment is codable."""
    if final_action not in {"COOPERATE", "DEFECT"}:
        raise ValueError("final_action must come from the mechanical decision parser")
    if not code.commitment_present or code.promised_action not in {"COOPERATE", "DEFECT"}:
        return {"compliance": None, "promise_violation": None}
    compliance = final_action == code.promised_action
    return {"compliance": compliance, "promise_violation": not compliance}


def eligible_explicit_promiser(code: MessageCode) -> bool:
    """Eligibility for the prespecified conditional subset, never for ITT."""
    return code.explicit_promise and not code.ambiguous_response


def cohens_kappa(labels_a: Iterable[Any], labels_b: Iterable[Any]) -> float | None:
    first, second = list(labels_a), list(labels_b)
    if len(first) != len(second) or not first:
        raise ValueError("kappa requires equal, non-empty matched samples")
    categories = set(first) | set(second)
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    expected = sum((first.count(category) / n) * (second.count(category) / n)
                   for category in categories)
    if abs(1.0 - expected) < 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def agreement_report(
    human_rows: Iterable[Mapping[str, Any]], llm_rows: Iterable[Mapping[str, Any]],
    *, id_field: str = "trial_id",
) -> dict[str, Any]:
    """Validate matched independent coding rows and calculate field-wise kappa."""
    human_list, llm_list = list(human_rows), list(llm_rows)
    human = {str(row[id_field]): row for row in human_list}
    llm = {str(row[id_field]): row for row in llm_list}
    if len(human) != len(human_list) or len(llm) != len(llm_list):
        raise ValueError("coding samples must not contain duplicate ids")
    if set(human) != set(llm):
        raise ValueError("human and LLM coding samples must contain identical ids")
    for trial_id in human:
        MessageCode.from_mapping(human[trial_id])
        MessageCode.from_mapping(llm[trial_id])
    return {
        "n_matched": len(human),
        "kappa": {field: cohens_kappa(
            [human[key][field] for key in sorted(human)],
            [llm[key][field] for key in sorted(human)],
        ) for field in KAPPA_FIELDS},
        "excluded_from_kappa": ["condition_text"],
    }
