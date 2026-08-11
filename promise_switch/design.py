"""Frozen researcher-side scientific design; labels never enter prompts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Condition(str, Enum):
    NO_COMMUNICATION = "no_communication"
    VOLUNTARY_PROMISE_NO_SWITCH = "voluntary_promise_no_switch"
    VOLUNTARY_PROMISE_SWITCH = "voluntary_promise_switch"
    INDUCED_PROMISE_NO_SWITCH = "induced_promise_no_switch"
    INDUCED_PROMISE_SWITCH = "induced_promise_switch"
    EXPECTATION_WITHOUT_OWN_PROMISE = "expectation_without_own_promise"


# Internal voluntary identifiers are retained for artifact compatibility.  These
# labels are the scientific/user-facing terminology and do not enter prompts.
SCIENTIFIC_CONDITION_LABELS = {
    Condition.NO_COMMUNICATION: "No Communication",
    Condition.VOLUNTARY_PROMISE_NO_SWITCH: "Voluntary Message--No-Switch",
    Condition.VOLUNTARY_PROMISE_SWITCH: "Voluntary Message--Switch",
    Condition.INDUCED_PROMISE_NO_SWITCH: "Induced Promise--No-Switch",
    Condition.INDUCED_PROMISE_SWITCH: "Induced Promise--Switch",
    Condition.EXPECTATION_WITHOUT_OWN_PROMISE: "Expectation Without Own Promise",
}


def scientific_condition_label(condition: Condition) -> str:
    """Return manuscript terminology without changing stable artifact identifiers."""
    return SCIENTIFIC_CONDITION_LABELS[condition]


class MessageRegime(str, Enum):
    NONE = "none"
    VOLUNTARY = "voluntary"
    INDUCED = "induced"


class OwnPromiseRegime(str, Enum):
    NONE = "none"
    VOLUNTARY_OPPORTUNITY = "voluntary_opportunity"
    INDUCED = "induced"


PAYOFFS = {
    "COOPERATE": {"participant_a": 8, "current_partner": 8},
    "DEFECT": {"participant_a": 10, "current_partner": 0},
}


@dataclass(frozen=True)
class TreatmentMetadata:
    condition: Condition
    message_regime: MessageRegime
    assigned_expectation: bool
    original_partner_id: str
    final_partner_id: str
    switched: bool
    own_promise_regime: OwnPromiseRegime


_BASE_TREATMENTS = {
    Condition.NO_COMMUNICATION: (MessageRegime.NONE, False, False, OwnPromiseRegime.NONE),
    Condition.VOLUNTARY_PROMISE_NO_SWITCH: (
        MessageRegime.VOLUNTARY, False, False, OwnPromiseRegime.VOLUNTARY_OPPORTUNITY,
    ),
    Condition.VOLUNTARY_PROMISE_SWITCH: (
        MessageRegime.VOLUNTARY, False, True, OwnPromiseRegime.VOLUNTARY_OPPORTUNITY,
    ),
    Condition.INDUCED_PROMISE_NO_SWITCH: (
        MessageRegime.INDUCED, True, False, OwnPromiseRegime.INDUCED,
    ),
    Condition.INDUCED_PROMISE_SWITCH: (
        MessageRegime.INDUCED, True, True, OwnPromiseRegime.INDUCED,
    ),
    Condition.EXPECTATION_WITHOUT_OWN_PROMISE: (
        MessageRegime.NONE, True, False, OwnPromiseRegime.NONE,
    ),
}


def treatment_metadata(condition: Condition, trial_id: str) -> TreatmentMetadata:
    message_regime, expectation, switched, own_promise = _BASE_TREATMENTS[condition]
    original = f"{trial_id}:B1"
    final = f"{trial_id}:B2" if switched else original
    return TreatmentMetadata(
        condition=condition,
        message_regime=message_regime,
        assigned_expectation=expectation,
        original_partner_id=original,
        final_partner_id=final,
        switched=switched,
        own_promise_regime=own_promise,
    )


def all_conditions() -> tuple[Condition, ...]:
    return tuple(Condition)


def fixed_conditions() -> tuple[Condition, ...]:
    return tuple(
        condition for condition in Condition
        if condition not in {
            Condition.VOLUNTARY_PROMISE_NO_SWITCH,
            Condition.VOLUNTARY_PROMISE_SWITCH,
        }
    )
