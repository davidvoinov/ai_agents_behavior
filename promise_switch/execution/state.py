"""Explicit durable trial and run state machines."""

from __future__ import annotations

from enum import Enum


class TrialState(str, Enum):
    PLANNED = "PLANNED"
    MESSAGE_PENDING = "MESSAGE_PENDING"
    MESSAGE_RECORDED = "MESSAGE_RECORDED"
    ASSIGNMENT_PENDING = "ASSIGNMENT_PENDING"
    ASSIGNED = "ASSIGNED"
    DECISION_PENDING = "DECISION_PENDING"
    COMPLETED = "COMPLETED"
    BEHAVIORAL_FAILURE = "BEHAVIORAL_FAILURE"
    INFRASTRUCTURE_PENDING_RETRY = "INFRASTRUCTURE_PENDING_RETRY"
    PAUSED_QUOTA = "PAUSED_QUOTA"
    FAILED_INFRASTRUCTURE_FINAL = "FAILED_INFRASTRUCTURE_FINAL"


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PAUSED_PROVIDER_QUOTA = "PAUSED_PROVIDER_QUOTA"
    PAUSED_INFRASTRUCTURE_REVIEW = "PAUSED_INFRASTRUCTURE_REVIEW"
    PAUSED_BEHAVIORAL_PARSER_REVIEW = "PAUSED_BEHAVIORAL_PARSER_REVIEW"
    FAILED = "FAILED"


_ALLOWED: dict[TrialState, set[TrialState]] = {
    TrialState.PLANNED: {TrialState.MESSAGE_PENDING, TrialState.ASSIGNED},
    TrialState.MESSAGE_PENDING: {
        TrialState.MESSAGE_RECORDED,
        TrialState.INFRASTRUCTURE_PENDING_RETRY,
        TrialState.PAUSED_QUOTA,
        TrialState.FAILED_INFRASTRUCTURE_FINAL,
    },
    TrialState.MESSAGE_RECORDED: {TrialState.ASSIGNMENT_PENDING},
    TrialState.ASSIGNMENT_PENDING: {TrialState.ASSIGNED},
    TrialState.ASSIGNED: {TrialState.DECISION_PENDING, TrialState.BEHAVIORAL_FAILURE},
    TrialState.DECISION_PENDING: {
        TrialState.COMPLETED,
        TrialState.BEHAVIORAL_FAILURE,
        TrialState.INFRASTRUCTURE_PENDING_RETRY,
        TrialState.PAUSED_QUOTA,
        TrialState.FAILED_INFRASTRUCTURE_FINAL,
    },
    TrialState.INFRASTRUCTURE_PENDING_RETRY: {
        TrialState.MESSAGE_PENDING,
        TrialState.DECISION_PENDING,
        TrialState.PAUSED_QUOTA,
        TrialState.FAILED_INFRASTRUCTURE_FINAL,
    },
    TrialState.PAUSED_QUOTA: {TrialState.MESSAGE_PENDING, TrialState.DECISION_PENDING},
    TrialState.COMPLETED: set(),
    TrialState.BEHAVIORAL_FAILURE: set(),
    TrialState.FAILED_INFRASTRUCTURE_FINAL: set(),
}


def transition(current: TrialState, target: TrialState) -> TrialState:
    if target not in _ALLOWED[current]:
        raise ValueError(f"Illegal Promise Switch transition: {current.value} -> {target.value}")
    return target


def is_terminal(state: TrialState) -> bool:
    return state in {
        TrialState.COMPLETED,
        TrialState.BEHAVIORAL_FAILURE,
        TrialState.FAILED_INFRASTRUCTURE_FINAL,
    }
