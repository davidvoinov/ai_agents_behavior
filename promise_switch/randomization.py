"""Seeded allocation with an enforced post-message boundary."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from .design import Condition, treatment_metadata
from .schemas import AssignmentRecord, MessageEvent, TrialPlan


def plan_fixed_trial(
    *, trial_id: str, run_id: str, seed: int, condition: Condition
) -> TrialPlan:
    if condition in {
        Condition.VOLUNTARY_PROMISE_NO_SWITCH,
        Condition.VOLUNTARY_PROMISE_SWITCH,
    }:
        raise ValueError("Voluntary final conditions cannot be assigned before message stage")
    return TrialPlan(
        trial_id=trial_id,
        run_id=run_id,
        randomization_seed=seed,
        voluntary_pool=False,
        condition=condition,
    )


def plan_voluntary_trial(*, trial_id: str, run_id: str, seed: int) -> TrialPlan:
    return TrialPlan(
        trial_id=trial_id,
        run_id=run_id,
        randomization_seed=seed,
        voluntary_pool=True,
        condition=None,
    )


def assign_fixed_trial(plan: TrialPlan) -> AssignmentRecord:
    if plan.voluntary_pool or plan.condition is None:
        raise ValueError("Fixed assignment requires a fixed-condition plan")
    return _assignment(plan.trial_id, plan.randomization_seed, plan.condition, None)


def allocate_voluntary_pool(
    *, seed: int, message_events: Sequence[MessageEvent]
) -> tuple[AssignmentRecord, ...]:
    """Allocate a recorded voluntary pool exactly 1:1, reproducibly.

    Raw message content is deliberately excluded from every allocation input.
    An even pool is required so the target ratio is exact.
    """
    if not message_events or len(message_events) % 2:
        raise ValueError("Voluntary allocation requires a non-empty even pool")
    trial_ids = [event.trial_id for event in message_events]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("Each voluntary trial must have one unique message event")
    for event in message_events:
        if not event.message_recorded or not event.message_event_id.strip():
            raise ValueError("Switch assignment requires a recorded message event")

    ordered_events = sorted(message_events, key=lambda event: event.trial_id)
    labels = [Condition.VOLUNTARY_PROMISE_NO_SWITCH] * (len(ordered_events) // 2)
    labels += [Condition.VOLUNTARY_PROMISE_SWITCH] * (len(ordered_events) // 2)
    random.Random(seed).shuffle(labels)
    return tuple(
        _assignment(event.trial_id, seed, condition, event.message_event_id)
        for event, condition in zip(ordered_events, labels, strict=True)
    )


def _assignment(
    trial_id: str,
    seed: int,
    condition: Condition,
    message_event_id: str | None,
) -> AssignmentRecord:
    metadata = treatment_metadata(condition, trial_id)
    digest = hashlib.sha256(
        f"{seed}|{trial_id}|{condition.value}|{message_event_id or 'fixed'}".encode("utf-8")
    ).hexdigest()[:20]
    return AssignmentRecord(
        assignment_id=f"assignment:{digest}",
        trial_id=trial_id,
        randomization_seed=seed,
        condition=condition,
        message_event_id=message_event_id,
        original_partner_id=metadata.original_partner_id,
        final_partner_id=metadata.final_partner_id,
        switched=metadata.switched,
    )
