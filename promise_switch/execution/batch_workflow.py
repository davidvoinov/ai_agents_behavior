"""Pure two-phase workflow gates shared by native provider Batch execution."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Mapping, Sequence

from ..client import ProviderAgnosticClient, RequestSpec
from ..design import Condition, treatment_metadata
from ..prompts import final_decision_messages, voluntary_message_messages
from ..randomization import allocate_voluntary_pool
from ..schemas import AssignmentRecord, MessageEvent, ModelConfig
from .planner import PlannedTrial


def build_message_batch_requests(
    plans: Sequence[PlannedTrial], *, config: ModelConfig, max_tokens: int,
    assignments: Mapping[str, AssignmentRecord | Mapping[str, object]] | None = None,
) -> tuple[RequestSpec, ...]:
    """Build Stage 1 only; fail if any voluntary assignment already exists."""
    client = ProviderAgnosticClient(config)
    output = []
    for plan in plans:
        if plan.planned_regime != "voluntary":
            continue
        if plan.final_condition is not None or plan.trial_id in (assignments or {}):
            raise ValueError("voluntary assignment exists before durable message ingestion")
        partner = treatment_metadata(
            Condition.VOLUNTARY_PROMISE_NO_SWITCH, plan.trial_id
        ).original_partner_id
        request = client.build_request(
            messages=voluntary_message_messages(partner),
            temperature=1.0, max_tokens=max_tokens,
        )
        output.append(_with_trial_metadata(request, plan, "message"))
    return tuple(output)


def allocate_recorded_voluntary_groups(
    plans: Sequence[PlannedTrial], *, seed: int,
    message_events: Mapping[str, MessageEvent],
) -> tuple[AssignmentRecord, ...]:
    """Cross the assignment boundary only when an entire group is durable."""
    groups: dict[tuple[str, str], list[PlannedTrial]] = defaultdict(list)
    for plan in plans:
        if plan.planned_regime == "voluntary":
            groups[(plan.config_id, str(plan.allocation_group_id))].append(plan)
    output = []
    for (config_id, group_id), group in sorted(groups.items()):
        required = sum(plan.plan_phase == "initial" for plan in group) or 2
        if len(group) != required or any(plan.trial_id not in message_events for plan in group):
            continue
        events = [message_events[plan.trial_id] for plan in group]
        if any(not event.message_recorded for event in events):
            raise ValueError("assignment requires durable recorded message events")
        output.extend(allocate_voluntary_pool(
            seed=allocation_seed(seed, config_id, group_id), message_events=events,
        ))
    return tuple(output)


def build_decision_batch_requests(
    plans: Sequence[PlannedTrial], *, config: ModelConfig, max_tokens: int,
    assignments: Mapping[str, AssignmentRecord],
    message_events: Mapping[str, MessageEvent],
) -> tuple[RequestSpec, ...]:
    """Build fresh Stage 2 contexts from assignment plus visible stored message only."""
    client = ProviderAgnosticClient(config)
    output = []
    for plan in plans:
        assignment = assignments.get(plan.trial_id)
        if assignment is None:
            continue
        event = message_events.get(plan.trial_id)
        is_voluntary = assignment.condition in {
            Condition.VOLUNTARY_PROMISE_NO_SWITCH,
            Condition.VOLUNTARY_PROMISE_SWITCH,
        }
        if is_voluntary and (
            event is None or not event.message_recorded
            or assignment.message_event_id != event.message_event_id
        ):
            raise ValueError("voluntary final request requires its recorded message event")
        messages = final_decision_messages(
            condition=assignment.condition,
            original_partner_id=assignment.original_partner_id,
            final_partner_id=assignment.final_partner_id,
            raw_message=event.raw_message if event is not None else "",
        )
        request = client.build_request(
            messages=messages, temperature=1.0, max_tokens=max_tokens,
        )
        output.append(_with_trial_metadata(request, plan, "final_decision"))
    return tuple(output)


def allocation_seed(seed: int, config_id: str, group_id: str) -> int:
    return int(hashlib.sha256(
        f"{seed}|{config_id}|voluntary|{group_id}".encode()
    ).hexdigest()[:16], 16)


def _with_trial_metadata(
    request: RequestSpec, plan: PlannedTrial, stage: str,
) -> RequestSpec:
    return RequestSpec(
        base_url=request.base_url, model_id=request.model_id,
        api_key_env=request.api_key_env, body=request.body,
        metadata={**request.metadata, "stage": stage, "trial_id": plan.trial_id},
    )
