"""Outcome-blind balanced scheduler."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

from .planner import PlannedTrial
from .state import TrialState, is_terminal


def balanced_pending_order(
    plans: Iterable[PlannedTrial], states: Mapping[str, TrialState],
    conditions: Mapping[str, str | None],
) -> list[str]:
    buckets: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    for plan in sorted(plans, key=lambda item: item.ordinal):
        state = states[plan.trial_id]
        if is_terminal(state):
            continue
        cell = conditions.get(plan.trial_id) or plan.planned_regime
        buckets[(plan.config_id, cell)].append(plan.trial_id)
    ordered: list[str] = []
    keys = sorted(buckets)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].popleft())
    return ordered
