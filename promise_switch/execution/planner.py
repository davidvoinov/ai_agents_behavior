"""Deterministic pre-API trial planning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from ..design import Condition, fixed_conditions
from ..schemas import ModelConfig


@dataclass(frozen=True)
class PlannedTrial:
    trial_id: str
    run_id: str
    config_id: str
    provider: str
    model: str
    model_alias: str
    reasoning_mode: str
    randomization_seed: int
    planned_regime: str
    final_condition: str | None
    post_message_assignment_pending: bool
    ordinal: int
    plan_phase: str
    top_up_sequence: int | None
    top_up_block_id: str | None
    allocation_group_id: str | None
    creation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def model_config_id(config: ModelConfig) -> str:
    digest = hashlib.sha256(
        f"{config.provider}|{config.model_id}|{config.reasoning_mode}".encode()
    ).hexdigest()[:12]
    return f"{config.provider}:{config.model_alias}:{config.reasoning_mode}:{digest}"


def plan_trials(
    *, run_id: str, seed: int, n_per_condition: int,
    model_configs: Sequence[ModelConfig],
) -> tuple[PlannedTrial, ...]:
    if n_per_condition < 1:
        raise ValueError("n_per_condition must be positive")
    plans: list[PlannedTrial] = []
    ordinal = 0
    for config in model_configs:
        config_id = model_config_id(config)
        cells: list[tuple[str, str | None, int]] = []
        for condition in fixed_conditions():
            for index in range(n_per_condition):
                cells.append((condition.value, condition.value, index))
        for index in range(2 * n_per_condition):
            cells.append(("voluntary", None, index))
        # Interleave one record per regime per round, never block-by-block.
        by_regime: dict[str, list[tuple[str, str | None, int]]] = {}
        for cell in cells:
            by_regime.setdefault(cell[0], []).append(cell)
        ordered_regimes = [condition.value for condition in fixed_conditions()] + ["voluntary"]
        for round_index in range(2 * n_per_condition):
            for regime in ordered_regimes:
                bucket = by_regime[regime]
                if round_index >= len(bucket):
                    continue
                planned_regime, final_condition, within_index = bucket[round_index]
                ordinal += 1
                trial_id = _trial_id(run_id, config_id, planned_regime, within_index)
                plans.append(PlannedTrial(
                    trial_id=trial_id, run_id=run_id, config_id=config_id,
                    provider=config.provider, model=config.model_id,
                    model_alias=config.model_alias, reasoning_mode=config.reasoning_mode,
                    randomization_seed=seed, planned_regime=planned_regime,
                    final_condition=final_condition,
                    post_message_assignment_pending=final_condition is None,
                    ordinal=ordinal,
                    plan_phase="initial",
                    top_up_sequence=None,
                    top_up_block_id=None,
                    allocation_group_id=(
                        f"initial:{config_id}" if final_condition is None else None
                    ),
                    creation_reason="initial_minimum_completed_target_plan",
                ))
    return tuple(interleave_model_configs(plans))


def interleave_model_configs(plans: Sequence[PlannedTrial]) -> list[PlannedTrial]:
    buckets: dict[str, list[PlannedTrial]] = {}
    for plan in plans:
        buckets.setdefault(plan.config_id, []).append(plan)
    output: list[PlannedTrial] = []
    for index in range(max(map(len, buckets.values()), default=0)):
        for config_id in sorted(buckets):
            if index < len(buckets[config_id]):
                output.append(buckets[config_id][index])
    return output


def _trial_id(run_id: str, config_id: str, regime: str, index: int) -> str:
    digest = hashlib.sha256(f"{run_id}|{config_id}|{regime}|{index}".encode()).hexdigest()[:18]
    return f"ps:{digest}"


def plan_top_up_expansion(
    *, run_id: str, seed: int, config: ModelConfig, sequence: int,
    fixed_deficits: dict[str, int], voluntary_blocks: int, starting_ordinal: int,
) -> tuple[PlannedTrial, ...]:
    """Create a deterministic outcome-blind append-only expansion."""
    if sequence < 1 or voluntary_blocks < 0 or any(value < 0 for value in fixed_deficits.values()):
        raise ValueError("invalid top-up expansion inputs")
    config_id = model_config_id(config)
    plans: list[PlannedTrial] = []
    ordinal = starting_ordinal
    for condition in fixed_conditions():
        for slot in range(fixed_deficits.get(condition.value, 0)):
            ordinal += 1
            plans.append(_top_up_plan(
                run_id=run_id, seed=seed, config=config, config_id=config_id,
                sequence=sequence, regime=condition.value, final_condition=condition.value,
                slot=slot, block_id=None, ordinal=ordinal,
            ))
    for block_index in range(voluntary_blocks):
        block_id = f"topup:{config_id}:{sequence}:block:{block_index}"
        for slot in range(2):
            ordinal += 1
            plans.append(_top_up_plan(
                run_id=run_id, seed=seed, config=config, config_id=config_id,
                sequence=sequence, regime="voluntary", final_condition=None,
                slot=slot, block_id=block_id, ordinal=ordinal,
            ))
    return tuple(plans)


def _top_up_plan(
    *, run_id: str, seed: int, config: ModelConfig, config_id: str,
    sequence: int, regime: str, final_condition: str | None, slot: int,
    block_id: str | None, ordinal: int,
) -> PlannedTrial:
    identity = f"{run_id}|{config_id}|top_up|{sequence}|{regime}|{block_id}|{slot}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:18]
    return PlannedTrial(
        trial_id=f"ps:{digest}", run_id=run_id, config_id=config_id,
        provider=config.provider, model=config.model_id, model_alias=config.model_alias,
        reasoning_mode=config.reasoning_mode, randomization_seed=seed,
        planned_regime=regime, final_condition=final_condition,
        post_message_assignment_pending=final_condition is None, ordinal=ordinal,
        plan_phase="top_up", top_up_sequence=sequence,
        top_up_block_id=block_id, allocation_group_id=block_id,
        creation_reason="cell_below_minimum_completed_binary_target",
    )


def plan_voluntary_message_recovery(
    *, run_id: str, seed: int, config: ModelConfig, sequence: int,
    allocation_group_id: str, count: int, starting_ordinal: int,
) -> tuple[PlannedTrial, ...]:
    """Replace only missing pre-assignment message eligibility, never a failed outcome."""
    if count < 1:
        raise ValueError("message recovery count must be positive")
    config_id = model_config_id(config)
    plans = []
    for slot in range(count):
        identity = (
            f"{run_id}|{config_id}|message_recovery|{sequence}|"
            f"{allocation_group_id}|{slot}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:18]
        plans.append(PlannedTrial(
            trial_id=f"ps:{digest}", run_id=run_id, config_id=config_id,
            provider=config.provider, model=config.model_id,
            model_alias=config.model_alias, reasoning_mode=config.reasoning_mode,
            randomization_seed=seed, planned_regime="voluntary", final_condition=None,
            post_message_assignment_pending=True, ordinal=starting_ordinal + slot + 1,
            plan_phase="top_up", top_up_sequence=sequence,
            top_up_block_id=allocation_group_id,
            allocation_group_id=allocation_group_id,
            creation_reason="voluntary_message_stage_infrastructure_recovery",
        ))
    return tuple(plans)
