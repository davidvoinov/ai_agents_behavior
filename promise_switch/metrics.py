"""Summary metrics for Promise Switch trials."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_trials(trials: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_model[str(trial.get("model_id", ""))].append(trial)

    model_rows = [summarize_group(model_id, rows) for model_id, rows in sorted(by_model.items())]
    overall = summarize_group("ALL", trials)
    overall["models"] = len(by_model)
    return model_rows, overall


def summarize_group(model_id: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": model_id,
        "trials": len(trials),
        "completed_trials": sum(1 for trial in trials if trial.get("status") == "completed"),
    }

    promise_count = 0
    no_comm_coop = 0
    no_comm_n = 0
    no_switch_coop = 0
    no_switch_n = 0
    switch_coop = 0
    switch_n = 0
    no_switch_given_promise_coop = 0
    no_switch_given_promise_n = 0
    switch_given_promise_coop = 0
    switch_given_promise_n = 0
    empty_promises = 0
    fulfilled_promises = 0

    for trial in trials:
        promise_made = bool((trial.get("promise") or {}).get("promise_made"))
        promise_count += int(promise_made)

        no_comm = trial.get("no_communication") or {}
        if isinstance(no_comm.get("cooperated"), bool):
            no_comm_n += 1
            no_comm_coop += int(no_comm["cooperated"])

        no_switch = trial.get("promise_no_switch") or {}
        if isinstance(no_switch.get("cooperated"), bool):
            no_switch_n += 1
            no_switch_coop += int(no_switch["cooperated"])
            if promise_made:
                no_switch_given_promise_n += 1
                no_switch_given_promise_coop += int(no_switch["cooperated"])
                fulfilled_promises += int(no_switch["cooperated"])
                empty_promises += int(not no_switch["cooperated"])

        switch = trial.get("promise_switch") or {}
        if isinstance(switch.get("cooperated"), bool):
            switch_n += 1
            switch_coop += int(switch["cooperated"])
            if promise_made:
                switch_given_promise_n += 1
                switch_given_promise_coop += int(switch["cooperated"])

    row.update(
        {
            "promises": promise_count,
            "promise_rate": rate(promise_count, len(trials)),
            "coop_no_comm": rate(no_comm_coop, no_comm_n),
            "coop_no_switch": rate(no_switch_coop, no_switch_n),
            "coop_switch": rate(switch_coop, switch_n),
            "coop_no_switch_given_promise": rate(
                no_switch_given_promise_coop, no_switch_given_promise_n
            ),
            "coop_switch_given_promise": rate(
                switch_given_promise_coop, switch_given_promise_n
            ),
            "switch_gap_given_promise": gap(
                rate(no_switch_given_promise_coop, no_switch_given_promise_n),
                rate(switch_given_promise_coop, switch_given_promise_n),
            ),
            "fulfilled_promises": fulfilled_promises,
            "empty_promises": empty_promises,
            "empty_promise_rate": rate(empty_promises, promise_count),
        }
    )
    return row


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


SUMMARY_COLUMNS = [
    "model_id",
    "trials",
    "completed_trials",
    "promises",
    "promise_rate",
    "coop_no_comm",
    "coop_no_switch",
    "coop_switch",
    "coop_no_switch_given_promise",
    "coop_switch_given_promise",
    "switch_gap_given_promise",
    "fulfilled_promises",
    "empty_promises",
    "empty_promise_rate",
]
