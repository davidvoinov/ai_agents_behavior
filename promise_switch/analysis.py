"""Offline, model-specific Promise Switch summaries and binary contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, erf, sqrt
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .design import Condition, scientific_condition_label

ModelKey = tuple[str, str, str]


@dataclass(frozen=True)
class Contrast:
    name: str
    treatment: Condition
    reference: Condition


# Frozen pre-confirmatory family. Holm correction is applied only across these
# three selection-free confirmatory contrasts.
PRIMARY_CONFIRMATORY_CONTRASTS = (
    Contrast("H8", Condition.INDUCED_PROMISE_NO_SWITCH, Condition.NO_COMMUNICATION),
    Contrast("H9", Condition.INDUCED_PROMISE_NO_SWITCH, Condition.INDUCED_PROMISE_SWITCH),
    Contrast("H10", Condition.INDUCED_PROMISE_NO_SWITCH, Condition.EXPECTATION_WITHOUT_OWN_PROMISE),
)
VOLUNTARY_ITT_CONTRAST = Contrast(
    "voluntary_no_switch_vs_switch_itt",
    Condition.VOLUNTARY_PROMISE_NO_SWITCH,
    Condition.VOLUNTARY_PROMISE_SWITCH,
)
VOLUNTARY_EXPLICIT_PROMISER_CONTRAST = Contrast(
    "voluntary_no_switch_vs_switch_explicit_promisers",
    Condition.VOLUNTARY_PROMISE_NO_SWITCH,
    Condition.VOLUNTARY_PROMISE_SWITCH,
)


def model_key(row: Mapping[str, Any]) -> ModelKey:
    return (str(row["provider"]), str(row["model"]), str(row["reasoning_mode"]))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt((p * (1 - p) / total) + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _newcombe_difference_ci(a: int, n_a: int, b: int, n_b: int) -> tuple[float, float]:
    """Newcombe score interval for p_a - p_b."""
    pa, pb = a / n_a, b / n_b
    la, ua = _wilson(a, n_a)
    lb, ub = _wilson(b, n_b)
    return (pa - pb - sqrt((pa - la) ** 2 + (ub - pb) ** 2),
            pa - pb + sqrt((ua - pa) ** 2 + (pb - lb) ** 2))


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p using fixed margins."""
    n1, n2, successes = a + b, c + d, a + c
    total = n1 + n2
    low, high = max(0, n1 - (total - successes)), min(n1, successes)
    denominator = comb(total, n1)

    def probability(x: int) -> float:
        return comb(successes, x) * comb(total - successes, n1 - x) / denominator

    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1)
                        if probability(x) <= observed + 1e-12))


def _chi_square_two_sided(a: int, b: int, c: int, d: int) -> float:
    total = a + b + c + d
    numerator = total * (a * d - b * c) ** 2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator == 0:
        return 1.0
    # Survival function for chi-square(df=1): erfc(sqrt(x/2)).
    return 1.0 - erf(sqrt((numerator / denominator) / 2))


def binary_contrast(rows: Sequence[Mapping[str, Any]], contrast: Contrast) -> dict[str, Any]:
    terminal = [row for row in rows if row.get("completion_status") in {"completed", "failed"}]
    selected = [row for row in rows if row.get("completion_status") == "completed"
                and row.get("final_action") in {"COOPERATE", "DEFECT"}]
    arms: dict[Condition, list[Mapping[str, Any]]] = {
        condition: [row for row in selected if row.get("condition") == condition.value]
        for condition in (contrast.treatment, contrast.reference)
    }
    n_t, n_r = len(arms[contrast.treatment]), len(arms[contrast.reference])
    if not n_t or not n_r:
        raise ValueError(f"contrast {contrast.name} has an empty arm")
    c_t = sum(row["final_action"] == "COOPERATE" for row in arms[contrast.treatment])
    c_r = sum(row["final_action"] == "COOPERATE" for row in arms[contrast.reference])
    terminal_n_t = sum(row.get("condition") == contrast.treatment.value for row in terminal)
    terminal_n_r = sum(row.get("condition") == contrast.reference.value for row in terminal)
    missing_t, missing_r = terminal_n_t - n_t, terminal_n_r - n_r
    if missing_t < 0 or missing_r < 0:
        raise ValueError(f"contrast {contrast.name} has inconsistent terminal denominators")
    lower_t, upper_t = c_t / terminal_n_t, (c_t + missing_t) / terminal_n_t
    lower_r, upper_r = c_r / terminal_n_r, (c_r + missing_r) / terminal_n_r
    table = (c_t, n_t - c_t, c_r, n_r - c_r)
    expected = [n_t * (c_t + c_r) / (n_t + n_r), n_t * (n_t + n_r - c_t - c_r) / (n_t + n_r),
                n_r * (c_t + c_r) / (n_t + n_r), n_r * (n_t + n_r - c_t - c_r) / (n_t + n_r)]
    use_fisher = any(value < 5 for value in expected)
    p_value = _fisher_two_sided(*table) if use_fisher else _chi_square_two_sided(*table)
    ci = _newcombe_difference_ci(c_t, n_t, c_r, n_r)
    return {
        "name": contrast.name,
        "treatment": contrast.treatment.value,
        "treatment_label": scientific_condition_label(contrast.treatment),
        "reference": contrast.reference.value,
        "reference_label": scientific_condition_label(contrast.reference),
        "treatment_n": n_t,
        "treatment_cooperate": c_t,
        "treatment_rate": c_t / n_t,
        "reference_n": n_r,
        "reference_cooperate": c_r,
        "reference_rate": c_r / n_r,
        "risk_difference": c_t / n_t - c_r / n_r,
        "risk_difference_ci_95": list(ci),
        "attrition_sensitivity": {
            "method": "worst_case_binary_outcome_bounds",
            "treatment_terminal_n": terminal_n_t,
            "treatment_missing_action_n": missing_t,
            "reference_terminal_n": terminal_n_r,
            "reference_missing_action_n": missing_r,
            "risk_difference_bounds": [lower_t - upper_r, upper_t - lower_r],
        },
        "test": "fisher_exact_two_sided" if use_fisher else "pearson_chi_square_two_sided",
        "p_value": p_value,
    }


def holm_adjust(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return copies with monotone Holm-adjusted p-values."""
    order = sorted(range(len(results)), key=lambda i: float(results[i]["p_value"]))
    adjusted = [0.0] * len(results)
    running = 0.0
    count = len(results)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(results[index]["p_value"])))
        adjusted[index] = running
    return [{**result, "p_value_holm": adjusted[index]} for index, result in enumerate(results)]


def analyze_by_model(
    rows: Iterable[Mapping[str, Any]],
    *,
    explicit_promiser_trial_ids_by_model: Mapping[ModelKey, Iterable[str]] | None = None,
) -> dict[ModelKey, dict[str, Any]]:
    """Analyze each provider/model/reasoning configuration independently."""
    groups: dict[ModelKey, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(model_key(row), []).append(row)
    output: dict[ModelKey, dict[str, Any]] = {}
    for key, group in groups.items():
        trial_ids = [str(row["trial_id"]) for row in group if row.get("trial_id") is not None]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError(f"duplicate trial_id within model configuration {key}")
        counts = {condition.value: {
            "terminal_n": 0, "completed_n": 0, "analyzed_n": 0,
            "cooperate": 0, "defect": 0,
            "infrastructure_attrition": 0, "behavioral_attrition": 0,
            "initial_terminal_n": 0, "top_up_terminal_n": 0,
        }
                  for condition in Condition}
        for row in group:
            condition = str(row.get("condition", ""))
            if condition not in counts:
                continue
            counts[condition]["terminal_n"] += 1
            phase_key = ("top_up_terminal_n" if row.get("plan_phase") == "top_up"
                         else "initial_terminal_n")
            counts[condition][phase_key] += 1
            failure_category = row.get("failure_category")
            if failure_category in {"infrastructure", "behavioral"}:
                counts[condition][f"{failure_category}_attrition"] += 1
            if row.get("completion_status") != "completed":
                continue
            counts[condition]["completed_n"] += 1
            action = str(row.get("final_action", "")).lower()
            if action in {"cooperate", "defect"}:
                counts[condition]["analyzed_n"] += 1
                counts[condition][action] += 1
        for condition_counts in counts.values():
            analyzed_n = condition_counts["analyzed_n"]
            condition_counts["cooperation_rate"] = (
                condition_counts["cooperate"] / analyzed_n if analyzed_n else None
            )
        confirmatory = holm_adjust([
            binary_contrast(group, contrast) for contrast in PRIMARY_CONFIRMATORY_CONTRASTS
        ])
        voluntary_itt = binary_contrast(group, VOLUNTARY_ITT_CONTRAST)
        selected: dict[str, Any] = {
            "name": VOLUNTARY_EXPLICIT_PROMISER_CONTRAST.name,
            "classification": "further_secondary_selected_subgroup",
        }
        if explicit_promiser_trial_ids_by_model is None or key not in explicit_promiser_trial_ids_by_model:
            selected["status"] = "requires_explicit_promise_codes"
        else:
            eligible_ids = set(explicit_promiser_trial_ids_by_model[key])
            eligible_rows = [row for row in group if str(row.get("trial_id")) in eligible_ids]
            try:
                selected.update(binary_contrast(eligible_rows, VOLUNTARY_EXPLICIT_PROMISER_CONTRAST))
                selected["status"] = "estimated"
            except ValueError:
                selected["status"] = "not_estimable_empty_arm"
        output[key] = {
            "analysis_scope": "model_configuration_specific_primary",
            "condition_labels": {
                condition.value: scientific_condition_label(condition)
                for condition in Condition
            },
            "condition_counts": counts,
            "confirmatory_contrasts": confirmatory,
            "secondary_analyses": {
                "voluntary_itt": {
                    **voluntary_itt,
                    "classification": "secondary_randomized_intention_to_treat",
                },
                "voluntary_explicit_promisers": selected,
            },
        }
    return output


def descriptive_cross_model_summary(
    model_results: Mapping[ModelKey, Mapping[str, Any]],
) -> dict[str, Any]:
    """Arrange model-specific estimates for inspection without pooling observations."""
    configurations: list[dict[str, Any]] = []
    effects: dict[str, list[float]] = {
        contrast.name: [] for contrast in PRIMARY_CONFIRMATORY_CONTRASTS
    }
    directions: dict[str, dict[str, int]] = {
        contrast.name: {"positive": 0, "negative": 0, "zero": 0}
        for contrast in PRIMARY_CONFIRMATORY_CONTRASTS
    }
    for key in sorted(model_results):
        provider, model, reasoning_mode = key
        result = model_results[key]
        contrasts = {
            str(item["name"]): {
                field: item[field] for field in (
                    "risk_difference", "risk_difference_ci_95", "p_value", "p_value_holm"
                )
            }
            for item in result["confirmatory_contrasts"]
        }
        conditions = {
            condition.value: {
                "label": result["condition_labels"][condition.value],
                "actual_n": result["condition_counts"][condition.value]["analyzed_n"],
                "cooperate": result["condition_counts"][condition.value]["cooperate"],
                "defect": result["condition_counts"][condition.value]["defect"],
                "cooperation_rate": result["condition_counts"][condition.value][
                    "cooperation_rate"
                ],
            }
            for condition in Condition
        }
        configurations.append({
            "provider": provider,
            "model": model,
            "reasoning_mode": reasoning_mode,
            "conditions": conditions,
            "confirmatory_contrasts": contrasts,
        })
        for name, contrast in contrasts.items():
            value = float(contrast["risk_difference"])
            effects[name].append(value)
            direction = "positive" if value > 0 else "negative" if value < 0 else "zero"
            directions[name][direction] += 1
    descriptors = {
        name: {
            "direction_counts": directions[name],
            "risk_difference_range": (
                [min(values), max(values)] if values else None
            ),
            "risk_difference_median": median(values) if values else None,
        }
        for name, values in effects.items()
    }
    return {
        "classification": "DESCRIPTIVE CROSS-MODEL SUMMARY",
        "primary_inference": "PRIMARY MODEL-SPECIFIC RESULTS",
        "pooled_observation_level_effect": None,
        "pooled_significance_test": None,
        "model_weighting": "one displayed row per provider/model/reasoning configuration",
        "configuration_count": len(configurations),
        "configurations": configurations,
        "directional_descriptors": descriptors,
    }
