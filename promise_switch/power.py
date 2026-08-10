"""Deterministic planning power for equal-arm binary Promise Switch contrasts.

Power uses the large-sample two-sided z test for two independent proportions:
the variance is pooled under the null and unpooled under the alternative.  This
is planning arithmetic, not Fisher-exact power and not the confirmatory test
implementation.
"""

from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist
from typing import Any, Iterable, Literal

Direction = Literal["increase", "decrease"]

REFERENCE_RATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
MDES = (0.15, 0.20, 0.25, 0.30)
CANDIDATE_N = (50, 75, 100)
ALPHAS = (0.05, 0.05 / 3)
TARGET_POWERS = (0.80, 0.90)
ATTRITION_RATES = (0.00, 0.02, 0.05, 0.10)
DIRECTIONS: tuple[Direction, ...] = ("increase", "decrease")


def _probability(value: float, name: str, *, interior: bool = False) -> float:
    lower_ok = value > 0 if interior else value >= 0
    upper_ok = value < 1 if interior else value <= 1
    if not lower_ok or not upper_ok:
        interval = "(0, 1)" if interior else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return value


def alternative_rate(p0: float, mde: float, direction: Direction) -> float:
    """Return p1 or reject a mathematically invalid absolute-MDE scenario."""
    _probability(p0, "p0")
    if mde <= 0 or mde > 1:
        raise ValueError("mde must be in (0, 1]")
    if direction not in DIRECTIONS:
        raise ValueError("direction must be increase or decrease")
    p1 = round(p0 + mde if direction == "increase" else p0 - mde, 12)
    _probability(p1, "p1")
    return p1


def two_proportion_z_power(*, p0: float, p1: float, n_per_arm: int, alpha: float) -> float:
    """Approximate two-sided power for an equal-arm pooled-null z test."""
    _probability(p0, "p0")
    _probability(p1, "p1")
    _probability(alpha, "alpha", interior=True)
    if p0 == p1:
        raise ValueError("p0 and p1 must differ")
    if not isinstance(n_per_arm, int) or isinstance(n_per_arm, bool) or n_per_arm <= 0:
        raise ValueError("n_per_arm must be a positive integer")

    difference = p1 - p0
    pooled = (p0 + p1) / 2
    std_null = sqrt(2 * pooled * (1 - pooled))
    std_alternative = sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    if std_null == 0 or std_alternative == 0:
        raise ValueError("scenario has zero variance")

    normal = NormalDist()
    critical = normal.inv_cdf(1 - alpha / 2)
    standardized_effect = difference * sqrt(n_per_arm) / std_alternative
    scaled_critical = critical * std_null / std_alternative
    upper_tail = 1 - normal.cdf(scaled_critical - standardized_effect)
    lower_tail = normal.cdf(-scaled_critical - standardized_effect)
    return min(1.0, max(0.0, upper_tail + lower_tail))


def minimum_n_per_arm(
    *, p0: float, p1: float, alpha: float, target_power: float, max_n: int = 100_000,
) -> int:
    """Find the smallest integer equal-arm n attaining target approximate power."""
    _probability(target_power, "target_power", interior=True)
    if max_n <= 0:
        raise ValueError("max_n must be positive")
    if two_proportion_z_power(p0=p0, p1=p1, n_per_arm=max_n, alpha=alpha) < target_power:
        raise ValueError("target power not reached within max_n")
    low, high = 1, max_n
    while low < high:
        middle = (low + high) // 2
        if two_proportion_z_power(p0=p0, p1=p1, n_per_arm=middle, alpha=alpha) >= target_power:
            high = middle
        else:
            low = middle + 1
    return low


def initiated_trials(target_completed: int, attrition_rate: float) -> int:
    """Planning-only initiated trials required for a completed-cell target."""
    if not isinstance(target_completed, int) or isinstance(target_completed, bool) or target_completed <= 0:
        raise ValueError("target_completed must be a positive integer")
    _probability(attrition_rate, "attrition_rate")
    if attrition_rate == 1:
        raise ValueError("attrition_rate must be less than 1")
    return ceil(target_completed / (1 - attrition_rate))


def call_plan(n_completed_per_condition: int, configuration_count: int) -> dict[str, int]:
    """Exact base calls, excluding retry attempts and attrition inflation."""
    if (not isinstance(n_completed_per_condition, int)
            or isinstance(n_completed_per_condition, bool) or n_completed_per_condition <= 0):
        raise ValueError("n_completed_per_condition must be a positive integer")
    if (not isinstance(configuration_count, int)
            or isinstance(configuration_count, bool) or configuration_count <= 0):
        raise ValueError("configuration_count must be a positive integer")
    final_per_configuration = 6 * n_completed_per_condition
    message_per_configuration = 2 * n_completed_per_condition
    total_per_configuration = final_per_configuration + message_per_configuration
    return {
        "n_completed_per_condition": n_completed_per_condition,
        "configuration_count": configuration_count,
        "final_calls_per_configuration": final_per_configuration,
        "message_calls_per_configuration": message_per_configuration,
        "total_calls_per_configuration": total_per_configuration,
        "total_calls_all_configurations": total_per_configuration * configuration_count,
    }


def _valid_alternatives(
    reference_rates: Iterable[float], mdes: Iterable[float], directions: Iterable[Direction],
) -> list[tuple[float, float, float, Direction]]:
    scenarios = []
    for p0 in reference_rates:
        for mde in mdes:
            for direction in directions:
                try:
                    p1 = alternative_rate(p0, mde, direction)
                except ValueError as error:
                    if "p1 must be" not in str(error):
                        raise
                    continue
                scenarios.append((p0, p1, mde, direction))
    return scenarios


def build_power_report(
    *,
    reference_rates: Iterable[float] = REFERENCE_RATES,
    mdes: Iterable[float] = MDES,
    candidate_n: Iterable[int] = CANDIDATE_N,
    alphas: Iterable[float] = ALPHAS,
    target_powers: Iterable[float] = TARGET_POWERS,
    directions: Iterable[Direction] = DIRECTIONS,
) -> dict[str, Any]:
    """Build the complete deterministic planning grid and arithmetic tables."""
    p0_values = tuple(reference_rates)
    mde_values = tuple(mdes)
    n_values = tuple(candidate_n)
    alpha_values = tuple(alphas)
    target_values = tuple(target_powers)
    direction_values = tuple(directions)
    alternatives = _valid_alternatives(p0_values, mde_values, direction_values)

    candidate_rows = []
    minimum_rows = []
    for p0, p1, mde, direction in alternatives:
        for alpha in alpha_values:
            for n_per_arm in n_values:
                candidate_rows.append({
                    "p0": p0, "p1": p1, "direction": direction, "mde": mde,
                    "alpha": alpha, "n_per_arm": n_per_arm,
                    "estimated_power": two_proportion_z_power(
                        p0=p0, p1=p1, n_per_arm=n_per_arm, alpha=alpha),
                })
            for target_power in target_values:
                minimum_rows.append({
                    "p0": p0, "p1": p1, "direction": direction, "mde": mde,
                    "alpha": alpha, "target_power": target_power,
                    "minimum_n_per_arm": minimum_n_per_arm(
                        p0=p0, p1=p1, alpha=alpha, target_power=target_power),
                })

    attrition_rows = [{
        "target_completed_per_condition": target,
        "attrition_rate": rate,
        "initiated_trials_per_condition": initiated_trials(target, rate),
    } for target in n_values for rate in ATTRITION_RATES]

    return {
        "method": {
            "name": "large_sample_two_independent_proportions_z_test",
            "sidedness": "two-sided",
            "allocation": "equal_n_per_arm",
            "null_variance": "pooled",
            "alternative_variance": "unpooled",
            "fisher_exact_power": False,
            "multiplicity_note": (
                "alpha=0.05/3 is a conservative Bonferroni planning sensitivity; "
                "it is not identical to and does not replace the confirmatory Holm procedure"
            ),
            "n_interpretation": "completed observations per condition; attrition inflation is separate",
        },
        "grid": {
            "reference_rates": list(p0_values), "mdes": list(mde_values),
            "candidate_n_per_arm": list(n_values), "alphas": list(alpha_values),
            "target_powers": list(target_values), "directions": list(direction_values),
        },
        "candidate_power": candidate_rows,
        "minimum_n": minimum_rows,
        "attrition_inflation": attrition_rows,
        "base_call_plans": {
            "current_8_configurations": [call_plan(n, 8) for n in n_values],
            "expanded_11_configurations": [call_plan(n, 11) for n in n_values],
        },
    }
