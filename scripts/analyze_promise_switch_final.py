#!/usr/bin/env python3
"""Reproduce the final model-specific Promise Switch statistical analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLOT_CACHE = Path("/tmp") / "promise_switch_plot_cache"
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from jsonschema import Draft202012Validator
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import confint_proportions_2indep

from promise_switch.artifacts import read_jsonl, validate_run_integrity
from promise_switch.codebook import (
    CODEBOOK_VERSION,
    MessageCode,
    derived_outcomes,
    eligible_explicit_promiser,
    validate_annotation_record,
)
from promise_switch.design import Condition, scientific_condition_label
from promise_switch.version import prompt_templates_hash


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ANALYSIS_VERSION = "promise_switch_final_analysis_v1"
ALPHA = 0.05
REQUIRED_COMMITS = ("9e8f59e", "45e8285", "2f8ed70")
RUN_ROOT = ROOT / "runs" / "promise_switch"
SEMANTIC_DIR = (
    ROOT / "notes" / "promise_switch" / "semantic_coding" / CODEBOOK_VERSION
)
ANNOTATION_PATH = SEMANTIC_DIR / "final_full_semantic_annotations_n2738.jsonl"
SEMANTIC_KEY_PATH = SEMANTIC_DIR / "private_full_semantic_key_n2738.json"
SEMANTIC_EXPECTED_COUNT = 2738
SEMANTIC_V2_ACTIVE = False
SCHEMA_PATH = ROOT / "promise_switch" / "coding" / "annotation_schema_v1.json"
CODEBOOK_PATH = ROOT / "promise_switch" / "coding" / "codebook_v1.md"
PROMPT_PATH = ROOT / "promise_switch" / "coding" / "judge_prompt_v1.txt"
PYTHON_VALIDATOR_PATH = ROOT / "promise_switch" / "codebook.py"
OUTPUT_DIR = ROOT / "notes" / "promise_switch" / "final_analysis"
EXCLUDED_RUN = "deepseek_v4_flash_on_confirmatory_qcv3_20260823"

BASE_RUN_SPECS = (
    ("qwen3_32b_on_confirmatory_clean_20260821", "Qwen3-32B — reasoning ON", "Qwen3-32B", "on", "Nebius"),
    ("qwen3_32b_off_confirmatory_clean_20260822", "Qwen3-32B — reasoning OFF", "Qwen3-32B", "off", "Nebius"),
    ("glm_5_2_on_confirmatory_qcv3_20260822", "GLM-5.2 — reasoning ON", "GLM-5.2", "on", "Nebius UK"),
    ("glm_5_2_off_confirmatory_qcv3_20260823", "GLM-5.2 — reasoning OFF", "GLM-5.2", "off", "Nebius UK"),
    ("minimax_m3_on_confirmatory_qcv3_20260823", "MiniMax M3 — reasoning ON", "MiniMax M3", "on", "Nebius"),
    ("minimax_m3_off_confirmatory_qcv3_20260823", "MiniMax M3 — reasoning OFF", "MiniMax M3", "off", "Nebius"),
    ("deepseek_v4_pro_on_confirmatory_qcv3_v2_20260823", "DeepSeek V4 Pro — reasoning ON", "DeepSeek V4 Pro", "on", "Nebius"),
    ("deepseek_v4_pro_off_confirmatory_qcv3_v2_20260823", "DeepSeek V4 Pro — reasoning OFF", "DeepSeek V4 Pro", "off", "Nebius"),
    ("deepseek_v4_flash_off_confirmatory_qcv3_20260823", "DeepSeek V4 Flash — reasoning OFF", "DeepSeek V4 Flash", "off", "Nebius"),
    ("kimi_k3_confirmatory_qcv3_20260823", "Kimi K3", "Kimi K3", "unavailable", "Nebius"),
    ("llama_3_3_70b_confirmatory_qcv3_20260823", "Llama 3.3 70B", "Llama 3.3 70B", "off", "Nebius"),
    ("gpt_oss_120b_nebius_confirmatory_qcv3_20260823", "GPT-OSS-120B — Nebius", "GPT-OSS-120B", "on", "Nebius"),
    ("gpt_oss_120b_groq_confirmatory_qcv3_10rpm_20260823", "GPT-OSS-120B — Groq", "GPT-OSS-120B", "on", "Groq"),
)
RUN_SPECS = BASE_RUN_SPECS

# Registered future order only. These are deliberately excluded from RUN_SPECS
# until each run is complete, audited, and the versioned semantic-v2 corpus exists.
FUTURE_RUN_SPECS = (
    (None, "GPT-5.6 Luna — no-think", "GPT-5.6 Luna", "off", "OpenAI"),
    (None, "GPT-5.6 Luna — think, medium", "GPT-5.6 Luna", "on", "OpenAI"),
    (None, "Claude Sonnet 5 — no-think", "Claude", "off", "Anthropic"),
    (None, "Claude Sonnet 5 — think, adaptive xHigh", "Claude", "on", "Anthropic"),
)
FUTURE_MODEL_IDS = {
    "GPT-5.6 Luna — no-think": "gpt-5.6-luna",
    "GPT-5.6 Luna — think, medium": "gpt-5.6-luna",
    "Claude Sonnet 5 — no-think": "claude-sonnet-5",
    "Claude Sonnet 5 — think, adaptive xHigh": "claude-sonnet-5",
}

CONTRASTS = {
    "H8": (Condition.INDUCED_PROMISE_NO_SWITCH, Condition.NO_COMMUNICATION),
    "H9": (Condition.INDUCED_PROMISE_NO_SWITCH, Condition.INDUCED_PROMISE_SWITCH),
    "H10": (
        Condition.INDUCED_PROMISE_NO_SWITCH,
        Condition.EXPECTATION_WITHOUT_OWN_PROMISE,
    ),
}
VOLUNTARY_CONTRAST = (
    Condition.VOLUNTARY_PROMISE_NO_SWITCH,
    Condition.VOLUNTARY_PROMISE_SWITCH,
)
SEMANTIC_CATEGORIES = {
    "explicit_promise": ("yes", "no", "unclear"),
    "commitment_present": ("yes", "no", "unclear"),
    "statement_type": (
        "proposal", "no_substantive_statement", "intention", "commitment",
        "explicit_promise", "refusal", "ambiguous",
    ),
    "promised_action": ("NOT_APPLICABLE", "COOPERATE", "DEFECT", "OTHER", "UNCLEAR"),
    "conditionality": ("not_applicable", "unconditional", "conditional", "unclear"),
    "refusal_present": (
        "no_refusal", "refusal_to_promise", "refusal_to_cooperate",
        "other_refusal", "unclear",
    ),
    "ambiguous_response": ("no", "yes"),
}
EXPECTED_SEMANTIC_COUNTS = {
    "explicit_promise": {"yes": 1, "no": 2737, "unclear": 0},
    "commitment_present": {"yes": 255, "no": 2483, "unclear": 0},
    "statement_type": {
        "proposal": 1224, "no_substantive_statement": 743, "intention": 516,
        "commitment": 254, "explicit_promise": 1, "refusal": 0, "ambiguous": 0,
    },
    "promised_action": {
        "NOT_APPLICABLE": 2483, "COOPERATE": 235, "DEFECT": 17,
        "OTHER": 3, "UNCLEAR": 0,
    },
    "conditionality": {
        "not_applicable": 1967, "unconditional": 653, "conditional": 118,
        "unclear": 0,
    },
    "refusal_present": {
        "no_refusal": 2737, "refusal_to_promise": 1,
        "refusal_to_cooperate": 0, "other_refusal": 0, "unclear": 0,
    },
    "ambiguous_response": {"no": 2738, "yes": 0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout.strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def expected_counts(table: list[list[int]]) -> list[float]:
    row_totals = [sum(row) for row in table]
    column_totals = [sum(row[i] for row in table) for i in range(2)]
    total = sum(row_totals)
    return [row_totals[r] * column_totals[c] / total for r in range(2) for c in range(2)]


def two_sample_test(c_t: int, n_t: int, c_c: int, n_c: int) -> tuple[str, float, float]:
    if min(n_t, n_c) <= 0:
        raise ValueError("two-sample comparison requires nonempty arms")
    table = [[c_t, n_t - c_t], [c_c, n_c - c_c]]
    minimum_expected = min(expected_counts(table))
    if minimum_expected < 5:
        p_value = float(fisher_exact(table, alternative="two-sided").pvalue)
        return "Fisher exact (two-sided)", p_value, minimum_expected
    p_value = float(chi2_contingency(table, correction=False).pvalue)
    return "Pearson chi-square (two-sided, no Yates correction)", p_value, minimum_expected


def newcombe_ci(c_t: int, n_t: int, c_c: int, n_c: int) -> tuple[float, float]:
    low, high = confint_proportions_2indep(
        c_t, n_t, c_c, n_c, method="newcomb", compare="diff", alpha=ALPHA,
    )
    return float(low), float(high)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    size = len(values)
    for rank, index in enumerate(order):
        running = max(running, (size - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def load_trials() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    key = json.loads(SEMANTIC_KEY_PATH.read_text(encoding="utf-8"))
    for commit in REQUIRED_COMMITS:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT, check=True, capture_output=True,
        )
    frozen_hashes = {
        "annotation_schema_sha256": sha256(SCHEMA_PATH),
        "codebook_markdown_sha256": sha256(CODEBOOK_PATH),
        "judge_prompt_sha256": sha256(PROMPT_PATH),
        "python_validator_sha256": sha256(PYTHON_VALIDATOR_PATH),
    }
    for name, observed in frozen_hashes.items():
        if key["file_hashes"].get(name) != observed:
            raise ValueError(f"frozen scientific artifact hash mismatch: {name}")
    expected_runs = [Path(spec[0]).name for spec in (
        RUN_SPECS if SEMANTIC_V2_ACTIVE else BASE_RUN_SPECS
    )]
    if key.get("source_runs") != expected_runs:
        raise ValueError("semantic source runs do not match the active completed configuration set")
    all_trials: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    for order, (run_name, label, family, display_reasoning, display_provider) in enumerate(RUN_SPECS):
        run_dir = Path(run_name)
        if not run_dir.is_absolute():
            run_dir = RUN_ROOT / run_dir
        manifest = validate_run_integrity(
            run_dir, required_dataset_role="confirmatory", require_completed=True,
        )
        config = manifest["model_configurations"][0]
        trials = read_jsonl(run_dir / "trials.jsonl")
        assignments = {row["trial_id"]: row for row in read_jsonl(run_dir / "assignments.jsonl")}
        attempts = read_jsonl(run_dir / "api_attempts.jsonl")
        attempt_ids = [row["attempt_id"] for row in attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError(f"duplicate API-attempt identity in {run_name}")
        for row in trials:
            trial = dict(row)
            trial.update({
                "configuration_order": order,
                "configuration": label,
                "model_family_display": family,
                "provider_display": display_provider,
                "reasoning_display": display_reasoning,
                "model_alias": config["model_alias"],
                "model_id": config["model_id"],
                "source_run": run_name,
                "source_run_name": Path(run_name).name,
            })
            assignment = assignments.get(row["trial_id"])
            trial["post_message_randomized"] = bool(
                assignment and row.get("message_regime") == "voluntary"
            )
            all_trials.append(trial)
        run_metadata.append({
            "configuration": label,
            "run_name": run_name,
            "provider": config["provider"],
            "provider_display": display_provider,
            "model_alias": config["model_alias"],
            "model_id": config["model_id"],
            "model_family": family,
            "reasoning_mode": config["reasoning_mode"],
            "terminal_trials": len(trials),
            "api_attempt_records": len(attempts),
        })
    trial_ids = [(row["source_run"], row["trial_id"]) for row in all_trials]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("duplicate trial identity across included runs")
    return all_trials, run_metadata, key


def status_row(rows: list[dict[str, Any]], spec: tuple[str, str, str, str, str], condition: Condition) -> dict[str, Any]:
    run_name, label, family, reasoning, provider = spec
    arm = [row for row in rows if row["source_run"] == run_name and row.get("condition") == condition.value]
    valid = [
        row for row in arm
        if row.get("completion_status") == "completed"
        and row.get("final_action") in {"COOPERATE", "DEFECT"}
    ]
    cooperate = sum(row["final_action"] == "COOPERATE" for row in valid)
    defect = sum(row["final_action"] == "DEFECT" for row in valid)
    behavioral = sum(row.get("failure_category") == "behavioral" for row in arm)
    infrastructure = sum(row.get("failure_category") == "infrastructure" for row in arm)
    missing = len(arm) - len(valid)
    other_missing = missing - behavioral - infrastructure
    if other_missing:
        raise ValueError(f"unclassified missing outcomes in {label} | {condition.value}")
    return {
        "configuration": label,
        "source_run": run_name,
        "model_family": family,
        "provider": provider,
        "reasoning_mode": reasoning,
        "condition": condition.value,
        "condition_label": scientific_condition_label(condition),
        "assigned_trials": len(arm),
        "valid_binary_n": len(valid),
        "cooperate_n": cooperate,
        "defect_n": defect,
        "cooperation_rate": cooperate / len(valid),
        "behavioral_missing_n": behavioral,
        "behavioral_missing_rate": behavioral / len(arm),
        "infrastructure_missing_n": infrastructure,
        "infrastructure_missing_rate": infrastructure / len(arm),
        "total_missing_n": missing,
        "total_missing_rate": missing / len(arm),
    }


def make_contrast(
    hypothesis: str,
    model_order: int,
    treatment: dict[str, Any],
    comparison: dict[str, Any],
    classification: str,
) -> dict[str, Any]:
    c_t, n_t = treatment["cooperate_n"], treatment["valid_binary_n"]
    c_c, n_c = comparison["cooperate_n"], comparison["valid_binary_n"]
    rate_t, rate_c = c_t / n_t, c_c / n_c
    low, high = newcombe_ci(c_t, n_t, c_c, n_c)
    test, p_value, minimum_expected = two_sample_test(c_t, n_t, c_c, n_c)
    return {
        "analysis_classification": classification,
        "configuration_order": model_order,
        "configuration": treatment["configuration"],
        "hypothesis": hypothesis,
        "treatment_condition": treatment["condition"],
        "treatment_label": treatment["condition_label"],
        "treatment_assigned_n": treatment["assigned_trials"],
        "treatment_valid_n": n_t,
        "treatment_cooperate_n": c_t,
        "treatment_defect_n": n_t - c_t,
        "treatment_rate": rate_t,
        "comparison_condition": comparison["condition"],
        "comparison_label": comparison["condition_label"],
        "comparison_assigned_n": comparison["assigned_trials"],
        "comparison_valid_n": n_c,
        "comparison_cooperate_n": c_c,
        "comparison_defect_n": n_c - c_c,
        "comparison_rate": rate_c,
        "risk_difference": rate_t - rate_c,
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_method": "Newcombe-Wilson difference of independent proportions",
        "test": test,
        "minimum_expected_cell_count": minimum_expected,
        "raw_p_value": p_value,
    }


def missingness_bound(contrast: dict[str, Any]) -> dict[str, Any]:
    n_t, n_c = contrast["treatment_valid_n"], contrast["comparison_valid_n"]
    total_t, total_c = contrast["treatment_assigned_n"], contrast["comparison_assigned_n"]
    m_t, m_c = total_t - n_t, total_c - n_c
    c_t, c_c = contrast["treatment_cooperate_n"], contrast["comparison_cooperate_n"]
    lower = c_t / total_t - (c_c + m_c) / total_c
    upper = (c_t + m_t) / total_t - c_c / total_c
    complete_t, complete_c = n_t, n_c
    missing_test, missing_p, missing_expected = two_sample_test(
        complete_t, total_t, complete_c, total_c,
    )
    return {
        "analysis_classification": contrast["analysis_classification"],
        "configuration": contrast["configuration"],
        "hypothesis": contrast["hypothesis"],
        "treatment_condition": contrast["treatment_condition"],
        "comparison_condition": contrast["comparison_condition"],
        "treatment_assigned_n": total_t,
        "treatment_valid_n": n_t,
        "treatment_missing_n": m_t,
        "treatment_missing_rate": m_t / total_t,
        "comparison_assigned_n": total_c,
        "comparison_valid_n": n_c,
        "comparison_missing_n": m_c,
        "comparison_missing_rate": m_c / total_c,
        "absolute_missingness_difference": m_t / total_t - m_c / total_c,
        "missingness_test": missing_test,
        "missingness_minimum_expected_cell_count": missing_expected,
        "missingness_diagnostic_p_value": missing_p,
        "observed_risk_difference": contrast["risk_difference"],
        "extreme_case_lower_risk_difference": lower,
        "extreme_case_upper_risk_difference": upper,
        "worst_case_remains_positive": lower > 0,
    }


def semantic_records(key: dict[str, Any], trials: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations = read_jsonl(ANNOTATION_PATH)
    if len(annotations) != SEMANTIC_EXPECTED_COUNT:
        raise ValueError(
            f"expected {SEMANTIC_EXPECTED_COUNT} semantic annotations, found {len(annotations)}"
        )
    annotation_ids = [row.get("item_id") for row in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("duplicate semantic annotation item_id")
    key_items = key.get("items", [])
    if len(key_items) != SEMANTIC_EXPECTED_COUNT or {
        row["item_id"] for row in key_items
    } != set(annotation_ids):
        raise ValueError("semantic annotation/key coverage mismatch")
    schema = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    by_id = {row["item_id"]: row for row in annotations}
    trial_by_key = {(row["source_run_name"], row["trial_id"]): row for row in trials}
    joined: list[dict[str, Any]] = []
    for item in key_items:
        annotation = by_id[item["item_id"]]
        errors = sorted(schema.iter_errors(annotation), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"JSON Schema invalid annotation {item['item_id']}: {errors[0].message}")
        validate_annotation_record(annotation)
        if annotation["codebook_version"] != CODEBOOK_VERSION:
            raise ValueError("semantic codebook version mismatch")
        if hashlib.sha256(annotation["message"].encode("utf-8")).hexdigest() != item["message_sha256"]:
            raise ValueError(f"semantic message hash mismatch: {item['item_id']}")
        trial = trial_by_key.get((item["source_run_name"], item["source_trial_id"]))
        if trial is None:
            raise ValueError(f"semantic source trial absent from included runs: {item['item_id']}")
        if trial.get("message_event_id") != item["source_message_event_id"]:
            raise ValueError(f"semantic source message-event mismatch: {item['item_id']}")
        joined.append({**annotation, **{
            "source_run": item["source_run_name"],
            "source_trial_id": item["source_trial_id"],
            "condition": trial.get("condition"),
            "final_action": trial.get("final_action"),
            "completion_status": trial.get("completion_status"),
            "configuration": trial["configuration"],
        }})
    prevalence: list[dict[str, Any]] = []
    observed_nested: dict[str, dict[str, int]] = {}
    for field, categories in SEMANTIC_CATEGORIES.items():
        counts = Counter(row[field] for row in annotations)
        observed_nested[field] = {category: counts[category] for category in categories}
        prevalence.extend(
            {"field": field, "category": category, "count": counts[category],
             "proportion": counts[category] / SEMANTIC_EXPECTED_COUNT}
            for category in categories
        )
    if not SEMANTIC_V2_ACTIVE and observed_nested != EXPECTED_SEMANTIC_COUNTS:
        raise ValueError("semantic prevalence counts do not match the frozen expected counts")
    return joined, prevalence


def analyze_subgroup(joined: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    explicit = [row for row in joined if eligible_explicit_promiser(MessageCode.from_mapping(row))]
    if not SEMANTIC_V2_ACTIVE and len(explicit) != 1:
        raise ValueError(f"expected one promise-conditioned record, found {len(explicit)}")
    promise_status = {
        "definition": "explicit_promise=yes AND promised_action=COOPERATE AND conditionality=unconditional AND ambiguous_response=no",
        "n": len(explicit),
        "counts_by_randomized_arm": dict(sorted(Counter(row.get("condition") or "unassigned" for row in explicit).items())),
        "formal_between_arm_analysis_estimable": False,
        "status": (
            "The pre-specified promise-conditioned behavioral comparison is not meaningfully "
            "estimable because only one qualifying voluntary explicit promise occurred."
            if not SEMANTIC_V2_ACTIVE else
            f"Explicit semantic-v2 activation contains {len(explicit)} qualifying voluntary "
            "explicit promise records; this remains a separately identified descriptive subset."
        ),
    }
    undertakings = [
        row for row in joined
        if (row["explicit_promise"] == "yes" or row["commitment_present"] == "yes")
        and row["promised_action"] == "COOPERATE"
        and row["conditionality"] == "unconditional"
        and row["ambiguous_response"] == "no"
        and row.get("condition") in {condition.value for condition in VOLUNTARY_CONTRAST}
    ]
    arm_rows = {}
    for condition in VOLUNTARY_CONTRAST:
        arm = [row for row in undertakings if row["condition"] == condition.value]
        valid = [
            row for row in arm
            if row.get("completion_status") == "completed"
            and row.get("final_action") in {"COOPERATE", "DEFECT"}
        ]
        arm_rows[condition] = {
            "n": len(arm), "valid_n": len(valid),
            "cooperate_n": sum(row["final_action"] == "COOPERATE" for row in valid),
        }
    treatment, comparison = (arm_rows[condition] for condition in VOLUNTARY_CONTRAST)
    c_t, n_t = treatment["cooperate_n"], treatment["valid_n"]
    c_c, n_c = comparison["cooperate_n"], comparison["valid_n"]
    low, high = newcombe_ci(c_t, n_t, c_c, n_c)
    test, p_value, minimum_expected = two_sample_test(c_t, n_t, c_c, n_c)
    undertaking = [{
        "analysis_classification": "SECONDARY SELECTED-SUBGROUP DESCRIPTIVE ROBUSTNESS; NOT ITT",
        "subset_n": len(undertakings),
        "no_switch_n": treatment["n"],
        "no_switch_valid_n": n_t,
        "no_switch_cooperate_n": c_t,
        "no_switch_rate": c_t / n_t,
        "switch_n": comparison["n"],
        "switch_valid_n": n_c,
        "switch_cooperate_n": c_c,
        "switch_rate": c_c / n_c,
        "risk_difference_no_switch_minus_switch": c_t / n_t - c_c / n_c,
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_method": "Newcombe-Wilson difference of independent proportions",
        "test": test,
        "minimum_expected_cell_count": minimum_expected,
        "raw_p_value": p_value,
    }]
    compliance: list[dict[str, Any]] = []
    scope_filters = {
        "strict_explicit_promises": lambda row: row["explicit_promise"] == "yes",
        "explicit_promises_plus_unequivocal_commitments": lambda row: (
            row["explicit_promise"] == "yes" or row["commitment_present"] == "yes"
        ),
    }
    arm_filters = {
        "all_voluntary_arms": lambda row: row.get("condition") in {c.value for c in VOLUNTARY_CONTRAST},
        "voluntary_no_switch": lambda row: row.get("condition") == VOLUNTARY_CONTRAST[0].value,
        "voluntary_switch": lambda row: row.get("condition") == VOLUNTARY_CONTRAST[1].value,
    }
    for scope, scope_filter in scope_filters.items():
        for arm_name, arm_filter in arm_filters.items():
            selected = [row for row in joined if scope_filter(row) and arm_filter(row)]
            stated = [row for row in selected if row["promised_action"] in {"COOPERATE", "DEFECT"}]
            final = [row for row in selected if row.get("final_action") in {"COOPERATE", "DEFECT"}]
            joint = [row for row in stated if row.get("final_action") in {"COOPERATE", "DEFECT"}]
            derived = [derived_outcomes(MessageCode.from_mapping(row), row["final_action"]) for row in joint]
            cooperate_statements = [row for row in joint if row["promised_action"] == "COOPERATE"]
            violations = sum(row["final_action"] != "COOPERATE" for row in cooperate_statements)
            compliance_n = sum(result["compliance"] is True for result in derived)
            compliance.append({
                "analysis_classification": "DESCRIPTIVE SELECTED-MESSAGE SUBGROUP; NO CAUSAL INTERPRETATION",
                "scope": scope,
                "arm": arm_name,
                "selected_n": len(selected),
                "classifiable_stated_action_n": len(stated),
                "classifiable_final_action_n": len(final),
                "jointly_classifiable_n": len(joint),
                "compliance_n": compliance_n,
                "compliance_denominator": len(joint),
                "compliance_rate": compliance_n / len(joint) if joint else None,
                "cooperate_statement_with_final_n": len(cooperate_statements),
                "cooperate_statement_violation_n": violations,
                "cooperate_statement_violation_rate": violations / len(cooperate_statements) if cooperate_statements else None,
            })
    return promise_status, undertaking, compliance


def excluded_diagnostic() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = RUN_ROOT / EXCLUDED_RUN
    manifest = validate_run_integrity(
        run_dir, required_dataset_role="confirmatory", require_completed=False,
    )
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    trials = read_jsonl(run_dir / "trials.jsonl")
    assignments = read_jsonl(run_dir / "assignments.jsonl")
    rows: list[dict[str, Any]] = []
    for condition in Condition:
        arm_trials = [row for row in trials if row.get("condition") == condition.value]
        arm_assignments = [row for row in assignments if row.get("condition") == condition.value]
        valid = [
            row for row in arm_trials
            if row.get("completion_status") == "completed"
            and row.get("final_action") in {"COOPERATE", "DEFECT"}
        ]
        behavioral = sum(row.get("failure_category") == "behavioral" for row in arm_trials)
        infrastructure = sum(row.get("failure_category") == "infrastructure" for row in arm_trials)
        rows.append({
            "analysis_status": "EXCLUDED FROM CONFIRMATORY INFERENCE",
            "configuration": "DeepSeek V4 Flash — reasoning ON",
            "source_run": EXCLUDED_RUN,
            "condition": condition.value,
            "condition_label": scientific_condition_label(condition),
            "assigned_before_pause_n": len(arm_assignments),
            "attempted_terminal_trials_n": len(arm_trials),
            "not_terminal_at_stop_n": len(arm_assignments) - len(arm_trials),
            "valid_binary_n": len(valid),
            "cooperate_n": sum(row["final_action"] == "COOPERATE" for row in valid),
            "defect_n": sum(row["final_action"] == "DEFECT" for row in valid),
            "behavioral_failure_n": behavioral,
            "infrastructure_failure_n": infrastructure,
            "missing_among_attempted_n": len(arm_trials) - len(valid),
            "missing_rate_among_attempted": (len(arm_trials) - len(valid)) / len(arm_trials) if arm_trials else None,
            "run_status": state["run_status"],
        })
    pre = [row for row in trials if row.get("condition") is None]
    summary = {
        "analysis_status": "EXCLUDED FROM CONFIRMATORY INFERENCE",
        "run_status": state["run_status"],
        "terminal_trials": len(trials),
        "assignments_before_pause": len(assignments),
        "valid_binary_outcomes": sum(row.get("final_action") in {"COOPERATE", "DEFECT"} for row in trials),
        "behavioral_failures": sum(row.get("failure_category") == "behavioral" for row in trials),
        "infrastructure_failures": sum(row.get("failure_category") == "infrastructure" for row in trials),
        "pre_randomization_voluntary_failures": len(pre),
        "interpretation": "Observed instability under this specific model/provider/reasoning configuration; no intrinsic model or provider claim.",
    }
    return rows, summary


def make_heterogeneity(confirmatory: list[dict[str, Any]], voluntary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_effects = confirmatory + voluntary
    lookup = {(row["configuration"], row["hypothesis"]): row for row in all_effects}
    rows: list[dict[str, Any]] = []
    for family in ("Qwen3-32B", "GLM-5.2", "MiniMax M3", "DeepSeek V4 Pro"):
        on = f"{family} — reasoning ON"
        off = f"{family} — reasoning OFF"
        for hypothesis in ("H8", "H9", "H10", "Voluntary ITT"):
            a, b = lookup[(on, hypothesis)], lookup[(off, hypothesis)]
            rows.append({
                "comparison_type": "matched_reasoning_mode_descriptive",
                "group": family,
                "hypothesis": hypothesis,
                "configuration_a": on,
                "risk_difference_a": a["risk_difference"],
                "configuration_b": off,
                "risk_difference_b": b["risk_difference"],
                "descriptive_difference_a_minus_b": a["risk_difference"] - b["risk_difference"],
                "formal_interaction_test": False,
            })
    for hypothesis in ("H8", "H9", "H10", "Voluntary ITT"):
        a = lookup[("GPT-OSS-120B — Nebius", hypothesis)]
        b = lookup[("GPT-OSS-120B — Groq", hypothesis)]
        rows.append({
            "comparison_type": "matched_provider_control_descriptive",
            "group": "GPT-OSS-120B",
            "hypothesis": hypothesis,
            "configuration_a": "GPT-OSS-120B — Nebius",
            "risk_difference_a": a["risk_difference"],
            "configuration_b": "GPT-OSS-120B — Groq",
            "risk_difference_b": b["risk_difference"],
            "descriptive_difference_a_minus_b": a["risk_difference"] - b["risk_difference"],
            "formal_interaction_test": False,
        })
    return rows


def forest_plot(rows: list[dict[str, Any]], hypothesis: str, filename: str) -> None:
    selected = sorted(
        (row for row in rows if row["hypothesis"] == hypothesis),
        key=lambda row: row["configuration_order"], reverse=True,
    )
    values = [row["risk_difference"] for row in selected]
    lower = [value - row["ci_95_low"] for value, row in zip(values, selected)]
    upper = [row["ci_95_high"] - value for value, row in zip(values, selected)]
    labels = [row["configuration"] for row in selected]
    figure, axis = plt.subplots(figsize=(9.5, max(7.2, 0.46 * len(selected) + 1.2)))
    positions = range(len(selected))
    axis.errorbar(values, positions, xerr=[lower, upper], fmt="o", color="#1f4e79", capsize=3)
    axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
    axis.set_yticks(list(positions), labels)
    axis.set_xlabel("Absolute risk difference (treatment minus comparison)")
    axis.set_title(f"{hypothesis}: model-configuration-specific estimates")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300)
    figure.savefig(OUTPUT_DIR / f"{filename}.pdf")
    plt.close(figure)


def save_publication_figure(figure: Any, filename: str) -> None:
    figure.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def publication_confirmatory_forest(rows: list[dict[str, Any]]) -> None:
    hypotheses = (
        ("H8", "Induced Promise--No-Switch\nminus No Communication"),
        ("H9", "Induced Promise--No-Switch\nminus Induced Promise--Switch"),
        ("H10", "Induced Promise--No-Switch\nminus Expectation Without\nOwn Promise"),
    )
    selected = {
        hypothesis: sorted(
            (row for row in rows if row["hypothesis"] == hypothesis),
            key=lambda row: row["configuration_order"],
        )
        for hypothesis, _ in hypotheses
    }
    limits = [
        100 * row[field]
        for hypothesis, _ in hypotheses
        for row in selected[hypothesis]
        for field in ("ci_95_low", "ci_95_high")
    ] + [0.0]
    span = max(limits) - min(limits)
    padding = max(4.0, 0.04 * span)
    x_limits = (min(limits) - padding, max(limits) + padding)
    labels = [row["configuration"] for row in selected["H8"]]
    positions = list(range(len(labels)))
    significant_color = "#1f4e79"
    figure, axes = plt.subplots(
        1, 3, figsize=(9.2, max(5.8, 0.34 * len(labels) + 1.4)), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    for axis, (hypothesis, comparison) in zip(axes, hypotheses, strict=True):
        for position, row in zip(positions, selected[hypothesis], strict=True):
            estimate = 100 * row["risk_difference"]
            error = [[estimate - 100 * row["ci_95_low"]],
                     [100 * row["ci_95_high"] - estimate]]
            significant = row["holm_significant_alpha_0_05"]
            axis.errorbar(
                estimate, position, xerr=error, fmt="o", markersize=5.2,
                markerfacecolor=significant_color if significant else "white",
                markeredgecolor=significant_color, markeredgewidth=1.1,
                ecolor=significant_color, elinewidth=1.1, capsize=2.5,
            )
        axis.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=0)
        axis.set_xlim(*x_limits)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_title(f"{hypothesis}\n{comparison}", fontsize=10.5, pad=8)
        axis.tick_params(axis="both", labelsize=10.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    figure.supxlabel("Absolute risk difference (percentage points)", fontsize=11, y=0.035)
    figure.legend(
        handles=(
            Line2D([0], [0], marker="o", color=significant_color,
                   markerfacecolor=significant_color, linestyle="none",
                   label="Holm-adjusted p < .05"),
            Line2D([0], [0], marker="o", color=significant_color,
                   markerfacecolor="white", linestyle="none",
                   label="Not significant after Holm adjustment"),
        ),
        loc="lower center", bbox_to_anchor=(0.57, -0.005), ncol=2,
        frameon=False, fontsize=10,
    )
    figure.subplots_adjust(left=0.29, right=0.99, top=0.88, bottom=0.15)
    save_publication_figure(figure, "fig_PS_confirmatory_forest")


def publication_condition_heatmap(rows: list[dict[str, Any]]) -> None:
    lookup = {(row["configuration"], row["condition"]): row for row in rows}
    labels = list(dict.fromkeys(row["configuration"] for row in rows))
    conditions = tuple(Condition)
    values = [
        [100 * lookup[(label, condition.value)]["cooperation_rate"] for condition in conditions]
        for label in labels
    ]
    column_labels = [scientific_condition_label(condition) for condition in conditions]
    figure, axis = plt.subplots(figsize=(8.8, max(5.7, 0.34 * len(labels) + 1.3)))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(conditions)), column_labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(axis="x", labelsize=10, length=0, pad=7, rotation=32)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.tick_params(axis="y", labelsize=10.5, length=0)
    axis.spines[:].set_visible(False)
    axis.set_xticks([value - 0.5 for value in range(1, len(conditions))], minor=True)
    axis.set_yticks([value - 0.5 for value in range(1, len(labels))], minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index, row_values in enumerate(values):
        for column_index, value in enumerate(row_values):
            axis.text(
                column_index, row_index, f"{value:.1f}%",
                ha="center", va="center", fontsize=10,
                color="white" if value >= 55 else "#202020",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Cooperation rate (%)", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=10)
    figure.subplots_adjust(left=0.32, right=0.94, top=0.98, bottom=0.34)
    save_publication_figure(figure, "fig_PS_condition_heatmap")


def publication_voluntary_dumbbell(rows: list[dict[str, Any]]) -> None:
    selected = sorted(rows, key=lambda row: row["configuration_order"])
    labels = [row["configuration"] for row in selected]
    positions = list(range(len(selected)))
    no_switch_color = "#1f4e79"
    switch_color = "#d27b26"
    figure, axis = plt.subplots(figsize=(8.2, max(5.7, 0.34 * len(selected) + 1.3)))
    for position, row in zip(positions, selected, strict=True):
        no_switch = 100 * row["treatment_rate"]
        switch = 100 * row["comparison_rate"]
        significant = row["holm_significant_alpha_0_05"]
        axis.plot(
            [switch, no_switch], [position, position],
            color="#555555" if significant else "#b8b8b8",
            linewidth=1.7 if significant else 1.0, zorder=1,
        )
        axis.scatter(no_switch, position, s=31, color=no_switch_color, zorder=2)
        axis.scatter(switch, position, s=31, color=switch_color, zorder=2)
        difference = 100 * row["risk_difference"]
        marker = "*" if significant else ""
        axis.text(
            1.055, position, f"{difference:+.1f}{marker}",
            transform=axis.get_yaxis_transform(), ha="left", va="center",
            fontsize=10, fontweight="bold" if significant else "normal",
        )
    axis.set_xlim(-2, 102)
    axis.set_xticks(range(0, 101, 20))
    axis.set_ylim(-0.7, len(selected) - 0.3)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Cooperation rate (%)", fontsize=11)
    axis.tick_params(axis="both", labelsize=10)
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        1.055, 1.005, "NS − S\n(pp)", transform=axis.transAxes,
        ha="left", va="bottom", fontsize=10,
    )
    axis.legend(
        handles=(
            Line2D([0], [0], marker="o", color=no_switch_color,
                   markerfacecolor=no_switch_color, linestyle="none",
                   label="Voluntary Message--No-Switch"),
            Line2D([0], [0], marker="o", color=switch_color,
                   markerfacecolor=switch_color, linestyle="none",
                   label="Voluntary Message--Switch"),
        ),
        loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1,
        columnspacing=1.0, handletextpad=0.45, frameon=False, fontsize=9.2,
    )
    figure.text(0.69, 0.025, "* Holm-adjusted p < .05", fontsize=9.5)
    figure.subplots_adjust(left=0.35, right=0.86, top=0.91, bottom=0.13)
    save_publication_figure(figure, "fig_PS_voluntary_itt")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != 0 and abs(value) < 10 ** -digits:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in fields) + " |"
    separator = "|" + "|".join("---" for _ in fields) + "|"
    body = ["| " + " | ".join(fmt(row.get(field)) for field, _ in fields) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_report(
    condition_rows: list[dict[str, Any]], confirmatory: list[dict[str, Any]],
    voluntary: list[dict[str, Any]], bounds: list[dict[str, Any]],
    promise_status: dict[str, Any], undertaking: list[dict[str, Any]],
    compliance: list[dict[str, Any]], excluded: dict[str, Any],
    pre_randomization: list[dict[str, Any]],
) -> str:
    contrast_fields = [
        ("configuration", "Configuration"), ("treatment_cooperate_n", "T coop"),
        ("treatment_valid_n", "T N"), ("treatment_rate", "T rate"),
        ("comparison_cooperate_n", "C coop"), ("comparison_valid_n", "C N"),
        ("comparison_rate", "C rate"), ("risk_difference", "RD"),
        ("ci_95_low", "CI low"), ("ci_95_high", "CI high"),
        ("test", "Test"), ("raw_p_value", "Raw p"),
        ("holm_adjusted_p_value", "Holm p"),
    ]
    lines = [
        "# Final Promise Switch statistical analysis",
        "",
        "All inferential estimates are model-configuration-specific. No pooled confirmatory test was calculated.",
        "",
        "## Included configurations",
        "",
        *[f"- {spec[1]} (`{spec[0]}`)" for spec in RUN_SPECS],
        "",
    ]
    for hypothesis in ("H8", "H9", "H10"):
        lines.extend([
            f"## {hypothesis}", "",
            markdown_table([row for row in confirmatory if row["hypothesis"] == hypothesis], contrast_fields),
            "",
        ])
    lines.extend([
        "## Secondary randomized Voluntary Message ITT", "",
        markdown_table(voluntary, contrast_fields), "",
        "## Hypothesis-support summary", "",
    ])
    for hypothesis in ("H8", "H9", "H10"):
        selected = [row for row in confirmatory if row["hypothesis"] == hypothesis]
        positive = sum(row["risk_difference"] > 0 for row in selected)
        significant = sum(row["support_classification"] == "directionally supported and Holm-significant" for row in selected)
        lines.append(f"- {hypothesis}: {positive}/{len(RUN_SPECS)} positive; {significant}/{len(RUN_SPECS)} Holm-significant in the predicted direction.")
    aggregate_assigned = sum(row["assigned_trials"] for row in condition_rows)
    aggregate_missing = sum(row["total_missing_n"] for row in condition_rows)
    pre_randomization_missing = sum(row["pre_randomization_terminal_failures_n"] for row in pre_randomization)
    worst_positive = Counter(row["hypothesis"] for row in bounds if row["worst_case_remains_positive"])
    lines.extend([
        "", "## Attrition and extreme-case bounds", "",
        f"Across assigned condition-level trials, {aggregate_missing}/{aggregate_assigned} final outcomes were missing. Detailed model-by-condition and contrast-level diagnostics are in the CSV outputs.",
        f"Separately, {pre_randomization_missing} voluntary-message trials failed before receiving a Switch/No-Switch assignment.",
        "",
        *[f"- {hypothesis}: worst-case RD remained positive for {worst_positive[hypothesis]}/{len(RUN_SPECS)} configurations." for hypothesis in ("H8", "H9", "H10", "Voluntary ITT")],
        "", "## Semantic analyses", "",
        f"The frozen semantic prevalence counts matched all expected counts across 2,738 annotations. Promise-conditioned subset n={promise_status['n']}.",
        "",
        promise_status["status"],
        "",
        f"The secondary undertaking-to-cooperate subset contained {undertaking[0]['subset_n']} observations; No-Switch minus Switch RD={fmt(undertaking[0]['risk_difference_no_switch_minus_switch'])}, 95% Newcombe CI [{fmt(undertaking[0]['ci_95_low'])}, {fmt(undertaking[0]['ci_95_high'])}], raw p={fmt(undertaking[0]['raw_p_value'])}. This is a selected post-message subgroup, not ITT.",
        "", "Compliance and violation numerators and denominators are reported in `compliance_descriptives.csv`.",
        "", "## Excluded configuration", "",
        f"DeepSeek V4 Flash reasoning ON is EXCLUDED FROM CONFIRMATORY INFERENCE. The run stopped with status `{excluded['run_status']}` after {excluded['terminal_trials']} terminal trials; {excluded['valid_binary_outcomes']} had valid binary outcomes, {excluded['behavioral_failures']} were behavioral failures, and {excluded['infrastructure_failures']} were infrastructure failures.",
        "", "The diagnostic describes only this specific model/provider/reasoning configuration.",
        "", "## Statistical definitions", "",
        "- Confidence intervals: 95% Newcombe-Wilson intervals for independent-proportion differences.",
        "- Tests: two-sided Fisher exact if any expected cell count was below 5; otherwise two-sided Pearson chi-square without Yates correction.",
        "- Confirmatory multiplicity: Holm correction within each configuration across H8, H9, and H10.",
        f"- Secondary multiplicity: Holm correction across the {len(RUN_SPECS)} Voluntary Message ITT comparisons.",
        "- Extreme-case missing-outcome bounds are deterministic bounds, not confidence intervals.",
        "",
    ])
    return "\n".join(lines)


def internal_checks(confirmatory: list[dict[str, Any]], voluntary: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    boundary_cases = {
        "zero_vs_zero": newcombe_ci(0, 100, 0, 100),
        "all_vs_all": newcombe_ci(100, 100, 100, 100),
        "all_vs_zero": newcombe_ci(100, 100, 0, 100),
        "zero_vs_all": newcombe_ci(0, 100, 100, 100),
    }
    for low, high in boundary_cases.values():
        if not (-1 <= low <= high <= 1):
            raise ValueError("Newcombe boundary-case validation failed")
    sample = [0.01, 0.03, 0.5]
    if any(abs(a - b) > 1e-12 for a, b in zip(holm_adjust(sample), [0.03, 0.06, 0.5])):
        raise ValueError("Holm implementation validation failed")
    fisher_name, _, fisher_min = two_sample_test(0, 100, 1, 100)
    chi_name, _, chi_min = two_sample_test(50, 100, 40, 100)
    if not (fisher_name.startswith("Fisher") and fisher_min < 5):
        raise ValueError("Fisher routing validation failed")
    if not (chi_name.startswith("Pearson") and chi_min >= 5):
        raise ValueError("Pearson routing validation failed")
    configuration_count = len(RUN_SPECS)
    if (len(confirmatory) != 3 * configuration_count
            or len(voluntary) != configuration_count
            or len(condition_rows) != len(Condition) * configuration_count):
        raise ValueError("analysis table coverage validation failed")
    for index in range(configuration_count):
        model_rows = [row for row in confirmatory if row["configuration_order"] == index]
        if {row["hypothesis"] for row in model_rows} != set(CONTRASTS):
            raise ValueError("confirmatory hypothesis coverage validation failed")
    for row in confirmatory + voluntary:
        independently_calculated = row["treatment_cooperate_n"] / row["treatment_valid_n"] - row["comparison_cooperate_n"] / row["comparison_valid_n"]
        if abs(independently_calculated - row["risk_difference"]) > 1e-12:
            raise ValueError("risk-difference arithmetic validation failed")
    return {
        "newcombe_boundary_cases": boundary_cases,
        "holm_test_input": sample,
        "holm_test_output": holm_adjust(sample),
        "fisher_route_minimum_expected": fisher_min,
        "pearson_route_minimum_expected": chi_min,
        "confirmatory_rows": len(confirmatory),
        "voluntary_itt_rows": len(voluntary),
        "condition_rows": len(condition_rows),
        "api_attempts_used_as_analysis_units": False,
        "excluded_configuration_entered_inference": False,
    }


def activation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activate-run-specs", type=Path,
        help="JSON activation file containing only completed, audited future runs",
    )
    parser.add_argument(
        "--activate-semantic-v2", type=Path,
        help="explicit finalized semantic-v2 activation manifest",
    )
    return parser


def activate_semantic_v2(path: Path, active_specs: tuple[tuple[Any, ...], ...]) -> None:
    """Activate a finalized v2 corpus only when it matches the activated future runs."""
    global ANNOTATION_PATH, SEMANTIC_KEY_PATH, SEMANTIC_EXPECTED_COUNT, SEMANTIC_V2_ACTIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("corpus_version") != "promise_switch_semantic_corpus_v2":
        raise ValueError("semantic-v2 activation has an incompatible version")
    annotations = Path(str(payload.get("annotations_path", ""))).resolve()
    key = Path(str(payload.get("private_key_path", ""))).resolve()
    if not annotations.is_file() or sha256(annotations) != payload.get("annotations_sha256"):
        raise ValueError("semantic-v2 annotation artifact hash mismatch")
    if not key.is_file() or sha256(key) != payload.get("private_key_sha256"):
        raise ValueError("semantic-v2 private-key artifact hash mismatch")
    if sha256(ANNOTATION_PATH) != payload.get("frozen_base_sha256"):
        raise ValueError("semantic-v2 activation does not reference the frozen v1 corpus")
    count = payload.get("combined_count")
    if not isinstance(count, int) or count <= 2738:
        raise ValueError("semantic-v2 activation has no validated new records")
    activated_paths = {str(Path(spec[0]).resolve()) for spec in active_specs}
    source_rows = payload.get("source_runs")
    source_paths = {
        str(Path(str(row.get("run_path", ""))).resolve())
        for row in source_rows if isinstance(row, dict)
    } if isinstance(source_rows, list) else set()
    if not source_paths or source_paths != activated_paths:
        raise ValueError("semantic-v2 sources do not match explicitly activated future runs")
    ANNOTATION_PATH = annotations
    SEMANTIC_KEY_PATH = key
    SEMANTIC_EXPECTED_COUNT = count
    SEMANTIC_V2_ACTIVE = True


def activated_specs(path: Path) -> tuple[tuple[Any, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("activation file must contain a non-empty runs list")
    allowed = {spec[1]: spec for spec in FUTURE_RUN_SPECS}
    output = []
    seen_labels: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("label") not in allowed:
            raise ValueError("activation contains an unknown future configuration label")
        label = str(row["label"])
        if label in seen_labels:
            raise ValueError("activation contains a duplicate future configuration label")
        seen_labels.add(label)
        if not isinstance(row.get("run_path"), str) or not row["run_path"].strip():
            raise ValueError("activated run requires an explicit nonempty run_path")
        template = allowed[label]
        run_dir = Path(row["run_path"]).resolve()
        manifest = validate_run_integrity(
            run_dir, required_dataset_role="confirmatory", require_completed=True,
        )
        if manifest.get("prompt_hash") != prompt_templates_hash():
            raise ValueError(f"activated run prompt hash mismatch: {run_dir}")
        if manifest.get("response_qc_version") != "promise_switch_response_qc_v3":
            raise ValueError(f"activated run QC version mismatch: {run_dir}")
        configurations = manifest.get("model_configurations")
        if not isinstance(configurations, list) or len(configurations) != 1:
            raise ValueError(f"activated run must contain exactly one model configuration: {run_dir}")
        if manifest.get("requested_completed_trials_per_final_condition") != 100:
            raise ValueError(f"activated run minimum completed target mismatch: {run_dir}")
        config = configurations[0]
        expected = {
            "model_id": FUTURE_MODEL_IDS[str(row["label"])],
            "reasoning_mode": template[3],
            "provider": str(template[4]).lower(),
        }
        observed = {
            "model_id": config.get("model_id"),
            "reasoning_mode": config.get("reasoning_mode"),
            "provider": config.get("provider"),
        }
        if observed != expected:
            raise ValueError(f"activated run model/reasoning identity mismatch: {run_dir}")
        output.append((str(run_dir), template[1], template[2], template[3], template[4]))
    order = {spec[1]: index for index, spec in enumerate(FUTURE_RUN_SPECS)}
    if [order[spec[1]] for spec in output] != sorted(order[spec[1]] for spec in output):
        raise ValueError("future activation order must match the registered append order")
    return tuple(output)


def main(argv: list[str] | None = None) -> int:
    args = activation_parser().parse_args(argv)
    global RUN_SPECS
    future_specs: tuple[tuple[Any, ...], ...] = ()
    if args.activate_run_specs:
        future_specs = activated_specs(args.activate_run_specs)
        RUN_SPECS = (*BASE_RUN_SPECS, *future_specs)
    if args.activate_semantic_v2:
        if not future_specs:
            raise ValueError("semantic-v2 activation requires explicit future run activation")
        activate_semantic_v2(args.activate_semantic_v2, future_specs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trials, run_metadata, semantic_key = load_trials()
    condition_rows = [
        status_row(trials, spec, condition)
        for spec in RUN_SPECS for condition in Condition
    ]
    by_arm = {(row["configuration"], row["condition"]): row for row in condition_rows}
    confirmatory: list[dict[str, Any]] = []
    for order, spec in enumerate(RUN_SPECS):
        for hypothesis, (treatment, comparison) in CONTRASTS.items():
            confirmatory.append(make_contrast(
                hypothesis, order, by_arm[(spec[1], treatment.value)],
                by_arm[(spec[1], comparison.value)], "CONFIRMATORY MODEL-SPECIFIC",
            ))
        family = [row for row in confirmatory if row["configuration"] == spec[1]]
        adjusted = holm_adjust(row["raw_p_value"] for row in family)
        for row, adjusted_p in zip(family, adjusted):
            row["holm_family"] = f"{spec[1]}: H8/H9/H10"
            row["holm_family_size"] = 3
            row["holm_adjusted_p_value"] = adjusted_p
            row["holm_significant_alpha_0_05"] = adjusted_p < ALPHA
            row["effect_sign"] = "positive" if row["risk_difference"] > 0 else "zero" if row["risk_difference"] == 0 else "negative"
            row["support_classification"] = (
                "directionally supported and Holm-significant"
                if row["risk_difference"] > 0 and adjusted_p < ALPHA
                else "positive RD but not Holm-significant"
                if row["risk_difference"] > 0
                else "zero/tie"
                if row["risk_difference"] == 0
                else "effect in opposite direction"
            )
    voluntary = [
        make_contrast(
            "Voluntary ITT", order,
            by_arm[(spec[1], VOLUNTARY_CONTRAST[0].value)],
            by_arm[(spec[1], VOLUNTARY_CONTRAST[1].value)],
            "SECONDARY RANDOMIZED ITT",
        )
        for order, spec in enumerate(RUN_SPECS)
    ]
    voluntary_adjusted = holm_adjust(row["raw_p_value"] for row in voluntary)
    for row, adjusted_p in zip(voluntary, voluntary_adjusted):
        row["holm_family"] = f"Secondary Voluntary Message ITT across {len(RUN_SPECS)} configurations"
        row["holm_family_size"] = len(RUN_SPECS)
        row["holm_adjusted_p_value"] = adjusted_p
        row["holm_significant_alpha_0_05"] = adjusted_p < ALPHA
    bounds = [missingness_bound(row) for row in confirmatory + voluntary]
    joined, prevalence = semantic_records(semantic_key, trials)
    promise_status, undertaking, compliance = analyze_subgroup(joined)
    heterogeneity = make_heterogeneity(confirmatory, voluntary)
    excluded_rows, excluded_summary = excluded_diagnostic()
    validation = internal_checks(confirmatory, voluntary, condition_rows)

    trial_level = [{
        "trial_id": row["trial_id"],
        "run_id": row["run_id"],
        "source_run": row["source_run"],
        "configuration": row["configuration"],
        "model_alias": row["model_alias"],
        "model_id": row["model_id"],
        "model_family": row["model_family_display"],
        "provider": row["provider"],
        "reasoning_mode": row["reasoning_mode"],
        "original_treatment_assignment": row.get("condition") or "voluntary_pre_randomization",
        "condition": row.get("condition"),
        "switched": row.get("switched"),
        "post_message_randomized": row["post_message_randomized"],
        "final_action": row.get("final_action"),
        "completion_status": row.get("completion_status"),
        "failure_category": row.get("failure_category"),
        "failure_reason": row.get("failure_reason"),
        "plan_phase": row.get("plan_phase"),
        "top_up_sequence": row.get("top_up_sequence"),
    } for row in trials]
    pre_randomization = []
    for spec in RUN_SPECS:
        rows = [
            row for row in trials
            if row["source_run"] == spec[0] and row.get("condition") is None
        ]
        pre_randomization.append({
            "configuration": spec[1],
            "source_run": spec[0],
            "pre_randomization_terminal_failures_n": len(rows),
            "behavioral_failures_n": sum(row.get("failure_category") == "behavioral" for row in rows),
            "infrastructure_failures_n": sum(row.get("failure_category") == "infrastructure" for row in rows),
        })

    write_csv(OUTPUT_DIR / "trial_level_analysis_dataset.csv", trial_level)
    write_csv(OUTPUT_DIR / "condition_counts.csv", condition_rows)
    write_csv(OUTPUT_DIR / "confirmatory_h8_h10.csv", confirmatory)
    write_json(OUTPUT_DIR / "confirmatory_h8_h10.json", confirmatory)
    write_csv(OUTPUT_DIR / "voluntary_itt.csv", voluntary)
    write_csv(OUTPUT_DIR / "attrition_by_condition.csv", condition_rows)
    write_csv(OUTPUT_DIR / "pre_randomization_attrition.csv", pre_randomization)
    write_csv(OUTPUT_DIR / "missingness_bounds.csv", bounds)
    write_csv(OUTPUT_DIR / "semantic_prevalence.csv", prevalence)
    write_json(OUTPUT_DIR / "promise_conditioned_status.json", promise_status)
    write_csv(OUTPUT_DIR / "undertaking_robustness.csv", undertaking)
    write_csv(OUTPUT_DIR / "compliance_descriptives.csv", compliance)
    write_csv(OUTPUT_DIR / "heterogeneity_descriptives.csv", heterogeneity)
    write_csv(OUTPUT_DIR / "deepseek_flash_on_excluded_diagnostic.csv", excluded_rows)
    for hypothesis, filename in (
        ("H8", "forest_h8"), ("H9", "forest_h9"), ("H10", "forest_h10"),
        ("Voluntary ITT", "forest_voluntary_itt"),
    ):
        forest_plot(confirmatory + voluntary, hypothesis, filename)
    publication_confirmatory_forest(confirmatory)
    publication_condition_heatmap(condition_rows)
    publication_voluntary_dumbbell(voluntary)
    report = build_report(
        condition_rows, confirmatory, voluntary, bounds, promise_status,
        undertaking, compliance, excluded_summary, pre_randomization,
    )
    (OUTPUT_DIR / "final_analysis_report.md").write_text(report, encoding="utf-8")

    input_paths = [
        ANNOTATION_PATH, SEMANTIC_KEY_PATH, SCHEMA_PATH, CODEBOOK_PATH,
        PROMPT_PATH, PYTHON_VALIDATOR_PATH,
    ] + sorted(
        path
        for run_name in [spec[0] for spec in RUN_SPECS] + [EXCLUDED_RUN]
        for path in (RUN_ROOT / run_name).iterdir()
        if path.is_file() and path.name != ".writer.lock"
    )
    generated_files = (
        "analysis_manifest.json", "attrition_by_condition.csv",
        "compliance_descriptives.csv", "condition_counts.csv",
        "confirmatory_h8_h10.csv", "confirmatory_h8_h10.json",
        "deepseek_flash_on_excluded_diagnostic.csv", "final_analysis_report.md",
        "forest_h10.pdf", "forest_h10.png", "forest_h8.pdf", "forest_h8.png",
        "forest_h9.pdf", "forest_h9.png", "forest_voluntary_itt.pdf",
        "forest_voluntary_itt.png", "fig_PS_condition_heatmap.pdf",
        "fig_PS_condition_heatmap.png", "fig_PS_confirmatory_forest.pdf",
        "fig_PS_confirmatory_forest.png", "fig_PS_voluntary_itt.pdf",
        "fig_PS_voluntary_itt.png", "heterogeneity_descriptives.csv",
        "missingness_bounds.csv", "pre_randomization_attrition.csv",
        "promise_conditioned_status.json", "semantic_prevalence.csv",
        "trial_level_analysis_dataset.csv", "undertaking_robustness.csv",
        "voluntary_itt.csv",
    )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "alpha": ALPHA,
        "semantic_codebook_version": CODEBOOK_VERSION,
        "required_ancestor_commits": list(REQUIRED_COMMITS),
        "analysis_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "included_configurations": run_metadata,
        "excluded_configuration": {"source_run": EXCLUDED_RUN, **excluded_summary},
        "hypothesis_definitions": {
            "H8": "Induced Promise--No-Switch minus No Communication",
            "H9": "Induced Promise--No-Switch minus Induced Promise--Switch",
            "H10": "Induced Promise--No-Switch minus Expectation Without Own Promise",
            "secondary_voluntary_itt": "Voluntary Message--No-Switch minus Voluntary Message--Switch",
        },
        "confidence_interval": "95% Newcombe-Wilson interval for the difference of two independent proportions",
        "test_selection": "Two-sided Fisher exact if any expected 2x2 cell is <5; otherwise Pearson chi-square without Yates correction",
        "holm_families": {
            "confirmatory": f"{len(RUN_SPECS)} separate within-configuration families of H8/H9/H10 (size 3)",
            "secondary": f"one family of {len(RUN_SPECS)} Voluntary Message ITT comparisons",
        },
        "analysis_unit": "randomized trial; API attempts are not analysis units",
        "input_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in input_paths
        ],
        "validation": validation,
        "generated_files": list(generated_files),
    }
    write_json(OUTPUT_DIR / "analysis_manifest.json", manifest)
    print(json.dumps({
        "analysis_version": ANALYSIS_VERSION,
        "included_configurations": len(RUN_SPECS),
        "trial_records": len(trials),
        "semantic_annotations": len(joined),
        "promise_conditioned_n": promise_status["n"],
        "output_directory": str(OUTPUT_DIR),
        "validation": validation,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
