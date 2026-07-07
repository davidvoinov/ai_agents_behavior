#!/usr/bin/env python3
"""
Run Experiment 1: Berg trust game without communication.

The runner reads model and experiment settings from YAML configs, creates an
independent investor-trustee pair for each dyad, records raw prompts and raw
model responses, validates JSON decisions, computes payoffs, and appends one
JSON object per dyad to the configured JSONL output file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency: PyYAML. Install dependencies with: pip install -r requirements.txt")

try:
    from berg import berg_trust_game as B
except ImportError:
    import berg_trust_game as B


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment_1_pilot.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config is empty or invalid: {path}")
    return data


def resolve_path(path_value: str | None, base: Path = ROOT) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def index_models(models_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = models_cfg.get("models", [])
    if not isinstance(models, list):
        raise ValueError("models_free.yaml must contain a list under 'models'.")
    by_name = {}
    for model in models:
        name = model.get("name")
        if not name:
            raise ValueError("Every model entry must have a name.")
        if name in by_name:
            raise ValueError(f"Duplicate model name in config: {name}")
        by_name[name] = model
    return by_name


def enabled_pairs(exp_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = exp_cfg.get("model_pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("Experiment config must contain a list under 'model_pairs'.")
    return [pair for pair in pairs if pair.get("enabled", False)]


def require_model_ready(model_name: str, model: dict[str, Any]) -> None:
    required_fields = ["provider", "model_family", "model_id", "api_key_env"]
    missing = [field for field in required_fields if not model.get(field)]
    if missing:
        raise ValueError(
            f"Model '{model_name}' is enabled but missing required fields: {', '.join(missing)}"
        )
    if model.get("provider") == "google":
        raise ValueError(
            f"Model '{model_name}' uses Google/Gemini. This runner currently supports "
            "OpenAI-compatible chat endpoints; add a Gemini adapter before enabling it."
        )


def check_api_key(model_name: str, model: dict[str, Any]) -> None:
    key_env = model["api_key_env"]
    key = os.environ.get(key_env, "").strip()
    if not key or key.startswith("your-"):
        raise RuntimeError(
            f"Missing API key for model '{model_name}'. Set {key_env}=... in .env."
        )


def make_model_config(model: dict[str, Any], temperature: float) -> B.ModelConfig:
    return B.ModelConfig(
        model=model["model_id"],
        base_url=model.get("base_url"),
        api_key_env=model["api_key_env"],
        temperature=temperature,
    )


def investor_decision_prompt(game: B.GameConfig) -> str:
    return (
        f"Now privately decide how many {game.unit_word} to pass to the "
        f"{game.trustee_label}. Your choice must be an integer from 0 to "
        f"{game.endowment}.\n"
        'Respond with ONLY a JSON object of the form '
        '{"sent": <integer>, "reasoning": "<brief explanation>"}.\n'
        "Do not include any text outside the JSON."
    )


def trustee_decision_prompt(game: B.GameConfig, sent: int, tripled_amount: int) -> str:
    return (
        f"The {game.investor_label} chose to pass X = {sent} {game.unit_word}. "
        f"After tripling, you received {tripled_amount} {game.unit_word} in addition "
        f"to your own starting {game.endowment}.\n"
        f"Now privately decide how many {game.unit_word} to return to the "
        f"{game.investor_label}. Your choice must be an integer from 0 to "
        f"{tripled_amount}.\n"
        'Respond with ONLY a JSON object of the form '
        '{"returned": <integer>, "reasoning": "<brief explanation>"}.\n'
        "Do not include any text outside the JSON."
    )


def apply_prompt_prefix(prompt: str, model: dict[str, Any]) -> str:
    prefix = str(model.get("prompt_prefix") or "").strip()
    if not prefix:
        return prompt
    return f"{prefix}\n{prompt}"


def normalize_response(text: str | None, model: dict[str, Any]) -> dict[str, Any]:
    if text is None:
        return {
            "text": None,
            "applied": False,
            "rule": None,
            "rules": [],
        }

    rules = model.get("response_normalization") or []
    if isinstance(rules, str):
        rules = [rules]

    normalized = text
    applied_rules: list[str] = []

    for rule in rules:
        before = normalized
        if rule == "strip_leading_think_block":
            normalized = re.sub(
                r"^\s*<think>.*?</think>\s*",
                "",
                normalized,
                count=1,
                flags=re.DOTALL,
            )
        elif rule == "unwrap_json_code_fence":
            match = re.match(
                r"^\s*```(?:json)?\s*(\{.*\})\s*```\s*$",
                normalized,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                normalized = match.group(1).strip()
        else:
            raise ValueError(f"Unknown response normalization rule: {rule}")

        if normalized != before:
            applied_rules.append(rule)

    before = normalized
    normalized = normalized.replace("\\$", "$")
    if normalized != before:
        applied_rules.append("repair_invalid_dollar_escape")

    applied = bool(applied_rules)
    return {
        "text": normalized,
        "applied": applied,
        "rule": ",".join(applied_rules) if applied else None,
        "rules": applied_rules,
    }


def parse_strict_decision(text: str | None, field: str, lo: int, hi: int) -> dict[str, Any]:
    if text is None:
        return {
            "value": None,
            "reasoning": "",
            "parsed_response": None,
            "parse_success": False,
            "validation_success": False,
            "error": "missing response text",
        }

    raw = text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "value": None,
            "reasoning": "",
            "parsed_response": None,
            "parse_success": False,
            "validation_success": False,
            "error": f"invalid JSON: {exc.msg}",
        }

    if not isinstance(data, dict):
        return {
            "value": None,
            "reasoning": "",
            "parsed_response": data,
            "parse_success": False,
            "validation_success": False,
            "error": "JSON response must be an object",
        }

    if field not in data:
        return {
            "value": None,
            "reasoning": str(data.get("reasoning", "")),
            "parsed_response": data,
            "parse_success": False,
            "validation_success": False,
            "error": f"missing required field: {field}",
        }

    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int):
        return {
            "value": None,
            "reasoning": str(data.get("reasoning", "")),
            "parsed_response": data,
            "parse_success": False,
            "validation_success": False,
            "error": f"{field} must be an integer",
        }

    if not lo <= value <= hi:
        return {
            "value": value,
            "reasoning": str(data.get("reasoning", "")),
            "parsed_response": data,
            "parse_success": True,
            "validation_success": False,
            "error": f"{field} must be between {lo} and {hi}",
        }

    return {
        "value": value,
        "reasoning": str(data.get("reasoning", "")),
        "parsed_response": data,
        "parse_success": True,
        "validation_success": True,
        "error": None,
    }


def run_no_communication_dyad(
    *,
    run_id: str,
    experiment_id: str,
    treatment: str,
    pair_id: str,
    dyad_index: int,
    investor_model: dict[str, Any],
    trustee_model: dict[str, Any],
    investor_mcfg: B.ModelConfig,
    trustee_mcfg: B.ModelConfig,
    investor_chat: B.ChatFn,
    trustee_chat: B.ChatFn,
    game: B.GameConfig,
) -> dict[str, Any]:
    investor_system = B.investor_system_prompt(game, with_comm=False)
    trustee_system = B.trustee_system_prompt(game, with_comm=False)
    investor = B.Agent("investor", investor_system, investor_chat)
    trustee = B.Agent("trustee", trustee_system, trustee_chat)

    investor_prompt = apply_prompt_prefix(investor_decision_prompt(game), investor_model)
    investor_raw = investor.respond(investor_prompt)
    investor_normalized = normalize_response(investor_raw, investor_model)
    investor_parsed = parse_strict_decision(
        investor_normalized["text"],
        "sent",
        0,
        game.endowment,
    )
    sent = investor_parsed["value"]

    if investor_parsed["validation_success"]:
        tripled_amount = sent * game.multiplier
        trustee_prompt = apply_prompt_prefix(
            trustee_decision_prompt(game, sent, tripled_amount),
            trustee_model,
        )
        trustee_raw = trustee.respond(trustee_prompt)
        trustee_normalized = normalize_response(trustee_raw, trustee_model)
        trustee_parsed = parse_strict_decision(
            trustee_normalized["text"],
            "returned",
            0,
            tripled_amount,
        )
    else:
        tripled_amount = None
        trustee_prompt = None
        trustee_raw = None
        trustee_normalized = {
            "text": None,
            "applied": False,
            "rule": None,
            "rules": [],
        }
        trustee_parsed = {
            "value": None,
            "reasoning": "",
            "parsed_response": None,
            "parse_success": False,
            "validation_success": False,
            "error": "skipped because investor choice was invalid",
        }

    returned = trustee_parsed["value"]
    valid_dyad = bool(
        investor_parsed["validation_success"]
        and trustee_parsed["validation_success"]
    )

    if valid_dyad:
        payoff_a = game.endowment - sent + returned
        payoff_b = game.endowment + tripled_amount - returned
        joint_payoff = payoff_a + payoff_b
    else:
        payoff_a = None
        payoff_b = None
        joint_payoff = None

    dyad_id = f"{pair_id}-{dyad_index:04d}"
    record = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "treatment": treatment,
        "pair_id": pair_id,
        "dyad_id": dyad_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider_A": investor_model["provider"],
        "provider_B": trustee_model["provider"],
        "model_id_A": investor_model["model_id"],
        "model_id_B": trustee_model["model_id"],
        "model_family_A": investor_model["model_family"],
        "model_family_B": trustee_model["model_family"],
        "analysis_label_A": investor_model.get("analysis_label"),
        "analysis_label_B": trustee_model.get("analysis_label"),
        "role_A": "investor",
        "role_B": "trustee",
        "temperature_A": investor_mcfg.temperature,
        "temperature_B": trustee_mcfg.temperature,
        "endowment": game.endowment,
        "multiplier": game.multiplier,
        "sent": sent,
        "tripled_amount": tripled_amount,
        "returned": returned,
        "payoff_A": payoff_a,
        "payoff_B": payoff_b,
        "joint_payoff": joint_payoff,
        "system_prompt_A": investor_system,
        "decision_prompt_A": investor_prompt,
        "raw_response_A": investor_raw,
        "normalized_response_A": investor_normalized["text"],
        "normalization_applied_A": investor_normalized["applied"],
        "normalization_rule_A": investor_normalized["rule"],
        "normalization_rules_A": investor_normalized["rules"],
        "parsed_response_A": investor_parsed["parsed_response"],
        "system_prompt_B": trustee_system,
        "decision_prompt_B": trustee_prompt,
        "raw_response_B": trustee_raw,
        "normalized_response_B": trustee_normalized["text"],
        "normalization_applied_B": trustee_normalized["applied"],
        "normalization_rule_B": trustee_normalized["rule"],
        "normalization_rules_B": trustee_normalized["rules"],
        "parsed_response_B": trustee_parsed["parsed_response"],
        "parse_success_A": investor_parsed["parse_success"],
        "parse_success_B": trustee_parsed["parse_success"],
        "validation_success_A": investor_parsed["validation_success"],
        "validation_success_B": trustee_parsed["validation_success"],
        "parse_error_A": investor_parsed["error"],
        "parse_error_B": trustee_parsed["error"],
        "sent_zero": sent == 0 if sent is not None else None,
        "sent_full": sent == game.endowment if sent is not None else None,
        "positive_trust": sent > 0 if sent is not None else None,
        "returned_zero_after_trust": sent > 0 and returned == 0 if valid_dyad else None,
        "returned_at_least_sent": sent > 0 and returned >= sent if valid_dyad else None,
        "returned_more_than_sent": sent > 0 and returned > sent if valid_dyad else None,
        "communication_rounds": 0,
        "communication_transcript": [],
        "agreement_detected": None,
        # Compatibility aliases for existing summary helpers.
        "model": f"{investor_model['model_id']} / {trustee_model['model_id']}",
        "temperature": investor_mcfg.temperature,
        "tripled": tripled_amount,
        "return_ratio": (returned / tripled_amount) if valid_dyad and tripled_amount > 0 else None,
        "investor_net": returned - sent if valid_dyad else None,
        "investor_payoff": payoff_a,
        "trustee_payoff": payoff_b,
        "trustee_received": tripled_amount > 0 if valid_dyad else None,
        "betrayal": bool(tripled_amount > 0 and returned < sent) if valid_dyad else None,
        "full_betrayal": bool(tripled_amount > 0 and returned == 0) if valid_dyad else None,
        "reciprocated": bool(tripled_amount > 0 and returned >= sent) if valid_dyad else None,
        "investor_parse_clean": investor_parsed["validation_success"],
        "trustee_parse_clean": trustee_parsed["validation_success"],
        "investor_reasoning": investor_parsed["reasoning"],
        "trustee_reasoning": trustee_parsed["reasoning"],
        "investor_prompt": investor_prompt,
        "trustee_prompt": trustee_prompt,
        "investor_raw": investor_raw,
        "trustee_raw": trustee_raw,
        "history_shown": False,
        "comm_rounds": 0,
        "transcript": [],
    }
    return record


def print_plan(
    exp_cfg: dict[str, Any],
    model_index: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    n_pairs = int(exp_cfg["sample"]["n_pairs_per_model_pair"])
    total = n_pairs * len(pairs)
    print(f"Experiment: {exp_cfg['experiment']['id']}")
    print(f"Treatment: {exp_cfg['experiment']['treatment']}")
    print(f"Dyads per model pair: {n_pairs}")
    print(f"Enabled model pairs: {len(pairs)}")
    print(f"Total dyads: {total}")
    for pair in pairs:
        inv = model_index[pair["investor_model"]]
        tru = model_index[pair["trustee_model"]]
        print(
            f"- {pair['pair_id']}: "
            f"{inv.get('analysis_label', inv['name'])} -> "
            f"{tru.get('analysis_label', tru['name'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 pilot from YAML config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to experiment YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Read configs and print the run plan without API calls.")
    parser.add_argument(
        "--only-pair",
        action="append",
        default=[],
        help="Run only the selected pair_id. Can be passed more than once.",
    )
    parser.add_argument("--output", help="Override the JSONL output path from the config.")
    args = parser.parse_args()

    config_path = resolve_path(args.config) if args.config else DEFAULT_CONFIG
    assert config_path is not None
    exp_cfg = load_yaml(config_path)

    model_source = exp_cfg.get("model_source", {}).get("file")
    models_path = resolve_path(model_source)
    if models_path is None:
        raise ValueError("Experiment config must define model_source.file")
    models_cfg = load_yaml(models_path)
    model_index = index_models(models_cfg)
    pairs = enabled_pairs(exp_cfg)
    if args.only_pair:
        requested = set(args.only_pair)
        pairs = [pair for pair in pairs if pair.get("pair_id") in requested]
        found = {pair.get("pair_id") for pair in pairs}
        missing = requested - found
        if missing:
            raise ValueError(f"Requested pair_id is not enabled or does not exist: {', '.join(sorted(missing))}")

    if not pairs:
        raise ValueError("No enabled model pairs found in experiment config.")

    for pair in pairs:
        for key in ("investor_model", "trustee_model"):
            model_name = pair[key]
            if model_name not in model_index:
                raise ValueError(f"Pair '{pair.get('pair_id')}' references unknown model: {model_name}")
            require_model_ready(model_name, model_index[model_name])

    print_plan(exp_cfg, model_index, pairs)
    if args.dry_run:
        print("Dry run only: no API calls were made.")
        return

    for pair in pairs:
        check_api_key(pair["investor_model"], model_index[pair["investor_model"]])
        check_api_key(pair["trustee_model"], model_index[pair["trustee_model"]])

    experiment_id = exp_cfg["experiment"]["id"]
    treatment = exp_cfg["experiment"]["treatment"]
    n_pairs = int(exp_cfg["sample"]["n_pairs_per_model_pair"])
    temperature = float(exp_cfg["generation"]["temperature"])
    output_path = resolve_path(args.output) if args.output else resolve_path(exp_cfg["output"]["raw_path"])
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    game = B.GameConfig()
    run_id = f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    chat_cache: dict[str, tuple[B.ModelConfig, B.ChatFn]] = {}

    records_by_pair: dict[str, list[dict[str, Any]]] = {}
    with output_path.open("a", encoding="utf-8") as fh:
        for pair in pairs:
            pair_id = pair["pair_id"]
            investor_name = pair["investor_model"]
            trustee_name = pair["trustee_model"]
            investor_model = model_index[investor_name]
            trustee_model = model_index[trustee_name]

            for model_name, model in ((investor_name, investor_model), (trustee_name, trustee_model)):
                if model_name not in chat_cache:
                    mcfg = make_model_config(model, temperature)
                    chat_cache[model_name] = (mcfg, B.make_chat_fn(mcfg))

            investor_mcfg, investor_chat = chat_cache[investor_name]
            trustee_mcfg, trustee_chat = chat_cache[trustee_name]
            records_by_pair[pair_id] = []

            for dyad_index in range(n_pairs):
                record = run_no_communication_dyad(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    treatment=treatment,
                    pair_id=pair_id,
                    dyad_index=dyad_index,
                    investor_model=investor_model,
                    trustee_model=trustee_model,
                    investor_mcfg=investor_mcfg,
                    trustee_mcfg=trustee_mcfg,
                    investor_chat=investor_chat,
                    trustee_chat=trustee_chat,
                    game=game,
                )
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                records_by_pair[pair_id].append(record)
                time.sleep(float(os.environ.get("SLEEP_BETWEEN", "0")))

            B.summarize(records_by_pair[pair_id], pair_id)

    print(f"\nDone. Raw records: {output_path}")


if __name__ == "__main__":
    main()
