"""Experiment orchestration."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import ApiResult, OpenRouterClient
from .io import append_jsonl, read_jsonl, write_csv, write_json
from .metrics import SUMMARY_COLUMNS, summarize_trials
from .models import (
    filter_zero_price_text_models,
    models_from_config_ids,
    supported_parameters,
)
from .parser import as_bool, parse_decision, parse_json_payload, parse_promise
from .prompts import (
    health_check_messages,
    no_communication_messages,
    no_switch_decision_messages,
    promise_messages,
    switch_decision_messages,
)


def run_experiment(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    dry_run = bool(overrides.get("dry_run"))
    run_id = overrides.get("run_id") or utc_run_id()
    output_dir = Path(str(config.get("output_dir", "runs")))
    run_dir = output_dir / run_id
    resume = bool(overrides.get("resume"))
    if run_dir.exists() and not resume and not dry_run:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key_env = str(overrides.get("api_key_env") or "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env)
    client = OpenRouterClient(
        api_key=api_key,
        timeout_s=int(config.get("request_timeout_s", 90)),
        http_referer=config.get("http_referer"),
    )

    key_info = None
    if api_key and not dry_run:
        key_result = client.get_key_info()
        key_info = key_result.data if key_result.ok else {"error": key_result.error}
        write_json(run_dir / "key_info.json", key_info)

    models = load_models(config, client, run_dir, dry_run=dry_run)
    rng = random.Random(int(config.get("seed", 20260706)))
    if bool(config.get("shuffle_models", True)):
        rng.shuffle(models)
    model_offset = int(config.get("model_offset", 0) or 0)
    if model_offset:
        models = models[model_offset:]
    max_models = config.get("max_models")
    if max_models is not None:
        models = models[: int(max_models)]
    if bool(config.get("health_check_models", False)) and not dry_run:
        models = run_health_checks(config, client, run_dir, models)

    trials = plan_trials(models, int(config.get("n_trials_per_model", 1)))
    write_json(run_dir / "planned_trials.json", trials)

    manifest = {
        "run_id": run_id,
        "experiment_name": config.get("experiment_name", "promise_switch"),
        "started_at": utc_now(),
        "dry_run": dry_run,
        "config": redact_config(config),
        "models_planned": len(models),
        "trials_planned": len(trials),
        "api_key_env": api_key_env,
        "key_info_available": key_info is not None,
    }
    write_json(run_dir / "manifest.json", manifest)

    if dry_run:
        return {
            "run_dir": str(run_dir),
            "dry_run": True,
            "models_planned": len(models),
            "trials_planned": len(trials),
        }

    completed_ids = {
        str(row.get("trial_id"))
        for row in read_jsonl(run_dir / "trials.jsonl")
        if row.get("status") == "completed"
    }

    completed = 0
    partial_or_failed = 0
    hard_stop = False
    for planned in trials:
        if planned["trial_id"] in completed_ids:
            completed += 1
            continue
        trial = run_trial(config, client, run_dir, planned)
        append_jsonl(run_dir / "trials.jsonl", trial)
        if trial["status"] == "completed":
            completed += 1
        else:
            partial_or_failed += 1
        if trial.get("hard_stop"):
            hard_stop = True
            break
        sleep_s = float(config.get("inter_call_sleep_s", 0.2))
        if sleep_s > 0:
            time.sleep(sleep_s)

    all_trials = read_jsonl(run_dir / "trials.jsonl")
    model_rows, overall = summarize_trials(all_trials)
    write_csv(run_dir / "summary_by_model.csv", model_rows, SUMMARY_COLUMNS)
    write_json(run_dir / "summary_overall.json", overall)

    manifest["finished_at"] = utc_now()
    manifest["trials_completed"] = completed
    manifest["trials_partial_or_failed"] = partial_or_failed
    manifest["hard_stop"] = hard_stop
    write_json(run_dir / "manifest.json", manifest)

    return {
        "run_dir": str(run_dir),
        "dry_run": False,
        "models_planned": len(models),
        "trials_planned": len(trials),
        "trials_completed": completed,
        "trials_partial_or_failed": partial_or_failed,
    }


def load_models(
    config: dict[str, Any],
    client: OpenRouterClient,
    run_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    source = str(config.get("model_source", "live"))
    exclude = {str(value) for value in config.get("exclude_model_ids", [])}
    if source in {"provider_config", "providers"}:
        models = models_from_provider_config(config.get("models", []), exclude)
        write_json(run_dir / "models_snapshot.json", {"source": source, "data": models})
        write_json(run_dir / "filtered_models.json", models)
        return models
    if source == "config":
        model_ids = [
            str(value)
            for value in config.get("model_ids", [])
            if str(value) not in exclude
        ]
        models = models_from_config_ids(model_ids)
        write_json(run_dir / "models_snapshot.json", {"source": "config", "data": models})
        write_json(run_dir / "filtered_models.json", models)
        return models

    result = client.get_models()
    if not result.ok or not result.data:
        if config.get("model_ids"):
            models = models_from_config_ids([str(value) for value in config["model_ids"]])
            write_json(
                run_dir / "models_snapshot.json",
                {
                    "source": "config_fallback",
                    "fetch_error": result.error,
                    "data": models,
                },
            )
            write_json(run_dir / "filtered_models.json", models)
            return models
        raise RuntimeError(f"Could not fetch OpenRouter models: {result.error}")

    data = result.data.get("data") or []
    if not isinstance(data, list):
        raise RuntimeError("OpenRouter /models response did not contain data list")
    write_json(run_dir / "models_snapshot.json", result.data)
    models = filter_zero_price_text_models(data, exclude)
    write_json(run_dir / "filtered_models.json", models)
    return models


def models_from_provider_config(
    raw_models: Any,
    exclude_model_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_models, list):
        raise ValueError("provider_config model_source requires a list under 'models'.")

    models = []
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ValueError(f"Model entry #{index + 1} must be an object.")
        model = dict(raw_model)
        model_id = str(model.get("model_id") or model.get("id") or "").strip()
        if not model_id:
            raise ValueError(f"Model entry #{index + 1} is missing model_id/id.")
        if model_id in exclude_model_ids:
            continue
        model["id"] = model_id
        model.setdefault("name", model.get("analysis_label") or model_id)
        model.setdefault("supported_parameters", [])
        model.setdefault("architecture", {"input_modalities": ["text"], "output_modalities": ["text"]})
        returnable = {
            key: value
            for key, value in model.items()
            if key not in {"api_key", "token"}
        }
        models.append(returnable)
    return models


def run_health_checks(
    config: dict[str, Any],
    client: OpenRouterClient,
    run_dir: Path,
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    healthy = []
    for model in models:
        planned = {
            "trial_id": f"health__{slugify(str(model['id']))}",
            "model_id": str(model["id"]),
            "trial_index": 0,
            "model": model,
        }
        result = call_phase(
            config,
            client,
            run_dir,
            planned,
            "health_check",
            health_check_messages(),
        )
        payload, parse_error = parse_json_payload(result.get("content", ""))
        payload_ok = False
        if payload is not None:
            payload_ok = as_bool(payload.get("ok")) is True
        is_healthy = bool(result.get("ok")) and payload_ok
        append_jsonl(
            run_dir / "health_checks.jsonl",
            {
                "at": utc_now(),
                "model_id": model["id"],
                "healthy": is_healthy,
                "status": result.get("status"),
                "error": result.get("error"),
                "parse_error": parse_error,
                "payload": payload,
            },
        )
        if is_healthy:
            healthy.append(model)
        if should_hard_stop(config, result):
            break
        sleep_s = float(config.get("inter_call_sleep_s", 0.2))
        if sleep_s > 0:
            time.sleep(sleep_s)
    write_json(run_dir / "healthy_models.json", healthy)
    return healthy


def plan_trials(models: list[dict[str, Any]], n_trials_per_model: int) -> list[dict[str, Any]]:
    trials = []
    for model in models:
        model_id = str(model["id"])
        for trial_index in range(1, n_trials_per_model + 1):
            trials.append(
                {
                    "trial_id": f"{slugify(model_id)}__{trial_index:03d}",
                    "model_id": model_id,
                    "trial_index": trial_index,
                    "model": model,
                }
            )
    return trials


def run_trial(
    config: dict[str, Any],
    client: OpenRouterClient,
    run_dir: Path,
    planned: dict[str, Any],
) -> dict[str, Any]:
    model = planned["model"]
    model_id = planned["model_id"]
    trial_index = int(planned["trial_index"])
    partner_message = str(config.get("scripted_partner_message", "Please choose COOPERATE."))

    trial: dict[str, Any] = {
        "trial_id": planned["trial_id"],
        "model_id": model_id,
        "model_name": model.get("name"),
        "provider": model.get("provider"),
        "model_family": model.get("model_family"),
        "analysis_label": model.get("analysis_label"),
        "trial_index": trial_index,
        "started_at": utc_now(),
        "partner_mode": config.get("partner_mode", "scripted"),
        "partner_message": partner_message,
        "status": "partial",
        "errors": [],
    }

    no_comm_result = call_phase(
        config,
        client,
        run_dir,
        planned,
        "no_communication",
        no_communication_messages(trial_index),
    )
    trial["no_communication_call"] = no_comm_result["call_summary"]
    trial["no_communication"] = parse_decision(no_comm_result.get("content", ""))
    collect_phase_error(run_dir, trial, "no_communication", no_comm_result, trial["no_communication"])
    if should_hard_stop(config, no_comm_result):
        trial["status"] = "failed"
        trial["hard_stop"] = True
        trial["finished_at"] = utc_now()
        return trial
    if not no_comm_result.get("ok") or not trial["no_communication"].get("ok"):
        return finish_trial(trial, "partial")

    promise_result = call_phase(
        config,
        client,
        run_dir,
        planned,
        "promise",
        promise_messages(partner_message, trial_index),
    )
    trial["promise_call"] = promise_result["call_summary"]
    trial["promise"] = parse_promise(promise_result.get("content", ""))
    collect_phase_error(run_dir, trial, "promise", promise_result, trial["promise"])
    if should_hard_stop(config, promise_result):
        trial["status"] = "failed"
        trial["hard_stop"] = True
        trial["finished_at"] = utc_now()
        return trial
    if not promise_result.get("ok") or not trial["promise"].get("ok"):
        return finish_trial(trial, "partial")

    message_to_b = trial["promise"].get("message_to_B") or promise_result.get("content", "")
    promise_made = bool(trial["promise"].get("promise_made"))

    no_switch_result = call_phase(
        config,
        client,
        run_dir,
        planned,
        "promise_no_switch",
        no_switch_decision_messages(
            partner_message=partner_message,
            message_to_b=message_to_b,
            promise_made=promise_made,
            trial_index=trial_index,
        ),
    )
    trial["promise_no_switch_call"] = no_switch_result["call_summary"]
    trial["promise_no_switch"] = parse_decision(no_switch_result.get("content", ""))
    collect_phase_error(run_dir, trial, "promise_no_switch", no_switch_result, trial["promise_no_switch"])
    if should_hard_stop(config, no_switch_result):
        trial["status"] = "failed"
        trial["hard_stop"] = True
        trial["finished_at"] = utc_now()
        return trial
    if not no_switch_result.get("ok") or not trial["promise_no_switch"].get("ok"):
        return finish_trial(trial, "partial")

    switch_result = call_phase(
        config,
        client,
        run_dir,
        planned,
        "promise_switch",
        switch_decision_messages(
            message_to_old_b=message_to_b,
            promise_made_to_old_b=promise_made,
            trial_index=trial_index,
        ),
    )
    trial["promise_switch_call"] = switch_result["call_summary"]
    trial["promise_switch"] = parse_decision(switch_result.get("content", ""))
    collect_phase_error(run_dir, trial, "promise_switch", switch_result, trial["promise_switch"])
    if should_hard_stop(config, switch_result):
        trial["hard_stop"] = True

    required_ok = [
        trial["no_communication"].get("ok"),
        trial["promise"].get("ok"),
        trial["promise_no_switch"].get("ok"),
        trial["promise_switch"].get("ok"),
    ]
    trial["status"] = "completed" if all(required_ok) else "partial"
    if trial.get("hard_stop"):
        trial["status"] = "failed"
    return finish_trial(trial, trial["status"])


def finish_trial(trial: dict[str, Any], status: str) -> dict[str, Any]:
    trial["status"] = status
    trial["finished_at"] = utc_now()
    return trial


def call_phase(
    config: dict[str, Any],
    client: OpenRouterClient,
    run_dir: Path,
    planned: dict[str, Any],
    phase: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    model = planned["model"]
    supported = supported_parameters(model)
    phase_client = client_for_model(config, client, model)
    phase_messages = apply_model_prompt_prefix(messages, model)
    max_retries = int(config.get("max_retries", 1))
    retry_sleep_s = float(config.get("retry_sleep_s", 2.0))
    result: ApiResult | None = None

    for attempt in range(max_retries + 1):
        result = phase_client.chat_completion(
            model_id=planned["model_id"],
            messages=phase_messages,
            max_tokens=int(config.get("max_tokens", 300)),
            temperature=float(config.get("temperature", 0.2)),
            supported_parameters=supported,
            use_response_format=bool(config.get("use_response_format", True)),
        )
        log_api_call(run_dir, planned, phase, attempt, phase_messages, result)
        if result.ok or result.status not in {408, 409, 429, 500, 502, 503, 504, 529}:
            break
        if attempt < max_retries and retry_sleep_s > 0:
            time.sleep(retry_sleep_s * (attempt + 1))

    assert result is not None
    content = extract_message_content(result.data)
    phase_sleep_s = float(model.get("phase_sleep_s", config.get("phase_sleep_s", 0.0)) or 0.0)
    if phase_sleep_s > 0:
        time.sleep(phase_sleep_s)
    return {
        "ok": result.ok,
        "status": result.status,
        "error": result.error,
        "content": content,
        "call_summary": {
            "ok": result.ok,
            "status": result.status,
            "error": result.error,
            "latency_s": round(result.latency_s, 3),
            "usage": (result.data or {}).get("usage") if result.data else None,
        },
    }


def apply_model_prompt_prefix(
    messages: list[dict[str, str]],
    model: dict[str, Any],
) -> list[dict[str, str]]:
    prefix = str(model.get("prompt_prefix") or "").strip()
    if not prefix:
        return messages
    out = []
    applied = False
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "user" and not applied:
            copied["content"] = f"{prefix}\n{copied.get('content', '')}"
            applied = True
        out.append(copied)
    return out


def client_for_model(
    config: dict[str, Any],
    default_client: OpenRouterClient,
    model: dict[str, Any],
) -> OpenRouterClient:
    base_url = model.get("base_url")
    api_key_env = model.get("api_key_env")
    if not base_url and not api_key_env:
        return default_client
    api_key = os.environ.get(str(api_key_env or ""))
    return OpenRouterClient(
        api_key=api_key,
        base_url=str(base_url or "https://openrouter.ai/api/v1"),
        timeout_s=int(config.get("request_timeout_s", 90)),
        app_title=str(config.get("app_title", "Promise Switch Experiment")),
        http_referer=config.get("http_referer"),
    )


def log_api_call(
    run_dir: Path,
    planned: dict[str, Any],
    phase: str,
    attempt: int,
    messages: list[dict[str, str]],
    result: ApiResult,
) -> None:
    append_jsonl(
        run_dir / "api_calls.jsonl",
        {
            "at": utc_now(),
            "trial_id": planned["trial_id"],
            "model_id": planned["model_id"],
            "provider": planned.get("model", {}).get("provider"),
            "analysis_label": planned.get("model", {}).get("analysis_label"),
            "phase": phase,
            "attempt": attempt,
            "request": {
                "messages": messages,
            },
            "response": {
                "ok": result.ok,
                "status": result.status,
                "data": result.data,
                "text": result.text,
                "error": result.error,
                "latency_s": result.latency_s,
                "headers": {
                    key: value
                    for key, value in result.headers.items()
                    if key.lower() in {"x-request-id", "retry-after"}
                },
            },
        },
    )


def extract_message_content(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    return ""


def collect_phase_error(
    run_dir: Path,
    trial: dict[str, Any],
    phase: str,
    call_result: dict[str, Any],
    parsed: dict[str, Any],
) -> None:
    if not call_result.get("ok"):
        error = {
            "at": utc_now(),
            "trial_id": trial.get("trial_id"),
            "model_id": trial.get("model_id"),
            "phase": phase,
            "type": "api",
            "status": call_result.get("status"),
            "error": call_result.get("error"),
        }
        trial["errors"].append(error)
        append_jsonl(run_dir / "errors.jsonl", error)
    if parsed.get("ok") is False:
        error = {
            "at": utc_now(),
            "trial_id": trial.get("trial_id"),
            "model_id": trial.get("model_id"),
            "phase": phase,
            "type": "parse",
            "error": parsed.get("parse_error") or parsed.get("error"),
        }
        trial["errors"].append(error)
        append_jsonl(run_dir / "errors.jsonl", error)


def should_hard_stop(config: dict[str, Any], call_result: dict[str, Any]) -> bool:
    if bool(config.get("stop_on_payment_required", True)) and call_result.get("status") == 402:
        return True
    return False


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config)


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("promise_switch_%Y%m%d_%H%M%S")
