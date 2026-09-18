#!/usr/bin/env python3
"""Offline-first exact-setting Promise Switch Batch technical preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.design import Condition
from promise_switch.execution.anthropic_batch import AnthropicBatchProvider
from promise_switch.execution.batch_transport import BatchCoordinator, make_mapping
from promise_switch.execution.batch_workflow import (
    build_decision_batch_requests,
)
from promise_switch.execution.openai_batch import OpenAIBatchProvider
from promise_switch.execution.attrition import (
    classify_batch_decision_response, classify_transport,
)
from promise_switch.execution.planner import plan_trials
from promise_switch.parser import parse_decision
from promise_switch.randomization import assign_fixed_trial, plan_fixed_trial
from promise_switch.providers import create_model_config
from promise_switch.version import PROMPT_VERSION, RESPONSE_QC_VERSION, prompt_templates_hash

EXPECTED_PROMPT_HASH = "17b5f31eaac9d9f2742d74a01fae17bc4a93a847d33f49d1016392035a8f175c"
LIVE_ACK = "I AUTHORIZE ONE PAID BATCH PREFLIGHT"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--provider", required=True, choices=("openai", "anthropic"))
    result.add_argument("--model", required=True, choices=(
        "gpt_5_6_luna_openai", "claude_sonnet_5_anthropic",
    ))
    result.add_argument("--reasoning-mode", required=True, choices=("off", "on"))
    result.add_argument("--candidate-cap", "--validated-cap", dest="candidate_cap", type=int)
    result.add_argument("--output-dir", type=Path, default=Path("runs/promise_switch_batch_preflight"))
    result.add_argument("--run-id")
    result.add_argument("--live-submit", action="store_true")
    result.add_argument("--poll-existing", metavar="BATCH_KEY")
    result.add_argument("--acknowledge-paid-provider-call")
    return result


def build_report(args: argparse.Namespace) -> tuple[dict[str, object], object, list[object]]:
    config = create_model_config(
        provider=args.provider, model_alias=args.model, reasoning_mode=args.reasoning_mode,
    )
    if config.production_max_output_tokens is not None:
        raise ValueError("new Batch configuration unexpectedly has a production cap")
    blockers = []
    if args.candidate_cap is None:
        blockers.append("production output cap unresolved pending paid Batch preflight")
        cap = 1
    elif args.candidate_cap < 1:
        raise ValueError("--candidate-cap must be positive")
    else:
        cap = args.candidate_cap
    technical_config = replace(config, production_max_output_tokens=cap)
    plans = plan_trials(
        run_id=args.run_id or "offline_batch_preflight", seed=20260808,
        n_per_condition=1, model_configs=(technical_config,),
    )
    fixed = next(plan for plan in plans if plan.final_condition == Condition.NO_COMMUNICATION.value)
    assignment = assign_fixed_trial(plan_fixed_trial(
        trial_id=fixed.trial_id, run_id=fixed.run_id, seed=fixed.randomization_seed,
        condition=Condition.NO_COMMUNICATION,
    ))
    decision_request = build_decision_batch_requests(
        [fixed], config=technical_config, max_tokens=cap,
        assignments={fixed.trial_id: assignment}, message_events={},
    )[0]
    provider = (
        OpenAIBatchProvider() if args.provider == "openai" else AnthropicBatchProvider()
    )
    requests = [decision_request]
    if len(requests) != 1:
        raise RuntimeError("minimal paid Batch preflight must contain exactly one request")
    serialized = [provider.serialize(make_mapping(decision_request, 1))]
    if prompt_templates_hash() != EXPECTED_PROMPT_HASH:
        raise RuntimeError("frozen Promise Switch prompt hash changed")
    report = {
        "offline_only": not args.live_submit,
        "provider": config.provider,
        "model_alias": config.model_alias,
        "model_id": config.model_id,
        "requested_model_id": config.model_id,
        "returned_model_id": None,
        "model_identity_status": "UNVERIFIABLE",
        "reasoning_mode": config.reasoning_mode,
        "provider_request_kwargs": config.provider_request_kwargs,
        "batch_protocol": config.batch_protocol,
        "output_token_field": config.output_token_field,
        "batch_only_confirmatory": config.batch_only_confirmatory,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_templates_hash(),
        "response_qc_version": RESPONSE_QC_VERSION,
        "temperature": 1.0,
        "candidate_cap": args.candidate_cap,
        "blockers": blockers,
        "request_count": len(serialized),
        "submitted_request_count": len(serialized),
        "preflight_passed": False,
        "preflight_status": "OFFLINE_ONLY" if not args.live_submit else "PENDING_RESULTS",
        "serialized_requests": serialized,
    }
    return report, provider, requests


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.live_submit and args.poll_existing:
        raise SystemExit("choose either --live-submit or --poll-existing")
    report, provider, requests = build_report(args)
    if args.live_submit or args.poll_existing:
        report["preflight_status"] = "PENDING_RESULTS"
        if args.candidate_cap is None:
            raise SystemExit("provider operations require --candidate-cap")
        if args.acknowledge_paid_provider_call != LIVE_ACK:
            raise SystemExit(
                f"provider operations require --acknowledge-paid-provider-call '{LIVE_ACK}'"
            )
        if not args.run_id:
            raise SystemExit("provider operations require --run-id")
        coordinator = BatchCoordinator(
            args.output_dir / args.run_id, provider,
            timestamp=lambda: datetime.now(timezone.utc).isoformat(),
        )
        if args.live_submit:
            batch_key = coordinator.prepare(
                run_id=args.run_id, model=str(report["model_id"]),
                reasoning_mode=args.reasoning_mode, batch_sequence=1,
                batch_phase="technical_preflight",
                requests=[(request, 1) for request in requests],
            )
            report["batch_key"] = batch_key
            report["provider_job_id"] = coordinator.submit(batch_key)
        else:
            normalized = coordinator.poll_and_retrieve(args.poll_existing)
            report["batch_key"] = args.poll_existing
            if normalized is None:
                latest = coordinator.latest(args.poll_existing) or {}
                report["provider_status"] = latest.get("status", "unknown")
                report["preflight_passed"] = False
                if latest.get("status") in {"FAILED", "CANCELED"}:
                    report["blockers"].append("provider batch terminated without usable results")
                    report["preflight_status"] = "FAILED"
            else:
                mappings = coordinator.mappings(args.poll_existing)
                stage_by_id = {row["custom_id"]: row["stage"] for row in mappings}
                checks = []
                for custom_id, response in normalized.items():
                    transport = classify_transport(response)
                    if transport.category == "success":
                        outcome = classify_batch_decision_response(
                            response, parse_decision(response.raw_response)
                        )
                    else:
                        outcome = transport
                    checks.append({
                        "custom_id": custom_id, "stage": stage_by_id[custom_id],
                        "transport_status": response.status,
                        "finish_reason": response.finish_reason,
                        "returned_model_id": response.returned_model_id,
                        "qc_category": outcome.category, "qc_cause": outcome.cause,
                        "reasoning_signal_present": response.error_metadata.get(
                            "reasoning_signal_present", response.error_metadata.get("thinking_present")
                        ),
                        "reasoning_token_count": response.error_metadata.get(
                            "reasoning_token_count", response.error_metadata.get("thinking_token_count")
                        ),
                        "usage": response.usage,
                    })
                report["result_checks"] = checks
                returned_ids = [row["returned_model_id"] for row in checks]
                report["returned_model_id"] = (
                    returned_ids[0] if len(returned_ids) == 1 else None
                )
                report["model_identity_status"] = model_identity_verification(
                    str(report["requested_model_id"]), returned_ids,
                )
                latest = coordinator.latest(args.poll_existing) or {}
                report["provider_status"] = latest.get("provider_status", latest.get("status"))
                report["provider_result_artifact_hash"] = latest.get("result_artifact_hash")
                report["provider_result_path"] = latest.get("result_path")
                coverage_complete = (
                    not latest.get("missing_custom_ids")
                    and len(checks) == len(requests)
                )
                report["result_coverage_complete"] = coverage_complete
                rebuilt_request = report["serialized_requests"][0]
                request_configuration, configuration_source = (
                    durable_request_configuration(
                        provider=args.provider, batch_key=args.poll_existing,
                        mappings=mappings, latest=latest,
                        rebuilt_request=rebuilt_request,
                    )
                )
                report["reasoning_configuration_source"] = configuration_source
                reasoning = reasoning_verification(
                    provider=args.provider,
                    mode=args.reasoning_mode,
                    request_configuration=request_configuration,
                    checks=checks,
                    model_identity_status=str(report["model_identity_status"]),
                )
                report.update(reasoning)
                report["truncation_status"] = (
                    "TRUNCATED" if any(
                        row.get("finish_reason") in {"length", "max_tokens"} for row in checks
                    ) else "NONE"
                )
                report["candidate_cap_sufficient"] = (
                    coverage_complete and report["truncation_status"] == "NONE"
                )
                report["preflight_passed"] = all(
                    row["qc_category"] == "success" for row in checks
                ) and coverage_complete and report["reasoning_verification"] in {
                    "VERIFIED_ON", "VERIFIED_OFF", "CONFIG_VERIFIED_ADAPTIVE",
                } and report["model_identity_status"] == "VERIFIED_MATCH"
                if not report["preflight_passed"]:
                    report["preflight_status"] = "FAILED"
                    report["blockers"].append(
                        "QC, reasoning, or returned-model verification failed"
                    )
                if report["preflight_passed"]:
                    report["preflight_status"] = "PASSED"
                    coordinator.mark_ingested(args.poll_existing)
        report["offline_only"] = False
    if args.poll_existing and report.get("preflight_status") in {"PASSED", "FAILED"}:
        _persist_terminal_report(args.output_dir / str(args.run_id), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["blockers"]:
        return 2
    if args.poll_existing and not report.get("preflight_passed", False):
        return 3
    return 0


def _persist_terminal_report(run_dir: Path, report: dict[str, object]) -> None:
    """Atomically retain the terminal, secret-free preflight verification evidence."""
    # Real coordinators create the run directory during prepare. Isolated CLI
    # fakes used by offline tests intentionally have no artifact directory.
    if not run_dir.is_dir():
        return
    path = run_dir / "preflight_verification.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def durable_request_configuration(
    *, provider: str, batch_key: str, mappings: tuple[dict[str, object], ...] | list[dict[str, object]],
    latest: dict[str, object], rebuilt_request: dict[str, object],
) -> tuple[dict[str, object], str]:
    """Load the exact serialized Batch request, verifying its durable hash first."""
    input_path = latest.get("input_path")
    expected_hash = latest.get("input_artifact_hash")
    if isinstance(input_path, str) and isinstance(expected_hash, str):
        path = Path(input_path)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("durable Batch input artifact hash mismatch")
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        expected_ids = {
            str(row["custom_id"]) for row in mappings
            if row.get("batch_key", batch_key) == batch_key
        }
        wire_rows = [row for row in rows if str(row.get("custom_id")) in expected_ids]
        if len(wire_rows) != 1 or len(rows) != len(expected_ids):
            raise ValueError("durable Batch input does not match request mappings")
        field = "params" if provider == "anthropic" else "body"
        configuration = wire_rows[0].get(field)
        if not isinstance(configuration, dict):
            raise ValueError("durable Batch input is missing provider request configuration")
        return configuration, "DURABLE_BATCH_INPUT"

    # Compatibility for isolated unit fakes and pre-input artifacts. Production
    # lifecycle rows always carry a hash-verified input artifact.
    for row in mappings:
        request = row.get("request")
        body = request.get("body") if isinstance(request, dict) else None
        if isinstance(body, dict):
            return body, "DURABLE_REQUEST_MAPPING"
    field = "params" if provider == "anthropic" else "body"
    configuration = rebuilt_request.get(field)
    if not isinstance(configuration, dict):
        raise ValueError("request configuration is unavailable")
    return configuration, "REBUILT_TEST_FALLBACK"


def reasoning_verification(
    *, provider: str, mode: str, request_configuration: dict[str, object],
    checks: list[dict[str, object]], model_identity_status: str,
) -> dict[str, object]:
    """Verify requested reasoning configuration separately from observed reasoning.

    Anthropic adaptive thinking is a request configuration: the model may choose
    whether to use thinking on an individual request.  Observation is retained as
    telemetry and is not used to relabel or retry a valid adaptive/xhigh response.
    """
    signals = [row.get("reasoning_signal_present") for row in checks]
    counts = [row.get("reasoning_token_count") for row in checks]
    thinking_observed = any(
        signal is True or isinstance(count, (int, float)) and count > 0
        for signal, count in zip(signals, counts, strict=True)
    )
    thinking_tokens = (
        int(counts[0]) if len(counts) == 1 and isinstance(counts[0], (int, float))
        else None
    )
    base: dict[str, object] = {
        "thinking_observed": thinking_observed,
        "thinking_tokens": thinking_tokens,
    }
    provider_accepted = bool(checks) and all(
        row.get("transport_status") == "success"
        and row.get("qc_category") == "success"
        for row in checks
    )
    if not provider_accepted or model_identity_status != "VERIFIED_MATCH":
        return {
            **base,
            "reasoning_verification": "CONFIG_VERIFICATION_FAILED",
            "reasoning_configuration_failure": (
                "provider response failed" if not provider_accepted
                else "returned model identity did not match"
            ),
        }

    if provider == "anthropic":
        thinking = request_configuration.get("thinking")
        output_config = request_configuration.get("output_config")
        if mode == "on":
            exact = (
                thinking == {"type": "adaptive", "display": "omitted"}
                and output_config == {"effort": "xhigh"}
            )
            return {
                **base,
                "configured_reasoning_mode": "adaptive/xhigh",
                "reasoning_verification": (
                    "CONFIG_VERIFIED_ADAPTIVE" if exact
                    else "CONFIG_VERIFICATION_FAILED"
                ),
                **({} if exact else {
                    "reasoning_configuration_failure": (
                        "expected thinking=adaptive/display=omitted and effort=xhigh"
                    ),
                }),
            }
        exact = thinking == {"type": "disabled"} and "output_config" not in request_configuration
        absent = all(
            signal in {None, False}
            and (count is None or isinstance(count, (int, float)) and count == 0)
            for signal, count in zip(signals, counts, strict=True)
        )
        return {
            **base,
            "configured_reasoning_mode": "disabled",
            "reasoning_verification": (
                "VERIFIED_OFF" if exact and absent else "CONFIG_VERIFICATION_FAILED"
            ),
            **({} if exact and absent else {
                "reasoning_configuration_failure": (
                    "disabled request was missing/altered or thinking was observed"
                ),
            }),
        }

    expected_effort = "medium" if mode == "on" else "none"
    exact = request_configuration.get("reasoning_effort") == expected_effort
    if mode == "on":
        status = "VERIFIED_ON" if exact and thinking_observed else "UNVERIFIABLE"
    else:
        absent = all(
            signal in {None, False}
            and (count is None or isinstance(count, (int, float)) and count == 0)
            for signal, count in zip(signals, counts, strict=True)
        )
        status = "VERIFIED_OFF" if exact and absent else "UNVERIFIABLE"
    return {
        **base,
        "configured_reasoning_mode": f"reasoning_effort={expected_effort}",
        "reasoning_verification": status,
    }


def model_identity_verification(expected: str, returned: list[object]) -> str:
    if len(returned) != 1 or not isinstance(returned[0], str) or not returned[0]:
        return "UNVERIFIABLE"
    return "VERIFIED_MATCH" if returned[0] == expected else "MISMATCH"


if __name__ == "__main__":
    raise SystemExit(main())
