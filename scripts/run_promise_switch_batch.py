#!/usr/bin/env python3
"""Operate the durable Promise Switch provider-Batch state machine explicitly."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.execution.anthropic_batch import AnthropicBatchProvider
from promise_switch.execution.batch_runner import PromiseSwitchBatchRunner
from promise_switch.execution.openai_batch import OpenAIBatchProvider
from promise_switch.execution.runner import ExecutionParameters, parameters_from_manifest
from promise_switch.providers import create_model_config

PAID_ACK = "I AUTHORIZE PAID PROMISE SWITCH BATCH EXECUTION"
POLL_ACK = "I AUTHORIZE PROMISE SWITCH BATCH RESULT RETRIEVAL"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action", choices=(
            "initialize", "prepare", "submit", "poll-ingest", "reconcile",
            "mark-interrupted", "retry-non-submission", "resume-partial", "resume-run",
        ),
    )
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-dir", type=Path, default=Path("runs/promise_switch"))
    result.add_argument("--provider", choices=("openai", "anthropic"))
    result.add_argument("--model")
    result.add_argument("--reasoning-mode", choices=("on", "off"))
    result.add_argument("--validated-cap", type=int)
    result.add_argument("--dataset-role", choices=("technical", "pilot", "confirmatory"))
    result.add_argument("--n-per-condition", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--batch-sequence", type=int)
    result.add_argument("--batch-key")
    result.add_argument("--provider-job-id")
    result.add_argument("--approve-parser-review-class")
    result.add_argument("--acknowledge-paid-provider-call")
    result.add_argument("--live-submit", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_dir = args.output_dir / args.run_id
    if args.action == "initialize":
        required = ("provider", "model", "reasoning_mode", "validated_cap",
                    "dataset_role", "n_per_condition", "seed")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit("initialize is missing: " + ", ".join(
                "--" + name.replace("_", "-") for name in missing
            ))
        config = replace(create_model_config(
            provider=args.provider, model_alias=args.model,
            reasoning_mode=args.reasoning_mode,
        ), production_max_output_tokens=args.validated_cap)
        parameters = ExecutionParameters(
            run_id=args.run_id, output_dir=args.output_dir,
            n_per_condition=args.n_per_condition, seed=args.seed,
            model_configs=(config,), dataset_role=args.dataset_role,
            execution_mode="provider_batch",
            cli_invocation=tuple(sys.argv if argv is None else [__file__, *argv]),
        )
    else:
        parameters = parameters_from_manifest(run_dir)
        if parameters.execution_mode != "provider_batch":
            raise SystemExit("run manifest is not a provider_batch execution")
        config = parameters.model_configs[0]
        for supplied, expected, flag in (
            (args.provider, config.provider, "--provider"),
            (args.model, config.model_alias, "--model"),
            (args.reasoning_mode, config.reasoning_mode, "--reasoning-mode"),
            (args.validated_cap, config.production_max_output_tokens, "--validated-cap"),
        ):
            if supplied is not None and supplied != expected:
                raise SystemExit(f"{flag} conflicts with the immutable run manifest")
    if args.action == "submit":
        if not args.live_submit:
            raise SystemExit("submit requires --live-submit")
        if args.acknowledge_paid_provider_call != PAID_ACK:
            raise SystemExit(
                "submit requires --acknowledge-paid-provider-call "
                f"'{PAID_ACK}'"
            )
    elif args.action == "poll-ingest":
        if args.acknowledge_paid_provider_call != POLL_ACK:
            raise SystemExit(
                "poll-ingest requires --acknowledge-paid-provider-call "
                f"'{POLL_ACK}'"
            )
    provider = (
        OpenAIBatchProvider() if config.provider == "openai" else AnthropicBatchProvider()
    )
    runner = PromiseSwitchBatchRunner(parameters, provider)
    if args.action == "initialize":
        result = runner.initialize()
    elif args.action == "prepare":
        if args.batch_sequence is None:
            raise SystemExit("prepare requires --batch-sequence")
        result = {"batch_key": runner.prepare_next_batch(batch_sequence=args.batch_sequence)}
    elif args.action in {"submit", "poll-ingest"}:
        if not args.batch_key:
            raise SystemExit(f"{args.action} requires --batch-key")
        if args.action == "submit":
            result = {"provider_job_id": runner.coordinator.submit(args.batch_key)}
        else:
            responses = runner.poll_and_retrieve(args.batch_key)
            result = (
                {"pending": True} if responses is None
                else {"pending": False, "outcomes": runner.ingest_retrieved(
                    args.batch_key, responses,
                )}
            )
    elif args.action == "reconcile":
        if not args.batch_key or not args.provider_job_id:
            raise SystemExit("reconcile requires --batch-key and --provider-job-id")
        runner.coordinator.reconcile_submission(args.batch_key, args.provider_job_id)
        result = {"reconciled": True}
    elif args.action == "mark-interrupted":
        if not args.batch_key:
            raise SystemExit("mark-interrupted requires --batch-key")
        runner.coordinator.mark_interrupted_submission_for_reconciliation(args.batch_key)
        result = {"reconciliation_required": True}
    elif args.action == "retry-non-submission":
        if not args.batch_key:
            raise SystemExit("retry-non-submission requires --batch-key")
        runner.coordinator.retry_known_non_submission(args.batch_key)
        result = {"operational_retry_prepared": True}
    elif args.action == "resume-partial":
        if not args.batch_key:
            raise SystemExit("resume-partial requires --batch-key")
        runner.coordinator.resume_partial_retrieval(args.batch_key)
        result = {"partial_retrieval_resumed": True}
    else:
        runner.resume_paused_run(
            approve_parser_review_class=args.approve_parser_review_class,
        )
        result = {"run_resumed": True}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
