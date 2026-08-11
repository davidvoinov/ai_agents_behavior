#!/usr/bin/env python3
"""Create or resume a crash-safe Promise Switch execution run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.execution.rate_limit import RateLimitPolicy
from promise_switch.execution.retry import RetryPolicy
from promise_switch.execution.runner import (
    ExecutionParameters,
    PromiseSwitchRunner,
    parameters_from_manifest,
)
from promise_switch.providers import create_model_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id")
    result.add_argument("--resume", metavar="RUN_ID")
    result.add_argument("--provider")
    result.add_argument("--model", help="Promise Switch model alias, e.g. qwen3_32b")
    result.add_argument("--reasoning-mode", choices=("on", "off", "unavailable"))
    result.add_argument(
        "--dataset-role",
        choices=("technical", "pilot", "confirmatory", "synthetic"),
        help="required scientific purpose for every new non-dry run",
    )
    result.add_argument("--n-per-condition", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--output-dir", type=Path, default=Path("runs/promise_switch"))
    result.add_argument("--temperature", type=float)
    result.add_argument("--max-tokens", type=int)
    result.add_argument("--message-max-tokens", type=int)
    result.add_argument("--requests-per-minute", type=int)
    result.add_argument("--tokens-per-minute", type=int)
    result.add_argument("--minimum-interval-seconds", type=float)
    result.add_argument("--dry-run", action="store_true", default=None)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    invocation = tuple(sys.argv if argv is None else ["run_promise_switch.py", *argv])
    if args.resume:
        if args.run_id and args.run_id != args.resume:
            raise SystemExit("--run-id cannot differ from --resume")
        run_dir = args.output_dir / args.resume
        parameters = parameters_from_manifest(run_dir)
        _reject_resume_overrides(args, parameters)
        runner = PromiseSwitchRunner(parameters)
        print(json.dumps(runner.preflight_estimate(), indent=2, sort_keys=True))
        print(json.dumps(runner.run(resume=True), indent=2, sort_keys=True))
        return 0

    missing = [name for name in ("run_id", "provider", "model", "n_per_condition", "seed")
               if getattr(args, name) is None]
    if missing:
        raise SystemExit("Missing required new-run arguments: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    if args.dry_run:
        if args.dataset_role not in {None, "synthetic"}:
            raise SystemExit("--dry-run requires --dataset-role synthetic")
        dataset_role = "synthetic"
    else:
        if args.dataset_role is None:
            raise SystemExit("Missing required new-run argument: --dataset-role")
        if args.dataset_role == "synthetic":
            raise SystemExit("--dataset-role synthetic requires --dry-run")
        dataset_role = args.dataset_role
    model_config = create_model_config(
        provider=args.provider, model_alias=args.model, reasoning_mode=args.reasoning_mode,
    )
    max_tokens = args.max_tokens or 256
    parameters = ExecutionParameters(
        run_id=args.run_id,
        output_dir=args.output_dir,
        n_per_condition=args.n_per_condition,
        seed=args.seed,
        model_configs=(model_config,),
        dataset_role=dataset_role,
        temperature=0.0 if args.temperature is None else args.temperature,
        message_max_tokens=args.message_max_tokens or max_tokens,
        decision_max_tokens=max_tokens,
        retry_policy=RetryPolicy(),
        rate_limit_policy=RateLimitPolicy(
            requests_per_minute=args.requests_per_minute or 30,
            tokens_per_minute=args.tokens_per_minute,
            max_concurrency=1,
            minimum_interval_seconds=(2.0 if args.minimum_interval_seconds is None
                                      else args.minimum_interval_seconds),
        ),
        dry_run=bool(args.dry_run),
        cli_invocation=invocation,
    )
    runner = PromiseSwitchRunner(parameters)
    print(json.dumps(runner.preflight_estimate(), indent=2, sort_keys=True))
    print(json.dumps(runner.run(), indent=2, sort_keys=True))
    return 0


def _reject_resume_overrides(args: argparse.Namespace, parameters: ExecutionParameters) -> None:
    expected = {
        "provider": parameters.model_configs[0].provider,
        "model": parameters.model_configs[0].model_alias,
        "reasoning_mode": parameters.model_configs[0].reasoning_mode,
        "dataset_role": parameters.dataset_role,
        "n_per_condition": parameters.n_per_condition,
        "seed": parameters.seed,
        "temperature": parameters.temperature,
        "message_max_tokens": parameters.message_max_tokens,
        "requests_per_minute": parameters.rate_limit_policy.requests_per_minute,
        "tokens_per_minute": parameters.rate_limit_policy.tokens_per_minute,
        "minimum_interval_seconds": parameters.rate_limit_policy.minimum_interval_seconds,
    }
    for name, value in expected.items():
        supplied = getattr(args, name, None)
        if supplied is not None and supplied != value:
            raise SystemExit(f"Incompatible resume override --{name.replace('_', '-')}")
    if args.max_tokens is not None and args.max_tokens != parameters.decision_max_tokens:
        raise SystemExit("Incompatible resume override --max-tokens")
    if args.dry_run is not None and bool(args.dry_run) != parameters.dry_run:
        raise SystemExit("Incompatible resume override --dry-run")


if __name__ == "__main__":
    raise SystemExit(main())
