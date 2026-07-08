#!/usr/bin/env python3
"""CLI entrypoint for Experiment 3: Promise Switch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "promise_switch" / "config.json"

from promise_switch.config import load_config
from promise_switch.experiment import run_experiment


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def has_real_api_key(env_var: str) -> bool:
    value = os.environ.get(env_var, "").strip()
    return len(value) > 30 and value.startswith("sk-or-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Vanberg-style Promise Switch experiment."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to JSON config.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory with the same run id.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create manifest and planned trials without API calls.",
    )
    parser.add_argument(
        "--model-source",
        choices=("live", "config"),
        default=None,
        help="Use live OpenRouter /models or static config model_ids.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override n_trials_per_model.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Limit models after filtering and shuffling.",
    )
    parser.add_argument(
        "--model-offset",
        type=int,
        default=None,
        help="Skip this many models after filtering/shuffling, for chunked runs.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep model order from the config/catalog for deterministic chunks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override randomization seed.",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run a short API health-check for each planned model before trials.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Local env file to load before reading the API key.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(resolve_project_path(args.env_file))
    config = load_config(resolve_project_path(args.config))

    overrides = {
        "run_id": args.run_id,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "api_key_env": args.api_key_env,
        "health_check": args.health_check,
    }
    if args.model_source is not None:
        config["model_source"] = args.model_source
    if args.n_trials is not None:
        config["n_trials_per_model"] = args.n_trials
    if args.max_models is not None:
        config["max_models"] = args.max_models
    if args.model_offset is not None:
        config["model_offset"] = args.model_offset
    if args.no_shuffle:
        config["shuffle_models"] = False
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.health_check:
        config["health_check_models"] = True
    if config.get("output_dir"):
        output_dir = Path(str(config["output_dir"]))
        if not output_dir.is_absolute():
            config["output_dir"] = str(PROJECT_ROOT / output_dir)

    if not args.dry_run and not has_real_api_key(args.api_key_env):
        print(
            f"Missing real {args.api_key_env}. Put it in {args.env_file} or run with --dry-run.",
            file=sys.stderr,
        )
        return 2

    result = run_experiment(config=config, overrides=overrides)
    print(f"Run directory: {result['run_dir']}")
    print(f"Models planned: {result['models_planned']}")
    print(f"Trials planned: {result['trials_planned']}")
    if result.get("trials_completed") is not None:
        print(f"Trials completed: {result['trials_completed']}")
        print(f"Trials partial/failed: {result['trials_partial_or_failed']}")
    if result.get("dry_run"):
        print("Dry-run only: no API calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
