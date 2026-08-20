#!/usr/bin/env python3
"""Make exactly one live request for one final Promise Switch configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.execution.transport import OpenAICompatibleTransport
from promise_switch.preflight import ExactFinalPreflight, PreflightParameters
from promise_switch.providers import create_model_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preflight-id", required=True)
    result.add_argument("--provider", required=True)
    result.add_argument("--model", required=True, help="one exact Promise Switch registry alias")
    result.add_argument("--reasoning-mode", choices=("on", "off", "unavailable"))
    result.add_argument(
        "--output-dir", type=Path, default=Path("preflights/promise_switch"),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = create_model_config(
        provider=args.provider, model_alias=args.model, reasoning_mode=args.reasoning_mode,
    )
    artifact = ExactFinalPreflight(
        PreflightParameters(
            preflight_id=args.preflight_id,
            output_dir=args.output_dir,
            model_config=config,
        ),
        transport=OpenAICompatibleTransport(),
    ).run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
