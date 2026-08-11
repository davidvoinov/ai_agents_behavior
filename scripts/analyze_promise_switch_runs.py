#!/usr/bin/env python3
"""Produce model-specific offline Promise Switch condition and contrast tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.analysis import analyze_by_model
from promise_switch.artifacts import read_jsonl, validate_run_integrity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--dataset-role", required=True, choices=("pilot", "confirmatory"),
        help="all input manifests must explicitly have this one scientific role",
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    resolved = [run_dir.resolve() for run_dir in args.run_dirs]
    if len(resolved) != len(set(resolved)):
        raise SystemExit("A run directory may be supplied only once")
    for run_dir in sorted(resolved):
        try:
            validate_run_integrity(
                run_dir,
                required_dataset_role=args.dataset_role,
                require_completed=True,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        rows.extend(read_jsonl(run_dir / "trials.jsonl"))
    output = {"|".join(key): value for key, value in analyze_by_model(rows).items()}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
