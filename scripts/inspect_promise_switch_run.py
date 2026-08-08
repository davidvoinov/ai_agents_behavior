#!/usr/bin/env python3
"""Inspect immutable configuration and technical state without running a trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    invalid_marker = args.run_dir / "INVALID_CONCURRENT_EXECUTION.json"
    if invalid_marker.exists():
        print("[INVALID RUN MARKER]")
        print(json.dumps(
            json.loads(invalid_marker.read_text(encoding="utf-8")),
            indent=2, sort_keys=True,
        ))
    for name in ("manifest.json", "run_state.json"):
        path = args.run_dir / name
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        print(f"[{name}]")
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
