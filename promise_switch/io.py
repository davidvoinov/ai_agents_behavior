"""File IO helpers for experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            values = [_csv_cell(row.get(column)) for column in columns]
            handle.write(",".join(values) + "\n")


def _csv_cell(value: Any) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        text = '"' + text.replace('"', '""') + '"'
    return text
