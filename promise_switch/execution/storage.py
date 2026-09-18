"""Atomic JSON and fsync-backed append-only JSONL storage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

FILES = (
    "manifest.json",
    "planned_trials.jsonl",
    "top_up_events.jsonl",
    "pause_events.jsonl",
    "parser_review_events.jsonl",
    "message_events.jsonl",
    "assignments.jsonl",
    "api_attempts.jsonl",
    "trials.jsonl",
    "errors.jsonl",
    "batch_jobs.jsonl",
    "batch_request_mappings.jsonl",
    "run_state.json",
)


class RunStorage:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            if name.endswith(".jsonl"):
                (self.run_dir / name).touch(exist_ok=True)

    def create_manifest(self, manifest: dict[str, Any]) -> None:
        path = self.run_dir / "manifest.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o444)
        try:
            payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fsync_directory()

    def read_manifest(self) -> dict[str, Any]:
        return json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))

    def append(self, name: str, record: dict[str, Any]) -> None:
        if name not in FILES or not name.endswith(".jsonl"):
            raise ValueError(f"Not an append-only run file: {name}")
        path = self.run_dir / name
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_many(self, name: str, records: Iterable[dict[str, Any]]) -> None:
        if name not in FILES or not name.endswith(".jsonl"):
            raise ValueError(f"Not an append-only run file: {name}")
        payload = b"".join(
            (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            for record in records
        )
        if not payload:
            return
        descriptor = os.open(
            self.run_dir / name, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644,
        )
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("short append while writing Promise Switch journal")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.run_dir / name
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def write_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.run_dir / "run_state.json", state)

    def read_state(self) -> dict[str, Any]:
        path = self.run_dir / "run_state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.run_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
