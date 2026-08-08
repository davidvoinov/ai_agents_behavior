"""Exclusive OS-level writer lock for one Promise Switch run directory."""

from __future__ import annotations

import fcntl
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any


class RunLockError(RuntimeError):
    """Raised before journal mutation when another process owns the run lock."""


class RunWriterLock:
    """Hold an advisory ``flock`` for the complete lifecycle of one writer."""

    filename = ".writer.lock"

    def __init__(self, run_dir: Path, *, run_id: str):
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / self.filename
        self._descriptor: int | None = None

    def acquire(self) -> "RunWriterLock":
        if self._descriptor is not None:
            raise RunLockError(f"Writer lock already held by this object for {self.run_id}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RunLockError(
                f"Another Promise Switch writer is active for run {self.run_id!r}"
            ) from exc
        self._descriptor = descriptor
        self._write_metadata({
            "run_id": self.run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "authority": "OS advisory flock; file existence alone does not indicate activity",
        })
        return self

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def metadata(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        if self._descriptor is None:
            raise RunLockError("Cannot write lock metadata before acquisition")
        payload = (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        os.ftruncate(self._descriptor, 0)
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        os.write(self._descriptor, payload)
        os.fsync(self._descriptor)

    def __enter__(self) -> "RunWriterLock":
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
