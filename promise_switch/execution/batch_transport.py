"""Durable, provider-independent Batch API primitives for Promise Switch.

Nothing in this module performs I/O at import time.  Provider calls happen only
through an explicit ``BatchCoordinator.submit`` or ``poll_and_retrieve`` call.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..client import RequestSpec
from ..schemas import NormalizedResponse, sanitize_for_serialization
from .lock import RunWriterLock
from .storage import RunStorage


class BatchLifecycleState(str, Enum):
    PREPARED = "PREPARED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    AWAITING_RESULTS = "AWAITING_RESULTS"
    RESULTS_RETRIEVED = "RESULTS_RETRIEVED"
    INGESTED = "INGESTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass(frozen=True)
class BatchRequestMapping:
    custom_id: str
    trial_id: str
    stage: str
    attempt_number: int
    request_body_hash: str
    request: RequestSpec

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_serialization({
            "custom_id": self.custom_id,
            "trial_id": self.trial_id,
            "stage": self.stage,
            "attempt_number": self.attempt_number,
            "request_body_hash": self.request_body_hash,
            "request": self.request.to_log_dict(),
        })


@dataclass(frozen=True)
class ProviderBatchStatus:
    state: str
    terminal: bool
    succeeded: bool
    raw: Mapping[str, Any]


class BatchSubmissionError(RuntimeError):
    """Submission failure with an explicit duplicate-risk classification."""

    def __init__(self, message: str, *, known_not_submitted: bool):
        super().__init__(message)
        self.known_not_submitted = known_not_submitted


class BatchProvider(Protocol):
    name: str

    def serialize(self, mapping: BatchRequestMapping) -> dict[str, Any]: ...
    def submit(self, input_path: Path, *, metadata: Mapping[str, str]) -> str: ...
    def poll(self, provider_job_id: str) -> ProviderBatchStatus: ...
    def retrieve(self, provider_job_id: str, status: ProviderBatchStatus) -> Sequence[dict[str, Any]]: ...
    def normalize(self, result: Mapping[str, Any]) -> NormalizedResponse: ...


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_custom_id(trial_id: str, stage: str, attempt_number: int) -> str:
    if stage not in {"message", "final_decision"}:
        raise ValueError(f"unsupported Promise Switch stage: {stage}")
    if attempt_number not in {1, 2, 3}:
        raise ValueError("Promise Switch permits attempt numbers 1, 2, and 3 only")
    trial_hash = hashlib.sha256(trial_id.encode("utf-8")).hexdigest()[:24]
    stage_code = "m" if stage == "message" else "d"
    return f"ps_{trial_hash}_{stage_code}_a{attempt_number}"


def make_mapping(request: RequestSpec, attempt_number: int) -> BatchRequestMapping:
    trial_id = str(request.metadata.get("trial_id", ""))
    stage = str(request.metadata.get("stage", ""))
    if not trial_id:
        raise ValueError("batch request requires trial_id metadata")
    return BatchRequestMapping(
        custom_id=deterministic_custom_id(trial_id, stage, attempt_number),
        trial_id=trial_id,
        stage=stage,
        attempt_number=attempt_number,
        request_body_hash=canonical_json_hash(request.body),
        request=request,
    )


class BatchCoordinator:
    """Append-only lifecycle coordinator with fail-closed submission recovery."""

    def __init__(
        self, run_dir: Path, provider: BatchProvider, *, timestamp: Callable[[], str]
    ) -> None:
        self.storage = RunStorage(run_dir)
        self.provider = provider
        self.timestamp = timestamp
        self.input_dir = run_dir / "batch_inputs"
        self.result_dir = run_dir / "batch_results"

    def prepare(
        self, *, run_id: str, model: str, reasoning_mode: str, batch_sequence: int,
        batch_phase: str, requests: Sequence[tuple[RequestSpec, int]],
    ) -> str:
        with self._lock():
            return self._prepare_locked(
                run_id=run_id, model=model, reasoning_mode=reasoning_mode,
                batch_sequence=batch_sequence, batch_phase=batch_phase, requests=requests,
            )

    def _prepare_locked(
        self, *, run_id: str, model: str, reasoning_mode: str, batch_sequence: int,
        batch_phase: str, requests: Sequence[tuple[RequestSpec, int]],
    ) -> str:
        if not requests:
            raise ValueError("cannot prepare an empty provider batch")
        mappings = [make_mapping(request, attempt) for request, attempt in requests]
        custom_ids = [item.custom_id for item in mappings]
        if len(custom_ids) != len(set(custom_ids)):
            raise ValueError("duplicate custom_id in provider batch")
        batch_key = f"{self.provider.name}:{run_id}:{batch_phase}:{batch_sequence}"
        serialized = [self.provider.serialize(item) for item in mappings]
        input_payload = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in serialized
        )
        input_hash = hashlib.sha256(input_payload.encode("utf-8")).hexdigest()
        input_path = self.input_dir / f"{hashlib.sha256(batch_key.encode()).hexdigest()[:20]}.jsonl"
        existing_mapping_rows = self.storage.read_jsonl("batch_request_mappings.jsonl")
        existing_attempts = {
            (str(row.get("trial_id")), str(row.get("stage")), int(row.get("attempt_number", 0)))
            for row in existing_mapping_rows
        }
        ingested_attempts = {
            (str(row.get("trial_id")), str(row.get("stage")), int(row.get("attempt_number", 0)))
            for row in self.storage.read_jsonl("api_attempts.jsonl")
        }
        existing_attempts.update(ingested_attempts)
        requested_attempts = {
            (item.trial_id, item.stage, item.attempt_number) for item in mappings
        }
        collision = sorted(requested_attempts & existing_attempts)
        if collision:
            orphan_rows = [row for row in existing_mapping_rows if row.get("batch_key") == batch_key]
            orphan_identity = {
                (str(row.get("trial_id")), str(row.get("stage")), int(row.get("attempt_number", 0)))
                for row in orphan_rows
            }
            mapping_by_identity = {
                (row.trial_id, row.stage, row.attempt_number): row for row in mappings
            }
            if (
                self._latest(batch_key) is None
                and not requested_attempts.intersection(ingested_attempts)
                and orphan_identity == requested_attempts
                and len(orphan_rows) == len(mappings)
                and all(
                    row.get("custom_id") == mapping_by_identity[identity].custom_id
                    and row.get("request_body_hash") == mapping_by_identity[identity].request_body_hash
                    for row in orphan_rows
                    for identity in [(
                        str(row.get("trial_id")), str(row.get("stage")),
                        int(row.get("attempt_number", 0)),
                    )]
                )
                and input_path.is_file()
                and hashlib.sha256(input_path.read_bytes()).hexdigest() == input_hash
            ):
                self._event(batch_key, BatchLifecycleState.PREPARED, {
                    "run_id": run_id, "provider": self.provider.name, "model": model,
                    "reasoning_mode": reasoning_mode, "batch_sequence": batch_sequence,
                    "batch_phase": batch_phase, "input_path": str(input_path),
                    "input_artifact_hash": input_hash, "request_count": len(mappings),
                    "provider_job_id": None, "result_artifact_hash": None,
                    "recovered_orphaned_prepare": True,
                })
                return batch_key
            raise ValueError(f"scientific attempt already exists in this run: {collision}")
        if self._latest(batch_key) is not None or self._mappings(batch_key):
            raise ValueError(f"batch key already exists: {batch_key}")
        _atomic_write_text(input_path, input_payload)
        self.storage.append_many("batch_request_mappings.jsonl", ({
                "batch_key": batch_key, "run_id": run_id, "provider": self.provider.name,
                "model": model, "reasoning_mode": reasoning_mode,
                "batch_sequence": batch_sequence, "batch_phase": batch_phase,
                **mapping.to_dict(),
            } for mapping in mappings))
        self._event(batch_key, BatchLifecycleState.PREPARED, {
            "run_id": run_id, "provider": self.provider.name, "model": model,
            "reasoning_mode": reasoning_mode, "batch_sequence": batch_sequence,
            "batch_phase": batch_phase, "input_path": str(input_path),
            "input_artifact_hash": input_hash, "request_count": len(mappings),
            "provider_job_id": None, "result_artifact_hash": None,
        })
        return batch_key

    def submit(self, batch_key: str) -> str:
        with self._lock():
            return self._submit_locked(batch_key)

    def _submit_locked(self, batch_key: str) -> str:
        current = self._require_state(batch_key, {BatchLifecycleState.PREPARED})
        mappings = self._mappings(batch_key)
        keys = [(row["trial_id"], row["stage"], int(row["attempt_number"])) for row in mappings]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate scientific attempts in prepared batch")
        all_mappings = self.storage.read_jsonl("batch_request_mappings.jsonl")
        all_custom_ids = [str(row.get("custom_id")) for row in all_mappings]
        if len(all_custom_ids) != len(set(all_custom_ids)):
            raise ValueError("batch mapping journal contains duplicate custom_ids")
        for key in keys:
            if sum(
                (row.get("trial_id"), row.get("stage"), int(row.get("attempt_number", 0))) == key
                for row in all_mappings
            ) != 1:
                raise ValueError(f"scientific attempt is not run-wide unique: {key}")
        attempt_rows = self.storage.read_jsonl("api_attempts.jsonl")
        attempt_ids = [str(row.get("attempt_id")) for row in attempt_rows]
        attempt_keys = [
            (row.get("trial_id"), row.get("stage"), int(row.get("attempt_number", 0)))
            for row in attempt_rows
        ]
        if len(attempt_ids) != len(set(attempt_ids)) or len(attempt_keys) != len(set(attempt_keys)):
            raise ValueError("API-attempt journal contains duplicate attempt identities")
        ingested = {
            (row.get("trial_id"), row.get("stage"), int(row.get("attempt_number", 0)))
            for row in attempt_rows
        }
        if any(key in ingested for key in keys):
            raise ValueError("prepared batch contains an already ingested scientific attempt")
        self._event(batch_key, BatchLifecycleState.SUBMITTING, current)
        try:
            job_id = self.provider.submit(
                Path(str(current["input_path"])),
                metadata={"promise_switch_batch_key": batch_key[:512]},
            )
        except BatchSubmissionError as exc:
            state = (BatchLifecycleState.FAILED if exc.known_not_submitted
                     else BatchLifecycleState.RECONCILIATION_REQUIRED)
            self._event(batch_key, state, {**current, "submission_error": str(exc)})
            raise
        except Exception as exc:
            self._event(batch_key, BatchLifecycleState.RECONCILIATION_REQUIRED, {
                **current, "submission_error_type": type(exc).__name__,
            })
            raise
        self._event(batch_key, BatchLifecycleState.SUBMITTED, {
            **current, "provider_job_id": job_id,
        })
        return job_id

    def poll_and_retrieve(self, batch_key: str) -> dict[str, NormalizedResponse] | None:
        with self._lock():
            return self._poll_and_retrieve_locked(batch_key)

    def _poll_and_retrieve_locked(
        self, batch_key: str,
    ) -> dict[str, NormalizedResponse] | None:
        latest = self._latest(batch_key)
        if latest and latest.get("status") in {
            BatchLifecycleState.RESULTS_RETRIEVED.value,
            BatchLifecycleState.INGESTED.value,
        }:
            return self._load_retrieved_results_locked(batch_key, latest)
        current = self._require_state(
            batch_key, {BatchLifecycleState.SUBMITTED, BatchLifecycleState.AWAITING_RESULTS}
        )
        job_id = str(current["provider_job_id"])
        status = self.provider.poll(job_id)
        if not status.terminal:
            self._event(batch_key, BatchLifecycleState.AWAITING_RESULTS, {
                **current, "provider_status": status.state,
            })
            return None
        if not status.succeeded:
            lowered = status.state.lower()
            if "expir" in lowered:
                return self._persist_terminal_synthetic_results(
                    batch_key, current, status, response_status="expired",
                )
            state = (BatchLifecycleState.CANCELED if "cancel" in lowered
                     else BatchLifecycleState.FAILED)
            self._event(batch_key, state, {**current, "provider_status": status.state})
            return None
        results = list(self.provider.retrieve(job_id, status))
        mappings = self._mappings(batch_key)
        expected = {row["custom_id"] for row in mappings}
        actual = [str(row.get("custom_id", "")) for row in results]
        if len(actual) != len(set(actual)) or set(actual) - expected:
            self._event(batch_key, BatchLifecycleState.RECONCILIATION_REQUIRED, {
                **current, "provider_status": status.state,
                "missing_custom_ids": sorted(expected - set(actual)),
                "unexpected_custom_ids": sorted(set(actual) - expected),
                "duplicate_result": len(actual) != len(set(actual)),
            })
            raise ValueError("provider batch results contain duplicate or unexpected custom_ids")
        missing = sorted(expected - set(actual))
        sanitized = sanitize_for_serialization(results)
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in sanitized)
        result_hash = hashlib.sha256(payload.encode()).hexdigest()
        result_path = self.result_dir / f"{hashlib.sha256(batch_key.encode()).hexdigest()[:20]}.jsonl"
        _atomic_write_text(result_path, payload)
        self._event(batch_key, BatchLifecycleState.RESULTS_RETRIEVED, {
            **current, "provider_status": status.state,
            "result_path": str(result_path.resolve()),
            "result_artifact_hash": result_hash,
            "missing_custom_ids": missing,
            "result_coverage_complete": not missing,
        })
        return {str(row["custom_id"]): self.provider.normalize(row) for row in results}

    def load_retrieved_results(self, batch_key: str) -> dict[str, NormalizedResponse]:
        """Load and validate an already durable result artifact without provider I/O."""
        with self._lock():
            current = self._require_state(
                batch_key, {
                    BatchLifecycleState.RESULTS_RETRIEVED, BatchLifecycleState.INGESTED,
                }
            )
            return self._load_retrieved_results_locked(batch_key, current)

    def _load_retrieved_results_locked(
        self, batch_key: str, current: Mapping[str, Any],
    ) -> dict[str, NormalizedResponse]:
        raw_path = current.get("result_path")
        expected_hash = current.get("result_artifact_hash")
        if not isinstance(raw_path, str) or not raw_path or not isinstance(expected_hash, str):
            raise ValueError("retrieved batch lacks a durable result artifact identity")
        result_path = Path(raw_path)
        if not result_path.is_file():
            raise ValueError("durable Batch result artifact is missing")
        payload = result_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("durable Batch result artifact hash mismatch")
        try:
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()
                    if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("durable Batch result artifact is malformed") from exc
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("durable Batch result artifact contains a non-object row")
        mappings = self._mappings(batch_key)
        expected = {str(row["custom_id"]) for row in mappings}
        actual = [str(row.get("custom_id", "")) for row in rows]
        if len(actual) != len(set(actual)) or set(actual) - expected:
            raise ValueError("durable Batch result artifact has duplicate or unexpected custom_ids")
        missing = sorted(expected - set(actual))
        if missing != sorted(str(value) for value in current.get("missing_custom_ids", [])):
            raise ValueError("durable Batch result coverage differs from lifecycle record")
        if bool(current.get("result_coverage_complete")) != (not missing):
            raise ValueError("durable Batch result completeness marker is inconsistent")
        responses: dict[str, NormalizedResponse] = {}
        for row in rows:
            custom_id = str(row["custom_id"])
            synthetic = row.get("synthetic_transport_status")
            responses[custom_id] = (
                NormalizedResponse(**synthetic)
                if isinstance(synthetic, dict) else self.provider.normalize(row)
            )
        return responses

    def mark_ingested(self, batch_key: str) -> None:
        with self._lock():
            current = self._require_state(batch_key, {
                BatchLifecycleState.RESULTS_RETRIEVED, BatchLifecycleState.INGESTED,
            })
            if current.get("status") == BatchLifecycleState.INGESTED.value:
                return
            state = (BatchLifecycleState.RECONCILIATION_REQUIRED
                     if current.get("missing_custom_ids") else BatchLifecycleState.INGESTED)
            self._event(batch_key, state, {
                **current, "known_results_ingested": True,
            })

    def reconcile_submission(self, batch_key: str, provider_job_id: str) -> None:
        """Explicitly attach a provider job found by an operator after ambiguity.

        There is deliberately no automatic search-or-resubmit path: a provider
        identifier must be supplied after independently reconciling the job.
        """
        if not provider_job_id.strip():
            raise ValueError("provider_job_id is required for reconciliation")
        with self._lock():
            current = self._require_state(batch_key, {
                BatchLifecycleState.SUBMITTING,
                BatchLifecycleState.RECONCILIATION_REQUIRED,
            })
            self._event(batch_key, BatchLifecycleState.SUBMITTED, {
                **current, "provider_job_id": provider_job_id,
                "reconciled_explicitly": True,
            })

    def mark_interrupted_submission_for_reconciliation(self, batch_key: str) -> None:
        """Recover a durable SUBMITTING marker left by a process crash."""
        with self._lock():
            current = self._require_state(batch_key, {BatchLifecycleState.SUBMITTING})
            self._event(batch_key, BatchLifecycleState.RECONCILIATION_REQUIRED, {
                **current, "reconciliation_reason": "submission_interrupted_before_job_id",
            })

    def retry_known_non_submission(self, batch_key: str) -> None:
        """Re-prepare only a create/upload operation known not to have submitted."""
        with self._lock():
            current = self._require_state(batch_key, {BatchLifecycleState.FAILED})
            if current.get("provider_job_id") or not current.get("submission_error"):
                raise ValueError("failed batch is not proven safe for operational resubmission")
            self._event(batch_key, BatchLifecycleState.PREPARED, {
                **current, "operational_retry": True,
            })

    def resume_partial_retrieval(self, batch_key: str) -> None:
        """Explicitly re-poll a provider job after known partial results were ingested."""
        with self._lock():
            current = self._require_state(
                batch_key, {BatchLifecycleState.RECONCILIATION_REQUIRED}
            )
            if not current.get("provider_job_id") or not current.get("known_results_ingested"):
                raise ValueError("batch is not an ingested partial-result reconciliation")
            self._event(batch_key, BatchLifecycleState.SUBMITTED, {
                **current, "partial_retrieval_resumed_explicitly": True,
            })

    def mappings(self, batch_key: str) -> tuple[dict[str, Any], ...]:
        return tuple(self._mappings(batch_key))

    def latest(self, batch_key: str) -> dict[str, Any] | None:
        return self._latest(batch_key)

    def _persist_terminal_synthetic_results(
        self, batch_key: str, current: Mapping[str, Any], status: ProviderBatchStatus,
        *, response_status: str,
    ) -> dict[str, NormalizedResponse]:
        mappings = self._mappings(batch_key)
        responses = {
            str(row["custom_id"]): NormalizedResponse(
                status=response_status, raw_response="",
                error_metadata={"provider_batch_status": status.state, "unexecuted": True},
            )
            for row in mappings
        }
        serializable = [
            {"custom_id": custom_id, "synthetic_transport_status": response.to_dict()}
            for custom_id, response in sorted(responses.items())
        ]
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in serializable)
        result_hash = hashlib.sha256(payload.encode()).hexdigest()
        result_path = self.result_dir / f"{hashlib.sha256(batch_key.encode()).hexdigest()[:20]}.jsonl"
        _atomic_write_text(result_path, payload)
        self._event(batch_key, BatchLifecycleState.RESULTS_RETRIEVED, {
            **dict(current), "provider_status": status.state,
            "result_path": str(result_path.resolve()),
            "result_artifact_hash": result_hash, "missing_custom_ids": [],
            "result_coverage_complete": True, "whole_batch_unexecuted": True,
        })
        return responses

    def _lock(self) -> RunWriterLock:
        return RunWriterLock(self.storage.run_dir, run_id=self.storage.run_dir.name)

    def _mappings(self, batch_key: str) -> list[dict[str, Any]]:
        return [row for row in self.storage.read_jsonl("batch_request_mappings.jsonl")
                if row.get("batch_key") == batch_key]

    def _latest(self, batch_key: str) -> dict[str, Any] | None:
        rows = [row for row in self.storage.read_jsonl("batch_jobs.jsonl")
                if row.get("batch_key") == batch_key]
        return rows[-1] if rows else None

    def _require_state(
        self, batch_key: str, allowed: set[BatchLifecycleState]
    ) -> dict[str, Any]:
        row = self._latest(batch_key)
        if row is None:
            raise ValueError(f"unknown batch key: {batch_key}")
        if BatchLifecycleState(str(row["status"])) not in allowed:
            raise ValueError(f"batch {batch_key} is in state {row['status']}")
        return row

    def _event(
        self, batch_key: str, state: BatchLifecycleState, values: Mapping[str, Any]
    ) -> None:
        timestamp = self.timestamp()
        prior_count = sum(row.get("batch_key") == batch_key
                          for row in self.storage.read_jsonl("batch_jobs.jsonl"))
        self.storage.append("batch_jobs.jsonl", sanitize_for_serialization({
            **dict(values), "batch_key": batch_key, "status": state.value,
            "timestamp": timestamp,
            "lifecycle_event_id": hashlib.sha256(
                f"{batch_key}|{prior_count}|{state.value}".encode()
            ).hexdigest(),
        }))


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
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
