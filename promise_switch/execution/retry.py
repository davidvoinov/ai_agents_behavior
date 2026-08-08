"""Bounded exponential backoff with jitter and Retry-After support."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter_fraction: float = 0.25

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def delay(
        self,
        *,
        failed_attempt_number: int,
        retry_after_seconds: float | None = None,
        random_value: Callable[[], float] = random.random,
    ) -> float:
        exponential = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** max(0, failed_attempt_number - 1)),
        )
        jitter = exponential * self.jitter_fraction * random_value()
        return max(exponential + jitter, retry_after_seconds or 0.0)

    def to_dict(self) -> dict[str, int | float]:
        return self.__dict__.copy()
