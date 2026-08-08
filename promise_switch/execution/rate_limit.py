"""Provider-aware proactive pacing independent of scientific allocation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RateLimitPolicy:
    requests_per_minute: int | None = 30
    tokens_per_minute: int | None = None
    max_concurrency: int = 1
    minimum_interval_seconds: float = 2.0

    def to_dict(self) -> dict[str, int | float | None]:
        return self.__dict__.copy()


class RateLimiter:
    def __init__(
        self,
        policy: RateLimitPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.policy = policy
        self.clock = clock
        self.sleeper = sleeper
        self._last_request: float | None = None
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()

    def before_request(self, estimated_tokens: int = 0) -> float:
        total_wait = 0.0
        while True:
            now = self.clock()
            self._evict(now)
            waits = [0.0]
            if self._last_request is not None:
                waits.append(self.policy.minimum_interval_seconds - (now - self._last_request))
            if self.policy.requests_per_minute and len(self._requests) >= self.policy.requests_per_minute:
                waits.append(60.0 - (now - self._requests[0]))
            if self.policy.tokens_per_minute and estimated_tokens:
                used = sum(tokens for _, tokens in self._tokens)
                if used + estimated_tokens > self.policy.tokens_per_minute and self._tokens:
                    running = used
                    for timestamp, tokens in self._tokens:
                        running -= tokens
                        if running + estimated_tokens <= self.policy.tokens_per_minute:
                            waits.append(60.0 - (now - timestamp))
                            break
            wait = max(0.0, max(waits))
            if not wait:
                break
            self.sleeper(wait)
            total_wait += wait
        now = self.clock()
        self._last_request = now
        self._requests.append(now)
        if estimated_tokens:
            self._tokens.append((now, estimated_tokens))
        return total_wait

    def _evict(self, now: float) -> None:
        while self._requests and now - self._requests[0] >= 60.0:
            self._requests.popleft()
        while self._tokens and now - self._tokens[0][0] >= 60.0:
            self._tokens.popleft()
