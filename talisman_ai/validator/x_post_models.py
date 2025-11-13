from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Protocol, TypeVar

import hashlib
import time


T = TypeVar("T")


@dataclass
class PublicMetrics:
    """
    Minimal engagement metrics needed by the grader.

    All values are non-negative integers coming from the upstream API
    (X, SN13, etc.), but we don't enforce that here beyond typing.
    """
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0


@dataclass
class AuthorInfo:
    """
    Author metadata as used by the grader.

    created_at may be None if the upstream API doesn't provide it
    (e.g., SN13 today). followers_count should be the best available
    value from the upstream API.
    """
    id: Optional[str]
    username: str
    display_name: str
    followers_count: int = 0
    created_at: Optional[datetime] = None


@dataclass
class PostRecord:
    """
    Unified representation of a single X post as seen by the grader.

    Any API client (tweepy-based, SN13-based, mock, etc.) should construct and
    return this shape so the rest of the validator code doesn't depend on
    specific SDK response types.
    """
    id: str
    text: str
    created_at: datetime
    public_metrics: PublicMetrics
    author: AuthorInfo


class PostProvider(Protocol):
    """
    Minimal interface the grader needs from any post data source.

    Implementations:
        - XAPIClient (tweepy-based)
        - SN13APIClient
        - In-memory / fake provider for tests
    """

    def fetch_post(self, post_id: str, attempts: int = 3) -> Optional[PostRecord]:
        """
        Fetch a post by ID.

        Implementations are free to use retries, backoff, etc. internally,
        but they should obey the 'attempts' hint as an upper bound on how
        many tries to perform before failing.

        Returns:
            PostRecord if the post was found and parsed successfully,
            or None if not found / inaccessible (after all retries).
            Exceptions are allowed to bubble up for non-retryable failures.
        """
        ...


def with_retries(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    post_id: str,
    base_delay: float = 0.5,
    retry_on: Optional[Callable[[Exception], bool]] = None,
) -> Optional[T]:
    """
    Run 'func' with bounded retries and deterministic jitter based on post_id.

    This is intentionally generic and does not know about X / SN13 / requests /
    tweepy. Callers decide which exceptions are retryable via 'retry_on'.

    Args:
        func: Zero-argument callable to execute (e.g., a single HTTP request).
        attempts: Maximum number of attempts (must be >= 1).
        post_id: Used to seed deterministic jitter so behavior is stable
                 across replays for the same post.
        base_delay: Base delay (seconds) before backoff/jitter modifiers.
        retry_on: If provided, only exceptions for which retry_on(e) is True
                  will be retried. Others are re-raised immediately.

    Returns:
        The value returned by 'func' on success, or None if all attempts
        are exhausted and 'func' consistently returns None without raising.

    Raises:
        The last encountered exception for non-retryable errors, or when
        all retry attempts fail due to exceptions.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    # Precompute jitter seed from post_id for deterministic behavior.
    jitter_seed = int(
        hashlib.md5(str(post_id).encode("utf-8")).hexdigest()[:8], 16
    ) % 21

    for attempt_index in range(attempts):
        try:
            result = func()
            # If func returns successfully (even if result is None), we stop.
            return result
        except Exception as e:
            # If caller says "do not retry this", re-raise immediately.
            if retry_on is not None and not retry_on(e):
                raise

            # If this was the last attempt, re-raise.
            if attempt_index == attempts - 1:
                raise

            # Deterministic jittered backoff:
            #  delay = base_delay * (attempt_number) + jitter_seed / 100.0
            # where attempt_number starts at 1.
            attempt_number = attempt_index + 1
            delay = base_delay * attempt_number + (jitter_seed / 100.0)
            time.sleep(delay)

    # If we somehow exit the loop without returning or raising,
    # treat it as a "no result" outcome.
    return None
