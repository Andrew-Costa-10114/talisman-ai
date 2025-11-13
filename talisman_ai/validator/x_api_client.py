# x_api_client.py
from __future__ import annotations
from typing import Optional

import tweepy
from datetime import datetime
from talisman_ai import config

from talisman_ai.validator.x_post_models import (
    PostProvider,
    PostRecord,
    PublicMetrics,
    AuthorInfo,
)
from talisman_ai.validator.x_post_models import with_retries


class XAPIClient(PostProvider):
    """Wrapper around tweepy.Client that returns PostRecord."""

    def __init__(self, client: tweepy.Client):
        self.client = client

    def _fetch_once(self, post_id: str) -> Optional[PostRecord]:
        resp = self.client.get_tweet(
            id=str(post_id),
            expansions=["author_id"],
            tweet_fields=["created_at", "public_metrics", "text"],
            user_fields=["username", "name", "created_at", "public_metrics"],
        )

        if not resp or not getattr(resp, "data", None):
            return None

        post = resp.data
        includes = getattr(resp, "includes", {}) or {}
        users = {u.id: u for u in includes.get("users", [])}
        author = users.get(post.author_id)

        # Build domain objects
        pm = getattr(post, "public_metrics", {}) or {}
        metrics = PublicMetrics(
            like_count=int(pm.get("like_count", 0) or 0),
            retweet_count=int(pm.get("retweet_count", 0) or 0),
            reply_count=int(pm.get("reply_count", 0) or 0),
        )

        author_metrics = getattr(author, "public_metrics", {}) or {}
        author_info = AuthorInfo(
            id=str(author.id) if author and getattr(author, "id", None) else None,
            username=(author.username if author else "") or "",
            display_name=(author.name if author else "") or "",
            followers_count=int(author_metrics.get("followers_count", 0) or 0),
            created_at=getattr(author, "created_at", None),
        )

        created_at = getattr(post, "created_at", None)
        if not isinstance(created_at, datetime):
            raise ValueError("Post created_at missing or invalid from X API")

        return PostRecord(
            id=str(post.id),
            text=post.text or "",
            created_at=created_at,
            public_metrics=metrics,
            author=author_info,
        )

    def fetch_post(self, post_id: str, attempts: int = 3) -> Optional[PostRecord]:
        def op():
            try:
                return self._fetch_once(post_id)
            except tweepy.TooManyRequests:
                raise  # will be retried by with_retries
            except tweepy.TweepyException as e:
                # Only retry transient-ish things; bubble others
                if any(
                    s in str(e).lower()
                    for s in ["500", "502", "503", "504", "timeout", "connection"]
                ):
                    raise
                raise

        return with_retries(op, attempts=attempts, post_id=post_id)


def create_client() -> XAPIClient:
    """Create an X API client. Uses talisman_ai.config.X_BEARER_TOKEN."""
    token = getattr(config, "X_BEARER_TOKEN", None)
    if not token or token == "null":
        raise ValueError("[GRADER] X_BEARER_TOKEN not set - X API is required for validation")
    try:
        tweepy_client = tweepy.Client(bearer_token=token)
        return XAPIClient(tweepy_client)
    except Exception as e:
        raise RuntimeError(f"[GRADER] Failed to initialize X API client: {e}") from e
