# sn13_api_client.py
from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
import requests
from dateutil.parser import isoparse

from talisman_ai import config
from talisman_ai.validator.x_post_models import (
    PostProvider,
    PostRecord,
    PublicMetrics,
    AuthorInfo,
)
from talisman_ai.validator.x_post_models import with_retries


def _make_x_url_from_post_id(post_id: str) -> str:
    if post_id.startswith("http"):
        return post_id
    if "/" in post_id or "@" in post_id:
        if not post_id.startswith("https://"):
            return f"https://x.com/{post_id}" if not post_id.startswith("x.com") else f"https://{post_id}"
        return post_id
    return f"https://x.com/i/web/status/{post_id}"


class SN13APIClient(PostProvider):
    """SN13 API client wrapper that returns PostRecord."""

    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        self._session = requests.Session()

    def _fetch_once(self, post_id: str) -> Optional[PostRecord]:
        post_url = _make_x_url_from_post_id(post_id)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"source": "X", "url": post_url}

        resp = self._session.post(self.api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        json_resp = resp.json()

        if json_resp.get("status") != "success":
            return None

        data_list = json_resp.get("data") or []
        if not data_list:
            return None

        item = data_list[0]

        # Parse created_at
        dt = item.get("datetime")
        created_at: Optional[datetime]
        if dt:
            created_at = isoparse(dt) if isinstance(dt, str) else dt
        else:
            created_at = None

        post_data = item.get("tweet", {}) or {}
        user_data = item.get("user", {}) or {}

        if created_at is None:
            raise ValueError("SN13 post datetime missing")

        metrics = PublicMetrics(
            like_count=int(post_data.get("like_count", 0) or 0),
            retweet_count=int(post_data.get("retweet_count", 0) or 0),
            reply_count=int(post_data.get("reply_count", 0) or 0),
        )

        author_info = AuthorInfo(
            id=str(user_data["id"]) if user_data.get("id") is not None else None,
            username=user_data.get("username", "") or "",
            display_name=user_data.get("display_name", "") or "",
            followers_count=int(user_data.get("followers_count", 0) or 0),
            created_at=None,  # SN13 doesn't give this
        )

        return PostRecord(
            id=str(post_data.get("id") or ""),
            text=item.get("text", "") or "",
            created_at=created_at,
            public_metrics=metrics,
            author=author_info,
        )

    def fetch_post(self, post_id: str, attempts: int = 3) -> Optional[PostRecord]:
        def op():
            try:
                return self._fetch_once(post_id)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and (
                    e.response.status_code == 429 or e.response.status_code >= 500
                ):
                    raise
                raise
            except requests.exceptions.RequestException as e:
                # Retry only “transient-ish” errors
                msg = str(e).lower()
                if any(s in msg for s in ["timeout", "connection", "500", "502", "503", "504"]):
                    raise
                raise

        return with_retries(op, attempts=attempts, post_id=post_id)


def create_client() -> SN13APIClient:
    api_key = getattr(config, "SN13_API_KEY", None)
    api_url = getattr(
        config,
        "SN13_API_URL",
        "https://constellation.api.cloud.macrocosmos.ai/sn13.v1.Sn13Service/OnDemandData",
    )

    if not api_key or api_key == "null":
        raise ValueError("[GRADER] SN13_API_KEY not set - SN13 API is required for validation")

    return SN13APIClient(api_key, api_url)
