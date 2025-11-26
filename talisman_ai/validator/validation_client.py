# talisman_ai/validator/validation_client.py
import asyncio
from typing import Any, Dict, List, Callable, Optional
import httpx
import bittensor as bt
import time

from talisman_ai import config


def create_auth_message(timestamp=None):
    """Create a standardized authentication message"""
    if timestamp is None:
        timestamp = time.time()
    return f"talisman-ai-auth:{int(timestamp)}"


def sign_message(wallet, message):
    """Sign a message with the wallet's hotkey"""
    signature = wallet.hotkey.sign(message)
    return signature.hex()


class ValidationClient:
    """
    Client for API v2 validation system.
    
    - Gets validation payloads from /v2/validation when done grading
    - Submits results to /v2/validation_result when grading is done
    - Gets scores from /v2/scores every N blocks (default 100) and sets hotkey rewards
    - Tracks last scores block window to avoid duplicates
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        poll_seconds: Optional[int] = None,
        http_timeout: Optional[float] = None,
        scores_block_interval: Optional[int] = None,
        wallet: Optional[bt.wallet] = None,
    ):
        """
        Initialize the ValidationClient.
        
        Args:
            api_url: Base URL for the miner API. Defaults to MINER_API_URL env var
            poll_seconds: Seconds between poll attempts. Defaults to VALIDATION_POLL_SECONDS env var or 10 seconds
            http_timeout: HTTP request timeout in seconds. Defaults to BATCH_HTTP_TIMEOUT env var or 30.0 seconds
            scores_block_interval: Blocks between score fetches. Defaults to SCORES_BLOCK_INTERVAL env var or 100
            wallet: Optional Bittensor wallet for authentication
        """
        self.api_url = api_url or config.MINER_API_URL
        self.validation_endpoint = f"{self.api_url}/v2/validation"
        self.validation_result_endpoint = f"{self.api_url}/v2/validation_result"
        self.scores_endpoint = f"{self.api_url}/v2/scores"
        self.poll_seconds = int(poll_seconds or config.VALIDATION_POLL_SECONDS)
        self.http_timeout = float(http_timeout or config.BATCH_HTTP_TIMEOUT)
        self.scores_block_interval = int(scores_block_interval or config.SCORES_BLOCK_INTERVAL)
        self.wallet = wallet

        self._running: bool = False
        self._last_scores_window: Optional[int] = None  # Last window number we fetched scores for (from API)
        self._current_validations: List[Dict[str, Any]] = []
        self._last_status_check: Optional[float] = None  # Timestamp of last status check
        self._status_check_interval: int = 60  # Check status every 60 seconds (scores change every ~20 minutes)

    def _create_auth_headers(self) -> Dict[str, str]:
        """Create authentication headers if wallet is available"""
        headers = {}
        if self.wallet:
            try:
                timestamp = time.time()
                message = create_auth_message(timestamp)
                signature = sign_message(self.wallet, message)
                headers = {
                    "X-Auth-SS58Address": self.wallet.hotkey.ss58_address,
                    "X-Auth-Signature": signature,
                    "X-Auth-Message": message,
                    "X-Auth-Timestamp": str(timestamp)
                }
            except Exception as e:
                bt.logging.warning(f"[VALIDATION] Failed to create auth headers: {e}")
        return headers

    async def fetch_validations(self) -> List[Dict[str, Any]]:
        """Fetch pending validation payloads from /v2/validation"""
        headers = self._create_auth_headers()
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            r = await client.get(self.validation_endpoint, headers=headers)
            r.raise_for_status()
            data = r.json()
            if data.get("available"):
                return data.get("payloads", [])
            return []

    async def submit_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit validation results to /v2/validation_result"""
        payload = {
            "validator_hotkey": str(self.wallet.hotkey.ss58_address),
            "results": results,
        }
        headers = self._create_auth_headers()
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            r = await client.post(self.validation_result_endpoint, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()

    async def fetch_status(self) -> Optional[Dict[str, Any]]:
        """Fetch status from /v2/status to sync block numbers"""
        status_endpoint = f"{self.api_url}/v2/status"
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            r = await client.get(status_endpoint)
            r.raise_for_status()
            return r.json()

    async def fetch_scores(self) -> Optional[Dict[str, Any]]:
        """Fetch scores from /v2/scores"""
        headers = self._create_auth_headers()
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            r = await client.get(self.scores_endpoint, headers=headers)
            r.raise_for_status()
            return r.json()

    async def _should_fetch_scores(self) -> bool:
        """
        Check if we should fetch scores by querying API's current window.
        
        Since the API now returns scores from the PREVIOUS window, we fetch scores
        when a new window starts (i.e., when current_window > last_scores_window).
        This ensures we get the completed previous window's scores.
        """
        import time
        
        # Only check status periodically (not every loop iteration)
        # Since scores change every ~20 minutes, checking every 60 seconds is sufficient
        now = time.time()
        if self._last_status_check is not None:
            time_since_last_check = now - self._last_status_check
            if time_since_last_check < self._status_check_interval:
                # Too soon to check again
                return False
        
        self._last_status_check = now
        
        try:
            status_data = await self.fetch_status()
            if status_data and status_data.get("status") == "ok":
                api_current_window = status_data.get("current_window")
                
                if api_current_window is not None:
                    # Fetch if we haven't fetched for this current window yet
                    # When a new window starts, we want to fetch the previous window's scores
                    if self._last_scores_window is None or self._last_scores_window < api_current_window:
                        return True
                    
                    # Already fetched for this window
                    return False
        except Exception as e:
            bt.logging.debug(f"[VALIDATION] Failed to check API window: {e}")
        
        # If we can't determine, don't fetch (will retry on next check interval)
        return False

    async def run(
        self,
        on_validations: Callable[[List[Dict[str, Any]]], Any],
        on_scores: Callable[[Dict[str, Any]], Any],
    ):
        """
        Main validation loop.
        
        Args:
            on_validations: Callback for batch of validation payloads (async or sync)
            on_scores: Callback when scores are fetched (async or sync)
        """
        self._running = True
        bt.logging.info(f"[VALIDATION] Starting validation client (poll_interval={self.poll_seconds}s, scores_interval={self.scores_block_interval} blocks)")

        try:
            while self._running:
                try:
                    # Check if we need to fetch scores (based on API's current window)
                    should_fetch = await self._should_fetch_scores()
                    
                    if should_fetch:
                        try:
                            # Get current window from status to confirm we should fetch
                            status_data = await self.fetch_status()
                            if status_data and status_data.get("status") == "ok":
                                api_current_window = status_data.get("current_window")
                                
                                if api_current_window is not None:
                                    # Fetch scores - API now returns scores from the PREVIOUS window
                                    scores_data = await self.fetch_scores()
                                    if scores_data:
                                        # Verify we got scores for the previous window
                                        api_window_start = scores_data.get("block_window_start")
                                        api_window_end = scores_data.get("block_window_end")
                                        api_current_block = scores_data.get("current_block", 0)
                                        
                                        if api_window_start is not None:
                                            # Use blocks_per_window from API response (source of truth)
                                            api_blocks_per_window = scores_data.get("blocks_per_window", self.scores_block_interval)
                                            
                                            # Calculate window from block_window_start using API's blocks_per_window
                                            # This should be the previous window (api_current_window - 1)
                                            calculated_window = api_window_start // api_blocks_per_window
                                            expected_previous_window = api_current_window - 1
                                            
                                            # Verify we got scores for the previous window
                                            if calculated_window == expected_previous_window or (api_current_window > 0 and calculated_window < api_current_window):
                                                # Mark that we've fetched scores for this current window
                                                # (the scores are from the previous window, but we fetched them now)
                                                self._last_scores_window = api_current_window
                                                bt.logging.info(
                                                    f"[VALIDATION] Fetched scores for previous block window {calculated_window} "
                                                    f"(blocks {api_window_start}-{api_window_end}, "
                                                    f"current window={api_current_window}, current block={api_current_block})"
                                                )
                                                
                                                # Call scores callback
                                                maybe_coro = on_scores(scores_data)
                                                if asyncio.iscoroutine(maybe_coro):
                                                    await maybe_coro
                                            else:
                                                bt.logging.warning(
                                                    f"[VALIDATION] Window mismatch: expected previous window {expected_previous_window}, "
                                                    f"got {calculated_window} (current={api_current_window}), skipping"
                                                )
                                        else:
                                            bt.logging.warning(
                                                f"[VALIDATION] Invalid scores response: missing block_window_start"
                                            )
                                else:
                                    bt.logging.debug(
                                        f"[VALIDATION] Status response missing current_window, will retry"
                                    )
                        except Exception as e:
                            bt.logging.warning(f"[VALIDATION] Failed to fetch scores: {e}")

                    # Fetch validations if we don't have any pending
                    if not self._current_validations:
                        try:
                            validations = await self.fetch_validations()
                            if validations:
                                self._current_validations = validations
                                bt.logging.info(f"[VALIDATION] Fetched {len(validations)} validation(s)")
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 404:
                                # No validations available, continue
                                pass
                            else:
                                bt.logging.warning(f"[VALIDATION] HTTP {e.response.status_code}: {e}")
                        except Exception as e:
                            bt.logging.warning(f"[VALIDATION] Failed to fetch validations: {e}")

                    # Process all validations at once
                    if self._current_validations:
                        validations = self._current_validations.copy()
                        self._current_validations.clear()
                        maybe_coro = on_validations(validations)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro
                        # Continue immediately to fetch more validations
                        continue

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    bt.logging.warning(f"[VALIDATION] Error in validation loop: {e}")

                # Sleep only if no validations to process
                await asyncio.sleep(self.poll_seconds)

        except asyncio.CancelledError:
            bt.logging.info("[VALIDATION] Validation client cancelled")
            raise
        finally:
            self._running = False
            bt.logging.info("[VALIDATION] Validation client stopped")

