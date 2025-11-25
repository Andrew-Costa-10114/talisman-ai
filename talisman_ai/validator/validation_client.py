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
        self._last_scores_block_window: Optional[int] = None
        self._current_validations: List[Dict[str, Any]] = []

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

    async def fetch_scores(self) -> Optional[Dict[str, Any]]:
        """Fetch scores from /v2/scores"""
        headers = self._create_auth_headers()
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            r = await client.get(self.scores_endpoint, headers=headers)
            r.raise_for_status()
            return r.json()

    def _get_current_block(self) -> int:
        """Get current block number"""
        try:
            network = getattr(config, "BT_NETWORK", "test")
            sub = bt.subtensor(network=network)
            return sub.get_current_block()
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to get current block: {e}")
            return 0

    def _should_fetch_scores(self, current_block: int) -> bool:
        """Check if we should fetch scores based on block interval"""
        if current_block == 0:
            return False
        
        # Calculate which block window we're in
        block_window = current_block // self.scores_block_interval
        
        # Fetch if we haven't fetched for this window yet
        if self._last_scores_block_window is None or self._last_scores_block_window < block_window:
            return True
        
        # Still in the same window - will check again on next loop iteration
        return False

    async def run(
        self,
        on_validation: Callable[[Dict[str, Any]], Any],
        on_scores: Callable[[Dict[str, Any]], Any],
    ):
        """
        Main validation loop.
        
        Args:
            on_validation: Callback for each validation payload (async or sync)
            on_scores: Callback when scores are fetched (async or sync)
        """
        self._running = True
        bt.logging.info(f"[VALIDATION] Starting validation client (poll_interval={self.poll_seconds}s, scores_interval={self.scores_block_interval} blocks)")

        try:
            while self._running:
                try:
                    # Check if we need to fetch scores
                    current_block = self._get_current_block()
                    should_fetch = self._should_fetch_scores(current_block)
                    
                    if not should_fetch and self._last_scores_block_window is not None:
                        # Still in same block window, will check again on next iteration
                        current_window = current_block // self.scores_block_interval
                        blocks_until_next = ((current_window + 1) * self.scores_block_interval) - current_block
                        bt.logging.debug(
                            f"[VALIDATION] Scores check: still in window {current_window} "
                            f"({blocks_until_next} blocks until next window)"
                        )
                    
                    if should_fetch:
                        try:
                            expected_window_start = (current_block // self.scores_block_interval) * self.scores_block_interval
                            
                            scores_data = await self.fetch_scores()
                            if scores_data:
                                # Verify the scores match the expected window
                                api_window_start = scores_data.get("block_window_start")
                                api_current_block = scores_data.get("current_block", 0)
                                
                                # Double-check we're still in the same window
                                verify_block = self._get_current_block()
                                verify_window_start = (verify_block // self.scores_block_interval) * self.scores_block_interval
                                
                                if api_window_start == expected_window_start and verify_window_start == expected_window_start:
                                    block_window = current_block // self.scores_block_interval
                                    self._last_scores_block_window = block_window
                                    bt.logging.info(
                                        f"[VALIDATION] Fetched scores for block window {block_window} "
                                        f"(blocks {api_window_start}-{scores_data.get('block_window_end')}, "
                                        f"current={api_current_block})"
                                    )
                                    
                                    # Call scores callback
                                    maybe_coro = on_scores(scores_data)
                                    if asyncio.iscoroutine(maybe_coro):
                                        await maybe_coro
                                else:
                                    bt.logging.warning(
                                        f"[VALIDATION] Block window mismatch: expected_start={expected_window_start}, "
                                        f"api_start={api_window_start}, verify_start={verify_window_start}, skipping"
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

                    # Process validations one at a time
                    if self._current_validations:
                        validation = self._current_validations.pop(0)
                        maybe_coro = on_validation(validation)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro
                        # Continue immediately to process next validation or fetch more
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

