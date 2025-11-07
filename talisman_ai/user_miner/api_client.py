"""
API client for submitting posts to the local FastAPI server.

This module handles HTTP communication with the subnet API server, including:
- POST requests to the /v1/submit endpoint
- Retry logic with exponential backoff (3s, 6s, 12s)
- Header authentication (X-Hotkey)
- Response handling and status validation
"""

import time
import requests
import bittensor as bt
from typing import Dict, Optional

from talisman_ai import config

# Import auth utilities for creating signed headers
import sys
import os
# Add api directory to path to import auth_utils
api_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "api")
if os.path.exists(api_path):
    sys.path.insert(0, api_path)
    try:
        from auth_utils import create_auth_message, sign_message
    except ImportError:
        # If auth_utils not available, we'll use a fallback
        create_auth_message = None
        sign_message = None

class APIClient:
    """
    Client for submitting analyzed posts to the subnet API server.
    
    Handles HTTP communication, retries, and error handling for post submissions.
    The API endpoint is idempotent, so duplicate submissions return a success status.
    """
    
    def __init__(self, wallet: Optional[bt.wallet] = None):
        """
        Initialize the API client.
        
        Sets up the base URL pointing to the local FastAPI server.
        The submission_count tracks total submission attempts (not just successes).
        
        Args:
            wallet: Optional Bittensor wallet for authentication. If provided, requests will include signed auth headers.
        """
        self.submission_count = 0
        self.base_url = config.MINER_API_URL
        self.wallet = wallet

    def _submit_to_api(self, post_data: Dict) -> bool:
        """
        Internal method to submit a post to the API server.
        
        Implements retry logic with exponential backoff:
        - Up to 3 attempts total (initial + 2 retries)
        - Wait times between retries: 3s after first failure, 6s after second failure
        - Wait time calculated as: 3 * (2^attempt) seconds
        - 10 second timeout per request
        
        Args:
            post_data: Dictionary containing post submission data including:
                      miner_hotkey, post_id, content, date, author, tokens, sentiment, etc.
        
        Returns:
            True if submission succeeded (status "new", "duplicate", or "ok"), False otherwise.
            Note: "duplicate" is treated as success since the API is idempotent.
        """
        url = f"{self.base_url}/v1/submit"
        post_id = post_data.get("post_id", "unknown")
        hotkey = post_data.get("miner_hotkey", "")
        
        # Create authentication headers if wallet is available
        headers = {}
        if self.wallet and create_auth_message and sign_message:
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
                bt.logging.debug(f"[APIClient] Added authentication headers for hotkey: {hotkey}")
            except Exception as e:
                bt.logging.warning(f"[APIClient] Failed to create auth headers: {e}, proceeding without auth")
                # No fallback - auth is required
                headers = {}
        else:
            # No fallback - auth is required
            headers = {}
        
        bt.logging.info(f"[APIClient] Submitting post {post_id} to {url} (hotkey: {hotkey})")
        
        # Retry logic: up to 3 attempts with exponential backoff
        # Wait time = 3 * (2^attempt) seconds (gives 3s, 6s, 12s)
        for attempt in range(3): #TODO make this configurable
            try:
                bt.logging.debug(f"[APIClient] Attempt {attempt+1}/3 for post {post_id}")
                resp = requests.post(url, json=post_data, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status")
                bt.logging.info(f"[APIClient] Response for {post_id}: status={status}, message={data.get('message', 'N/A')}")
                # The API returns "new" for first-time submissions, "duplicate" for already-submitted posts
                # Both are considered success since the API is idempotent per (miner_hotkey, post_id)
                return status in ("new", "duplicate", "ok")
            except requests.RequestException as e:
                bt.logging.warning(f"[APIClient] Submit attempt {attempt+1}/3 for {post_id} failed: {e}")
                if attempt < 2:
                    wait_time = 3 * (2 ** attempt)
                    bt.logging.debug(f"[APIClient] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        bt.logging.error(f"[APIClient] All 3 attempts failed for post {post_id}")
        return False

    def submit_post(self, post_data: Dict) -> bool:
        """
        Submit a post to the API server.
        
        This is the public interface for submitting posts. It tracks submission attempts
        and logs the result.
        
        Args:
            post_data: Dictionary containing post submission data (see _submit_to_api for details).
        
        Returns:
            True if submission succeeded, False otherwise.
        """
        self.submission_count += 1
        success = self._submit_to_api(post_data)
        if success:
            bt.logging.trace(f"[APIClient] Submitted post '{post_data.get('post_id')}' successfully")
        else:
            bt.logging.warning(f"[APIClient] API submission failed for post '{post_data.get('post_id')}'")
        return success
