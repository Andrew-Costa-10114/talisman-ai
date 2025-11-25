# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

import asyncio
import time
from typing import List, Dict, Any, Optional
import numpy as np
import bittensor as bt
from talisman_ai.base.validator import BaseValidatorNeuron
from talisman_ai.validator.forward import forward
from talisman_ai.validator.validation_client import ValidationClient
from talisman_ai.validator.grader import grade_hotkey, CONSENSUS_VALID, CONSENSUS_INVALID


class Validator(BaseValidatorNeuron):
    """
    Validator neuron for API v2 probabilistic validation system.
    
    The validator:
    1. Gets validation payloads from /v2/validation
    2. Grades individual posts using the grading system
    3. Submits results to /v2/validation_result
    4. Gets scores from /v2/scores every N blocks and sets hotkey rewards (only source of score updates)
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # Initialize validation client
        self._validation_client = ValidationClient(wallet=self.wallet)
        self._validation_task: Optional[asyncio.Task] = None
        self._pending_results: List[Dict[str, Any]] = []

    async def _on_validation(self, validation: Dict[str, Any]):
        """
        Process a single validation payload.
        
        Args:
            validation: Validation payload with validation_id, miner_hotkey, post, selected_at
        """
        validation_id = validation.get("validation_id")
        miner_hotkey = validation.get("miner_hotkey")
        post = validation.get("post", {})
        
        bt.logging.info(f"[VALIDATION] Processing validation_id={validation_id}, miner_hotkey={miner_hotkey}")
        
        # Grade the post
        label, grade_result = grade_hotkey([post])
        
        # Determine success and failure_reason
        success = label == CONSENSUS_VALID
        failure_reason = None
        
        if not success:
            error_info = grade_result.get("error", {})
            failure_reason = {
                "code": error_info.get("code", "unknown_error"),
                "message": error_info.get("message", "Unknown error"),
                "post_id": error_info.get("post_id", post.get("post_id", "unknown")),
                "details": error_info.get("details", {})
            }
            bt.logging.warning(
                f"[VALIDATION] ✗ Validation FAILED for {miner_hotkey}: "
                f"{failure_reason['code']} - {failure_reason['message']}"
            )
        else:
            bt.logging.info(f"[VALIDATION] ✓ Validation PASSED for {miner_hotkey}")
        
        # Add result to pending results
        result = {
            "validator_hotkey": str(self.wallet.hotkey.ss58_address),
            "validation_id": validation_id,
            "miner_hotkey": miner_hotkey,
            "success": success,
            "failure_reason": failure_reason,
        }
        self._pending_results.append(result)
        
        # Submit results if we have some accumulated (or immediately for single validation)
        await self._submit_pending_results()

    async def _submit_pending_results(self):
        """Submit pending validation results to the API"""
        if not self._pending_results:
            return
        
        results = self._pending_results.copy()
        self._pending_results.clear()
        
        bt.logging.info(f"[VALIDATION] Submitting {len(results)} validation result(s)")
        
        try:
            response = await self._validation_client.submit_results(results)
            bt.logging.info(f"[VALIDATION] ✓ Submitted results: {response}")
        except Exception as e:
            bt.logging.error(f"[VALIDATION] ✗ Failed to submit results: {e}", exc_info=True)
            # Re-add results to pending for retry
            self._pending_results.extend(results)

    async def _on_scores(self, scores_data: Dict[str, Any]):
        """
        Handle scores fetched from /v2/scores endpoint.
        
        Sets hotkey rewards based on scores for the current block window.
        
        Args:
            scores_data: Scores response with scores dict, block_window metadata, etc.
        """
        scores = scores_data.get("scores", {})
        current_block = scores_data.get("current_block")
        block_window_start = scores_data.get("block_window_start")
        block_window_end = scores_data.get("block_window_end")
        
        bt.logging.info(
            f"[SCORES] Processing scores for block window {block_window_start}-{block_window_end} "
            f"(current={current_block}, {len(scores)} hotkeys)"
        )
        
        if not scores:
            bt.logging.warning("[SCORES] No scores in response")
            return
        
        uids_to_update = []
        rewards_array = []
        
        for hotkey, score in scores.items():
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
                uids_to_update.append(uid)
                rewards_array.append(float(score))
            except ValueError:
                bt.logging.debug(f"[SCORES] Hotkey {hotkey} not found in metagraph, skipping")
        
        if uids_to_update and rewards_array:
            rewards_np = np.array(rewards_array)
            uids_np = np.array(uids_to_update)
            bt.logging.info(f"[SCORES] Setting rewards for {len(uids_to_update)} miner(s)...")
            self.update_scores(rewards_np, uids_np.tolist())
            bt.logging.info(
                f"[SCORES] ✓ Set rewards: range {rewards_np.min():.3f} - {rewards_np.max():.3f}, "
                f"mean={rewards_np.mean():.3f}"
            )
        else:
            bt.logging.warning("[SCORES] No valid hotkeys found to update")

    async def forward(self):
        """
        Main validator forward loop.
        
        Starts the validation client on first invocation. The client runs independently
        in the background, processing validations and fetching scores as needed.
        """
        if self._validation_task is None:
            self._validation_task = asyncio.create_task(
                self._validation_client.run(
                    on_validation=self._on_validation,
                    on_scores=self._on_scores,
                )
            )
            bt.logging.info("[VALIDATION] Started validation client")

        self.save_state()
        return await forward(self)


# Entrypoint
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
