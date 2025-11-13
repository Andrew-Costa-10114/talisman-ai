"""
User miner implementation that orchestrates the post processing pipeline:
scrape -> analyze -> submit.

Runs in a single background thread for simplicity and reliability.
The miner continuously scrapes posts, analyzes them for subnet relevance and sentiment,
and submits them to the API server.
"""

import threading
import time
import bittensor as bt
from typing import Dict
from datetime import datetime, timezone

from talisman_ai import config
from talisman_ai.user_miner.api_client import APIClient
from talisman_ai.user_miner.post_scraper import PostScraper
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.analyzer.scoring import score_post_entry
from talisman_ai.utils.normalization import norm_text

class MyMiner:
    """
    User miner that processes posts in a background thread.
    
    The miner follows a simple pipeline:
    1. Scrapes posts from the configured source (PostScraper)
    2. Analyzes each post for subnet relevance and sentiment (Analyzer)
    3. Submits analyzed posts to the API server (APIClient)
    
    Runs continuously until max_posts is reached or stopped explicitly.
    """
    
    def __init__(self, hotkey: str = "HOTKEY_PLACEHOLDER", wallet: bt.wallet = None):
        """
        Initialize the miner with required components.
        
        Args:
            hotkey: Miner hotkey identifier, typically provided by the parent neuron.
                   Defaults to placeholder if not provided.
            wallet: Optional Bittensor wallet for API authentication.
        """
        self.scraper = PostScraper()
        self.analyzer = setup_analyzer()
        self.api_client = APIClient(wallet=wallet)

        self.miner_hotkey = hotkey

        # Track post IDs we've already processed to avoid duplicate submissions
        self._seen_post_ids: set[str] = set()

        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        self.posts_processed = 0
        # Maximum number of posts to process before stopping
        # Set via MAX_POSTS environment variable in .miner_env file (default: 1000)
        self.max_posts = config.MAX_POSTS

    def start(self):
        """
        Starts the miner's background processing thread.
        Safe to call multiple times (idempotent).
        """
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            bt.logging.info("[MyMiner] Started in background thread")

    def stop(self):
        """
        Stops the miner's background processing thread.
        Waits up to 5 seconds for the thread to finish gracefully.
        """
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=5)
            bt.logging.info("[MyMiner] Stopped")

    def _run(self):
        """
        Main processing loop running in the background thread.
        
        Continuously scrapes posts, analyzes them, and submits to the API
        until max_posts is reached or running is set to False.
        Handles errors gracefully and logs progress.
        """
        bt.logging.info("[MyMiner] Background thread started")
        max_posts_target = "unlimited" if self.max_posts <= 0 else f"{self.max_posts} post(s)"
        bt.logging.info(f"[MyMiner] Target: {max_posts_target}, hotkey: {self.miner_hotkey}")
        scrape_interval = config.SCRAPE_INTERVAL_SECONDS
        posts_per_cycle = config.POSTS_PER_SCRAPE
        posts_to_submit = config.POSTS_TO_SUBMIT
        max_posts_display = "unlimited" if self.max_posts <= 0 else str(self.max_posts)
        bt.logging.info(f"[MyMiner] Configuration: scrape_interval={scrape_interval}s, posts_per_cycle={posts_per_cycle}, posts_to_submit={posts_to_submit}, max_posts={max_posts_display}")
        
        first_cycle = True
        while self.running and (self.max_posts <= 0 or self.posts_processed < self.max_posts):
            try:
                max_posts_display = "unlimited" if self.max_posts <= 0 else str(self.max_posts)
                bt.logging.info(f"[MyMiner] Starting scrape cycle (processed: {self.posts_processed}/{max_posts_display})")
                
                # Delay between scrape cycles (skip delay on first cycle for immediate action)
                if not first_cycle:
                    bt.logging.info(f"[MyMiner] Waiting {scrape_interval} seconds before scraping...")
                    time.sleep(scrape_interval)
                else:
                    first_cycle = False
                    bt.logging.info("[MyMiner] First cycle - scraping immediately (no delay)")

                bt.logging.info(f"[MyMiner] Scraping {posts_per_cycle} post(s)...")
                posts = self.scraper.scrape_posts(count=posts_per_cycle) or []
                bt.logging.info(f"[MyMiner] Scraped {len(posts)} post(s)")
                
                # Filter out posts we've already submitted
                new_posts = []
                for post in posts:
                    pid = str(post.get("id", "unknown"))
                    if pid not in self._seen_post_ids:
                        new_posts.append(post)
                    else:
                        bt.logging.debug(f"[MyMiner] Post {pid} already submitted, skipping")
                
                bt.logging.info(f"[MyMiner] {len(new_posts)} new post(s) to process (after filtering duplicates)")
                
                submitted_this_cycle = 0
                if len(new_posts) == 0:
                    bt.logging.info("[MyMiner] No new posts to process, skipping to next cycle")
                    continue
                
                for post in new_posts:
                    # Limit submissions per cycle (check this first before processing)
                    if submitted_this_cycle >= posts_to_submit:
                        bt.logging.info(f"[MyMiner] Reached POSTS_TO_SUBMIT limit ({posts_to_submit}) for this cycle, stopping submissions")
                        break
                    
                    # Check max_posts limit (0 or negative means unlimited)
                    if self.max_posts > 0 and self.posts_processed >= self.max_posts:
                        bt.logging.info(f"[MyMiner] Reached max_posts limit ({self.max_posts}), stopping")
                        break

                    pid = str(post.get("id", "unknown"))
                    bt.logging.debug(f"[MyMiner] Processing post ID: {pid}")

                    # Normalize content to match what validator will compare against
                    # This ensures miners analyze and submit the exact text that validators will check
                    raw_content = post.get("content", "")
                    content = norm_text(raw_content)
                    if not content:
                        bt.logging.warning(f"[MyMiner] Post {pid} content is empty after normalization, skipping")
                        continue
                    
                    # Log if normalization changed the content (for debugging)
                    if raw_content != content:
                        bt.logging.debug(f"[MyMiner] Content normalized for {pid} (length: {len(raw_content)} -> {len(content)} chars)")

                    # Analyze post content for subnet relevance and sentiment
                    # Returns a dict with subnet_relevance (per-subnet scores) and sentiment
                    bt.logging.info(f"[MyMiner] Analyzing post {pid} (content length: {len(content)} chars)")
                    analysis = self.analyzer.analyze_post_complete(content)
                    bt.logging.debug(f"[MyMiner] Analysis complete for {pid}: {analysis}")
                    
                    # Extract subnet relevance scores (0.0 to 1.0) for each subnet
                    # Each subnet gets a relevance score indicating how relevant the post is to that subnet
                    tokens = {}
                    for subnet_name, relevance_data in analysis.get("subnet_relevance", {}).items():
                        relevance_score = relevance_data.get("relevance", 0.0)
                        tokens[subnet_name] = float(relevance_score)
                    bt.logging.info(f"[MyMiner] Extracted tokens for {pid}: {list(tokens.keys())} (scores: {tokens})")
                    
                    # Extract sentiment score: -1.0 (negative) to 1.0 (positive)
                    # This matches the format expected by validators for scoring
                    sentiment = float(analysis.get("sentiment", 0.0))
                    bt.logging.info(f"[MyMiner] Extracted sentiment for {pid}: {sentiment:.3f}")

                    # Calculate post score using score_post_entry (same scoring logic as validator)
                    # Convert timestamp to ISO format for scoring function
                    post_date = post.get("timestamp", 0)
                    if isinstance(post_date, int):
                        dt = datetime.fromtimestamp(post_date, tz=timezone.utc)
                        post_date_iso = dt.isoformat()
                    else:
                        post_date_iso = datetime.now(timezone.utc).isoformat()
                    
                    post_entry = {
                        "url": f"post_{pid}",
                        "post_info": {
                            "post_text": content,
                            "post_date": post_date_iso,
                            "like_count": int(post.get("likes", 0) or 0),
                            "retweet_count": int(post.get("retweets", 0) or 0),
                            "quote_count": 0,  # Not provided in post data
                            "reply_count": int(post.get("responses", 0) or 0),
                            "author_followers": int(post.get("followers", 0) or 0),
                            "account_age_days": int(post.get("account_age", 0)),
                        }
                    }
                    
                    try:
                        # Pass the already-computed analysis to avoid re-analyzing all subnets
                        scored_result = score_post_entry(post_entry, self.analyzer, k=5, analysis_result=analysis)
                        post_score = scored_result.get("score", 0.0)
                        bt.logging.info(f"[MyMiner] Calculated score for {pid}: {post_score:.3f}")
                    except Exception as e:
                        bt.logging.warning(f"[MyMiner] Error calculating score for {pid}: {e}, using 0.0")
                        post_score = 0.0

                    post_data = {
                        "miner_hotkey": self.miner_hotkey,
                        "post_id": pid,
                        "content": content,  # Use normalized content (already normalized above)
                        "date": int(post.get("timestamp", 0)),
                        "author": str(post.get("author", "unknown")),
                        "account_age": int(post.get("account_age", 0)),
                        "retweets": int(post.get("retweets", 0) or 0),
                        "likes": int(post.get("likes", 0) or 0),
                        "responses": int(post.get("responses", 0) or 0),
                        "followers": int(post.get("followers", 0) or 0),
                        "tokens": tokens,
                        "sentiment": sentiment,
                        "score": post_score,  # Miner-provided score
                    }
                    bt.logging.info(f"[MyMiner] Prepared post_data for {pid}: hotkey={post_data['miner_hotkey']}, author={post_data['author']}, date={post_data['date']}, score={post_score:.3f}")

                    # Submit post to API server
                    # On success, track the post ID and increment processed count
                    bt.logging.info(f"[MyMiner] Submitting post {pid} to API...")
                    success = self.api_client.submit_post(post_data)
                    if success:
                        self._seen_post_ids.add(pid)
                        self.posts_processed += 1
                        submitted_this_cycle += 1
                        bt.logging.info(
                            f"[MyMiner] ✓ Submitted post '{pid}' successfully "
                            f"({self.posts_processed}/{self.max_posts}, cycle: {submitted_this_cycle}/{posts_to_submit})"
                        )
                    else:
                        bt.logging.warning(f"[MyMiner] ✗ Failed to submit post '{pid}'")

            except Exception as e:
                bt.logging.error(f"[MyMiner] Error in loop: {e}")
                time.sleep(2)

        bt.logging.info(f"[MyMiner] Background thread stopped. Processed {self.posts_processed} posts")

    def get_stats(self) -> Dict:
        """
        Returns current statistics about the miner's operation.
        
        Returns:
            Dict containing:
                - posts_processed: Number of posts successfully submitted
                - max_posts: Maximum posts to process before stopping
                - running: Whether the miner is currently running
                - thread_alive: Whether the background thread is alive
        """
        with self.lock:
            return {
                "posts_processed": self.posts_processed,
                "max_posts": self.max_posts,
                "running": self.running,
                "thread_alive": self.thread.is_alive() if self.thread else False,
            }
