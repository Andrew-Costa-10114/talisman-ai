"""
X Post Scoring and Validator Batch Verification

Provides functions to:
1. Score post components (value, recency)
2. Validate miner batches via sampling and exact canonical string matching

Validator Flow:
- Miner submits batch of N posts with classifications
- Validator samples M posts (e.g., 10-20 from 100)
- Validator runs classification on sampled posts
- Validator compares canonical strings for exact match
- If all match → accept batch, else → reject batch
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple
import random
import bittensor as bt

from relevance import SubnetRelevanceAnalyzer, PostClassification


# ===== Normalization Caps =====
CAPS = {
    "likes": 5_000,
    "retweets": 1_000,
    "quotes": 300,
    "replies": 600,
    "followers": 200_000,
    "account_age_days": 7 * 365,
}


def _clamp01(x: float) -> float:
    """Clamp a float value to the range [0.0, 1.0]"""
    return max(0.0, min(1.0, float(x)))


def _norm(value: float, cap: float) -> float:
    """
    Normalize a value to [0.0, 1.0] using linear scaling with a hard cap
    
    Args:
        value: The raw value to normalize
        cap: The cap threshold - values at or above this threshold yield 1.0
        
    Returns:
        Normalized value in [0.0, 1.0]
    """
    return _clamp01(value / cap)


def recency_score(post_date_iso: str, horizon_hours: float = 24.0) -> float:
    """
    Compute recency score based on post age using linear time decay
    
    Args:
        post_date_iso: ISO format date string (e.g., "2024-01-01T12:00:00Z")
        horizon_hours: Time window in hours (default: 24.0)
        
    Returns:
        Recency score in [0.0, 1.0]
    """
    dt = datetime.fromisoformat(post_date_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    return _clamp01(1.0 - age_hours / horizon_hours)


def value_score(
    like_count: int,
    retweet_count: int,
    quote_count: int,
    reply_count: int,
    author_followers: int,
    account_age_days: int,
    caps: Dict = CAPS,
) -> float:
    """
    Compute value score based on engagement metrics and author credibility
    
    The value score is an equal-weight average of six normalized components:
    1-4. Engagement signals (likes, retweets, quotes, replies)
    5-6. Author credibility (follower count, account age)
    
    Args:
        like_count: Number of likes on the tweet
        retweet_count: Number of retweets
        quote_count: Number of quote tweets
        reply_count: Number of replies
        author_followers: Number of followers the author has
        account_age_days: Age of the author's account in days
        caps: Dictionary of cap values for normalization (defaults to CAPS)
        
    Returns:
        Value score in [0.0, 1.0]
    """
    comps = [
        _norm(like_count or 0, caps["likes"]),
        _norm(retweet_count or 0, caps["retweets"]),
        _norm(quote_count or 0, caps["quotes"]),
        _norm(reply_count or 0, caps["replies"]),
        _norm(author_followers or 0, caps["followers"]),
        _norm(account_age_days or 0, caps["account_age_days"]),
    ]
    return sum(comps) / len(comps)


# ===== Validator Batch Verification =====

def validate_miner_batch(
    miner_batch: List[Dict],
    analyzer: SubnetRelevanceAnalyzer,
    sample_size: int = 10,
    seed: int = None
) -> Tuple[bool, Dict]:
    """
    Validate a miner's batch by sampling posts and checking for exact classification matches
    
    Validator Logic:
    1. Sample N posts from miner's batch
    2. Run LLM classification on each sampled post
    3. Compare miner's canonical string vs validator's canonical string
    4. If all match exactly → accept batch
    5. If any deviate → reject batch
    
    Args:
        miner_batch: List of post dicts with keys:
            - post_text: The post content
            - miner_classification: Dict with miner's claimed classification
                - Must contain all fields to build canonical string
        analyzer: SubnetRelevanceAnalyzer instance
        sample_size: Number of posts to sample for validation (default: 10)
        seed: Random seed for reproducible sampling (optional)
        
    Returns:
        Tuple of (is_valid, result_dict):
            - is_valid: True if all sampled posts match exactly, False otherwise
            - result_dict: Contains 'matches', 'total_sampled', 'discrepancies'
    """
    
    # Sample posts
    if seed is not None:
        random.seed(seed)
    
    sample_size = min(sample_size, len(miner_batch))
    sampled_posts = random.sample(miner_batch, sample_size)
    
    bt.logging.info(f"[Validator] Sampling {sample_size} posts from batch of {len(miner_batch)}")
    
    matches = 0
    discrepancies = []
    
    for i, post_data in enumerate(sampled_posts):
        post_text = post_data.get("post_text", "")
        miner_classification = post_data.get("miner_classification", {})
        
        # Validator runs classification
        validator_result = analyzer.classify_post(post_text)
        
        if validator_result is None:
            bt.logging.warning(f"[Validator] Failed to classify sampled post {i+1}")
            discrepancies.append({
                "post_index": i,
                "reason": "validator_classification_failed",
                "post_preview": post_text[:100]
            })
            continue
        
        # Build miner's canonical string from their claimed classification
        try:
            miner_canonical = _build_canonical_from_dict(miner_classification)
        except Exception as e:
            bt.logging.warning(f"[Validator] Invalid miner classification format: {e}")
            discrepancies.append({
                "post_index": i,
                "reason": "invalid_miner_format",
                "error": str(e)
            })
            continue
        
        # Get validator's canonical string
        validator_canonical = validator_result.to_canonical_string()
        
        # Exact match check
        if miner_canonical == validator_canonical:
            matches += 1
            bt.logging.debug(f"[Validator] Post {i+1}: MATCH")
        else:
            bt.logging.warning(f"[Validator] Post {i+1}: MISMATCH")
            bt.logging.debug(f"  Miner:     {miner_canonical}")
            bt.logging.debug(f"  Validator: {validator_canonical}")
            discrepancies.append({
                "post_index": i,
                "reason": "canonical_mismatch",
                "miner_canonical": miner_canonical,
                "validator_canonical": validator_canonical,
                "post_preview": post_text[:100]
            })
    
    is_valid = (matches == sample_size) and (len(discrepancies) == 0)
    
    result = {
        "is_valid": is_valid,
        "matches": matches,
        "total_sampled": sample_size,
        "discrepancies": discrepancies,
        "match_rate": matches / sample_size if sample_size > 0 else 0.0
    }
    
    if is_valid:
        bt.logging.success(f"[Validator] Batch ACCEPTED: {matches}/{sample_size} matches")
    else:
        bt.logging.warning(f"[Validator] Batch REJECTED: {matches}/{sample_size} matches, {len(discrepancies)} discrepancies")
    
    return is_valid, result


def _build_canonical_from_dict(classification: Dict) -> str:
    """
    Build canonical string from miner's classification dictionary
    
    This must match the exact format from PostClassification.to_canonical_string()
    
    Args:
        classification: Dict with keys matching PostClassification fields
        
    Returns:
        Canonical string for exact matching
    """
    # Extract fields
    subnet_id = int(classification["subnet_id"])
    content_type = classification["content_type"]
    sentiment = classification["sentiment"]
    technical_quality = classification["technical_quality"]
    market_analysis = classification["market_analysis"]
    impact_potential = classification["impact_potential"]
    relevance_confidence = classification["relevance_confidence"]
    evidence_spans = classification.get("evidence_spans", [])
    anchors_detected = classification.get("anchors_detected", [])
    
    # Sort evidence for determinism (same as PostClassification)
    sorted_evidence = "|".join(sorted([s.lower() for s in evidence_spans]))
    sorted_anchors = "|".join(sorted([s.lower() for s in anchors_detected]))
    
    return f"{subnet_id}|{content_type}|{sentiment}|{technical_quality}|{market_analysis}|{impact_potential}|{relevance_confidence}|{sorted_evidence}|{sorted_anchors}"


# ===== Scoring Weights (for future use) =====
# These can be used if you want to compute final post scores based on classification + engagement
RELEVANCE_WEIGHT = 0.50  # 50% weight on subnet relevance (binary: subnet_id != 0)
VALUE_WEIGHT = 0.40      # 40% weight on signal value/quality
RECENCY_WEIGHT = 0.10    # 10% weight on recency


def compute_post_score(
    classification: PostClassification,
    post_info: Dict,
    weights: Dict = None
) -> float:
    """
    Compute final post score combining classification + engagement + recency
    
    Args:
        classification: PostClassification result
        post_info: Dict with engagement metrics and post_date
        weights: Optional custom weights dict
        
    Returns:
        Final score in [0.0, 1.0]
    """
    if weights is None:
        weights = {
            "relevance": RELEVANCE_WEIGHT,
            "value": VALUE_WEIGHT,
            "recency": RECENCY_WEIGHT
        }
    
    # Relevance: binary (1.0 if subnet_id != 0, else 0.0)
    relevance = 1.0 if classification.subnet_id != 0 else 0.0
    
    # Value: engagement + author credibility
    val = value_score(
        like_count=post_info.get("like_count", 0) or 0,
        retweet_count=post_info.get("retweet_count", 0) or 0,
        quote_count=post_info.get("quote_count", 0) or 0,
        reply_count=post_info.get("reply_count", 0) or 0,
        author_followers=post_info.get("author_followers", 0) or 0,
        account_age_days=post_info.get("account_age_days", 0) or 0,
    )
    
    # Recency
    rec = recency_score(post_info.get("post_date", datetime.now(timezone.utc).isoformat()))
    
    # Combine
    final = weights["relevance"] * relevance + weights["value"] * val + weights["recency"] * rec
    
    return _clamp01(final)

