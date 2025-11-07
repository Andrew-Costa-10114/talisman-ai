"""
Scoring module for the BitTensor subnet incentive mechanism.

This module provides functions to score tweets based on three components:
1. Recency: How recent the tweet is (newer tweets score higher)
2. Relevance: How relevant the tweet is to BitTensor subnets
3. Value: Signal quality based on engagement metrics and author credibility

The primary scoring function is `score_tweet_entry()`, which uses an LLM-based
analyzer to determine subnet relevance. The legacy `compile_tweet_score()` function
uses simple keyword matching and is kept for backward compatibility.
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple
import re

# Normalization caps for engagement metrics and author attributes.
# Values above these caps are clamped to 1.0 in the normalized score.
# These thresholds are calibrated for BitTensor-scale accounts.
CAPS = {
    "likes": 5_000,           # Likes cap: accounts with 5k+ likes get max score
    "retweets": 1_000,        # Retweets cap: accounts with 1k+ retweets get max score
    "quotes": 300,            # Quotes cap: accounts with 300+ quotes get max score
    "replies": 600,           # Replies cap: accounts with 600+ replies get max score
    "followers": 200_000,     # Followers cap: accounts with 200k+ followers get max score
    "account_age_days": 7 * 365,  # Account age cap: accounts older than 7 years get max score
}

# Regex pattern for splitting keywords (used by legacy keyword-based relevance)
KEYWORD_SPLIT_RE = re.compile(r"[^\w#+]+", re.UNICODE)


def _clamp01(x: float) -> float:
    """Clamp a float value to the range [0.0, 1.0]."""
    return max(0.0, min(1.0, float(x)))


def _norm(value: float, cap: float) -> float:
    """
    Normalize a value to [0.0, 1.0] using linear scaling with a hard cap.
    
    Args:
        value: The raw value to normalize
        cap: The cap threshold - values at or above this threshold yield 1.0
        
    Returns:
        Normalized value in [0.0, 1.0], where value/cap is clamped to 1.0
    """
    return _clamp01(value / cap)

def recency_score(tweet_date_iso: str, horizon_hours: float = 24.0) -> float:
    """
    Compute recency score based on tweet age using linear time decay.
    
    The score decreases linearly from 1.0 (for tweets posted now) to 0.0
    (for tweets older than horizon_hours). Tweets older than the horizon
    receive a score of 0.0.
    
    Args:
        tweet_date_iso: ISO format date string (e.g., "2024-01-01T12:00:00Z")
        horizon_hours: Time window in hours (default: 24.0). Tweets older than
                       this receive a score of 0.0.
                       
    Returns:
        Recency score in [0.0, 1.0], where 1.0 = just posted, 0.0 = older than horizon
    """
    dt = datetime.fromisoformat(tweet_date_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    return _clamp01(1.0 - age_hours / horizon_hours)

def relevance_score(tweet_text: str, keywords: List[str]) -> float:
    """
    Legacy keyword-based relevance scoring (case-insensitive OR search).
    
    Returns 1.0 if any keyword appears in the tweet text, otherwise 0.0.
    This is a simple binary relevance check used by the legacy scoring function.
    
    Note: The primary scoring function uses `score_tweet_entry()` which employs
    an LLM-based analyzer for more sophisticated relevance detection.
    
    Args:
        tweet_text: The tweet text to search
        keywords: List of keywords to search for (e.g., ['TAO', 'BitTensor', 'subnet'])
        
    Returns:
        1.0 if any keyword is found, 0.0 otherwise
    """
    text = tweet_text.lower()
    kws = [k.lower() for k in keywords]
    return 1.0 if any(k in text for k in kws) else 0.0

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
    Compute value score based on engagement metrics and author credibility.
    
    The value score is an equal-weight average of six normalized components:
    1. Like count (engagement signal)
    2. Retweet count (engagement signal)
    3. Quote count (engagement signal)
    4. Reply count (engagement signal)
    5. Author follower count (author credibility)
    6. Account age in days (author credibility)
    
    Each component is normalized to [0.0, 1.0] using the caps defined in CAPS,
    then all components are averaged together.
    
    Args:
        like_count: Number of likes on the tweet
        retweet_count: Number of retweets
        quote_count: Number of quote tweets
        reply_count: Number of replies
        author_followers: Number of followers the author has
        account_age_days: Age of the author's account in days
        caps: Dictionary of cap values for normalization (defaults to CAPS)
        
    Returns:
        Value score in [0.0, 1.0], representing overall signal quality
    """
    comps = [
        _norm(like_count, caps["likes"]),
        _norm(retweet_count, caps["retweets"]),
        _norm(quote_count, caps["quotes"]),
        _norm(reply_count, caps["replies"]),
        _norm(author_followers, caps["followers"]),
        _norm(account_age_days, caps["account_age_days"]),
    ]
    return sum(comps) / len(comps)

def compile_tweet_score(tweet_info: Dict, keywords: List[str], weights: Dict = None) -> Dict:
    """
    Legacy scoring function using keyword-based relevance matching.
    
    This function combines recency, relevance (keyword-based), and value scores
    into a final score. It uses simple keyword matching for relevance detection.
    
    Note: For production use, prefer `score_tweet_entry()` which uses an LLM-based
    analyzer for more accurate relevance detection.
    
    Args:
        tweet_info: Dictionary containing tweet metadata with keys:
            - tweet_text: The tweet text content
            - tweet_date: ISO format date string
            - like_count: Number of likes (optional, defaults to 0)
            - retweet_count: Number of retweets (optional, defaults to 0)
            - quote_count: Number of quote tweets (optional, defaults to 0)
            - reply_count: Number of replies (optional, defaults to 0)
            - author_followers: Author's follower count (optional, defaults to 0)
            - account_age_days: Author's account age in days (optional, defaults to 0)
        keywords: List of keywords to search for in tweet text
        weights: Dictionary with keys "recency", "relevance", "value" specifying
                 component weights. If None, uses equal weights (1/3 each).
                 
    Returns:
        Dictionary containing:
            - recency: Recency score component [0.0, 1.0]
            - relevance: Relevance score component [0.0, 1.0]
            - value: Value score component [0.0, 1.0]
            - final_score: Weighted combination of all components [0.0, 1.0]
    """
    if weights is None:
        weights = {"recency": 1/3, "relevance": 1/3, "value": 1/3}

    r_recent = recency_score(tweet_info["tweet_date"])
    r_rel = relevance_score(tweet_info["tweet_text"], keywords)
    r_val = value_score(
        like_count=tweet_info.get("like_count", 0) or 0,
        retweet_count=tweet_info.get("retweet_count", 0) or 0,
        quote_count=tweet_info.get("quote_count", 0) or 0,
        reply_count=tweet_info.get("reply_count", 0) or 0,
        author_followers=tweet_info.get("author_followers", 0) or 0,
        account_age_days=tweet_info.get("account_age_days", 0) or 0,
    )

    final = (
        weights["recency"] * r_recent
        + weights["relevance"] * r_rel
        + weights["value"] * r_val
    )

    return {
        "recency": r_recent,
        "relevance": r_rel,
        "value": r_val,
        "final_score": _clamp01(final),
    }


# Default weights for production scoring (used by score_tweet_entry)
# These weights prioritize relevance over value, with recency as a minor factor
RELEVANCE_WEIGHT = 0.50  # 50% weight on subnet relevance
VALUE_WEIGHT = 0.40      # 40% weight on signal value/quality
RECENCY_WEIGHT = 0.10    # 10% weight on recency


def top_k_relevance_from_analyzer(text: str, analyzer, k: int = 5, analysis_result: Dict = None) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Compute subnet relevance scores using the analyzer and return top-k results.
    
    This function uses the SubnetRelevanceAnalyzer to determine how relevant a tweet
    is to different BitTensor subnets, then returns the mean relevance score of the
    top-k most relevant subnets.
    
    Args:
        text: The tweet text to analyze (only used if analysis_result is None)
        analyzer: SubnetRelevanceAnalyzer instance configured with subnet registry
        k: Number of top subnets to consider when computing the mean (default: 5)
        analysis_result: Optional pre-computed analysis result dict. If provided,
                        this will be used instead of calling analyze_tweet_complete again.
        
    Returns:
        Tuple of:
            - mean_top: Mean relevance score of top-k subnets [0.0, 1.0]
            - top: List of (subnet_name, relevance_score) tuples for top-k subnets,
                   sorted by relevance (highest first)
    """
    if analysis_result is None:
        out = analyzer.analyze_tweet_complete(text)
    else:
        out = analysis_result
    items = [(name, data.get("relevance", 0.0)) for name, data in out.get("subnet_relevance", {}).items()]
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[:k]
    mean_top = float(sum(s for _, s in top) / len(top)) if top else 0.0
    return mean_top, top


def score_tweet_entry(entry: Dict, analyzer, k: int = 5, analysis_result: Dict = None) -> Dict:
    """
    Score a single tweet entry using LLM-based subnet relevance analysis.
    
    This is the primary scoring function for production use. It computes three
    component scores (relevance, value, recency) and combines them into a final
    score using the default weights defined above.
    
    The relevance component uses an LLM-based analyzer to determine how relevant
    the tweet is to BitTensor subnets, providing more accurate results than
    simple keyword matching.
    
    Args:
        entry: Dictionary containing tweet entry with keys:
            - url: Tweet URL or identifier
            - tweet_info: Dictionary with tweet metadata containing:
                - tweet_text: The tweet text content
                - tweet_date: ISO format date string
                - like_count: Number of likes (optional, defaults to 0)
                - retweet_count: Number of retweets (optional, defaults to 0)
                - quote_count: Number of quote tweets (optional, defaults to 0)
                - reply_count: Number of replies (optional, defaults to 0)
                - author_followers: Author's follower count (optional, defaults to 0)
                - account_age_days: Author's account age in days (optional, defaults to 0)
        analyzer: SubnetRelevanceAnalyzer instance configured with subnet registry
        k: Number of top subnets to consider when computing relevance (default: 5)
        analysis_result: Optional pre-computed analysis result dict. If provided,
                        this will be used instead of calling analyze_tweet_complete again.
        
    Returns:
        Dictionary containing:
            - url: Original tweet URL/identifier
            - top_subnets: List of (subnet_name, relevance_score) tuples for top-k subnets
            - relevance: Mean relevance score of top-k subnets [0.0, 1.0]
            - value: Value score based on engagement and author credibility [0.0, 1.0]
            - recency: Recency score based on tweet age [0.0, 1.0]
            - score: Final weighted score combining all components [0.0, 1.0]
    """
    info = entry["tweet_info"]

    # Compute component scores
    rel_mean, rel_top = top_k_relevance_from_analyzer(info["tweet_text"], analyzer, k=k, analysis_result=analysis_result)
    val = value_score(
        like_count=info.get("like_count", 0) or 0,
        retweet_count=info.get("retweet_count", 0) or 0,
        quote_count=info.get("quote_count", 0) or 0,
        reply_count=info.get("reply_count", 0) or 0,
        author_followers=info.get("author_followers", 0) or 0,
        account_age_days=info.get("account_age_days", 0) or 0,
    )
    rec = recency_score(info["tweet_date"])

    # Combine components using default weights
    final = RELEVANCE_WEIGHT * rel_mean + VALUE_WEIGHT * val + RECENCY_WEIGHT * rec

    return {
        "url": entry.get("url", ""),
        "top_subnets": rel_top,
        "relevance": rel_mean,
        "value": val,
        "recency": rec,
        "score": max(0.0, min(1.0, float(final)))
    }

