
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import bittensor as bt
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.utils.normalization import norm_text

# =============================================================================
# Miner-Facing Overview
# =============================================================================
# This file validates a batch of your posts and then scores you.
#
# Philosophy (plain language):
# - Deterministic: everyone is checked the same way, in the same order.
# - Simple: each post is validated step-by-step; we stop at the first failure.
# - Public-friendly: tolerances and checks are explicit and documented below.
# - Ground truth: live X (Twitter) API + validator's analyzer. The X API is REQUIRED.
#
# What you must submit per post:
#   post_id, content, author, date (unix seconds), likes, retweets, replies,
#   followers, tokens (dict[str->float]), sentiment (float), score (float).
#
# Validation order (for each post, stop on first error):
#   1) X API check (post exists and data matches)
#      - content text matches exactly after normalization
#      - author username matches exactly (lowercase)
#      - timestamp must match exactly (Unix seconds)
#      - likes/retweets/replies/followers are NOT overstated beyond tolerance
#        (you may understate; you may NOT overstate beyond 10% or +1, whichever is larger)
#   2) Content analysis check (validator analyzes your text)
#      - tokens must match validator's tokens within ±0.05 absolute
#      - sentiment must match within ±0.05 absolute
#   3) Score check
#      - validator computes its own score using the same algorithm
#      - your score may be <= validator_score + 0.05; larger -> error
#
# Scoring:
#   - The validator validates sampled posts to determine VALID/INVALID
#   - If VALID, the validator uses avg_score_all_posts (API-calculated average of ALL posts)
#   - This grader only validates posts; it does not calculate final scores
#
# Return shape:
#   - If a post fails: (CONSENSUS_INVALID, { "error": {...}, "final_score": 0.0 })
#     Error includes code, message, failing post_index and post_id, plus details.
#   - If all pass: (CONSENSUS_VALID, { n_posts, tolerances, analyzer })
# Keep reading comments inline to see exactly how each check works.
# =============================================================================

# === Constants (publicly documented tolerances) ===
CONSENSUS_VALID, CONSENSUS_INVALID = 1, 0
TOKEN_TOLERANCE = 0.05           # tokens in [0,1] must be within ±0.05 of validator
SENTIMENT_TOLERANCE = 0.05       # sentiment in [-1,1] must be within ±0.05 of validator

# === Error helper: standardizes INVALID responses ===
def _err(code: str, message: str, post_id=None, details=None, post_index: Optional[int] = None):
    e = {"code": code, "message": message, "post_id": post_id, "details": details or {}}
    if post_index is not None:
        e["post_index"] = post_index
    return CONSENSUS_INVALID, {"error": e, "final_score": 0.0}

# === Analyzer creation ===
def make_analyzer():
    """Create the validator's analyzer. If this fails, grading cannot proceed."""
    try:
        return setup_analyzer()
    except Exception as e:
        bt.logging.error(f"[GRADER] Analyzer init failed: {e}")
        return None

# === Utilities ===
def analyze_text(text: str, analyzer) -> Tuple[Dict[str, float], float]:
    """Run validator analysis on your text -> (tokens, sentiment)."""
    if analyzer is None:
        return {}, 0.0
    try:
        out = analyzer.analyze_post_complete(text)
    except Exception as e:
        bt.logging.error(f"[GRADER] Analyzer error: {e}")
        return {}, 0.0
    tokens_raw = (out.get("subnet_relevance") or {})
    # Keys are normalized here so later comparisons are deterministic.
    tokens = {str(k).strip().lower(): float(v.get("relevance", 0.0)) for k, v in tokens_raw.items()}
    sentiment = float(out.get("sentiment", 0.0))
    return tokens, sentiment

def normalize_keys(d: Dict) -> Dict[str, float]:
    """Normalize dict keys: string, stripped, lowercase; values -> float."""
    return {str(k).strip().lower(): float(v) for k, v in (d or {}).items()}

def select_tokens(miner_raw: Dict, ref_raw: Dict, k: int = 128, eps: float = 0.05) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Token selection policy used for comparison:
      - Remove tiny values (< eps) on both sides (treated as noise).
      - Always keep ALL validator tokens.
      - Add additional miner tokens (largest magnitude first) until we reach cap k.
    This prevents you from dropping validator-relevant tokens via truncation.
    """
    mt = {k: v for k, v in normalize_keys(miner_raw).items() if abs(v) >= eps}
    rt = {k: v for k, v in normalize_keys(ref_raw).items() if abs(v) >= eps}
    keep = set(rt.keys())
    extras = sorted((set(mt) - keep), key=lambda x: (-abs(mt[x]), x))
    for x in extras:
        if len(keep) >= k:
            break
        keep.add(x)
    return {k: mt.get(k, 0.0) for k in keep}, rt

def tokens_match_within(miner: Dict[str, float], ref: Dict[str, float], abs_tol: float, eps: float = 0.05) -> Tuple[bool, Dict]:
    """
    Compare tokens with absolute tolerance. Pairs below eps on BOTH sides are ignored.
    Returns (ok, diffs). diffs shows where you exceed tolerance.
    """
    diffs = {}
    for k in (set(miner) | set(ref)):
        a, b = float(miner.get(k, 0.0)), float(ref.get(k, 0.0))
        if a < eps and b < eps:  # ignore noise
            continue
        if abs(a - b) > abs_tol:
            diffs[k] = {"miner": a, "validator": b, "allowed": abs_tol, "diff": abs(a - b)}
    return (len(diffs) == 0, diffs)

# === Batch-level grading entry point ===
def grade_hotkey(posts: List[Dict], analyzer=None, x_client=None) -> Tuple[int, Dict]:
    """
    Grade a list of posts using LLM validation only (tokens and sentiment).
    X API validation is now done on the API side before posts are sent to validators.
    
    Stops on first failure; otherwise returns VALID.
    """
    # Basic sanity
    if not posts:
        return _err("no_posts", "no posts submitted")
    try:
        analyzer = analyzer or make_analyzer()
        if analyzer is None:
            return _err("analyzer_unavailable", "Analyzer not initialized")
    except Exception as e:
        return _err("analyzer_unavailable", str(e))

    for i, post in enumerate(posts):
        post_id = post.get("post_id")
        if not post_id:
            return _err("missing_post_id", "post_id is required", None, {}, i)

        # --- LLM validation: content analysis (tokens and sentiment) ---
        # Note: X API validation is now done on the API side, so we only validate
        # the LLM analysis results (tokens/relevance and sentiment)
        content = post.get("content") or ""
        if not content:
            return _err("empty_content", "post content is empty", post_id, {}, i)

        miner_tokens_raw = post.get("tokens") or {}
        miner_sent = float(post.get("sentiment") or 0.0)
        
        # Get analysis result from LLM
        try:
            analysis_result = analyzer.analyze_post_complete(content)
        except Exception as e:
            bt.logging.error(f"[GRADER] Analyzer error: {e}")
            return _err("analyzer_error", f"Analyzer failed: {e}", post_id, {}, i)
        
        # Extract tokens and sentiment from analysis result
        ref_tokens_raw = (analysis_result.get("subnet_relevance") or {})
        # Keys are normalized here so later comparisons are deterministic.
        ref_tokens_raw_normalized = {str(k).strip().lower(): float(v.get("relevance", 0.0)) for k, v in ref_tokens_raw.items()}
        ref_sent = float(analysis_result.get("sentiment", 0.0))
        
        # Validate tokens/relevance match
        miner_tokens, ref_tokens = select_tokens(miner_tokens_raw, ref_tokens_raw_normalized, k=128, eps=0.05)
        matches, token_diffs = tokens_match_within(miner_tokens, ref_tokens, TOKEN_TOLERANCE)
        if not matches:
            # We include top mismatches to help you debug
            top = dict(sorted(token_diffs.items(), key=lambda kv: kv[1]["diff"], reverse=True)[:5])
            return _err("tokens_mismatch", "subnet relevance differs beyond tolerance", post_id,
                        {"mismatches": top, "total_mismatches": len(token_diffs)}, i)

        # Validate sentiment match
        if abs(miner_sent - ref_sent) > SENTIMENT_TOLERANCE:
            return _err("sentiment_mismatch", "sentiment differs beyond tolerance", post_id,
                        {"miner": miner_sent, "validator": ref_sent, "allowed": SENTIMENT_TOLERANCE, "diff": abs(miner_sent - ref_sent)}, i)

        # If we get here, this post passed all LLM validation checks
        # Note: Score validation is removed since it requires X API metrics,
        # which are now validated on the API side

    # All posts passed LLM validation
    # Note: We don't calculate final_score here because the validator uses
    # avg_score_all_posts (API-calculated average of ALL posts) instead.
    n = len(posts)

    analyzer_version = "unknown"
    if analyzer and hasattr(analyzer, "model"):
        analyzer_version = str(analyzer.model)

    return CONSENSUS_VALID, {
        "n_posts": n,
        "tolerances": {
            "token": TOKEN_TOLERANCE,
            "sentiment": SENTIMENT_TOLERANCE,
        },
        "analyzer": {"version": analyzer_version},
    }
