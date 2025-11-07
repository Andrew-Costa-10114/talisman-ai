# Validator Module

The validator is responsible for independently verifying miner submissions and enforcing quality standards. Unlike traditional Bittensor validators that query miners directly, this validator polls a coordination API for batches of miner posts, validates them against ground truth (X/Twitter API), and submits votes back to the API for consensus tracking.

## Overview

The validator operates in two phases:

1. **Batch Processing**: Continuously polls the API for batches of miner posts, grades each miner's submissions using a three-stage validation system, and submits votes back to the API
2. **Score Management**: Updates miner scores based on validation results, which are used by the base validator to set weights on-chain

The validator runs asynchronously in the background, processing batches independently of the main validator forward loop.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Validator Neuron                           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────── ┐  │
│  │ BatchClient  │──▶ |    Grader    │──▶│  Vote Submit  │  │
│  │              │    │              │    │               │  │
│  │ - Polls API  │    │ - X API      │    │ - Submits     │  │
│  │ - Tracks     │    │   validation │    │   votes       │  │
│  │   batches    │    │ - Analysis   │    │ - Updates     │  │
│  │              │    │   validation │    │   scores      │  │
│  └──────────────┘    │ - Score      │    └────────────── ┘  │
│                      │   validation │                       │
│                      └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Validator (`neurons/validator.py`)

The main validator neuron class that orchestrates batch processing and score management.

**Key Methods:**
- `__init__()`: Initializes the batch client and loads validator state
- `forward()`: Starts the batch polling client on first invocation
- `_on_batch()`: Processes a batch of miner submissions (callback from BatchClient)
- `_submit_hotkey_votes()`: Submits validator votes to the API
- `close()`: Gracefully shuts down the validator

**Processing Flow:**
1. BatchClient polls API and calls `_on_batch()` when a new batch is available
2. For each miner in the batch:
   - Grades their posts using `grade_hotkey()`
   - Calculates reward based on label (VALID/INVALID) and final_score
   - Maps hotkey to UID for score updates
   - Collects votes for API submission
3. Updates validator scores using `update_scores()`
4. Submits votes to API for consensus tracking

**Reward Logic:**
- **VALID miners**: Receive full incentive score (`reward = final_score`)
- **INVALID miners**: Receive 10% penalty (`reward = final_score * 0.1`)

### BatchClient (`batch_client.py`)

Asynchronously polls the coordination API for batches of miner posts.

**Features:**
- Polls `/v1/batch` endpoint at configurable intervals (`BATCH_POLL_SECONDS`)
- Tracks last processed `batch_id` to avoid duplicate processing
- Handles HTTP errors gracefully (logs warnings, continues polling)
- Calls callback function when new batch is detected
- Supports both sync and async callbacks

**Configuration:**
- `api_url`: Base URL for the miner API (default: `MINER_API_URL` env var)
- `poll_seconds`: Seconds between poll attempts (default: `BATCH_POLL_SECONDS` env var or 10s)
- `http_timeout`: HTTP request timeout (default: `BATCH_HTTP_TIMEOUT` env var or 10.0s)

**Batch Format:**
Each batch contains:
- `batch_id`: Unique identifier for the batch
- `batch`: List of miner entries, each containing:
  - `hotkey`: Miner's hotkey identifier
  - `posts`: List of post submissions to validate (API-selected sample)
  - `total_posts`: Total count of posts for this miner

### Grader (`grader.py`)

Core validation logic that performs three-stage validation of miner posts.

**Validation Stages:**

#### Stage 1: X API Validation
Validates posts against live X/Twitter API data:

1. **Post Existence**: Post must exist and be accessible via X API
2. **Text Match**: Content must match exactly after normalization (NFC, whitespace)
3. **Author Match**: Author username must match exactly (lowercase)
4. **Timestamp Match**: Timestamp must match exactly (Unix seconds)
5. **Metric Inflation Check**: Engagement metrics may NOT be overstated beyond tolerance:
   - Likes, retweets, replies, followers: max(1, ceil(10% of live value))
   - Understatement is allowed; overstatement beyond tolerance fails validation

**Tolerance:**
- `POST_METRIC_TOLERANCE = 0.1` (10% relative, with floor of 1)

#### Stage 2: Content Analysis Validation
Validates miner's analysis against validator's independent analysis:

1. **Token Matching**: Subnet relevance scores must match within tolerance
   - Tolerance: `TOKEN_TOLERANCE = 0.05` (absolute)
   - Uses token selection policy: removes noise (< 0.05), keeps all validator tokens, adds miner tokens up to cap (k=128)
   - Compares normalized keys (lowercase, stripped)

2. **Sentiment Matching**: Sentiment score must match within tolerance
   - Tolerance: `SENTIMENT_TOLERANCE = 0.05` (absolute)
   - Sentiment range: -1.0 (very bearish) to +1.0 (very bullish)

**Tolerance:**
- `TOKEN_TOLERANCE = 0.05` (absolute)
- `SENTIMENT_TOLERANCE = 0.05` (absolute)

#### Stage 3: Score Validation
Cross-checks miner's score against validator's computed score:

1. **Score Computation**: Validator computes its own score using the same algorithm as miners
   - Uses `score_tweet_entry()` with same weights (50% relevance, 40% value, 10% recency)
   - Uses live metrics from X API (not miner-provided metrics)

2. **Score Inflation Check**: Miner score may not exceed validator score beyond tolerance
   - Tolerance: `SCORE_TOLERANCE = 0.05` (absolute)
   - Miner score must be ≤ validator_score + 0.05

**Tolerance:**
- `SCORE_TOLERANCE = 0.05` (absolute)

**Validation Order:**
- Posts are validated sequentially
- **Stops at first failure** - if any post fails, the entire batch is marked INVALID
- Only if ALL posts pass does the miner receive VALID

**Final Scoring:**
If all posts pass validation:
1. Calculate average of miner's post scores
2. Apply quantity modifier:
   - 1–5 posts: 1.00×
   - 6–20 posts: 0.95×
   - 21+ posts: 0.90×
3. Final score = `avg_post_score × quantity_modifier`

**Return Format:**
- **INVALID**: `(CONSENSUS_INVALID, { "error": {...}, "final_score": 0.0 })`
  - Error includes: code, message, post_id, post_index, details
- **VALID**: `(CONSENSUS_VALID, { n_posts, avg_post_score, quantity_modifier, final_score, tolerances, analyzer })`

### Forward (`forward.py`)

Minimal forward function that delegates to base validator logic. The actual validation work is done by batch processing in the background.

**Note**: This subnet doesn't use traditional synapse-based querying. Instead, validation happens via batch processing.

### Reward (`reward.py`)

Template reward function (not used in this subnet). Rewards are calculated in `_on_batch()` based on validation results.

## Validation Process

### Complete Validation Flow

```
1. BatchClient polls API
   ↓
2. New batch detected (batch_id changed)
   ↓
3. For each miner in batch:
   ├─> For each post:
   │   ├─> Stage 1: X API validation
   │   │   ├─> Post exists?
   │   │   ├─> Text matches?
   │   │   ├─> Author matches?
   │   │   ├─> Timestamp matches?
   │   │   └─> Metrics not inflated?
   │   │
   │   ├─> Stage 2: Analysis validation
   │   │   ├─> Tokens match within ±0.05?
   │   │   └─> Sentiment matches within ±0.05?
   │   │
   │   └─> Stage 3: Score validation
   │       └─> Score not inflated beyond +0.05?
   │
   └─> If all posts pass:
       ├─> Calculate final_score
       ├─> Map hotkey to UID
       └─> Collect vote
   ↓
4. Update validator scores
   ↓
5. Submit votes to API
```

### Error Handling

The validator handles errors gracefully:

- **X API Errors**: Returns `x_api_error` or `x_api_no_response`
- **Post Not Found**: Returns `post_not_found`
- **Text Mismatch**: Returns `text_mismatch` with preview of differences
- **Author Mismatch**: Returns `author_mismatch` with both values
- **Timestamp Mismatch**: Returns `timestamp_mismatch` with difference
- **Metric Inflation**: Returns specific error code (`metric_inflation_likes`, etc.)
- **Token Mismatch**: Returns `tokens_mismatch` with top 5 mismatches
- **Sentiment Mismatch**: Returns `sentiment_mismatch` with both values
- **Score Inflation**: Returns `score_inflation` with both scores
- **Score Compute Error**: Returns `score_compute_error` if validator can't compute score

All errors include:
- `code`: Error code identifier
- `message`: Human-readable error message
- `post_id`: Post that failed validation
- `post_index`: Index of post in batch (0-based)
- `details`: Additional context (mismatches, values, etc.)

## Vote Submission

Votes are submitted to the API endpoint (`VOTE_ENDPOINT`) with the following format:

```python
{
    "validator_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "batch_id": 1234567890,
    "votes": [
        {
            "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "label": 1,  # 1=VALID, 0=INVALID
            "score": 0.75,  # Final incentive score
            "failure_reason": {  # Only present if INVALID
                "code": "tokens_mismatch",
                "message": "subnet relevance differs beyond tolerance",
                "post_id": "1234567890",
                "post_index": 2,
                "details": {...}
            }
        },
        ...
    ]
}
```

**Headers:**
- `x-hotkey`: Validator's hotkey address

**Response Handling:**
- Success: Logs response data
- Timeout: Logs error, continues processing
- Other errors: Logs error with exception info

## Configuration

The validator respects the following environment variables from `.vali_env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `BATCH_POLL_SECONDS` | Interval between batch polls | `10` |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout for API requests | `30.0` |
| `MINER_API_URL` | Coordination API server URL | `http://127.0.0.1:8000` |
| `VOTE_ENDPOINT` | Endpoint for submitting votes | `http://127.0.0.1:8000/v1/validate_hotkeys` |
| `X_BEARER_TOKEN` | X/Twitter API bearer token | Required |
| `X_API_BASE` | X/Twitter API base URL | `https://api.twitter.com/2` |
| `MODEL` | LLM model identifier | `deepseek-ai/DeepSeek-V3-0324` |
| `API_KEY` | LLM API key | Required |
| `LLM_BASE` | LLM API base URL | `https://llm.chutes.ai/v1` |

## Text Normalization

The validator normalizes post content using `norm_text()` to ensure consistency with miner submissions:

- Unicode normalization (NFC)
- Line ending normalization (`\r\n` → `\n`)
- Whitespace collapse (multiple spaces → single space)
- Trim leading/trailing whitespace

This ensures that minor formatting differences don't cause false mismatches.

## X API Requirements

The validator **requires** access to the X/Twitter API for validation:

- **Bearer Token**: Must be set in `X_BEARER_TOKEN` environment variable
- **Rate Limits**: Handled with deterministic retries (up to 3 attempts)
- **Retry Logic**: Includes deterministic jitter seeded by post_id for consistent behavior

**X API Usage:**
- Fetches tweet data including: text, author, timestamp, public metrics
- Fetches author data including: username, followers, account creation date
- Uses deterministic retries for rate limits and transient errors

## Score Updates

Validator scores are updated using the base validator's `update_scores()` method:

- Maps miner hotkeys to UIDs using metagraph
- Updates scores for all miners in the batch
- Scores are used by base validator to set weights on-chain
- Only updates scores for miners found in metagraph

**Score Range:**
- VALID miners: `[0.0, 1.0]` (based on final_score)
- INVALID miners: `[0.0, 0.1]` (10% of final_score)

## Error Codes Reference

| Code | Description |
|------|-------------|
| `no_posts` | No posts submitted |
| `missing_post_id` | Post ID is required |
| `analyzer_unavailable` | Analyzer not initialized |
| `x_api_unavailable` | X API client unavailable |
| `x_api_error` | X API request failed |
| `x_api_no_response` | X API gave no response after retries |
| `post_not_found` | Post not found or inaccessible |
| `text_mismatch` | Content doesn't match live post text |
| `author_mismatch` | Author doesn't match |
| `timestamp_missing` | Timestamp is missing |
| `timestamp_mismatch` | Timestamp doesn't match exactly |
| `metric_inflation_likes` | Likes overstated beyond tolerance |
| `metric_inflation_retweets` | Retweets overstated beyond tolerance |
| `metric_inflation_replies` | Replies overstated beyond tolerance |
| `metric_inflation_followers` | Followers overstated beyond tolerance |
| `empty_content` | Post content is empty |
| `tokens_mismatch` | Subnet relevance differs beyond tolerance |
| `sentiment_mismatch` | Sentiment differs beyond tolerance |
| `score_compute_error` | Validator couldn't compute score |
| `score_inflation` | Miner score exceeds validator tolerance |

## Integration

The validator is integrated into the main neuron (`neurons/validator.py`):

```python
class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self._batch_client = BatchClient()
        self._batch_task = None
    
    async def forward(self):
        if self._batch_task is None:
            self._batch_task = asyncio.create_task(
                self._batch_client.run(self._on_batch)
            )
        return await forward(self)
```

The batch client runs independently in the background, processing batches as they become available, while the base validator handles weight setting automatically based on epoch timing.

