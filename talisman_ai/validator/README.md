# Validator Module

The validator is responsible for independently verifying miner submissions and enforcing quality standards. Unlike traditional Bittensor validators that query miners directly, this validator polls a coordination API for validation payloads, validates them using LLM analysis, and submits results back to the API.

## Overview

The validator operates in two phases:

1. **Validation Processing**: Continuously polls the API for validation payloads, grades individual posts using LLM analysis, and submits results back to the API
2. **Score Management**: Updates miner scores based on validation results, which are used by the base validator to set weights on-chain

The validator runs asynchronously in the background, processing batches independently of the main validator forward loop.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Validator Neuron                           │
│                                                             │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ ValidationClient │──▶ |    Grader    │──▶│  Submit   │  │
│  │                  │    │              │    │  Results  │  │
│  │ - Polls /v2/    │    │ - LLM        │    │ - Updates │  │
│  │   validation     │    │   analysis   │    │   scores  │  │
│  │ - Fetches scores │    │   (tokens &   │    │           │  │
│  │   from /v2/     │    │   sentiment)  │    │           │  │
│  │   scores         │    │              │    │           │  │
│  └──────────────────┘    └──────────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Validator (`neurons/validator.py`)

The main validator neuron class that orchestrates batch processing and score management.

**Key Methods:**
- `__init__()`: Initializes the validation client and analyzer
- `forward()`: Starts the validation client on first invocation
- `_on_validations()`: Processes validation payloads from API v2
- `_process_single_validation()`: Grades a single post using LLM analysis
- `_submit_pending_results()`: Submits validation results to /v2/validation_result
- `_on_scores()`: Updates hotkey rewards based on scores from /v2/scores

**Processing Flow:**
1. ValidationClient polls /v2/validation and fetches validation payloads
2. For each validation payload:
   - Grades the post using `grade_hotkey()` (LLM analysis only - tokens and sentiment)
   - X API validation is done on the API side before posts are sent to validators
   - Collects results for batch submission
3. Submits all results to /v2/validation_result
4. Fetches scores from /v2/scores every N blocks and updates hotkey rewards

**Reward Logic:**
- **VALID miners**: Receive full incentive score (`reward = final_score`)
- **INVALID miners**: Receive 10% penalty (`reward = final_score * 0.1`)

### ValidationClient (`validation_client.py`)

Client for API v2 validation system that handles fetching validations and scores.

**Features:**
- Polls `/v2/validation` endpoint for validation payloads
- Submits results to `/v2/validation_result` endpoint
- Fetches scores from `/v2/scores` every N blocks (configurable via `SCORES_BLOCK_INTERVAL`)
- Tracks last processed scores window to avoid duplicates
- Handles HTTP errors gracefully (logs warnings, continues polling)
- Supports authentication via Bittensor wallet signatures

**Configuration:**
- `api_url`: Base URL for the miner API (default: `MINER_API_URL` env var)
- `poll_seconds`: Seconds between poll attempts (default: `VALIDATION_POLL_SECONDS` env var or 10s)
- `http_timeout`: HTTP request timeout (default: `BATCH_HTTP_TIMEOUT` env var or 30.0s)
- `scores_block_interval`: Blocks between score fetches (default: `SCORES_BLOCK_INTERVAL` env var or 100)

**Validation Payload Format:**
Each validation payload contains:
- `validation_id`: Unique identifier for the validation
- `miner_hotkey`: Miner's hotkey identifier
- `post`: Post data to validate (content, tokens, sentiment, etc.)
- `selected_at`: Timestamp when post was selected for validation

### Grader (`grader.py`)

Core validation logic that performs three-stage validation of miner posts.

**Validation Stages:**

#### Stage 1: Content Analysis Validation
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

**Note:** X API validation (post existence, text/author/timestamp matching, metric inflation checks) is now performed on the API side before posts are sent to validators. Validators only perform LLM-based analysis validation.

**Validation Order:**
- Posts are validated sequentially
- **Stops at first failure** - if any post fails validation, the result is marked as failed
- Only if the post passes all checks does it receive a success result

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
1. ValidationClient polls API for validation payloads
   ↓
2. Posts received (already passed X API validation on API side)
   ↓
3. For each validation payload:
   ├─> Stage 1: LLM Analysis validation
   │   ├─> Tokens match within ±0.05?
   │   └─> Sentiment matches within ±0.05?
   │
   └─> If post passes:
       └─> Collect result (success=true)
   ↓
4. Submit results to /v2/validation_result
   ↓
5. Fetch scores from /v2/scores and update hotkey rewards
```

**Note:** X API validation (post existence, text/author/timestamp matching, metric inflation checks) is performed on the API side before posts are sent to validators. Validators only perform LLM-based analysis validation (tokens/relevance and sentiment).

### Error Handling

The validator handles errors gracefully:

- **Analyzer Errors**: Returns `analyzer_error` if LLM analysis fails
- **Empty Content**: Returns `empty_content` if post has no content
- **Token Mismatch**: Returns `tokens_mismatch` with top mismatches
- **Sentiment Mismatch**: Returns `sentiment_mismatch` with both values

All errors include:
- `code`: Error code identifier
- `message`: Human-readable error message
- `post_id`: Post that failed validation
- `details`: Additional context (mismatches, values, etc.)

**Note:** X API validation errors (post not found, text/author/timestamp mismatches, metric inflation) are handled on the API side before posts are sent to validators.

## Validation Result Submission

Validation results are submitted to the `/v2/validation_result` endpoint with the following format:

```python
{
    "validator_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "results": [
        {
            "validation_id": "uuid-here",
            "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "success": true,  # true if validation passed, false if failed
            "failure_reason": {  # Only present if success=false
                "code": "tokens_mismatch",
                "message": "subnet relevance differs beyond tolerance",
                "post_id": "1234567890",
                "details": {...}
            }
        },
        ...
    ]
}
```

**Note:** The `validator_hotkey` is included in the payload (not in each result), and each result includes `validation_id`, `miner_hotkey`, `success`, and optionally `failure_reason`.

**Headers:**
- `X-Auth-SS58Address`: Validator's hotkey address (SS58 format)
- `X-Auth-Signature`: Signature of auth message
- `X-Auth-Message`: Auth message (timestamp-based)
- `X-Auth-Timestamp`: Timestamp used in auth message

**Response Handling:**
- Success: Logs response data
- HTTP errors: Logs error, results are kept for retry
- Other errors: Logs error with exception info

## Configuration

The validator respects the following environment variables from `.vali_env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `VALIDATION_POLL_SECONDS` | Interval between validation polls | `10` |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout for API requests | `30.0` |
| `MINER_API_URL` | Coordination API server URL | `https://talisman.rizzo.network/api` |
| `SCORES_BLOCK_INTERVAL` | Blocks between score fetches | `100` |
| `MODEL` | LLM model identifier | `Qwen/Qwen3-32B` |
| `API_KEY` | LLM API key | Required |
| `LLM_BASE` | LLM API base URL | `https://llm.chutes.ai/v1` |

## Text Normalization

The validator normalizes post content using `norm_text()` to ensure consistency with miner submissions:

- Unicode normalization (NFC)
- Line ending normalization (`\r\n` → `\n`)
- Whitespace collapse (multiple spaces → single space)
- Trim leading/trailing whitespace

This ensures that minor formatting differences don't cause false mismatches.

## X API Validation

**Note:** X API validation is now performed on the API side before posts are sent to validators. The validator no longer requires X API access and only performs LLM-based analysis validation (tokens/relevance and sentiment).

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
| `empty_content` | Post content is empty |
| `analyzer_error` | Analyzer failed to process post |
| `tokens_mismatch` | Subnet relevance differs beyond tolerance |
| `sentiment_mismatch` | Sentiment differs beyond tolerance |

## Integration

The validator is integrated into the main neuron (`neurons/validator.py`):

```python
class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self._analyzer = setup_analyzer()
        self._validation_client = ValidationClient(wallet=self.wallet)
        self._validation_task = None
    
    async def forward(self):
        if self._validation_task is None:
            self._validation_task = asyncio.create_task(
                self._validation_client.run(
                    on_validations=self._on_validations,
                    on_scores=self._on_scores,
                )
            )
        return await forward(self)
```

The validation client runs independently in the background, processing validations and fetching scores as they become available, while the base validator handles weight setting automatically based on epoch timing.

