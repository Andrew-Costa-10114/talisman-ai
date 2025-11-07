# User Miner Module

The user miner is the core component that orchestrates the post processing pipeline: **scrape → analyze → submit**. It runs continuously in a background thread, fetching posts from X/Twitter, analyzing them for subnet relevance and sentiment, and submitting high-quality posts to the coordination API.

## Overview

The miner follows a simple but effective pipeline:

1. **Scrape** posts from X/Twitter API using configurable keywords
2. **Analyze** each post for subnet relevance and sentiment using LLM-based analysis
3. **Score** posts using a weighted combination of relevance, value, and recency
4. **Submit** analyzed posts to the API server for validation

The miner runs in a single background thread for simplicity and reliability, continuously processing posts until reaching the configured limit or being stopped.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MyMiner (Background Thread)            │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ PostScraper  │───▶│   Analyzer   │───▶│ APIClient   │  │
│  │              │    │              │    │             │  │
│  │ - Fetches    │    │ - Relevance  │    │ - Submits   │  │
│  │   tweets     │    │ - Sentiment  │    │   to API    │  │
│  │ - Filters    │    │ - Scoring    │    │ - Retries   │  │
│  │   duplicates │    │              │    │             │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### MyMiner (`my_miner.py`)

The main orchestrator class that runs the processing loop. It:

- Manages the background thread lifecycle (`start()`, `stop()`)
- Coordinates scraping, analysis, and submission
- Tracks processed posts to avoid duplicates
- Respects configuration limits (`MAX_POSTS`, `POSTS_PER_SCRAPE`, `POSTS_TO_SUBMIT`)
- Handles errors gracefully with logging

**Key Methods:**
- `start()`: Starts the background processing thread
- `stop()`: Stops the thread gracefully (waits up to 5 seconds)
- `get_stats()`: Returns current statistics (posts processed, running status, etc.)
- `_run()`: Main processing loop (runs in background thread)

### PostScraper (`post_scraper.py`)

Fetches posts from X/Twitter API using the Tweepy client.

**Features:**
- Searches for tweets matching configurable keywords (default: `["omron", "bittensor"]`)
- Fetches tweets from the last 72 hours
- Excludes retweets and filters by English language
- Tracks seen tweets to avoid duplicates
- Automatically refetches when running low on tweets (< 10 remaining)
- Returns random samples from the fetched pool

**Configuration:**
- Keywords are configurable in `post_scraper.py` (line 24): `self.keywords = ["omron", "bittensor"]`
- Uses `X_BEARER_TOKEN` and `X_API_BASE` from `.miner_env` configuration

**Data Format:**
Each post includes:
- `id`: Tweet ID
- `content`: Tweet text
- `author`: Author username
- `timestamp`: Unix timestamp
- `account_age`: Account age in days
- `retweets`, `likes`, `responses`: Engagement metrics
- `followers`: Author follower count

### Analyzer (`analyzer/`)

Uses LLM-based analysis to determine subnet relevance and sentiment.

**Subnet Relevance Analysis:**
- Analyzes how relevant a post is to each registered BitTensor subnet
- Uses hybrid approach: LLM understanding + deterministic scoring
- Returns relevance scores (0.0 to 1.0) for each subnet
- Subnets are loaded from `analyzer/data/subnets.json`

**Sentiment Analysis:**
- Classifies sentiment on a scale from -1.0 (very bearish) to +1.0 (very bullish)
- Categories:
  - `1.0`: Very bullish (excitement, major positive developments)
  - `0.5`: Moderately positive (optimistic, good news)
  - `0.0`: Neutral (factual, informative, balanced)
  - `-0.5`: Moderately negative (concerns, skepticism)
  - `-1.0`: Very bearish (major issues, strong criticism)

**Scoring:**
Posts are scored using `score_tweet_entry()` which combines three components:

1. **Relevance** (50% weight): Mean relevance score of top-k subnets
   - Uses LLM-based analyzer to determine subnet relevance
   - Considers top 5 most relevant subnets by default

2. **Value** (40% weight): Signal quality based on engagement and author credibility
   - Normalizes: likes, retweets, quotes, replies, followers, account age
   - Uses caps: 5k likes, 1k retweets, 300 quotes, 600 replies, 200k followers, 7 years account age

3. **Recency** (10% weight): How recent the tweet is
   - Linear decay from 1.0 (just posted) to 0.0 (older than 24 hours)
   - Formula: `1.0 - (age_hours / 24.0)`

**Final Score:**
```
post_score = 0.50 × relevance + 0.40 × value + 0.10 × recency
```

### APIClient (`api_client.py`)

Handles HTTP communication with the coordination API server.

**Features:**
- Submits posts to `/v1/submit` endpoint
- Includes miner hotkey in `X-Hotkey` header for authentication
- Implements retry logic with exponential backoff:
  - Up to 3 attempts total
  - Wait times: 3s after first failure, 6s after second failure
  - 10 second timeout per request
- Treats "duplicate" status as success (API is idempotent)

**Submission Format:**
```python
{
    "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "post_id": "1234567890",
    "content": "Normalized tweet text...",
    "date": 1234567890,  # Unix timestamp
    "author": "username",
    "account_age": 365,
    "retweets": 10,
    "likes": 50,
    "responses": 5,
    "followers": 1000,
    "tokens": {  # Subnet relevance scores
        "subnet_name": 0.85,
        ...
    },
    "sentiment": 0.5,  # -1.0 to +1.0
    "score": 0.75  # Final weighted score
}
```

## Processing Flow

1. **Initialization:**
   - Miner starts with hotkey from parent neuron
   - Initializes `PostScraper`, `Analyzer`, and `APIClient`
   - Loads subnet registry from `analyzer/data/subnets.json`

2. **Scrape Cycle:**
   - Waits `SCRAPE_INTERVAL_SECONDS` between cycles (default: 300s = 5 minutes)
   - Scrapes `POSTS_PER_SCRAPE` posts (default: 5)
   - Filters out posts already seen (tracked in `_seen_post_ids`)

3. **Analysis:**
   - For each new post:
     - Normalizes content using `norm_text()` (ensures consistency with validator)
     - Analyzes for subnet relevance and sentiment using LLM
     - Calculates post score using `score_tweet_entry()`
     - Prepares submission data

4. **Submission:**
   - Submits up to `POSTS_TO_SUBMIT` posts per cycle (default: 2)
   - Tracks successful submissions in `_seen_post_ids`
   - Increments `posts_processed` counter

5. **Limits:**
   - Stops if `posts_processed >= MAX_POSTS` (0 = unlimited)
   - Respects `POSTS_TO_SUBMIT` per cycle limit
   - Continues until stopped or limit reached

## Configuration

The miner respects the following environment variables from `.miner_env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `SCRAPE_INTERVAL_SECONDS` | Interval between scrape cycles | `300` (5 minutes) |
| `POSTS_PER_SCRAPE` | Number of posts to scrape per cycle | `5` |
| `POSTS_TO_SUBMIT` | Number of posts to submit per cycle | `2` |
| `MAX_POSTS` | Maximum posts to process (0 = unlimited) | `0` |
| `X_BEARER_TOKEN` | X/Twitter API bearer token | Required |
| `X_API_BASE` | X/Twitter API base URL | `https://api.twitter.com/2` |
| `MINER_API_URL` | Coordination API server URL | `http://127.0.0.1:8000` |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout for API requests | `30.0` |
| `MODEL` | LLM model identifier | `deepseek-ai/DeepSeek-V3-0324` |
| `API_KEY` | LLM API key | Required |
| `LLM_BASE` | LLM API base URL | `https://llm.chutes.ai/v1` |

## Text Normalization

The miner normalizes post content using `norm_text()` to ensure consistency with validator analysis:

- Unicode normalization (NFC)
- Line ending normalization (`\r\n` → `\n`)
- Whitespace collapse (multiple spaces → single space)
- Trim leading/trailing whitespace

This ensures that minor formatting differences don't cause false mismatches during validation.

## Error Handling

The miner handles errors gracefully:

- **Scraping errors**: Logs warning and continues to next cycle
- **Analysis errors**: Logs warning, uses default values (score = 0.0), continues
- **API submission errors**: Retries up to 3 times with exponential backoff
- **Thread errors**: Logs error, waits 2 seconds, continues loop

## Statistics

The miner tracks:
- `posts_processed`: Number of posts successfully submitted
- `max_posts`: Maximum posts to process before stopping
- `running`: Whether the miner is currently running
- `thread_alive`: Whether the background thread is alive

Access via `my_miner.get_stats()`.

## Integration

The miner is integrated into the main neuron (`neurons/miner.py`):

```python
class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        hotkey = self.wallet.hotkey.ss58_address
        self.my_miner = MyMiner(hotkey=hotkey)
        self.my_miner.start()
```

The miner runs independently in a background thread and doesn't interfere with the neuron's synapse processing.

## Customization

### Changing Search Keywords

Edit `post_scraper.py` line 24:
```python
self.keywords = ["your", "keywords", "here"]
```