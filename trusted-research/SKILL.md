---
name: trusted-research
description: |
  **Trusted Research**: Search for content on a specific topic from a curated list of trusted YouTube creators and podcasts.
  Use this skill whenever the user asks to find content, videos, episodes, or articles about a topic from their favorite creators. Also trigger when the user mentions "trusted research", "what have my creators said about...", "find me content on...", or names a topic and expects results filtered through their trusted sources list.
---

# Trusted Research

You help the user discover content on a specific topic from creators they trust. The primary interface is a **Telegram bot** (`bot.py`) that the user runs on their machine.

## Architecture

The bot (`bot.py`) uses long-polling against the Telegram Bot API. It supports two source types:

**YouTube channels** (keyword search):
1. Queries each channel via the YouTube Data API v3 `search` endpoint (100 units per channel)
2. Batch-fetches full video descriptions via `videos.list` (1 unit per 50 videos)
3. Filters by relevance — keeps only videos whose title or description mentions the topic

**Podcasts** (semantic search):
1. RSS feeds are fetched and cached locally (refreshed once per day or on `/refresh`)
2. Episode titles + descriptions are embedded using `all-MiniLM-L6-v2` (sentence-transformers)
3. On search, the topic is embedded and compared via cosine similarity against all episodes
4. Results above a similarity threshold are returned as rich Telegram photo cards

## Key files

- `creators.csv` — 96 YouTube channels + 59 podcasts (type, name, url, channel_id, apple_podcasts_id)
- `creators_test.csv` — 2 YouTube channels for development testing (saves API quota)
- `podcasts_test.csv` — 3 podcasts for development testing
- `podcasts_all.csv` — all 59 podcasts (no YouTube quota used)
- `.env` — contains `YOUTUBE_API_KEY` and `TELEGRAM_BOT_TOKEN`
- `bot.py` — the main Telegram bot script
- `.podcast_cache.json` — auto-generated RSS feed cache
- `.podcast_embeddings.npz` — auto-generated episode embeddings

## Setup

The bot requires a Python virtual environment with sentence-transformers:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers numpy
```

## Running the bot

```bash
source .venv/bin/activate
python3 bot.py                              # all creators
python3 bot.py --creators creators_test.csv # test YouTube (2 channels)
python3 bot.py --creators podcasts_all.csv  # all podcasts only
```

## Bot commands

- `/start` — show help
- `/refresh` — re-fetch all podcast RSS feeds and recompute embeddings

## Important notes

- Don't summarize the content — links and previews are enough for the user to decide what to watch/listen to.
- Prioritize relevance over recency.
- YouTube API quota is 10,000 units/day (free tier). Podcast search is free (local RSS + embeddings).
- The bot auto-reloads when `bot.py` is modified — no need to restart manually.
