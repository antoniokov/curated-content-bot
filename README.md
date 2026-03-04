# Curated Content Bot

I regularly struggle to find quality content once I get fascinated with a certain topic: airships, AI agents orchestration, etc. At the same time, I have a list of people and teams that I enjoy reading or listening to, but I'm usually consuming content while it's fresh even if the topic is not especially hot for me at the moment.

**Curated Content Bot** lets me name a topic in a Telegram chat and get back relevant content from my trusted creators — YouTube videos with auto-unfurled previews, and podcast episodes with thumbnails and descriptions.

## How it works

Telegram bot (`@curated_content_bot`). Both YouTube and podcasts follow the same pattern: **cache locally, search via embeddings**.

1. On startup (or `/refresh`), all content is fetched and cached locally:
   - **YouTube**: all videos from each channel's uploads playlist via `playlistItems.list` (incremental — only new videos on subsequent refreshes)
   - **Podcasts**: all episodes from each podcast's RSS feed
2. Titles and descriptions are embedded using the `all-MiniLM-L6-v2` model (sentence-transformers)
3. On each search, the topic is embedded and compared against all cached embeddings using cosine similarity, with keyword fallback
4. Results are sent back grouped by creator, capped at 7 per source

YouTube links auto-unfurl into rich previews. Podcast episodes show as photo cards with the episode title (tap-to-copy) and a short description.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers numpy pytest
```

Create a `.env` file with your API keys:

```
YOUTUBE_API_KEY=...
TELEGRAM_BOT_TOKEN=...
```

Add your creators to `data/creators.csv` (columns: type, name, url, channel_id, apple_podcasts_id).

## Running

```
source .venv/bin/activate
python3 bot.py                              # all creators
python3 bot.py --creators creators_test.csv # test YouTube (2 channels)
python3 bot.py --creators podcasts_test.csv # test podcasts (3 podcasts)
python3 bot.py --creators podcasts_all.csv  # all podcasts only (no YouTube quota)
```

## Bot commands

- `/start` — show help
- `/refresh` — re-fetch YouTube videos and podcast feeds, recompute embeddings

## API quota

YouTube uses `playlistItems.list` (1 unit per 50 videos) instead of `search.list` (100 units per channel). Initial full cache build costs ~576 units; incremental daily refreshes cost ~96–200 units. Searches themselves cost 0 units. The free tier is 10,000 units/day (resets midnight Pacific). Podcast search is free (local RSS + embeddings).

## Project structure

```
bot.py                  — entry point
src/
  config.py             — paths, constants, load_env(), load_creators()
  utils.py              — strip_html(), truncate()
  embeddings.py         — lazy-loaded sentence-transformers model
  youtube.py            — YouTube fetch, cache, search
  podcast.py            — podcast RSS fetch, cache, search
  telegram.py           — Telegram Bot API helpers
  main.py               — main loop, command handling, auto-reload
tests/
  test_bot.py           — 10 pytest tests covering the core pipeline
data/
  creators.csv          — 96 YouTube channels + 59 podcasts
  (auto-generated cache and embedding files)
.env                    — YouTube Data API key + Telegram bot token
```

## Testing

```
python -m pytest tests/test_bot.py -v
```
