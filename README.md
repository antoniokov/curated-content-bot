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
4. Results from all sources are ranked by similarity and the top 10 are sent back, grouped by creator

YouTube links auto-unfurl into rich previews. Podcast episodes show as photo cards with the episode title (tap-to-copy) and a short description.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
- `/refresh` — incremental update: fetch only new YouTube videos since last refresh, re-fetch all podcast RSS feeds, recompute embeddings
- `/rebuild` — full rebuild from scratch: discard all cached data, re-fetch every video and episode, recompute embeddings. Use after code changes or when the cache seems stale.
- `/updatemodel` — re-download the embedding model from HuggingFace (use after upgrading sentence-transformers or when a newer model version is available)

## API quota

YouTube uses `playlistItems.list` (1 unit per 50 videos) and `videos.list` (1 unit per 50 videos) for metadata. Initial full cache build costs ~960 units; incremental daily refreshes cost ~100–250 units. Searches themselves cost 0 units. The free tier is 10,000 units/day (resets midnight Pacific). Podcast search is free (local RSS + embeddings).

## Project structure

```
bot.py                  — entry point
src/
  config.py             — paths, constants, load_env(), load_creators()
  utils.py              — text helpers, duration/views formatting
  embeddings.py         — lazy-loaded sentence-transformers model
  youtube.py            — YouTube fetch, cache, search
  podcast.py            — podcast RSS fetch, cache, search
  telegram.py           — Telegram Bot API helpers
  main.py               — main loop, command handling, auto-reload
tests/
  test_bot.py           — 18 pytest tests covering the core pipeline
data/
  creators.csv          — 96 YouTube channels + 59 podcasts
  (auto-generated cache and embedding files)
.env                    — YouTube Data API key + Telegram bot token
```

## Testing

```
source .venv/bin/activate
python3 -m pytest tests/test_bot.py -v
```
