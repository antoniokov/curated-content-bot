### Problem

On one side, I regularly struggle to find quality content once I get fascinated with a certain topic: airships, AI agents orchestration, etc.

On the other side, I have a list of people and teams that I enjoy reading or listening to, but I'm usually consuming content while its fresh even if the topic is not especially hot for me at the moment.

### Solution

On demand, I name a topic in a Telegram chat and get back relevant content from my trusted creators — YouTube videos with auto-unfurled previews, and podcast episodes with thumbnails and descriptions.

### UI

Telegram bot (`@TrustedResearchBot`). YouTube links auto-unfurl into rich previews. Podcast episodes show as photo cards with the episode title (tap-to-copy) and a short description.

### How it works

Both YouTube and podcasts follow the same pattern: **cache locally, search via embeddings**.

1. On startup (or `/refresh`), all content is fetched and cached locally:
   - **YouTube**: all videos from each channel's uploads playlist via `playlistItems.list` (incremental — only new videos on subsequent refreshes)
   - **Podcasts**: all episodes from each podcast's RSS feed
2. Titles and descriptions are embedded using the `all-MiniLM-L6-v2` model (sentence-transformers)
3. On each search, the topic is embedded and compared against all cached embeddings using cosine similarity, with keyword fallback
4. Results are sent back grouped by creator, capped at 7 per source

### Data

- `creators.csv` — 96 YouTube channels + 59 podcasts (type, name, url, channel_id, apple_podcasts_id)
- `.env` — YouTube Data API key + Telegram bot token
- `bot.py` — the main bot script (long-polling, no server needed)
- `.youtube_cache.json` — cached YouTube video metadata (auto-generated)
- `.youtube_embeddings.npz` — cached YouTube video embeddings (auto-generated)
- `.podcast_cache.json` — cached RSS feed data (auto-generated)
- `.podcast_embeddings.npz` — cached episode embeddings (auto-generated)

### Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers numpy
```

### Running

```
source .venv/bin/activate
python3 bot.py                              # all creators
python3 bot.py --creators creators_test.csv # test YouTube (2 channels)
python3 bot.py --creators podcasts_test.csv # test podcasts (3 podcasts)
python3 bot.py --creators podcasts_all.csv  # all podcasts only (no YouTube quota)
```

### Bot commands

- `/start` — show help
- `/refresh` — re-fetch YouTube videos and podcast feeds, recompute embeddings

### API quota

YouTube uses `playlistItems.list` (1 unit per 50 videos) instead of `search.list` (100 units per channel). Initial full cache build costs ~576 units; incremental daily refreshes cost ~96–200 units. Searches themselves cost 0 units. The free tier is 10,000 units/day (resets midnight Pacific). Podcast search is free (local RSS + embeddings).

### TODO

[x] Optimize YouTube quota usage via caching
[] Add a command to check the current daily YouTube quota available
[] Store local logs to help with future troubleshooting
[] Allow to add a creator from Telegram bot (does it need AI to enrich things?)
