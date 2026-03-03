### Problem

On one side, I regularly struggle to find quality content once I get fascinated with a certain topic: airships, AI agents orchestration, etc.

On the other side, I have a list of people and teams that I enjoy reading or listening to, but I'm usually consuming content while its fresh even if the topic is not especially hot for me at the moment.

### Solution

On demand, I name a topic in a Telegram chat and get back relevant content from my trusted creators — YouTube videos with auto-unfurled previews, and podcast episodes with thumbnails and descriptions.

### UI

Telegram bot (`@TrustedResearchBot`). YouTube links auto-unfurl into rich previews. Podcast episodes show as photo cards with the episode title (tap-to-copy) and a short description.

### How it works

**YouTube** (keyword search via API):

1. I send a topic to the Telegram bot (e.g. `airships`)
2. The bot searches all 96 YouTube channels via the YouTube Data API v3
3. Results go through a 3-phase pipeline:
   - **Search** each channel (100 API units per channel)
   - **Fetch full descriptions** for all candidate videos (1 unit per 50 videos)
   - **Filter by relevance** — only keep videos whose title or full description mentions the topic
4. Results are sent back grouped by creator, capped at 7 total

**Podcasts** (semantic search via embeddings):

1. RSS feeds for all 59 podcasts are fetched and cached locally (refreshed once per day, or on `/refresh`)
2. Episode titles and descriptions are embedded using the `all-MiniLM-L6-v2` model (sentence-transformers)
3. On each search, the topic is embedded and compared against all episode embeddings using cosine similarity
4. Results above the similarity threshold are returned as rich cards (thumbnail + title + description)

### Data

- `creators.csv` — 96 YouTube channels + 59 podcasts (type, name, url, channel_id, apple_podcasts_id)
- `.env` — YouTube Data API key + Telegram bot token
- `bot.py` — the main bot script (long-polling, no server needed)
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
- `/refresh` — re-fetch all podcast RSS feeds and recompute embeddings

### API quota

Each full YouTube search costs ~9,600 units (96 channels × 100 units). The free tier is 10,000 units/day (resets midnight Pacific). Podcast search is free (local RSS + embeddings).
