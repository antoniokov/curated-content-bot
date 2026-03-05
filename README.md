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
pip install -r requirements-dev.txt
```

Create a `.env` file with your API keys:

```
YOUTUBE_API_KEY=...
TELEGRAM_BOT_TOKEN=...
ALLOWED_CHAT_IDS=123456789
```

`ALLOWED_CHAT_IDS` is a comma-separated list of Telegram user IDs allowed to use the bot. In dev mode (`--dev`), this check is skipped.

Add your creators to `data/creators.csv` (columns: type, name, url, channel_id, apple_podcasts_id). See `data/creators_sample.csv` for an example.

## Running

```
source .venv/bin/activate
python3 bot.py --dev                              # dev mode (auto-reload, no auth, DEBUG logging)
python3 bot.py --dev --creators creators_test.csv # dev with test YouTube (2 channels)
python3 bot.py --dev --creators podcasts_test.csv # dev with test podcasts (3 podcasts)
python3 bot.py                                    # production mode (no auto-reload, auth enforced)
python3 bot.py --refresh                          # rebuild caches and exit (for cron/systemd timer)
python3 scripts/check_creators.py                 # check creators.csv for broken channels/feeds
python3 scripts/extract_subscriptions.py          # extract YouTube subscriptions HTML to CSV
```

## Bot commands

- `/start` — show help
- `/refresh` — incremental update: fetch only new YouTube videos since last refresh, re-fetch all podcast RSS feeds, recompute embeddings
- `/rebuild` — full rebuild from scratch: discard all cached data, re-fetch every video and episode, recompute embeddings. Use after code changes or when the cache seems stale.
- `/updatemodel` — re-download the embedding model from HuggingFace (use after upgrading sentence-transformers or when a newer model version is available)

## API quota

YouTube uses `playlistItems.list` (1 unit per 50 videos) and `videos.list` (1 unit per 50 videos) for metadata. Searches themselves cost 0 units. The free tier is 10,000 units/day (resets midnight Pacific). Podcast search is free (local RSS + embeddings).

## Project structure

```
bot.py                  — entry point
scripts/
  check_creators.py     — check creators.csv for broken channels/feeds
  extract_subscriptions.py — extract YouTube subscriptions HTML to CSV
src/
  config.py             — paths, constants, load_env(), load_creators()
  utils.py              — text helpers, duration/views formatting
  embeddings.py         — lazy-loaded sentence-transformers model
  youtube.py            — YouTube fetch, cache, search
  podcast.py            — podcast RSS fetch, cache, search
  telegram.py           — Telegram Bot API helpers
  main.py               — main loop, command handling, auto-reload
tests/
  test_bot.py           — 20 pytest tests covering the core pipeline
data/
  creators.csv          — your YouTube channels + podcasts (not in repo)
  creators_sample.csv   — example creators file with a few channels + podcasts
cache/
  (auto-generated cache and embedding files)
.env                    — YouTube Data API key + Telegram bot token
```

## Testing

```
source .venv/bin/activate
pip install -r requirements-dev.txt
python3 -m pytest tests/test_bot.py -v
```

## Deployment (any Linux server)

### Prerequisites

- Ubuntu/Debian server with systemd (2 GB RAM recommended — the embedding model + caches need ~200 MB at peak, and 2 GB gives comfortable headroom)
- Python 3.10+
- Your API keys ready: `YOUTUBE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `ALLOWED_CHAT_IDS`

### Step 1: SSH into your server

```bash
ssh root@<your-server-ip>
```

### Step 2: Install system dependencies

```bash
apt update && apt upgrade -y
apt install -y python3 python3-venv python3-pip git
python3 --version  # verify 3.10+
```

### Step 3: Create a dedicated `deploy` user

```bash
useradd -r -s /usr/sbin/nologin deploy
```

This is a system user with no login shell — it only runs the bot service. If the bot is ever compromised, the attacker is confined to `/opt/curated-content-bot` and can't access the rest of the system.

### Step 4: Clone the repo

```bash
git clone https://github.com/antoniokov/curated-content-bot.git /opt/curated-content-bot
cd /opt/curated-content-bot
```

### Step 5: Set up Python venv and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**CPU-only server** (most VPS instances) — saves ~2 GB of disk by skipping NVIDIA/CUDA packages:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**GPU server:**

```bash
pip install -r requirements.txt
```

This installs `sentence-transformers` (which pulls in PyTorch), `numpy`, and `defusedxml`. The embedding model (~80 MB) downloads automatically on first run.

### Step 6: Create the `.env` file

```bash
nano /opt/curated-content-bot/.env
```

Paste the following, replacing the placeholder values with your actual keys:

```
YOUTUBE_API_KEY=your_youtube_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
ALLOWED_CHAT_IDS=your_telegram_user_id_here
```

Save with Ctrl+O, Enter, Ctrl+X. `ALLOWED_CHAT_IDS` is comma-separated if you have multiple users.

### Step 7: Add your creators CSV

The creators list is specific to each deployment and is not included in the repo. Create it on the server:

```bash
nano /opt/curated-content-bot/data/creators.csv
```

Paste your CSV contents (columns: `type,name,url,channel_id,apple_podcasts_id`) and save with Ctrl+O, Enter, Ctrl+X.

Alternatively, upload the file from your local machine:

```bash
scp data/creators.csv root@<your-server-ip>:/opt/curated-content-bot/data/creators.csv
```

### Step 8: Build initial caches

```bash
source .venv/bin/activate
python3 bot.py --refresh
```

This fetches all YouTube videos and podcast episodes, computes embeddings, and saves cache files. Takes a few minutes on first run.

### Step 9: Set file ownership

```bash
chown -R deploy:deploy /opt/curated-content-bot
```

### Step 10: Install and start systemd services

```bash
# Install service files
cp deploy/curated-content-bot.service /etc/systemd/system/
cp deploy/daily-refresh.service /etc/systemd/system/
cp deploy/daily-refresh.timer /etc/systemd/system/
systemctl daemon-reload

# Enable and start
systemctl enable --now curated-content-bot
systemctl enable --now daily-refresh.timer
```

Three units are installed:
- `curated-content-bot.service` — runs the bot continuously, restarts on failure
- `daily-refresh.service` — oneshot that rebuilds caches
- `daily-refresh.timer` — triggers the refresh daily at 03:00 UTC

### Step 11: Verify it's running

```bash
systemctl status curated-content-bot
journalctl -u curated-content-bot -f        # live logs
systemctl list-timers daily-refresh.timer    # next scheduled refresh
```

Send a message to your bot in Telegram to confirm it responds.

### Deploying changes from your dev machine

Create a `.env.dev` file in the project root (it's gitignored):

```
SERVER=<your-server-ip>
```

Then deploy with:

```bash
make deploy          # git pull + restart the bot
make deploy-rebuild  # git pull + restart + rebuild caches
make deploy-csv CSV=data/creators.csv  # upload CSV + full cache rebuild
```

Since the repo is owned by `deploy` but you run `git pull` as root, you'll need to allow this once on the server:

```bash
git config --global --add safe.directory /opt/curated-content-bot
```

### Server-side operations

```bash
# Trigger a manual cache refresh
systemctl start daily-refresh

# View refresh logs
journalctl -u daily-refresh -e

# Check bot logs (also at /opt/curated-content-bot/logs/bot.log)
journalctl -u curated-content-bot -n 100
```

### CLI: `--refresh`

`python3 bot.py --refresh` rebuilds all caches and exits (no Telegram polling). This is what the daily systemd timer runs. The running bot picks up fresh cache files automatically on the next search query.
