#!/usr/bin/env python3
"""
Trusted Research Telegram Bot.

Send a topic — get back relevant content from your trusted creators.
Supports YouTube channels (via API) and Podcasts (via RSS + semantic search).

Setup:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install sentence-transformers numpy

Usage:
    source .venv/bin/activate
    python3 bot.py                                # all creators
    python3 bot.py --creators creators_test.csv   # test YouTube subset
    python3 bot.py --creators podcasts_test.csv   # test podcasts subset
    python3 bot.py --creators podcasts_all.csv    # all podcasts only

Reads YOUTUBE_API_KEY and TELEGRAM_BOT_TOKEN from .env in the same directory.
Reads creators from creators.csv (or a custom file via --creators) in the same directory.
"""

import csv
import html as html_module
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET

import logging
import numpy as np

# Suppress "UNEXPECTED key" warnings from transformers model loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

# --- Config ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_env():
    """Load key=value pairs from .env file."""
    env_path = os.path.join(SCRIPT_DIR, ".env")
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env

def load_creators(csv_filename="creators.csv"):
    """Load creators from a CSV file, split by type."""
    csv_path = os.path.join(SCRIPT_DIR, csv_filename)
    with open(csv_path) as f:
        creators = list(csv.DictReader(f))
    youtube = [c for c in creators if c["type"] == "YouTube channel" and c.get("channel_id")]
    podcasts = [c for c in creators if c["type"] == "Podcast" and c.get("url")]
    return youtube, podcasts


# --- YouTube API ---

def fetch_channel_videos(channel_id, api_key, max_per_page=50, known_ids=None):
    """Fetch all videos from a channel's uploads playlist via playlistItems.list.

    Costs 1 API unit per page of 50 videos (vs 100 units for search.list).
    Stops early if a video in known_ids is encountered (for incremental refresh).
    """
    # Derive uploads playlist ID: UC... -> UU...
    playlist_id = "UU" + channel_id[2:]

    videos = []
    page_token = None

    while True:
        params = {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": max_per_page,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"https://www.googleapis.com/youtube/v3/playlistItems?{urllib.parse.urlencode(params)}"
        try:
            resp = urllib.request.urlopen(url)
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"  playlistItems error {e.code} for {channel_id}: {e.read().decode()}", file=sys.stderr)
            break

        hit_known = False
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            video_id = snippet.get("resourceId", {}).get("videoId", "")
            if not video_id:
                continue
            if known_ids and video_id in known_ids:
                hit_known = True
                break

            thumbs = snippet.get("thumbnails", {})
            thumb = (thumbs.get("high") or thumbs.get("medium")
                     or thumbs.get("default") or {}).get("url", "")

            videos.append({
                "title": snippet.get("title", ""),
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "description": snippet.get("description", ""),
                "thumbnail": thumb,
            })

        if hit_known:
            break
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return videos


# --- Podcast RSS ---

# XML namespaces used in podcast feeds
NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "media": "http://search.yahoo.com/mrss/",
}


def _strip_html(text):
    """Remove HTML tags and decode entities, return plain text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text, max_len=200):
    """Truncate text to max_len, ending at a word boundary."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len].rsplit(" ", 1)[0]
    return truncated + "…"


def fetch_rss_episodes(feed_url, timeout=10):
    """Fetch and parse episodes from a podcast RSS feed."""
    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "TrustedResearchBot/1.0"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        xml_bytes = resp.read()
    except Exception as e:
        print(f"  RSS fetch error for {feed_url}: {e}", file=sys.stderr)
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"  RSS parse error for {feed_url}: {e}", file=sys.stderr)
        return []

    episodes = []
    channel = root.find("channel")
    if channel is None:
        return []

    # Channel-level fallback image
    channel_image = (
        channel.find("itunes:image", NS)
        or channel.find("image/url")
    )
    fallback_thumb = ""
    if channel_image is not None:
        fallback_thumb = channel_image.get("href", "") or channel_image.text or ""

    for item in channel.findall("item"):
        title = item.findtext("title", "").strip()
        description_raw = item.findtext("description", "").strip()
        description = _strip_html(description_raw)
        link = item.findtext("link", "").strip()
        guid = item.findtext("guid", "").strip()

        # Episode thumbnail: try itunes:image, then media:thumbnail, then channel fallback
        thumb = ""
        itunes_img = item.find("itunes:image", NS)
        if itunes_img is not None:
            thumb = itunes_img.get("href", "")
        if not thumb:
            media_thumb = item.find("media:thumbnail", NS)
            if media_thumb is not None:
                thumb = media_thumb.get("url", "")
        if not thumb:
            thumb = fallback_thumb

        # Try to get the audio enclosure URL as fallback link
        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url", "") if enclosure is not None else ""

        # Pick the best episode-specific URL
        best_link = ""
        if link and _is_episode_url(link):
            best_link = link
        elif guid and guid.startswith("http") and _is_episode_url(guid):
            best_link = guid
        else:
            best_link = audio_url

        if title:
            episodes.append({
                "title": title,
                "description": description,
                "link": best_link,
                "thumbnail": thumb,
            })
    return episodes


def _is_episode_url(url):
    """Check if a URL looks episode-specific (not just a homepage)."""
    try:
        from urllib.parse import urlparse
        path = urlparse(url).path.rstrip("/")
        return len(path) > 1  # more than just "/"
    except Exception:
        return bool(url)


# --- Podcast cache ---

CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours in seconds
SIMILARITY_THRESHOLD = 0.45    # minimum cosine similarity to consider relevant

# Lazy-loaded embedding model (loaded once on first use)
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("  Loading embedding model (first time may download ~80MB)...", file=sys.stderr)
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Model loaded.", file=sys.stderr)
    return _embed_model


def _cache_path():
    return os.path.join(SCRIPT_DIR, ".podcast_cache.json")


def _embeddings_path():
    return os.path.join(SCRIPT_DIR, ".podcast_embeddings.npz")


def load_podcast_cache():
    """Load cached podcast episodes and embeddings from disk, if fresh enough."""
    path = _cache_path()
    try:
        with open(path) as f:
            cache = json.load(f)
        age = time.time() - cache.get("timestamp", 0)
        if age < CACHE_MAX_AGE:
            feeds = cache["feeds"]
            # Load embeddings
            emb_path = _embeddings_path()
            if os.path.exists(emb_path):
                data = np.load(emb_path, allow_pickle=True)
                embeddings = data["embeddings"]
                index = data["index"].tolist()  # list of (feed_url, episode_idx)
            else:
                embeddings, index = None, None
            print(f"  Podcast cache loaded ({len(feeds)} feeds, {int(age/60)}min old, embeddings={'yes' if embeddings is not None else 'no'})", file=sys.stderr)
            return feeds, embeddings, index
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None, None, None


def build_podcast_cache(podcasts):
    """Fetch all podcast RSS feeds, compute embeddings, and cache everything."""
    print(f"  Building podcast cache for {len(podcasts)} feeds...", file=sys.stderr)
    feeds = {}
    for podcast in podcasts:
        name = podcast["name"]
        url = podcast["url"]
        print(f"    Fetching: {name}...", file=sys.stderr)
        episodes = fetch_rss_episodes(url)
        feeds[url] = {
            "name": name,
            "apple_podcasts_id": podcast.get("apple_podcasts_id", ""),
            "episodes": episodes,
        }

    # Save feed metadata as JSON
    cache = {"timestamp": time.time(), "feeds": feeds}
    try:
        with open(_cache_path(), "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"  Cache save error: {e}", file=sys.stderr)

    # Build embeddings for all episodes
    model = _get_embed_model()
    texts = []
    index = []  # (feed_url, episode_index) for each text
    for feed_url, feed_data in feeds.items():
        for i, ep in enumerate(feed_data["episodes"]):
            # Combine title and truncated description for embedding
            embed_text = ep["title"]
            if ep.get("description"):
                embed_text += ". " + _truncate(ep["description"], 300)
            texts.append(embed_text)
            index.append((feed_url, i))

    total_eps = len(texts)
    if texts:
        print(f"  Computing embeddings for {total_eps} episodes...", file=sys.stderr)
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
        # Normalize for cosine similarity (dot product on normalized vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        np.savez(_embeddings_path(), embeddings=embeddings, index=np.array(index, dtype=object))
        print(f"  Embeddings saved: {embeddings.shape}", file=sys.stderr)
    else:
        embeddings = None

    print(f"  Cache complete: {len(feeds)} feeds, {total_eps} episodes", file=sys.stderr)
    return feeds, embeddings, index


def get_podcast_cache(podcasts, force_refresh=False):
    """Get cached podcast data with embeddings, rebuilding if stale or forced."""
    if not force_refresh:
        feeds, embeddings, index = load_podcast_cache()
        if feeds is not None:
            return feeds, embeddings, index
    return build_podcast_cache(podcasts)


def search_all_podcasts(topic, podcasts, feeds=None, embeddings=None, index=None, max_total=7):
    """Search all podcasts using semantic similarity and return grouped results."""
    if feeds is None:
        feeds, embeddings, index = get_podcast_cache(podcasts)

    if embeddings is None or index is None:
        print("  No embeddings available, skipping podcast search.", file=sys.stderr)
        return []

    # Embed the query
    model = _get_embed_model()
    query_emb = model.encode([topic])
    query_emb = query_emb / np.linalg.norm(query_emb)

    # Compute cosine similarities (dot product since vectors are normalized)
    similarities = (embeddings @ query_emb.T).flatten()

    # Get top results above threshold, sorted by similarity
    ranked_indices = np.argsort(similarities)[::-1]

    # Group by podcast, respecting max_total
    results_by_feed = {}  # feed_url -> list of (similarity, episode)
    total = 0
    for idx in ranked_indices:
        if total >= max_total:
            break
        sim = float(similarities[idx])
        if sim < SIMILARITY_THRESHOLD:
            break
        feed_url, ep_idx = index[idx]
        feed_data = feeds.get(feed_url)
        if not feed_data:
            continue
        ep = feed_data["episodes"][int(ep_idx)]
        if feed_url not in results_by_feed:
            results_by_feed[feed_url] = {
                "creator": feed_data["name"],
                "feed_url": feed_url,
                "apple_podcasts_id": feed_data.get("apple_podcasts_id", ""),
                "episodes": [],
            }
        results_by_feed[feed_url]["episodes"].append({**ep, "_similarity": sim})
        total += 1

    # Return as list, ordered by best match per group
    all_results = sorted(results_by_feed.values(),
                         key=lambda g: max(e["_similarity"] for e in g["episodes"]),
                         reverse=True)

    # Fallback: keyword search if semantic search found nothing
    if not all_results:
        print(f"  Semantic search found nothing, falling back to keyword search for: {topic}", file=sys.stderr)
        kw = topic.lower()
        for feed_url, feed_data in feeds.items():
            for ep in feed_data["episodes"]:
                text = (ep.get("title", "") + " " + ep.get("description", "")).lower()
                if kw in text:
                    if feed_url not in results_by_feed:
                        results_by_feed[feed_url] = {
                            "creator": feed_data["name"],
                            "feed_url": feed_url,
                            "apple_podcasts_id": feed_data.get("apple_podcasts_id", ""),
                            "episodes": [],
                        }
                    if len(results_by_feed[feed_url]["episodes"]) < 3:
                        results_by_feed[feed_url]["episodes"].append({**ep, "_similarity": 0.0})
                        total += 1
                if total >= max_total:
                    break
            if total >= max_total:
                break
        all_results = list(results_by_feed.values())

    return all_results


# --- YouTube cache ---

def _youtube_cache_path():
    return os.path.join(SCRIPT_DIR, ".youtube_cache.json")


def _youtube_embeddings_path():
    return os.path.join(SCRIPT_DIR, ".youtube_embeddings.npz")


def load_youtube_cache():
    """Load cached YouTube videos and embeddings from disk, if fresh enough."""
    path = _youtube_cache_path()
    try:
        with open(path) as f:
            cache = json.load(f)
        age = time.time() - cache.get("timestamp", 0)
        if age < CACHE_MAX_AGE:
            channels = cache["channels"]
            emb_path = _youtube_embeddings_path()
            if os.path.exists(emb_path):
                data = np.load(emb_path, allow_pickle=True)
                embeddings = data["embeddings"]
                index = data["index"].tolist()
            else:
                embeddings, index = None, None
            total_videos = sum(len(ch["videos"]) for ch in channels.values())
            print(f"  YouTube cache loaded ({len(channels)} channels, {total_videos} videos, "
                  f"{int(age/60)}min old, embeddings={'yes' if embeddings is not None else 'no'})",
                  file=sys.stderr)
            return channels, embeddings, index
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None, None, None


def build_youtube_cache(yt_creators, api_key, existing_channels=None):
    """Fetch all YouTube channels' videos, compute embeddings, and cache."""
    print(f"  Building YouTube cache for {len(yt_creators)} channels...", file=sys.stderr)
    channels = {}

    for creator in yt_creators:
        name = creator["name"]
        channel_id = creator["channel_id"]
        channel_url = creator.get("url", "")

        # For incremental refresh: known video IDs from existing cache
        known_ids = None
        existing_videos = []
        if existing_channels and channel_url in existing_channels:
            existing_videos = existing_channels[channel_url].get("videos", [])
            known_ids = {v["video_id"] for v in existing_videos}

        print(f"    Fetching: {name}...", file=sys.stderr)
        new_videos = fetch_channel_videos(channel_id, api_key, known_ids=known_ids)

        # Merge: new videos first (newest), then existing videos not already in new
        all_videos = new_videos
        if existing_videos:
            new_ids = {v["video_id"] for v in new_videos}
            all_videos = new_videos + [v for v in existing_videos if v["video_id"] not in new_ids]

        channels[channel_url] = {
            "name": name,
            "channel_id": channel_id,
            "videos": all_videos,
        }
        print(f"      {len(new_videos)} new + {len(all_videos) - len(new_videos)} existing "
              f"= {len(all_videos)} total", file=sys.stderr)

    # Save channel metadata as JSON
    cache = {"timestamp": time.time(), "channels": channels}
    try:
        with open(_youtube_cache_path(), "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"  YouTube cache save error: {e}", file=sys.stderr)

    # Build embeddings for all videos
    model = _get_embed_model()
    texts = []
    index = []  # (channel_url, video_index) for each text
    for channel_url, ch_data in channels.items():
        for i, vid in enumerate(ch_data["videos"]):
            embed_text = vid["title"]
            if vid.get("description"):
                embed_text += ". " + _truncate(vid["description"], 300)
            texts.append(embed_text)
            index.append((channel_url, i))

    total_videos = len(texts)
    if texts:
        print(f"  Computing embeddings for {total_videos} videos...", file=sys.stderr)
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        np.savez(_youtube_embeddings_path(), embeddings=embeddings,
                 index=np.array(index, dtype=object))
        print(f"  YouTube embeddings saved: {embeddings.shape}", file=sys.stderr)
    else:
        embeddings = None

    print(f"  YouTube cache complete: {len(channels)} channels, {total_videos} videos",
          file=sys.stderr)
    return channels, embeddings, index


def get_youtube_cache(yt_creators, api_key, force_refresh=False):
    """Get cached YouTube data with embeddings, rebuilding if stale or forced."""
    existing_channels = None
    if not force_refresh:
        channels, embeddings, index = load_youtube_cache()
        if channels is not None:
            return channels, embeddings, index
        # Load stale cache for incremental refresh
        try:
            with open(_youtube_cache_path()) as f:
                existing_channels = json.load(f).get("channels")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return build_youtube_cache(yt_creators, api_key, existing_channels=existing_channels)


def search_youtube_cache(topic, channels, embeddings, index, max_total=7):
    """Search cached YouTube videos using semantic similarity, with keyword fallback."""
    if embeddings is None or index is None:
        print("  No YouTube embeddings available, skipping YouTube search.", file=sys.stderr)
        return []

    model = _get_embed_model()
    query_emb = model.encode([topic])
    query_emb = query_emb / np.linalg.norm(query_emb)

    similarities = (embeddings @ query_emb.T).flatten()
    ranked_indices = np.argsort(similarities)[::-1]

    results_by_channel = {}
    total = 0
    for idx in ranked_indices:
        if total >= max_total:
            break
        sim = float(similarities[idx])
        if sim < SIMILARITY_THRESHOLD:
            break
        channel_url, vid_idx = index[idx]
        ch_data = channels.get(channel_url)
        if not ch_data:
            continue
        vid = ch_data["videos"][int(vid_idx)]
        if channel_url not in results_by_channel:
            results_by_channel[channel_url] = {
                "creator": ch_data["name"],
                "videos": [],
            }
        results_by_channel[channel_url]["videos"].append({**vid, "_similarity": sim})
        total += 1

    all_results = sorted(results_by_channel.values(),
                         key=lambda g: max(v["_similarity"] for v in g["videos"]),
                         reverse=True)

    # Fallback: keyword search
    if not all_results:
        print(f"  YouTube semantic search found nothing, falling back to keywords: {topic}",
              file=sys.stderr)
        kw = topic.lower()
        keywords = {kw}
        if kw.endswith("s"):
            keywords.add(kw[:-1])
        else:
            keywords.add(kw + "s")

        for channel_url, ch_data in channels.items():
            for vid in ch_data["videos"]:
                text = (vid.get("title", "") + " " + vid.get("description", "")).lower()
                if any(k in text for k in keywords):
                    if channel_url not in results_by_channel:
                        results_by_channel[channel_url] = {
                            "creator": ch_data["name"],
                            "videos": [],
                        }
                    if len(results_by_channel[channel_url]["videos"]) < 3:
                        results_by_channel[channel_url]["videos"].append(
                            {**vid, "_similarity": 0.0})
                        total += 1
                if total >= max_total:
                    break
            if total >= max_total:
                break
        all_results = list(results_by_channel.values())

    return all_results


# --- Telegram API ---

def tg_request(method, token, data=None):
    """Make a Telegram Bot API request."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    if data:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def send_message(token, chat_id, text, disable_preview=False, parse_mode=None):
    """Send a message to a Telegram chat."""
    data = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_preview
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    tg_request("sendMessage", token, data)


def send_video_url(token, chat_id, url):
    """Send a single YouTube URL (Telegram will unfurl it with a preview)."""
    tg_request("sendMessage", token, {
        "chat_id": chat_id,
        "text": url
    })


def send_photo(token, chat_id, photo_url, caption="", parse_mode=None):
    """Send a photo with optional caption to a Telegram chat."""
    data = {
        "chat_id": chat_id,
        "photo": photo_url,
        "caption": caption,
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        tg_request("sendPhoto", token, data)
    except Exception as e:
        # Fallback to text if photo fails
        print(f"  sendPhoto failed: {e}, falling back to text", file=sys.stderr)
        send_message(token, chat_id, caption, disable_preview=True)


# --- Auto-reload ---

def _file_mtime(path):
    """Get file modification time, or 0 if not found."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def _check_reload():
    """Restart the process if bot.py has been modified since startup."""
    current_mtime = _file_mtime(__file__)
    if current_mtime > _check_reload.start_mtime:
        print("\n🔄 bot.py changed — reloading...", file=sys.stderr)
        os.execv(sys.executable, [sys.executable] + sys.argv)

_check_reload.start_mtime = _file_mtime(__file__)


# --- Main loop ---

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--creators", default="creators.csv", help="Path to creators CSV (default: creators.csv)")
    args = parser.parse_args()

    env = load_env()
    yt_key = env.get("YOUTUBE_API_KEY")
    tg_token = env.get("TELEGRAM_BOT_TOKEN")

    if not yt_key or not tg_token:
        print("Error: YOUTUBE_API_KEY and TELEGRAM_BOT_TOKEN must be set in .env", file=sys.stderr)
        sys.exit(1)

    yt_creators, podcasts = load_creators(args.creators)
    print(f"Bot started. Watching {len(yt_creators)} YouTube channels + {len(podcasts)} podcasts.", file=sys.stderr)

    # Pre-build caches + embeddings on startup if needed
    yt_channels, yt_embeddings, yt_index = None, None, None
    if yt_creators:
        yt_channels, yt_embeddings, yt_index = get_youtube_cache(yt_creators, yt_key)

    pod_feeds, pod_embeddings, pod_index = None, None, None
    if podcasts:
        pod_feeds, pod_embeddings, pod_index = get_podcast_cache(podcasts)

    offset = 0
    while True:
        _check_reload()
        try:
            updates = tg_request("getUpdates", tg_token, {
                "offset": offset,
                "timeout": 30
            })
        except Exception as e:
            print(f"Polling error: {e}", file=sys.stderr)
            time.sleep(5)
            continue

        for update in updates.get("result", []):
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            text = msg.get("text", "").strip()
            chat_id = msg.get("chat", {}).get("id")

            if not text or not chat_id:
                continue

            # Handle /start command
            if text == "/start":
                send_message(tg_token, chat_id,
                    "Send me a topic and I'll find relevant content from your trusted creators.\n\n"
                    "Commands:\n/refresh — re-fetch YouTube videos and podcast feeds")
                continue

            # Handle /refresh command
            if text == "/refresh":
                if yt_creators:
                    send_message(tg_token, chat_id, f"🔄 Refreshing {len(yt_creators)} YouTube channels...", disable_preview=True)
                    yt_channels, yt_embeddings, yt_index = build_youtube_cache(yt_creators, yt_key, existing_channels=yt_channels)
                    total_vids = sum(len(ch["videos"]) for ch in yt_channels.values())
                    send_message(tg_token, chat_id, f"✅ YouTube cache refreshed: {len(yt_channels)} channels, {total_vids} videos", disable_preview=True)
                if podcasts:
                    send_message(tg_token, chat_id, f"🔄 Refreshing {len(podcasts)} podcast feeds...", disable_preview=True)
                    pod_feeds, pod_embeddings, pod_index = build_podcast_cache(podcasts)
                    total_eps = sum(len(f["episodes"]) for f in pod_feeds.values())
                    send_message(tg_token, chat_id, f"✅ Podcast cache refreshed: {len(pod_feeds)} feeds, {total_eps} episodes", disable_preview=True)
                if not yt_creators and not podcasts:
                    send_message(tg_token, chat_id, "No creators in current file.", disable_preview=True)
                continue

            # Search
            print(f"Searching for \"{text}\"...", file=sys.stderr)

            parts = []
            if yt_creators:
                parts.append(f"{len(yt_creators)} channels")
            if podcasts:
                parts.append(f"{len(podcasts)} podcasts")
            send_message(tg_token, chat_id, f"🔍 Searching {' + '.join(parts)} for \"{text}\"...", disable_preview=True)

            total_results = 0

            # YouTube search (from cache)
            if yt_creators:
                yt_channels, yt_embeddings, yt_index = get_youtube_cache(yt_creators, yt_key)
                yt_results = search_youtube_cache(text, yt_channels, yt_embeddings, yt_index)
                for group in yt_results:
                    send_message(tg_token, chat_id, f"📺 {group['creator']}", disable_preview=True)
                    for video in group["videos"]:
                        send_video_url(tg_token, chat_id, video["url"])
                total_results += sum(len(g["videos"]) for g in yt_results)

            # Podcast search (semantic)
            if podcasts:
                pod_feeds, pod_embeddings, pod_index = get_podcast_cache(podcasts)
                pod_results = search_all_podcasts(text, podcasts,
                    feeds=pod_feeds, embeddings=pod_embeddings, index=pod_index)
                for group in pod_results:
                    creator = group["creator"]
                    send_message(tg_token, chat_id, f"🎙 {creator}", disable_preview=True)
                    for ep in group["episodes"]:
                        short_desc = _truncate(ep.get("description", ""), 200)
                        caption = f"<code>{ep['title']}</code>"
                        if short_desc:
                            caption += f"\n\n{short_desc}"
                        thumb = ep.get("thumbnail", "")
                        if thumb:
                            send_photo(tg_token, chat_id, thumb, caption=caption, parse_mode="HTML")
                        else:
                            send_message(tg_token, chat_id, caption, parse_mode="HTML", disable_preview=True)
                total_results += sum(len(g["episodes"]) for g in pod_results)

            if total_results == 0:
                send_message(tg_token, chat_id, "No relevant content found. Try a broader topic?")
            else:
                send_message(tg_token, chat_id, f"✅ Found {total_results} results", disable_preview=True)


if __name__ == "__main__":
    main()
