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

def youtube_search(channel_id, query, api_key, max_results=5):
    """Search a single YouTube channel for videos matching a query."""
    params = urllib.parse.urlencode({
        "part": "snippet",
        "channelId": channel_id,
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
        "key": api_key
    })
    url = f"https://www.googleapis.com/youtube/v3/search?{params}"
    try:
        resp = urllib.request.urlopen(url)
        data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"YouTube API error {e.code}: {e.read().decode()}", file=sys.stderr)
        return []

    if not data.get("items"):
        return []

    results = []
    for item in data["items"]:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        description = item["snippet"].get("description", "")
        results.append({
            "title": title,
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "description": description
        })
    return results


def fetch_full_descriptions(video_ids, api_key):
    """Batch-fetch full video descriptions. Costs 1 unit per 50 videos."""
    descriptions = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        params = urllib.parse.urlencode({
            "part": "snippet",
            "id": ",".join(batch),
            "key": api_key
        })
        url = f"https://www.googleapis.com/youtube/v3/videos?{params}"
        try:
            resp = urllib.request.urlopen(url)
            data = json.loads(resp.read())
            for item in data.get("items", []):
                descriptions[item["id"]] = item["snippet"].get("description", "")
        except urllib.error.HTTPError as e:
            print(f"Videos API error {e.code}: {e.read().decode()}", file=sys.stderr)
    return descriptions


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


def parse_topic(text):
    """Parse user input into search query and relevance keywords."""
    query = text.strip().lower()
    keywords = {query}
    # Add basic plural/singular variant
    if query.endswith("s"):
        keywords.add(query[:-1])
    else:
        keywords.add(query + "s")
    return query, keywords


def is_relevant(video, keywords):
    """Check if a video's title or description mentions any of the keywords."""
    text = (video["title"] + " " + video.get("description", "")).lower()
    return any(kw in text for kw in keywords)


def search_all_creators(topic, creators, api_key, max_total=7):
    """Search all creators and return grouped results."""
    search_query, keywords = parse_topic(topic)
    print(f"  Search query: \"{search_query}\"", file=sys.stderr)
    print(f"  Relevance keywords: {keywords}", file=sys.stderr)

    # Phase 1: search all channels
    all_results = []
    all_video_ids = []
    for creator in creators:
        videos = youtube_search(creator["channel_id"], search_query, api_key)
        if videos:
            all_results.append({"creator": creator["name"], "videos": videos})
            all_video_ids.extend(v["video_id"] for v in videos)

    # Phase 2: fetch full descriptions for all candidate videos
    if all_video_ids:
        print(f"  Fetching full descriptions for {len(all_video_ids)} videos...", file=sys.stderr)
        full_descriptions = fetch_full_descriptions(all_video_ids, api_key)
        for group in all_results:
            for video in group["videos"]:
                if video["video_id"] in full_descriptions:
                    video["description"] = full_descriptions[video["video_id"]]

    # Phase 3: filter by relevance using full descriptions
    filtered_results = []
    for group in all_results:
        relevant = [v for v in group["videos"] if is_relevant(v, keywords)]
        if relevant:
            filtered_results.append({"creator": group["creator"], "videos": relevant})

    # Cap at max_total videos
    total = 0
    capped = []
    for group in filtered_results:
        if total >= max_total:
            break
        remaining = max_total - total
        videos = group["videos"][:remaining]
        if videos:
            capped.append({"creator": group["creator"], "videos": videos})
            total += len(videos)
    return capped


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

    # Pre-build podcast cache + embeddings on startup if needed
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
                    "Commands:\n/refresh — re-fetch all podcast feeds")
                continue

            # Handle /refresh command
            if text == "/refresh":
                if podcasts:
                    send_message(tg_token, chat_id, f"🔄 Refreshing {len(podcasts)} podcast feeds + embeddings...", disable_preview=True)
                    pod_feeds, pod_embeddings, pod_index = build_podcast_cache(podcasts)
                    total_eps = sum(len(f["episodes"]) for f in pod_feeds.values())
                    send_message(tg_token, chat_id, f"✅ Cache refreshed: {len(pod_feeds)} feeds, {total_eps} episodes", disable_preview=True)
                else:
                    send_message(tg_token, chat_id, "No podcasts in current creators file.", disable_preview=True)
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

            # YouTube search
            if yt_creators:
                yt_results = search_all_creators(text, yt_creators, yt_key)
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
