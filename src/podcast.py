"""Podcast RSS fetching, caching, and search."""

import json
import logging
import os
import time
import urllib.request
import defusedxml.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np

logger = logging.getLogger(__name__)

from src.config import CACHE_DIR, CACHE_MAX_AGE, SIMILARITY_THRESHOLD
from src.embeddings import get_embed_model
from src.utils import strip_html, truncate, parse_podcast_duration


# XML namespaces used in podcast feeds
NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "media": "http://search.yahoo.com/mrss/",
}


# --- Fetching ---

def _is_episode_url(url):
    """Check if a URL looks episode-specific (not just a homepage)."""
    try:
        from urllib.parse import urlparse
        path = urlparse(url).path.rstrip("/")
        return len(path) > 1  # more than just "/"
    except Exception:
        return bool(url)


def fetch_rss_episodes(feed_url, timeout=10):
    """Fetch and parse episodes from a podcast RSS feed."""
    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "CuratedContentBot/1.0"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        xml_bytes = resp.read()
    except Exception as e:
        logger.error("RSS fetch error for %s: %s", feed_url, e)
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.error("RSS parse error for %s: %s", feed_url, e)
        return []

    episodes = []
    channel = root.find("channel")
    if channel is None:
        return []

    # Channel-level fallback image
    channel_image = channel.find("itunes:image", NS)
    if channel_image is None:
        channel_image = channel.find("image/url")
    fallback_thumb = ""
    if channel_image is not None:
        fallback_thumb = channel_image.get("href", "") or channel_image.text or ""

    for item in channel.findall("item"):
        title = item.findtext("title", "").strip()
        description_raw = item.findtext("description", "").strip()
        description = strip_html(description_raw)
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

        # Parse pubDate (RFC 822) → "YYYY-MM-DD"
        pub_date_raw = item.findtext("pubDate", "").strip()
        published_at = ""
        if pub_date_raw:
            try:
                published_at = parsedate_to_datetime(pub_date_raw).strftime("%Y-%m-%d")
            except Exception:
                pass

        # Parse duration (itunes:duration can be seconds, MM:SS, or HH:MM:SS)
        duration_raw = item.findtext("itunes:duration", "", NS).strip()
        duration = parse_podcast_duration(duration_raw) if duration_raw else None

        if title:
            episodes.append({
                "title": title,
                "description": description,
                "link": best_link,
                "thumbnail": thumb,
                "published_at": published_at,
                "duration": duration,
            })
    return episodes


# --- Caching ---

def _cache_path():
    return os.path.join(CACHE_DIR, ".podcast_cache.json")


def _embeddings_path():
    return os.path.join(CACHE_DIR, ".podcast_embeddings.npz")


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
            logger.info("Podcast cache loaded (%d feeds, %dmin old, embeddings=%s)",
                        len(feeds), int(age/60), "yes" if embeddings is not None else "no")
            return feeds, embeddings, index
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None, None, None


def build_podcast_cache(podcasts):
    """Fetch all podcast RSS feeds, compute embeddings, and cache everything."""
    logger.info("Building podcast cache for %d feeds...", len(podcasts))
    feeds = {}
    for podcast in podcasts:
        name = podcast["name"]
        url = podcast["url"]
        logger.info("Fetching: %s...", name)
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
        logger.error("Cache save error: %s", e)

    # Build embeddings for all episodes
    model = get_embed_model()
    texts = []
    index = []  # (feed_url, episode_index) for each text
    for feed_url, feed_data in feeds.items():
        for i, ep in enumerate(feed_data["episodes"]):
            # Combine title and truncated description for embedding
            embed_text = ep["title"]
            if ep.get("description"):
                embed_text += ". " + truncate(ep["description"], 300)
            texts.append(embed_text)
            index.append((feed_url, i))

    total_eps = len(texts)
    if texts:
        logger.info("Computing embeddings for %d episodes...", total_eps)
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        np.savez(_embeddings_path(), embeddings=embeddings, index=np.array(index, dtype=object))
        logger.info("Embeddings saved: %s", embeddings.shape)
    else:
        embeddings = None

    logger.info("Cache complete: %d feeds, %d episodes", len(feeds), total_eps)
    return feeds, embeddings, index


def get_podcast_cache(podcasts, force_refresh=False):
    """Get cached podcast data with embeddings, rebuilding if stale or forced."""
    if not force_refresh:
        feeds, embeddings, index = load_podcast_cache()
        if feeds is not None:
            return feeds, embeddings, index
    return build_podcast_cache(podcasts)


# --- Search ---

def search_all_podcasts(topic, podcasts, feeds=None, embeddings=None, index=None, max_total=None):
    """Search all podcasts using semantic similarity and return grouped results.

    When max_total is None, returns all results above the similarity threshold.
    """
    if feeds is None:
        feeds, embeddings, index = get_podcast_cache(podcasts)

    if embeddings is None or index is None:
        logger.warning("No embeddings available, skipping podcast search.")
        return []

    # Embed the query
    model = get_embed_model()
    query_emb = model.encode([topic])

    # Compute cosine similarities (dot product since vectors are normalized)
    similarities = (embeddings @ query_emb.T).flatten()

    # Get top results above threshold, sorted by similarity
    ranked_indices = np.argsort(similarities)[::-1]

    # Group by podcast, respecting max_total
    results_by_feed = {}  # feed_url -> list of (similarity, episode)
    total = 0
    for idx in ranked_indices:
        if max_total is not None and total >= max_total:
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
        logger.warning("Semantic search found nothing, falling back to keyword search for: %s", topic)
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
                if max_total is not None and total >= max_total:
                    break
            if max_total is not None and total >= max_total:
                break
        all_results = list(results_by_feed.values())

    return all_results
