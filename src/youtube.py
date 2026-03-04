"""YouTube video fetching, caching, and search."""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

import numpy as np

from src.config import DATA_DIR, CACHE_MAX_AGE, SIMILARITY_THRESHOLD
from src.embeddings import get_embed_model
from src.utils import truncate


# --- Fetching ---

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


# --- Caching ---

def _youtube_cache_path():
    return os.path.join(DATA_DIR, ".youtube_cache.json")


def _youtube_embeddings_path():
    return os.path.join(DATA_DIR, ".youtube_embeddings.npz")


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
    model = get_embed_model()
    texts = []
    index = []  # (channel_url, video_index) for each text
    for channel_url, ch_data in channels.items():
        for i, vid in enumerate(ch_data["videos"]):
            embed_text = vid["title"]
            if vid.get("description"):
                embed_text += ". " + truncate(vid["description"], 300)
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


# --- Search ---

def search_youtube_cache(topic, channels, embeddings, index, max_total=7):
    """Search cached YouTube videos using semantic similarity, with keyword fallback."""
    if embeddings is None or index is None:
        print("  No YouTube embeddings available, skipping YouTube search.", file=sys.stderr)
        return []

    model = get_embed_model()
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
