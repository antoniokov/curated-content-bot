"""Bot main loop: Telegram long-polling, command handling, search dispatch."""

import argparse
import glob
import logging
import os
import signal
import sys
import time

from src.config import load_env, load_creators, setup_logging, MAX_RESULTS, MAX_PER_CREATOR, cache_usage, CACHE_WARN_RATIO
from src.utils import truncate, format_date, format_duration, format_views, escape_html
from src.embeddings import update_embed_model, maybe_unload_model
from src.youtube import get_youtube_cache, build_youtube_cache, search_youtube_cache
from src.podcast import get_podcast_cache, build_podcast_cache, search_all_podcasts
from src.telegram import tg_request, send_message, send_video_url, send_photo


# --- Auto-reload ---

logger = logging.getLogger(__name__)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _file_mtime(path):
    """Get file modification time, or 0 if not found."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def _max_src_mtime():
    """Get the newest mtime of any .py file in src/."""
    return max((_file_mtime(f) for f in glob.glob(os.path.join(_SRC_DIR, "*.py"))), default=0)


_start_mtime = _max_src_mtime()


def send_search_results(tg_token, chat_id, results):
    """Send search results to a Telegram chat. Returns the total number of items sent."""
    total_results = 0
    for group in results:
        if group["source"] == "youtube":
            send_message(tg_token, chat_id, f"📺 {group['creator']}", disable_preview=True)
            for video in group["videos"]:
                meta = [p for p in [
                    format_date(video.get("published_at", "")),
                    format_duration(video.get("duration")),
                    format_views(video.get("views")),
                ] if p]
                url = video["url"]
                if meta:
                    url = f"{' · '.join(meta)}\n\n{url}"
                send_video_url(tg_token, chat_id, url)
            total_results += len(group["videos"])
        else:
            send_message(tg_token, chat_id, f"🎙 {group['creator']}", disable_preview=True)
            for ep in group["episodes"]:
                short_desc = escape_html(truncate(ep.get("description", ""), 200))
                meta = [p for p in [
                    format_date(ep.get("published_at", "")),
                    format_duration(ep.get("duration")),
                ] if p]
                caption = f"<code>{escape_html(ep['title'])}</code>"
                if meta:
                    caption += f"\n\n{' · '.join(meta)}"
                if short_desc:
                    caption += f"\n\n{short_desc}"
                thumb = ep.get("thumbnail", "")
                if thumb:
                    send_photo(tg_token, chat_id, thumb, caption=caption, parse_mode="HTML")
                else:
                    send_message(tg_token, chat_id, caption, parse_mode="HTML", disable_preview=True)
            total_results += len(group["episodes"])

    if total_results == 0:
        send_message(tg_token, chat_id, "No relevant content found. Try a broader topic?")
    else:
        send_message(tg_token, chat_id, f"✅ Found {total_results} results", disable_preview=True)
    return total_results


def merge_search_results(yt_results, pod_results, max_results=MAX_RESULTS, max_per_creator=MAX_PER_CREATOR):
    """Merge YouTube and podcast results, rank by similarity, return top N grouped by creator."""
    # Flatten all individual items with metadata
    flat = []
    for group in yt_results:
        for vid in group["videos"]:
            flat.append({
                "source": "youtube",
                "creator": group["creator"],
                "similarity": vid.get("_similarity", 0.0),
                "item": vid,
            })
    for group in pod_results:
        for ep in group["episodes"]:
            flat.append({
                "source": "podcast",
                "creator": group["creator"],
                "similarity": ep.get("_similarity", 0.0),
                "item": ep,
                "apple_podcasts_id": group.get("apple_podcasts_id", ""),
            })

    # Sort by similarity descending, take top N with per-creator cap
    flat.sort(key=lambda x: x["similarity"], reverse=True)
    selected = []
    creator_counts = {}
    for entry in flat:
        key = (entry["source"], entry["creator"])
        if creator_counts.get(key, 0) >= max_per_creator:
            continue
        selected.append(entry)
        creator_counts[key] = creator_counts.get(key, 0) + 1
        if len(selected) >= max_results:
            break
    flat = selected

    # Re-group by (source, creator), preserving order of first appearance
    groups = []
    seen = {}
    for entry in flat:
        key = (entry["source"], entry["creator"])
        if key not in seen:
            if entry["source"] == "youtube":
                group = {"source": "youtube", "creator": entry["creator"], "videos": []}
            else:
                group = {"source": "podcast", "creator": entry["creator"],
                         "episodes": [], "apple_podcasts_id": entry.get("apple_podcasts_id", "")}
            seen[key] = len(groups)
            groups.append(group)
        group = groups[seen[key]]
        if entry["source"] == "youtube":
            group["videos"].append(entry["item"])
        else:
            group["episodes"].append(entry["item"])

    return groups


def _check_reload():
    """Restart the process if any src/*.py file has been modified since startup."""
    if _max_src_mtime() > _start_mtime:
        logger.info("src/ changed — reloading...")
        os.execv(sys.executable, [sys.executable] + sys.argv)


# --- Main loop ---

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Received signal %s, shutting down gracefully...", signum)
    _shutdown = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--creators", default="creators.csv", help="Path to creators CSV (default: creators.csv)")
    parser.add_argument("--dev", action="store_true", help="Dev mode: enable auto-reload, skip auth check, DEBUG logging")
    parser.add_argument("--refresh", action="store_true", help="Rebuild caches and exit (for scheduled runs)")
    parser.add_argument("--refresh-youtube", action="store_true", help="Rebuild YouTube cache only and exit")
    parser.add_argument("--refresh-podcasts", action="store_true", help="Rebuild podcast cache only and exit")
    args = parser.parse_args()

    setup_logging(debug=args.dev)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    env = load_env()
    yt_key = env.get("YOUTUBE_API_KEY")
    tg_token = env.get("TELEGRAM_BOT_TOKEN")
    allowed_chat_ids = env.get("ALLOWED_CHAT_IDS", set())

    if not yt_key:
        logger.error("YOUTUBE_API_KEY must be set in .env")
        sys.exit(1)

    refresh_any = args.refresh or args.refresh_youtube or args.refresh_podcasts

    if not refresh_any and not tg_token:
        logger.error("TELEGRAM_BOT_TOKEN must be set in .env")
        sys.exit(1)

    yt_creators, podcasts = load_creators(args.creators)

    # --refresh / --refresh-youtube / --refresh-podcasts: rebuild caches and exit
    if refresh_any:
        refresh_yt = args.refresh or args.refresh_youtube
        refresh_pod = args.refresh or args.refresh_podcasts
        logger.info("Refreshing caches (youtube=%s, podcasts=%s)...", refresh_yt, refresh_pod)
        if refresh_yt and yt_creators:
            build_youtube_cache(yt_creators, yt_key)
            logger.info("YouTube cache rebuilt.")
        if refresh_pod and podcasts:
            build_podcast_cache(podcasts)
            logger.info("Podcast cache rebuilt.")
        logger.info("Cache refresh complete.")
        return

    mode = "dev" if args.dev else "production"
    logger.info("Bot started (%s). Watching %d YouTube channels + %d podcasts.", mode, len(yt_creators), len(podcasts))

    # Pre-build caches + embeddings on startup if needed
    yt_channels, yt_embeddings, yt_index = None, None, None
    if yt_creators:
        yt_channels, yt_embeddings, yt_index = get_youtube_cache(yt_creators, yt_key)

    pod_feeds, pod_embeddings, pod_index = None, None, None
    if podcasts:
        pod_feeds, pod_embeddings, pod_index = get_podcast_cache(podcasts)

    tg_request("setMyCommands", tg_token, {"commands": [
        {"command": "start", "description": "Show welcome message"},
        {"command": "refresh", "description": "Fetch new videos and episodes"},
        {"command": "rebuild", "description": "Full rebuild of all caches"},
        {"command": "updatemodel", "description": "Re-download the embedding model"},
    ]})

    offset = 0
    while not _shutdown:
        if args.dev:
            _check_reload()
        maybe_unload_model()
        try:
            updates = tg_request("getUpdates", tg_token, {
                "offset": offset,
                "timeout": 30
            })
        except Exception as e:
            logger.error("Polling error: %s", e)
            time.sleep(5)
            continue

        for update in updates.get("result", []):
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            text = msg.get("text", "").strip()
            chat_id = msg.get("chat", {}).get("id")

            if not text or not chat_id:
                continue

            # Auth check (skipped in dev mode)
            if not args.dev and allowed_chat_ids and chat_id not in allowed_chat_ids:
                logger.warning("Unauthorized chat_id: %s", chat_id)
                continue

            # Handle /start command
            if text == "/start":
                send_message(tg_token, chat_id,
                    "Send me a topic and I'll find relevant content from your trusted creators.\n\n"
                    "Commands:\n"
                    "/refresh — fetch new videos and episodes (incremental)\n"
                    "/rebuild — full rebuild of all caches from scratch\n"
                    "/updatemodel — re-download the embedding model from HuggingFace")
                continue

            # Handle /updatemodel command
            if text == "/updatemodel":
                send_message(tg_token, chat_id, "🔄 Updating embedding model...")
                update_embed_model()
                send_message(tg_token, chat_id, "✅ Embedding model updated.")
                continue

            # Handle /refresh (incremental) and /rebuild (full) commands
            if text in ("/refresh", "/rebuild"):
                full = text == "/rebuild"
                label = "Rebuilding" if full else "Refreshing"
                if yt_creators:
                    send_message(tg_token, chat_id, f"🔄 {label} {len(yt_creators)} YouTube channels...", disable_preview=True)
                    yt_channels, yt_embeddings, yt_index = build_youtube_cache(
                        yt_creators, yt_key,
                        existing_channels=None if full else yt_channels)
                    total_vids = sum(len(ch["videos"]) for ch in yt_channels.values())
                    send_message(tg_token, chat_id, f"✅ YouTube: {len(yt_channels)} channels, {total_vids} videos", disable_preview=True)
                if podcasts:
                    send_message(tg_token, chat_id, f"🔄 {label} {len(podcasts)} podcast feeds...", disable_preview=True)
                    pod_feeds, pod_embeddings, pod_index = build_podcast_cache(podcasts)
                    total_eps = sum(len(f["episodes"]) for f in pod_feeds.values())
                    send_message(tg_token, chat_id, f"✅ Podcasts: {len(pod_feeds)} feeds, {total_eps} episodes", disable_preview=True)
                if not yt_creators and not podcasts:
                    send_message(tg_token, chat_id, "No creators in current file.", disable_preview=True)
                continue

            # Search
            text = text[:200]
            logger.info('Searching for "%s"...', text)

            parts = []
            if yt_creators:
                parts.append(f"{len(yt_creators)} channels")
            if podcasts:
                parts.append(f"{len(podcasts)} podcasts")
            send_message(tg_token, chat_id, f"🔍 Searching {' + '.join(parts)} for \"{text}\"...", disable_preview=True)

            # Search all sources and merge results
            yt_results = []
            if yt_creators:
                yt_channels, yt_embeddings, yt_index = get_youtube_cache(yt_creators, yt_key)
                yt_results = search_youtube_cache(text, yt_channels, yt_embeddings, yt_index)

            pod_results = []
            if podcasts:
                pod_feeds, pod_embeddings, pod_index = get_podcast_cache(podcasts)
                pod_results = search_all_podcasts(text, podcasts,
                    feeds=pod_feeds, embeddings=pod_embeddings, index=pod_index)

            results = merge_search_results(yt_results, pod_results)

            send_search_results(tg_token, chat_id, results)

            # Warn if cache is getting large
            total_bytes, ratio = cache_usage()
            if ratio >= CACHE_WARN_RATIO:
                pct = int(ratio * 100)
                mb = total_bytes // (1024 * 1024)
                send_message(tg_token, chat_id,
                    f"⚠️ Cache is at {pct}% ({mb} MB). Ping Anton to take a look.",
                    disable_preview=True)

    logger.info("Shutdown complete.")
