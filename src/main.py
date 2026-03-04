"""Bot main loop: Telegram long-polling, command handling, search dispatch."""

import argparse
import glob
import os
import sys
import time

from src.config import load_env, load_creators
from src.utils import truncate
from src.youtube import get_youtube_cache, build_youtube_cache, search_youtube_cache
from src.podcast import get_podcast_cache, build_podcast_cache, search_all_podcasts
from src.telegram import tg_request, send_message, send_video_url, send_photo


# --- Auto-reload ---

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


def _check_reload():
    """Restart the process if any src/*.py file has been modified since startup."""
    if _max_src_mtime() > _start_mtime:
        print("\n🔄 src/ changed — reloading...", file=sys.stderr)
        os.execv(sys.executable, [sys.executable] + sys.argv)


# --- Main loop ---

def main():
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
                        short_desc = truncate(ep.get("description", ""), 200)
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
