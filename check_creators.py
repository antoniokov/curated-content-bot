#!/usr/bin/env python3
"""Check creators.csv for issues: broken YouTube playlists and unreachable podcast feeds."""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from src.config import load_env, load_creators
from src.podcast import fetch_rss_episodes


def check_youtube(yt_creators, api_key):
    """Check each YouTube channel's uploads playlist. Returns list of issue strings."""
    issues = []
    for creator in yt_creators:
        name = creator["name"]
        channel_id = creator["channel_id"]
        playlist_id = "UU" + channel_id[2:]
        params = {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": 1,
            "key": api_key,
        }
        url = f"https://www.googleapis.com/youtube/v3/playlistItems?{urllib.parse.urlencode(params)}"
        try:
            resp = urllib.request.urlopen(url, timeout=15)
            data = json.loads(resp.read())
            if not data.get("items"):
                issues.append(f"  YOUTUBE  {name}: playlist {playlist_id} is empty")
                print(f"  x {name}")
            else:
                print(f"  ok {name}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            issues.append(f"  YOUTUBE  {name}: playlist {playlist_id} HTTP {e.code} — {body}")
            print(f"  x {name}")
    return issues


def check_podcasts(podcasts):
    """Check each podcast RSS feed is reachable and returns episodes. Returns list of issue strings."""
    issues = []
    for podcast in podcasts:
        name = podcast["name"]
        feed_url = podcast["url"]
        episodes = fetch_rss_episodes(feed_url)
        if not episodes:
            issues.append(f"  PODCAST  {name}: {feed_url} — 0 episodes")
            print(f"  x {name}")
        else:
            print(f"  ok {name} ({len(episodes)} episodes)")
    return issues


def main():
    env = load_env()
    yt_key = env.get("YOUTUBE_API_KEY")
    if not yt_key:
        print("Error: YOUTUBE_API_KEY must be set in .env")
        sys.exit(1)

    yt_creators, podcasts = load_creators()

    issues = []

    print(f"\nChecking {len(yt_creators)} YouTube channels...\n")
    issues.extend(check_youtube(yt_creators, yt_key))

    print(f"\nChecking {len(podcasts)} podcast feeds...\n")
    issues.extend(check_podcasts(podcasts))

    if issues:
        print(f"\n{len(issues)} issue(s) found:\n")
        for issue in issues:
            print(issue)
        print()
        sys.exit(1)
    else:
        print("\nAll creators OK.\n")


if __name__ == "__main__":
    main()
