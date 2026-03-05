#!/usr/bin/env python3
"""Extract YouTube subscriptions from a saved YouTube subscriptions page HTML.

Usage:
    python scripts/extract_subscriptions.py [input_html] [output_csv]

Defaults:
    input_html:  data/subscriptions.html
    output_csv:  data/youtube_subscriptions.csv

To get the HTML file:
    1. Go to https://www.youtube.com/feed/channels
    2. Scroll to the bottom to load all subscriptions
    3. Save the page as HTML (Ctrl+S / Cmd+S)
"""

import csv
import json
import re
import sys


def extract_yt_initial_data(html):
    match = re.search(r"ytInitialData\s*=\s*", html)
    if not match:
        raise ValueError("Could not find ytInitialData in HTML")
    start = match.end()
    depth = 0
    for i in range(start, len(html)):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start : i + 1])
    raise ValueError("Could not parse ytInitialData JSON")


def find_channels(obj, results=None):
    if results is None:
        results = []
    if isinstance(obj, dict):
        if "channelId" in obj and "title" in obj:
            title = obj["title"]
            if isinstance(title, dict) and "simpleText" in title:
                handle = (
                    obj.get("navigationEndpoint", {})
                    .get("browseEndpoint", {})
                    .get("canonicalBaseUrl", "")
                )
                results.append((title["simpleText"], obj["channelId"], handle))
        for v in obj.values():
            find_channels(v, results)
    elif isinstance(obj, list):
        for item in obj:
            find_channels(item, results)
    return results


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/subscriptions.html"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/youtube_subscriptions.csv"

    with open(input_path, "r") as f:
        html = f.read()

    data = extract_yt_initial_data(html)
    channels = find_channels(data)

    # Deduplicate by channelId, preserving order
    seen = set()
    unique = []
    for name, channel_id, handle in channels:
        if channel_id not in seen:
            seen.add(channel_id)
            url = f"https://www.youtube.com{handle}" if handle else ""
            unique.append((name, channel_id, url))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "url", "channel_id", "apple_podcasts_id"])
        for name, channel_id, url in unique:
            writer.writerow(["YouTube channel", name, url, channel_id, ""])

    print(f"Extracted {len(unique)} channels to {output_path}")


if __name__ == "__main__":
    main()
