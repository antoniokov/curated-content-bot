#!/usr/bin/env python3
"""
Trusted Research - YouTube channel search script.

Searches all creators in creators.csv for videos matching a topic
using the YouTube Data API v3.

Usage:
    python search.py "airships" --key-file key.txt --creators creators.csv
    python search.py "airships" --key-file key.txt --creators creators.csv --format html -o results.html

Output: JSON (default) or HTML file with clickable thumbnail cards.

API quota: each channel search costs 100 units. With 91 channels, a full scan
costs 9,100 units out of the 10,000/day free quota. Channel IDs are read
directly from the channel_id column in creators.csv (no resolution step needed).
"""

import argparse
import csv
import html
import json
import sys
import urllib.request
import urllib.parse
import urllib.error


def load_api_key(key=None, key_file=None):
    if key:
        return key.strip()
    if key_file:
        with open(key_file) as f:
            return f.read().strip()
    raise ValueError("Provide --key or --key-file")


def api_get(endpoint, params, api_key):
    params["key"] = api_key
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?{urllib.parse.urlencode(params)}"
    try:
        resp = urllib.request.urlopen(url)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"API error {e.code}: {body}", file=sys.stderr)
        return None


def search_channel(channel_id, query, api_key, max_results=3):
    """Search a single channel for videos matching query."""
    data = api_get("search", {
        "part": "snippet",
        "channelId": channel_id,
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": "relevance"
    }, api_key)

    if not data or not data.get("items"):
        return []

    results = []
    for item in data["items"]:
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]
        results.append({
            "title": snippet["title"],
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            "published_at": snippet["publishedAt"],
            "description": snippet["description"][:200]
        })
    return results


def render_html(topic, results):
    """Render results as a standalone HTML page with thumbnail cards."""
    total_videos = sum(len(g["videos"]) for g in results)
    total_creators = len(results)

    cards_html = ""
    for group in results:
        creator = html.escape(group["creator"])
        cards_html += f'  <h2>{creator}</h2>\n  <div class="grid">\n'
        for v in group["videos"]:
            title = html.escape(v["title"])
            cards_html += f'''    <div class="card"><a href="{v["url"]}" target="_blank">
      <img src="{v["thumbnail"]}" alt="" loading="lazy">
      <div class="title">{title}</div>
    </a></div>\n'''
        cards_html += "  </div>\n\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trusted Research: {html.escape(topic)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8f8f8; color: #1a1a1a; padding: 2rem; max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 0.3rem; }}
  .subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 0.75rem; padding-bottom: 0.3rem; border-bottom: 1px solid #e0e0e0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }}
  .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: box-shadow 0.2s; }}
  .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
  .card a {{ text-decoration: none; color: inherit; display: block; }}
  .card img {{ width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; }}
  .card .title {{ padding: 0.6rem 0.75rem; font-size: 0.85rem; line-height: 1.3; color: #1a1a1a; }}
</style>
</head>
<body>
  <h1>Trusted Research: {html.escape(topic)}</h1>
  <p class="subtitle">{total_videos} results from {total_creators} creators</p>

{cards_html}</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Search trusted creators for a topic")
    parser.add_argument("topic", help="Topic to search for")
    parser.add_argument("--key", help="YouTube Data API key")
    parser.add_argument("--key-file", help="Path to file containing API key")
    parser.add_argument("--creators", default="creators.csv", help="Path to creators.csv")
    parser.add_argument("--max-results", type=int, default=3, help="Max results per channel")
    parser.add_argument("--max-total", type=int, default=7, help="Max total results to show")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="Output format")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout for JSON, results.html for HTML)")
    args = parser.parse_args()

    api_key = load_api_key(args.key, args.key_file)

    # Load creators
    with open(args.creators) as f:
        creators = list(csv.DictReader(f))

    yt_creators = [c for c in creators if c["type"] == "YouTube channel" and c.get("channel_id")]
    print(f"Searching {len(yt_creators)} YouTube channels for \"{args.topic}\"...", file=sys.stderr)

    # Search each channel
    all_results = []
    for creator in yt_creators:
        name = creator["name"]
        print(f"  Searching {name}...", file=sys.stderr)
        videos = search_channel(creator["channel_id"], args.topic, api_key, args.max_results)
        if videos:
            all_results.append({
                "creator": name,
                "channel_url": creator["url"],
                "videos": videos
            })

    # Keep grouping by creator but limit total videos
    total = 0
    capped_results = []
    for group in all_results:
        if total >= args.max_total:
            break
        remaining = args.max_total - total
        capped_videos = group["videos"][:remaining]
        if capped_videos:
            capped_results.append({
                "creator": group["creator"],
                "channel_url": group["channel_url"],
                "videos": capped_videos
            })
            total += len(capped_videos)

    # Output
    if args.format == "html":
        output_html = render_html(args.topic, capped_results)
        output_path = args.output or "results.html"
        with open(output_path, "w") as f:
            f.write(output_html)
        print(f"Results saved to {output_path}", file=sys.stderr)
    else:
        output = json.dumps(capped_results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Results saved to {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == "__main__":
    main()
