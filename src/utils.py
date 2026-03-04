"""Shared text utilities."""

import html as html_module
import re
from datetime import datetime


def strip_html(text):
    """Remove HTML tags and decode entities, return plain text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text, max_len=200):
    """Truncate text to max_len, ending at a word boundary."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len].rsplit(" ", 1)[0]
    return truncated + "…"


def format_date(date_str):
    """Format a YYYY-MM-DD date string as 'Jan 15, 2024'. Returns '' on failure."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%b %-d, %Y")
    except (ValueError, TypeError):
        return ""


def parse_iso8601_duration(s):
    """Parse ISO 8601 duration (e.g. 'PT1H23M45S') → total seconds, or None."""
    if not s:
        return None
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", s)
    if not m:
        return None
    h, mins, secs = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mins * 60 + secs


def parse_podcast_duration(s):
    """Parse itunes:duration value → total seconds, or None.

    Handles: pure seconds ("2978"), "MM:SS", "HH:MM:SS" / "H:MM:SS".
    """
    if not s:
        return None
    s = s.strip()
    try:
        if ":" not in s:
            return int(s)
        parts = s.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return None


def format_duration(total_seconds):
    """Format seconds as human-readable duration: '1h 23m', '45m', '30s'. Returns '' for None/0."""
    if not total_seconds:
        return ""
    h, remainder = divmod(total_seconds, 3600)
    m = remainder // 60
    s = remainder % 60
    if h:
        return f"{h}h {m}m" if m else f"{h}h"
    if m:
        return f"{m}m"
    return f"{s}s"


def format_views(count):
    """Format view count: '1.2M views', '150K views', '8.5K views'. Returns '' for None."""
    if count is None:
        return ""
    count = int(count)
    if count >= 1_000_000:
        val = count / 1_000_000
        return f"{val:.1f}M views".replace(".0M", "M")
    if count >= 1_000:
        val = count / 1_000
        return f"{val:.1f}K views".replace(".0K", "K")
    return f"{count} views"
