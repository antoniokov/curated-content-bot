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
