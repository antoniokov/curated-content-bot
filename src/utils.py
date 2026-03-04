"""Shared text utilities."""

import html as html_module
import re


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
