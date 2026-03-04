"""Telegram Bot API helpers."""

import json
import sys
import urllib.request


def tg_request(method, token, data=None):
    """Make a Telegram Bot API request."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    if data:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def send_message(token, chat_id, text, disable_preview=False, parse_mode=None):
    """Send a message to a Telegram chat."""
    data = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_preview
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    tg_request("sendMessage", token, data)


def send_video_url(token, chat_id, url):
    """Send a single YouTube URL (Telegram will unfurl it with a preview)."""
    tg_request("sendMessage", token, {
        "chat_id": chat_id,
        "text": url
    })


def send_photo(token, chat_id, photo_url, caption="", parse_mode=None):
    """Send a photo with optional caption to a Telegram chat."""
    data = {
        "chat_id": chat_id,
        "photo": photo_url,
        "caption": caption,
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        tg_request("sendPhoto", token, data)
    except Exception as e:
        # Fallback to text if photo fails
        print(f"  sendPhoto failed: {e}, falling back to text", file=sys.stderr)
        send_message(token, chat_id, caption, disable_preview=True)
