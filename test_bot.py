"""Tests for the core bot pipeline: load → fetch → cache → embed → search."""

import csv
import io
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

import bot

# --- Shared fixtures ---

MODEL = None


def get_model():
    """Load embedding model once across all tests."""
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return MODEL


def encode_and_normalize(texts):
    """Encode texts and L2-normalize (same as bot.py does)."""
    model = get_model()
    emb = model.encode(texts)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return emb / norms


YOUTUBE_API_RESPONSE = {
    "items": [
        {
            "snippet": {
                "title": "How Neural Networks Learn",
                "resourceId": {"videoId": "vid_abc"},
                "description": "A deep dive into backpropagation and gradient descent.",
                "thumbnails": {
                    "high": {"url": "https://img.youtube.com/vi/vid_abc/hq.jpg"}
                },
            }
        },
        {
            "snippet": {
                "title": "Cooking Italian Pasta",
                "resourceId": {"videoId": "vid_def"},
                "description": "Traditional recipes from Tuscany.",
                "thumbnails": {
                    "medium": {"url": "https://img.youtube.com/vi/vid_def/mq.jpg"}
                },
            }
        },
    ],
    "nextPageToken": None,
}

RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Test Pod</title>
    <itunes:image href="https://example.com/pod.jpg"/>
    <item>
      <title>Machine Learning Fundamentals</title>
      <description>&lt;p&gt;An intro to &lt;b&gt;ML&lt;/b&gt; &amp; deep learning.&lt;/p&gt;</description>
      <link>https://example.com/ep/ml-fundamentals</link>
      <itunes:image href="https://example.com/ep1.jpg"/>
    </item>
    <item>
      <title>Best Coffee Brewing Methods</title>
      <description>Pour-over vs French press vs espresso.</description>
      <link>https://example.com/ep/coffee</link>
    </item>
  </channel>
</rss>"""


# --- Tests ---


def test_load_creators():
    """CSV parsing splits YouTube/podcast correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir=bot.SCRIPT_DIR) as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "url", "channel_id", "apple_podcasts_id"])
        writer.writerow(["YouTube channel", "Chan A", "https://yt.com/@a", "UCxxxxA", ""])
        writer.writerow(["YouTube channel", "Chan B", "https://yt.com/@b", "UCxxxxB", ""])
        writer.writerow(["Podcast", "Pod X", "https://feed.example.com/x", "", "12345"])
        tmp_name = os.path.basename(f.name)
    try:
        yt, pods = bot.load_creators(tmp_name)
        assert len(yt) == 2
        assert len(pods) == 1
        assert yt[0]["name"] == "Chan A"
        assert yt[0]["channel_id"] == "UCxxxxA"
        assert pods[0]["name"] == "Pod X"
        assert pods[0]["url"] == "https://feed.example.com/x"
    finally:
        os.unlink(os.path.join(bot.SCRIPT_DIR, tmp_name))


def test_fetch_channel_videos():
    """YouTube API response → video dicts with all required fields."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(YOUTUBE_API_RESPONSE).encode()

    with patch("urllib.request.urlopen", return_value=mock_resp):
        videos = bot.fetch_channel_videos("UCxxxxA", "fake_key")

    assert len(videos) == 2
    v = videos[0]
    assert v["title"] == "How Neural Networks Learn"
    assert v["video_id"] == "vid_abc"
    assert v["url"] == "https://www.youtube.com/watch?v=vid_abc"
    assert "backpropagation" in v["description"]
    assert v["thumbnail"] == "https://img.youtube.com/vi/vid_abc/hq.jpg"
    # Second video falls back to medium thumbnail
    assert videos[1]["thumbnail"] == "https://img.youtube.com/vi/vid_def/mq.jpg"


def test_fetch_channel_videos_incremental():
    """known_ids stops pagination early — known video is excluded."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(YOUTUBE_API_RESPONSE).encode()

    with patch("urllib.request.urlopen", return_value=mock_resp):
        videos = bot.fetch_channel_videos("UCxxxxA", "fake_key", known_ids={"vid_def"})

    # Should stop at vid_def, only vid_abc returned
    assert len(videos) == 1
    assert videos[0]["video_id"] == "vid_abc"


def test_fetch_rss_episodes():
    """RSS XML → episode dicts with all required fields."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = RSS_XML
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        episodes = bot.fetch_rss_episodes("https://feed.example.com/test")

    assert len(episodes) == 2
    ep = episodes[0]
    assert ep["title"] == "Machine Learning Fundamentals"
    assert ep["link"] == "https://example.com/ep/ml-fundamentals"
    assert ep["thumbnail"] == "https://example.com/ep1.jpg"
    # HTML stripped and entities decoded
    assert "<p>" not in ep["description"]
    assert "<b>" not in ep["description"]
    assert "&amp;" not in ep["description"]
    assert "ML" in ep["description"]
    assert "&" in ep["description"]
    # Second episode falls back to channel-level itunes:image
    assert episodes[1]["thumbnail"] == "https://example.com/pod.jpg"


def test_strip_html():
    """HTML tags stripped, entities decoded."""
    assert bot._strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert bot._strip_html("&amp; &lt;test&gt;") == "& <test>"
    assert bot._strip_html("no tags here") == "no tags here"


def test_truncate():
    """Truncation at word boundary."""
    short = "hello world"
    assert bot._truncate(short, 200) == short

    long_text = "word " * 100  # 500 chars
    result = bot._truncate(long_text, 50)
    assert len(result) <= 51  # 50 + ellipsis char
    assert result.endswith("…")
    assert "  " not in result  # clean word boundary


def test_search_youtube_cache_semantic():
    """Semantic search returns results grouped by creator, with all video fields."""
    videos = [
        {"title": "How Neural Networks Learn", "video_id": "v1",
         "url": "https://youtube.com/watch?v=v1",
         "description": "Backpropagation and gradient descent", "thumbnail": "https://img/v1.jpg"},
        {"title": "Cooking Pasta from Scratch", "video_id": "v2",
         "url": "https://youtube.com/watch?v=v2",
         "description": "Italian recipes", "thumbnail": "https://img/v2.jpg"},
        {"title": "Transformer Architecture Explained", "video_id": "v3",
         "url": "https://youtube.com/watch?v=v3",
         "description": "Attention mechanism in deep learning", "thumbnail": "https://img/v3.jpg"},
    ]
    channels = {
        "https://yt.com/@ai": {"name": "AI Channel", "channel_id": "UC1", "videos": videos[:2]},
        "https://yt.com/@ml": {"name": "ML Channel", "channel_id": "UC2", "videos": [videos[2]]},
    }

    # Build embeddings the same way bot.py does
    texts = []
    index = []
    for ch_url, ch_data in channels.items():
        for i, vid in enumerate(ch_data["videos"]):
            embed_text = vid["title"]
            if vid.get("description"):
                embed_text += ". " + vid["description"]
            texts.append(embed_text)
            index.append((ch_url, i))

    embeddings = encode_and_normalize(texts)

    with patch.object(bot, "_get_embed_model", return_value=get_model()):
        results = bot.search_youtube_cache("deep learning neural networks", channels, embeddings, index, max_total=3)

    assert len(results) > 0
    # Should find ML-related videos, not cooking
    all_video_titles = [v["title"] for g in results for v in g["videos"]]
    assert any("Neural" in t or "Transformer" in t for t in all_video_titles)
    assert "Cooking Pasta from Scratch" not in all_video_titles

    # Check grouping and required fields
    for group in results:
        assert "creator" in group
        assert "videos" in group
        for v in group["videos"]:
            for field in ("title", "video_id", "url", "description", "thumbnail"):
                assert field in v, f"Missing field: {field}"


def test_search_youtube_cache_keyword_fallback():
    """Falls back to keywords when no semantic match passes threshold."""
    videos = [
        {"title": "My Trip to Zarquon", "video_id": "v1",
         "url": "https://youtube.com/watch?v=v1",
         "description": "Visiting the planet Zarquon", "thumbnail": "https://img/v1.jpg"},
    ]
    channels = {
        "https://yt.com/@travel": {"name": "Travel", "channel_id": "UC1", "videos": videos},
    }

    # Embeddings about something completely unrelated to our query keyword
    texts = ["My Trip to Zarquon. Visiting the planet Zarquon"]
    index = [("https://yt.com/@travel", 0)]
    embeddings = encode_and_normalize(texts)

    with patch.object(bot, "_get_embed_model", return_value=get_model()):
        results = bot.search_youtube_cache("Zarquon", channels, embeddings, index)

    # Semantic search won't match the threshold, but keyword search should find "Zarquon"
    assert len(results) > 0
    assert results[0]["videos"][0]["title"] == "My Trip to Zarquon"


def test_search_all_podcasts_semantic():
    """Semantic search returns results grouped by podcast, with all episode fields."""
    feeds = {
        "https://feed.example.com/ai": {
            "name": "AI Podcast",
            "apple_podcasts_id": "123",
            "episodes": [
                {"title": "Machine Learning Fundamentals", "description": "Intro to ML and deep learning",
                 "link": "https://example.com/ep/ml", "thumbnail": "https://img/ml.jpg"},
                {"title": "Best Pizza in New York", "description": "Food tour of NYC pizzerias",
                 "link": "https://example.com/ep/pizza", "thumbnail": "https://img/pizza.jpg"},
            ],
        },
        "https://feed.example.com/dl": {
            "name": "Deep Learning Show",
            "apple_podcasts_id": "456",
            "episodes": [
                {"title": "Convolutional Neural Networks", "description": "CNNs for image recognition",
                 "link": "https://example.com/ep/cnn", "thumbnail": "https://img/cnn.jpg"},
            ],
        },
    }

    texts = []
    index = []
    for feed_url, feed_data in feeds.items():
        for i, ep in enumerate(feed_data["episodes"]):
            embed_text = ep["title"]
            if ep.get("description"):
                embed_text += ". " + ep["description"]
            texts.append(embed_text)
            index.append((feed_url, i))

    embeddings = encode_and_normalize(texts)
    podcasts = []  # not used when feeds are passed directly

    with patch.object(bot, "_get_embed_model", return_value=get_model()):
        results = bot.search_all_podcasts(
            "neural networks and deep learning", podcasts,
            feeds=feeds, embeddings=embeddings, index=index, max_total=3)

    assert len(results) > 0
    all_titles = [ep["title"] for g in results for ep in g["episodes"]]
    assert any("Neural" in t or "Machine Learning" in t for t in all_titles)
    assert "Best Pizza in New York" not in all_titles

    # Check grouping and required fields
    for group in results:
        assert "creator" in group
        assert "episodes" in group
        assert "apple_podcasts_id" in group
        for ep in group["episodes"]:
            for field in ("title", "description", "link", "thumbnail"):
                assert field in ep, f"Missing field: {field}"


def test_search_all_podcasts_keyword_fallback():
    """Falls back to keywords when no semantic match passes threshold."""
    feeds = {
        "https://feed.example.com/misc": {
            "name": "Misc Pod",
            "apple_podcasts_id": "",
            "episodes": [
                {"title": "The Zarquon Incident", "description": "A strange event on Zarquon",
                 "link": "https://example.com/ep/zarquon", "thumbnail": ""},
            ],
        },
    }

    texts = ["The Zarquon Incident. A strange event on Zarquon"]
    index = [("https://feed.example.com/misc", 0)]
    embeddings = encode_and_normalize(texts)
    podcasts = []

    with patch.object(bot, "_get_embed_model", return_value=get_model()):
        results = bot.search_all_podcasts(
            "zarquon", podcasts,
            feeds=feeds, embeddings=embeddings, index=index)

    assert len(results) > 0
    assert results[0]["episodes"][0]["title"] == "The Zarquon Incident"
