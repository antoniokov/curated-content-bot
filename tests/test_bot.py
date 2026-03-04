"""Tests for the core bot pipeline: load → fetch → cache → embed → search."""

import csv
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR, load_creators, MAX_RESULTS
from src.utils import strip_html, truncate, format_date
from src.youtube import fetch_channel_videos, search_youtube_cache
from src.podcast import fetch_rss_episodes, search_all_podcasts
from src.main import merge_search_results, send_search_results

# --- Shared fixtures ---

MODEL = None


def get_model():
    """Load embedding model once across all tests."""
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return MODEL


def encode_and_normalize(texts):
    """Encode texts and L2-normalize (same as the bot does)."""
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
                "publishedAt": "2024-03-15T10:00:00Z",
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
                "publishedAt": "2024-02-20T08:30:00Z",
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
      <pubDate>Fri, 15 Mar 2024 10:00:00 +0000</pubDate>
      <link>https://example.com/ep/ml-fundamentals</link>
      <itunes:image href="https://example.com/ep1.jpg"/>
    </item>
    <item>
      <title>Best Coffee Brewing Methods</title>
      <description>Pour-over vs French press vs espresso.</description>
      <pubDate>Wed, 20 Feb 2024 08:30:00 +0000</pubDate>
      <link>https://example.com/ep/coffee</link>
    </item>
  </channel>
</rss>"""


# --- Tests ---


def test_load_creators():
    """CSV parsing splits YouTube/podcast correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir=DATA_DIR) as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "url", "channel_id", "apple_podcasts_id"])
        writer.writerow(["YouTube channel", "Chan A", "https://yt.com/@a", "UCxxxxA", ""])
        writer.writerow(["YouTube channel", "Chan B", "https://yt.com/@b", "UCxxxxB", ""])
        writer.writerow(["Podcast", "Pod X", "https://feed.example.com/x", "", "12345"])
        tmp_name = os.path.basename(f.name)
    try:
        yt, pods = load_creators(tmp_name)
        assert len(yt) == 2
        assert len(pods) == 1
        assert yt[0]["name"] == "Chan A"
        assert yt[0]["channel_id"] == "UCxxxxA"
        assert pods[0]["name"] == "Pod X"
        assert pods[0]["url"] == "https://feed.example.com/x"
    finally:
        os.unlink(os.path.join(DATA_DIR, tmp_name))


def test_fetch_channel_videos():
    """YouTube API response → video dicts with all required fields."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(YOUTUBE_API_RESPONSE).encode()

    with patch("urllib.request.urlopen", return_value=mock_resp):
        videos = fetch_channel_videos("UCxxxxA", "fake_key")

    assert len(videos) == 2
    v = videos[0]
    assert v["title"] == "How Neural Networks Learn"
    assert v["video_id"] == "vid_abc"
    assert v["url"] == "https://www.youtube.com/watch?v=vid_abc"
    assert "backpropagation" in v["description"]
    assert v["thumbnail"] == "https://img.youtube.com/vi/vid_abc/hq.jpg"
    assert v["published_at"] == "2024-03-15"
    # Second video falls back to medium thumbnail
    assert videos[1]["thumbnail"] == "https://img.youtube.com/vi/vid_def/mq.jpg"
    assert videos[1]["published_at"] == "2024-02-20"


def test_fetch_channel_videos_incremental():
    """known_ids stops pagination early — known video is excluded."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(YOUTUBE_API_RESPONSE).encode()

    with patch("urllib.request.urlopen", return_value=mock_resp):
        videos = fetch_channel_videos("UCxxxxA", "fake_key", known_ids={"vid_def"})

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
        episodes = fetch_rss_episodes("https://feed.example.com/test")

    assert len(episodes) == 2
    ep = episodes[0]
    assert ep["title"] == "Machine Learning Fundamentals"
    assert ep["link"] == "https://example.com/ep/ml-fundamentals"
    assert ep["thumbnail"] == "https://example.com/ep1.jpg"
    assert ep["published_at"] == "2024-03-15"
    # HTML stripped and entities decoded
    assert "<p>" not in ep["description"]
    assert "<b>" not in ep["description"]
    assert "&amp;" not in ep["description"]
    assert "ML" in ep["description"]
    assert "&" in ep["description"]
    # Second episode falls back to channel-level itunes:image
    assert episodes[1]["thumbnail"] == "https://example.com/pod.jpg"
    assert episodes[1]["published_at"] == "2024-02-20"


def test_strip_html():
    """HTML tags stripped, entities decoded."""
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert strip_html("&amp; &lt;test&gt;") == "& <test>"
    assert strip_html("no tags here") == "no tags here"


def test_format_date():
    """Date string formatted as 'Jan 15, 2024'."""
    assert format_date("2024-03-15") == "Mar 15, 2024"
    assert format_date("2024-01-01") == "Jan 1, 2024"
    assert format_date("") == ""
    assert format_date(None) == ""
    assert format_date("not-a-date") == ""


def test_truncate():
    """Truncation at word boundary."""
    short = "hello world"
    assert truncate(short, 200) == short

    long_text = "word " * 100  # 500 chars
    result = truncate(long_text, 50)
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

    # Build embeddings the same way the bot does
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

    with patch("src.youtube.get_embed_model", return_value=get_model()):
        results = search_youtube_cache("deep learning neural networks", channels, embeddings, index, max_total=3)

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

    with patch("src.youtube.get_embed_model", return_value=get_model()):
        results = search_youtube_cache("Zarquon", channels, embeddings, index)

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

    with patch("src.podcast.get_embed_model", return_value=get_model()):
        results = search_all_podcasts(
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

    with patch("src.podcast.get_embed_model", return_value=get_model()):
        results = search_all_podcasts(
            "zarquon", podcasts,
            feeds=feeds, embeddings=embeddings, index=index)

    assert len(results) > 0
    assert results[0]["episodes"][0]["title"] == "The Zarquon Incident"


def test_merge_search_results_ranks_across_sources():
    """Merge ranks all items by similarity and respects max_results limit."""
    yt_results = [
        {"creator": "AI Channel", "videos": [
            {"title": "Video A", "_similarity": 0.9, "url": "https://yt.com/a"},
            {"title": "Video B", "_similarity": 0.6, "url": "https://yt.com/b"},
        ]},
        {"creator": "ML Channel", "videos": [
            {"title": "Video C", "_similarity": 0.5, "url": "https://yt.com/c"},
        ]},
    ]
    pod_results = [
        {"creator": "AI Podcast", "apple_podcasts_id": "123", "episodes": [
            {"title": "Episode X", "_similarity": 0.85, "link": "https://pod.com/x"},
            {"title": "Episode Y", "_similarity": 0.55, "link": "https://pod.com/y"},
        ]},
    ]

    # With max_results=3, should get: Video A (0.9), Episode X (0.85), Video B (0.6)
    results = merge_search_results(yt_results, pod_results, max_results=3)

    total_items = sum(
        len(g.get("videos", [])) + len(g.get("episodes", []))
        for g in results
    )
    assert total_items == 3

    # Video A and Episode X should be in results (top 2 by similarity)
    all_titles = []
    for g in results:
        all_titles.extend(v["title"] for v in g.get("videos", []))
        all_titles.extend(e["title"] for e in g.get("episodes", []))
    assert "Video A" in all_titles
    assert "Episode X" in all_titles
    assert "Video B" in all_titles
    # Video C (0.5) and Episode Y (0.55) should be cut
    assert "Video C" not in all_titles
    assert "Episode Y" not in all_titles


def test_merge_search_results_preserves_group_structure():
    """Merged results are grouped by source+creator with correct fields."""
    yt_results = [
        {"creator": "Chan", "videos": [
            {"title": "V1", "_similarity": 0.8, "url": "https://yt.com/1", "video_id": "v1",
             "description": "desc", "thumbnail": "https://img/1.jpg"},
        ]},
    ]
    pod_results = [
        {"creator": "Pod", "apple_podcasts_id": "99", "episodes": [
            {"title": "E1", "_similarity": 0.7, "link": "https://pod.com/1",
             "description": "desc", "thumbnail": "https://img/e1.jpg"},
        ]},
    ]

    results = merge_search_results(yt_results, pod_results, max_results=10)
    assert len(results) == 2

    yt_group = next(g for g in results if g["source"] == "youtube")
    assert yt_group["creator"] == "Chan"
    assert len(yt_group["videos"]) == 1

    pod_group = next(g for g in results if g["source"] == "podcast")
    assert pod_group["creator"] == "Pod"
    assert pod_group["apple_podcasts_id"] == "99"
    assert len(pod_group["episodes"]) == 1


def test_published_dates_flow_from_search_to_telegram():
    """End-to-end: published dates survive search → merge → display for both sources."""
    # YouTube channel with published_at (as built by fetch_channel_videos)
    yt_videos = [
        {"title": "How Neural Networks Learn", "video_id": "v1",
         "url": "https://youtube.com/watch?v=v1", "published_at": "2024-03-15",
         "description": "Backpropagation and gradient descent", "thumbnail": "https://img/v1.jpg"},
        {"title": "Cooking Italian Pasta", "video_id": "v2",
         "url": "https://youtube.com/watch?v=v2", "published_at": "2024-02-20",
         "description": "Traditional recipes from Tuscany", "thumbnail": "https://img/v2.jpg"},
    ]
    channels = {
        "https://yt.com/@ai": {"name": "AI Channel", "channel_id": "UC1", "videos": yt_videos},
    }

    # Podcast feed with published_at (as built by fetch_rss_episodes)
    pod_episodes = [
        {"title": "Machine Learning Fundamentals", "description": "Intro to ML and deep learning",
         "link": "https://example.com/ep/ml", "thumbnail": "https://img/ml.jpg",
         "published_at": "2024-01-10"},
        {"title": "Best Pizza in New York", "description": "Food tour of NYC pizzerias",
         "link": "https://example.com/ep/pizza", "thumbnail": "https://img/pizza.jpg",
         "published_at": "2024-01-05"},
    ]
    feeds = {
        "https://feed.example.com/ai": {
            "name": "AI Podcast", "apple_podcasts_id": "123", "episodes": pod_episodes,
        },
    }

    # Build embeddings for YouTube
    yt_texts = []
    yt_index = []
    for ch_url, ch_data in channels.items():
        for i, vid in enumerate(ch_data["videos"]):
            yt_texts.append(vid["title"] + ". " + vid["description"])
            yt_index.append((ch_url, i))
    yt_embeddings = encode_and_normalize(yt_texts)

    # Build embeddings for podcasts
    pod_texts = []
    pod_index = []
    for feed_url, feed_data in feeds.items():
        for i, ep in enumerate(feed_data["episodes"]):
            pod_texts.append(ep["title"] + ". " + ep["description"])
            pod_index.append((feed_url, i))
    pod_embeddings = encode_and_normalize(pod_texts)

    model = get_model()

    # Step 1: Search both sources
    with patch("src.youtube.get_embed_model", return_value=model):
        yt_results = search_youtube_cache("neural networks", channels, yt_embeddings, yt_index)

    with patch("src.podcast.get_embed_model", return_value=model):
        pod_results = search_all_podcasts("machine learning", [],
            feeds=feeds, embeddings=pod_embeddings, index=pod_index)

    # Step 2: Merge
    results = merge_search_results(yt_results, pod_results)

    # Step 3: Send to Telegram (mocked)
    with patch("src.main.send_message") as mock_msg, \
         patch("src.main.send_video_url") as mock_video, \
         patch("src.main.send_photo") as mock_photo:

        send_search_results("token", 123, results)

        # YouTube: date must be in the video URL message
        video_calls = mock_video.call_args_list
        assert len(video_calls) >= 1, "No YouTube video messages sent"
        video_text = video_calls[0].args[2]
        assert "Mar 15, 2024" in video_text, (
            f"YouTube message missing date. Got: {video_text!r}")

        # Podcast: date must be in the photo caption
        photo_calls = mock_photo.call_args_list
        assert len(photo_calls) >= 1, "No podcast photo messages sent"
        caption = photo_calls[0].kwargs["caption"]
        assert "Jan 10, 2024" in caption, (
            f"Podcast caption missing date. Got: {caption!r}")
