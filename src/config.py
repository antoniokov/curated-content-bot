"""Project configuration: paths, constants, and data loading."""

import csv
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours in seconds
SIMILARITY_THRESHOLD = 0.45    # minimum cosine similarity to consider relevant


def load_env():
    """Load key=value pairs from .env file."""
    env_path = os.path.join(PROJECT_DIR, ".env")
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env


def load_creators(csv_filename="creators.csv"):
    """Load creators from a CSV file, split by type."""
    csv_path = os.path.join(DATA_DIR, csv_filename)
    with open(csv_path) as f:
        creators = list(csv.DictReader(f))
    youtube = [c for c in creators if c["type"] == "YouTube channel" and c.get("channel_id")]
    podcasts = [c for c in creators if c["type"] == "Podcast" and c.get("url")]
    return youtube, podcasts
