"""Project configuration: paths, constants, and data loading."""

import csv
import logging
import os
from logging.handlers import RotatingFileHandler

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours in seconds
SIMILARITY_THRESHOLD = 0.45    # minimum cosine similarity to consider relevant
MAX_RESULTS = 10               # top N results across all sources
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "bot.log")


def setup_logging(debug=False):
    """Configure logging to write to both stderr and a rotating log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


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

    # Parse ALLOWED_CHAT_IDS as a set of ints
    raw_ids = env.get("ALLOWED_CHAT_IDS", "")
    if raw_ids:
        env["ALLOWED_CHAT_IDS"] = {int(x.strip()) for x in raw_ids.split(",") if x.strip()}
    else:
        env["ALLOWED_CHAT_IDS"] = set()

    return env


def load_creators(csv_filename="creators.csv"):
    """Load creators from a CSV file, split by type."""
    csv_path = os.path.join(DATA_DIR, csv_filename)
    with open(csv_path) as f:
        creators = list(csv.DictReader(f))
    youtube = [c for c in creators if c["type"] == "YouTube channel" and c.get("channel_id")]
    podcasts = [c for c in creators if c["type"] == "Podcast" and c.get("url")]
    return youtube, podcasts
