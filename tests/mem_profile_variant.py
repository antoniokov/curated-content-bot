#!/usr/bin/env python3
"""Run a single memory-profiling variant for the embedding pipeline.

Usage: python3 tests/mem_profile_variant.py <variant> <output_json>
Variants: baseline, batch32, gc_between, unload_between, all_fixes, subprocess_iso
"""

import gc
import json
import os
import random
import resource
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_YOUTUBE_VIDEOS = 19168
NUM_PODCAST_EPISODES = 5000
SAMPLE_INTERVAL = 0.5  # seconds


def get_current_rss_mb():
    """Get current RSS in MB using ps (works on macOS)."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        ).strip()
        return int(out) / 1024
    except Exception:
        return 0


class MemoryMonitor:
    def __init__(self):
        self.samples = []
        self._stop = False
        self._thread = None
        self._start_time = None

    def start(self):
        self._start_time = time.time()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop:
            elapsed = time.time() - self._start_time
            rss = get_current_rss_mb()
            self.samples.append({"t": round(elapsed, 1), "rss_mb": round(rss, 1)})
            time.sleep(SAMPLE_INTERVAL)

    @property
    def peak_mb(self):
        return max(s["rss_mb"] for s in self.samples) if self.samples else 0


def generate_texts(n):
    """Generate synthetic texts similar to real video/podcast titles + descriptions."""
    random.seed(42)
    words = (
        "the quick brown fox jumps over lazy dog and then runs away fast into "
        "forest where birds sing loudly every morning while sun rises behind "
        "mountains creating beautiful scenery that inspires artists and musicians "
        "around world to create amazing works of art and music for everyone"
    ).split()
    texts = []
    for _ in range(n):
        # Title (~10 words) + description (~40 words) = ~250 chars avg
        title_len = random.randint(5, 15)
        desc_len = random.randint(20, 60)
        title = " ".join(random.choices(words, k=title_len))
        desc = " ".join(random.choices(words, k=desc_len))
        texts.append(f"{title}. {desc}")
    return texts


def _get_child_peak_rss_mb(pid):
    """Get RSS of a child process in MB using ps."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True
        ).strip()
        return int(out) / 1024
    except Exception:
        return 0


class ChildMemoryMonitor:
    """Monitor RSS of a child process by PID."""
    def __init__(self, pid):
        self.pid = pid
        self.samples = []
        self._stop = False
        self._thread = None
        self._start_time = None

    def start(self):
        self._start_time = time.time()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop:
            elapsed = time.time() - self._start_time
            rss = _get_child_peak_rss_mb(self.pid)
            self.samples.append({"t": round(elapsed, 1), "rss_mb": round(rss, 1)})
            time.sleep(SAMPLE_INTERVAL)

    @property
    def peak_mb(self):
        return max(s["rss_mb"] for s in self.samples) if self.samples else 0


def run_subprocess_iso(variant):
    """Run with subprocess isolation: each phase in its own child process.

    The parent monitors child RSS. When the child exits, the OS reclaims all
    its memory, so the two phases don't accumulate.
    """
    import numpy as np

    t0 = time.time()
    events = []
    all_samples = []
    phase_peaks = []

    def event(name, rss=None):
        elapsed = time.time() - t0
        if rss is None:
            rss = get_current_rss_mb()
        events.append({"t": round(elapsed, 1), "event": name, "rss_mb": round(rss, 1)})
        print(f"  [{elapsed:6.1f}s] {name} (RSS: {rss:.0f} MB)")

    event("start")

    batch_size = 32 if "batch32" in variant else 128

    # Run each phase as a child process
    phases = [
        ("youtube", NUM_YOUTUBE_VIDEOS, batch_size),
        ("podcast", NUM_PODCAST_EPISODES, batch_size),
    ]
    for phase_name, num_texts, batch_size in phases:
        event(f"{phase_name}_subprocess_starting")

        # Spawn child that does: generate texts -> load model -> encode -> exit
        child_script = f"""
import gc, json, os, random, resource, sys, time
sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))})
from tests.mem_profile_variant import generate_texts, get_current_rss_mb
import numpy as np
from src.embeddings import get_embed_model

events = []
t0 = time.time()
def ev(name):
    elapsed = time.time() - t0
    rss = get_current_rss_mb()
    events.append({{"t": round(elapsed, 1), "event": name, "rss_mb": round(rss, 1)}})

ev("child_start")
texts = generate_texts({num_texts})
ev("texts_generated")
model = get_embed_model()
ev("model_loaded")
embeddings = model.encode(texts, show_progress_bar=True, batch_size={batch_size})
ev("encoded")
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
embeddings = embeddings / norms
ev("normalized")
# In real code, save to .npz here
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
ev("done")
json.dump({{"events": events, "peak_mb": round(peak, 1)}}, sys.stdout)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", child_script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # Monitor child RSS from parent
        child_monitor = ChildMemoryMonitor(proc.pid)
        child_monitor.start()

        stdout, stderr = proc.communicate()
        child_monitor.stop()

        child_peak_from_monitor = child_monitor.peak_mb

        # Parse child's self-reported data
        try:
            child_data = json.loads(stdout)
            child_peak_self = child_data["peak_mb"]
            for ce in child_data["events"]:
                events.append({
                    "t": round(time.time() - t0, 1),
                    "event": f"{phase_name}:{ce['event']}",
                    "rss_mb": ce["rss_mb"],
                })
        except (json.JSONDecodeError, KeyError):
            child_peak_self = 0

        # Use the higher of the two measurements
        child_peak = max(child_peak_from_monitor, child_peak_self)
        phase_peaks.append(child_peak)

        # Offset child monitor samples by parent elapsed time
        phase_start = time.time() - t0 - (child_monitor.samples[-1]["t"] if child_monitor.samples else 0)
        for s in child_monitor.samples:
            all_samples.append({"t": round(phase_start + s["t"], 1), "rss_mb": s["rss_mb"]})

        event(f"{phase_name}_subprocess_done (child peak: {child_peak:.0f} MB)", rss=get_current_rss_mb())

    total_time = time.time() - t0
    # Peak is max of individual phase peaks (not cumulative!)
    peak = max(phase_peaks)

    event("done")

    return {
        "variant": variant,
        "config": {"batch_size": batch_size, "subprocess_isolation": True},
        "peak_rss_mb": round(peak, 1),
        "total_time_s": round(total_time, 1),
        "yt_time_s": 0,  # tracked per-child
        "passed": peak <= 1024,
        "events": events,
        "samples": all_samples,
        "phase_peaks": [round(p, 1) for p in phase_peaks],
    }


def run(variant):
    if variant.startswith("subprocess_iso"):
        return run_subprocess_iso(variant)

    import numpy as np
    import src.embeddings as emb_mod
    from src.embeddings import get_embed_model

    # Parse variant config
    configs = {
        "baseline":        {"batch_size": 128, "gc_between": False, "unload_between": False},
        "batch32":         {"batch_size": 32,  "gc_between": False, "unload_between": False},
        "gc_between":      {"batch_size": 128, "gc_between": True,  "unload_between": False},
        "unload_between":  {"batch_size": 128, "gc_between": True,  "unload_between": True},
        "all_fixes":       {"batch_size": 32,  "gc_between": True,  "unload_between": True},
    }
    cfg = configs[variant]
    batch_size = cfg["batch_size"]

    gc.collect()
    monitor = MemoryMonitor()
    monitor.start()
    events = []

    def event(name):
        elapsed = time.time() - t0
        rss = get_current_rss_mb()
        events.append({"t": round(elapsed, 1), "event": name, "rss_mb": round(rss, 1)})
        print(f"  [{elapsed:6.1f}s] {name} (RSS: {rss:.0f} MB)")

    t0 = time.time()
    event("start")

    # --- Phase 1: YouTube ---
    yt_texts = generate_texts(NUM_YOUTUBE_VIDEOS)
    event(f"youtube_texts_generated ({len(yt_texts)} texts)")

    model = get_embed_model()
    event("model_loaded")

    yt_embeddings = model.encode(yt_texts, show_progress_bar=True, batch_size=batch_size)
    event("youtube_encoded")

    norms = np.linalg.norm(yt_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    yt_embeddings = yt_embeddings / norms
    event("youtube_normalized")

    # Simulate saving (real code saves to .npz, freeing is implicit after function returns)
    del yt_texts, yt_embeddings, norms
    event("youtube_cleanup")

    yt_time = time.time() - t0

    # --- Between phases ---
    if cfg["unload_between"]:
        emb_mod._embed_model = None
        event("model_unloaded")

    if cfg["gc_between"]:
        gc.collect()
        event("gc_collected")

    # --- Phase 2: Podcasts ---
    pod_texts = generate_texts(NUM_PODCAST_EPISODES)
    event(f"podcast_texts_generated ({len(pod_texts)} texts)")

    if cfg["unload_between"]:
        model = get_embed_model()
        event("model_reloaded")
    else:
        model = get_embed_model()  # no-op, already loaded

    pod_embeddings = model.encode(pod_texts, show_progress_bar=True, batch_size=batch_size)
    event("podcast_encoded")

    norms = np.linalg.norm(pod_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    pod_embeddings = pod_embeddings / norms
    event("podcast_normalized")

    del pod_texts, pod_embeddings, norms
    event("podcast_cleanup")

    total_time = time.time() - t0

    # Final cleanup
    emb_mod._embed_model = None
    gc.collect()

    monitor.stop()
    event("done")

    peak = monitor.peak_mb
    return {
        "variant": variant,
        "config": cfg,
        "peak_rss_mb": round(peak, 1),
        "total_time_s": round(total_time, 1),
        "yt_time_s": round(yt_time, 1),
        "passed": peak <= 1024,
        "events": events,
        "samples": monitor.samples,
    }


if __name__ == "__main__":
    variant = sys.argv[1]
    output_path = sys.argv[2]
    print(f"Running variant: {variant}")
    result = run(variant)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n{status}: peak={result['peak_rss_mb']:.0f} MB, time={result['total_time_s']:.1f}s")
