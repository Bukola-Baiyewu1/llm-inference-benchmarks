"""Shared helpers: environment info, timing, percentiles and result files."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    """Hardware and library versions, saved next to every result so it can be reproduced."""
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "git_commit": git_commit(),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def percentiles(values: list[float], qs=(50, 95, 99)) -> dict:
    arr = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def time_fn(fn, warmup: int = 3, repeats: int = 20, cuda: bool = False) -> list[float]:
    """Run ``fn`` several times and return each run's wall time in seconds.

    GPU work is asynchronous: the CPU queues kernels and moves on. So on CUDA we
    synchronise before reading the clock, otherwise we'd time the queueing, not the work.
    """
    sync = None
    if cuda:
        import torch
        sync = torch.cuda.synchronize
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return times


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path
