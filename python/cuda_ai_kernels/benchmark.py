from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch


def cuda_event_benchmark(fn: Callable[[], Any], warmup: int = 50, repeat: int = 200) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return {
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

