"""Device configuration, memory release and progress logging for final training."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
REGISTRY = LOGS / "training.jsonl"
DEVICE = os.environ.get("PRC_DEVICE", "cpu").strip().lower()
if DEVICE not in {"cpu", "gpu"}:
    raise SystemExit(f"PRC_DEVICE must be cpu or gpu (received {DEVICE!r})")

try:
    _trim = ctypes.CDLL("libc.so.6").malloc_trim
    _trim.argtypes = [ctypes.c_size_t]
except (OSError, AttributeError):
    _trim = None


def release_memory() -> None:
    """Collect unreachable objects and return unused glibc heap pages."""
    gc.collect()
    if _trim is not None:
        _trim(0)


def src_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def gpu_usage() -> tuple[int, int] | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
        ).stdout
        utilization, memory = output.splitlines()[0].split(", ")
        return int(utilization), int(memory)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class Run:
    """Record training progress without reading an experiment registry."""

    def __init__(self, name: str, config: dict, registry: Path = REGISTRY,
                 logs: Path = LOGS, sample_s: float = 30.0) -> None:
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        self.rec = {"id": self.id, "name": name, "config": config,
                    "src_hash": src_hash(), "device": DEVICE, "phases": {}}
        self._registry = registry
        registry.parent.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        self._log = (logs / f"{self.id}.log").open("a")
        self._proc = psutil.Process()
        self._proc.cpu_percent()
        self._started = time.perf_counter()
        self._sample_s = sample_s
        self._stop = threading.Event()
        self._done = self._weight = self._fraction = self._rss_peak = 0.0
        self._phase = "start"
        self._gpu = None

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._log.write(message + "\n")
        self._log.flush()

    def _resources(self) -> str:
        rss = self._proc.memory_info().rss / 2**30
        self._rss_peak = max(self._rss_peak, rss)
        text = f"RAM {rss:.1f} GB | CPU {self._proc.cpu_percent():.0f}%"
        if self._gpu:
            text += f" | GPU {self._gpu[0]}% {self._gpu[1]} MB"
        return text

    def _status(self) -> str:
        total = 100 * (self._done + self._weight * self._fraction)
        return f"[{self._phase} {100 * self._fraction:3.0f}% | total {total:3.0f}%]"

    def _watch(self) -> None:
        while not self._stop.wait(self._sample_s):
            self._gpu = gpu_usage()
            elapsed = duration(time.perf_counter() - self._started)
            self.log(f"{self._status()} {self._resources()} | elapsed {elapsed}")

    def __enter__(self) -> Run:
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        self.log(f"== {self.id} (source {self.rec['src_hash']}, LightGBM {DEVICE.upper()})")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self._resources()
        self.rec["total_s"] = round(time.perf_counter() - self._started, 1)
        kernel_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        self.rec["peak_rss_gb"] = round(max(self._rss_peak, kernel_peak), 2)
        self.rec["ok"] = exc_type is None
        if exc is not None:
            self.rec["error"] = repr(exc)
        with self._registry.open("a") as stream:
            stream.write(json.dumps(self.rec) + "\n")
        self.log(f"== finished {self.id}: {duration(self.rec['total_s'])} | ok={self.rec['ok']}")
        self._log.close()

    @contextmanager
    def phase(self, label: str, weight: float):
        self._phase, self._weight, self._fraction = label, weight, 0.0
        started = time.perf_counter()
        self.log(f"{self._status()} {label}...")
        yield
        self._fraction = 1.0
        elapsed = time.perf_counter() - started
        self.rec["phases"][label] = round(elapsed, 1)
        self.log(f"{self._status()} {label} completed in {duration(elapsed)} | {self._resources()}")
        self._done += weight
        self._weight, self._fraction = 0.0, 0.0

    def progress(self, fraction: float, detail: str) -> None:
        self._fraction = min(max(fraction, 0.0), 1.0)
        self.log(f"{self._status()} {detail} | {self._resources()}")

    def lgb_callback(self, rounds: int, label: str = "round", every: int = 50,
                     start: float = 0.0, span: float = 1.0):
        started = time.perf_counter()

        def callback(environment) -> None:
            iteration = environment.iteration + 1
            if iteration % every and iteration != rounds:
                return
            elapsed = time.perf_counter() - started
            self.progress(start + span * iteration / rounds,
                          f"{label} {iteration}/{rounds} | ETA "
                          f"{duration(elapsed * (rounds - iteration) / iteration)}")

        callback.order = 30
        return callback

    def set(self, **values) -> None:
        self.rec.update(values)