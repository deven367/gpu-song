"""Apple Silicon GPU usage sampling via powermetrics."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

_ACTIVE_RE = re.compile(r"GPU HW active residency:\s*([\d.]+)%")
_IDLE_RE = re.compile(r"GPU idle residency:\s*([\d.]+)%")


class GPUSampleError(RuntimeError):
    """Raised when GPU usage cannot be sampled."""


def parse_powermetrics_output(text: str) -> float:
    """Parse GPU utilization percent from powermetrics gpu_power output."""
    match = _ACTIVE_RE.search(text)
    if match:
        return _clamp(float(match.group(1)))

    match = _IDLE_RE.search(text)
    if match:
        return _clamp(100.0 - float(match.group(1)))

    raise GPUSampleError(
        "Could not find GPU residency in powermetrics output. "
        "Is this an Apple Silicon Mac?"
    )


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def sample_once(interval_ms: int = 500, timeout: float | None = None) -> float:
    """Run one powermetrics sample and return GPU usage 0–100."""
    if timeout is None:
        timeout = max(5.0, interval_ms / 1000.0 + 3.0)

    cmd = [
        "powermetrics",
        "--samplers",
        "gpu_power",
        "-i",
        str(interval_ms),
        "-n",
        "1",
        "--hide-cpu-duty-cycle",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GPUSampleError(
            "powermetrics not found. This tool requires macOS."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise GPUSampleError("powermetrics timed out.") from exc

    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        lowered = combined.lower()
        if "superuser" in lowered or "root" in lowered or result.returncode == 1:
            raise GPUSampleError(
                "powermetrics requires root. Re-run with sudo, e.g.\n"
                "  sudo uv run gpu-song"
            )
        raise GPUSampleError(
            f"powermetrics failed (exit {result.returncode}): {combined.strip()}"
        )

    return parse_powermetrics_output(combined)


@dataclass
class GPUSampler:
    """Background poller that keeps the latest GPU usage percent."""

    interval_ms: int = 500
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _usage: float = field(default=0.0, init=False)
    _error: str | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    @property
    def usage(self) -> float:
        with self._lock:
            return self._usage

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self._thread is not None:
            return
        # Fail fast on first sample so callers get a clear sudo message.
        usage = sample_once(self.interval_ms)
        with self._lock:
            self._usage = usage
            self._error = None
        self._thread = threading.Thread(
            target=self._run, name="gpu-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                usage = sample_once(self.interval_ms)
                with self._lock:
                    self._usage = usage
                    self._error = None
            except GPUSampleError as exc:
                with self._lock:
                    self._error = str(exc)
                # Brief pause before retry so we don't spin on hard failures.
                self._stop.wait(1.0)
            except Exception as exc:  # noqa: BLE001 — keep drone alive
                with self._lock:
                    self._error = f"Unexpected sampler error: {exc}"
                self._stop.wait(1.0)
            # sample_once already blocks for ~interval_ms; tiny yield only.
            time.sleep(0.01)
