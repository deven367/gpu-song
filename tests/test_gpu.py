"""Tests for powermetrics parsing and sampling."""

from __future__ import annotations

import subprocess
import threading
from unittest.mock import MagicMock, patch

import pytest

from gpu_song.gpu import (
    GPUSampleError,
    GPUSampler,
    parse_powermetrics_output,
    sample_once,
)

SAMPLE_OUTPUT = """
**** GPU usage ****
GPU HW active frequency: 444 MHz
GPU HW active residency:  22.25% (444 MHz:  22%)
GPU idle residency:  77.75%
GPU Power: 223 mW
"""


def test_parse_active_residency() -> None:
    assert parse_powermetrics_output(SAMPLE_OUTPUT) == pytest.approx(22.25)


def test_parse_idle_fallback() -> None:
    text = "GPU idle residency:  90.00%\n"
    assert parse_powermetrics_output(text) == pytest.approx(10.0)


def test_parse_clamps_high_values() -> None:
    text = "GPU HW active residency: 150.0%\n"
    assert parse_powermetrics_output(text) == 100.0


def test_parse_missing_raises() -> None:
    with pytest.raises(GPUSampleError, match="residency"):
        parse_powermetrics_output("no gpu here")


def test_sample_once_success() -> None:
    result = MagicMock(
        returncode=0,
        stdout=SAMPLE_OUTPUT,
        stderr="",
    )
    with patch("gpu_song.gpu.subprocess.run", return_value=result) as run:
        usage = sample_once(interval_ms=200)
    assert usage == pytest.approx(22.25)
    cmd = run.call_args.args[0]
    assert cmd[0] == "powermetrics"
    assert "-i" in cmd and "200" in cmd


def test_sample_once_requires_root() -> None:
    result = MagicMock(
        returncode=1,
        stdout="",
        stderr="powermetrics must be invoked as the superuser\n",
    )
    with (
        patch("gpu_song.gpu.subprocess.run", return_value=result),
        pytest.raises(GPUSampleError, match="requires root"),
    ):
        sample_once()


def test_sample_once_missing_binary() -> None:
    with (
        patch(
            "gpu_song.gpu.subprocess.run",
            side_effect=FileNotFoundError("powermetrics"),
        ),
        pytest.raises(GPUSampleError, match="not found"),
    ):
        sample_once()


def test_sample_once_timeout() -> None:
    with (
        patch(
            "gpu_song.gpu.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="powermetrics", timeout=1),
        ),
        pytest.raises(GPUSampleError, match="timed out"),
    ):
        sample_once()


def test_sampler_stop_waits_for_blocked_worker() -> None:
    sampler = GPUSampler()
    sampling = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    joining = threading.Event()

    def sample(interval_ms: int) -> float:
        if threading.current_thread() is threading.main_thread():
            return 12.0
        sampling.set()
        release.wait()
        return 34.0

    def stop() -> None:
        try:
            sampler.stop()
        finally:
            stopped.set()
            joining.set()

    stopper = threading.Thread(target=stop)
    worker = None
    with patch("gpu_song.gpu.sample_once", side_effect=sample):
        try:
            sampler.start()
            worker = sampler._thread
            assert worker is not None
            assert sampling.wait(1.0)
            real_join = worker.join

            def join(timeout: float | None = None) -> None:
                if timeout is None:
                    joining.set()
                # Expire finite joins immediately, without waiting two seconds.
                real_join(timeout=0.0 if timeout is not None else None)

            with patch.object(worker, "join", side_effect=join):
                stopper.start()
                assert joining.wait(1.0)
                assert not stopped.is_set()
                assert sampler._thread is worker
                assert worker.is_alive()
                sampler.start()
                assert sampler._thread is worker
        finally:
            release.set()
            if stopper.ident is not None:
                stopper.join()
            sampler.stop()
            if worker is not None:
                worker.join()

    assert stopped.is_set()
    assert not worker.is_alive()
    assert sampler._thread is None
    assert sampler.usage == 34.0


def test_stopped_sampler_restarts_background_sampling() -> None:
    sampler = GPUSampler()
    first_sampling = threading.Event()
    restarted_sampling = threading.Event()
    initial_usages = iter([10.0, 30.0])
    background_samples = iter([(first_sampling, 20.0), (restarted_sampling, 40.0)])

    def sample(interval_ms: int) -> float:
        if threading.current_thread() is threading.main_thread():
            return next(initial_usages)
        sampling, usage = next(background_samples)
        sampling.set()
        sampler._stop.wait()
        return usage

    with patch("gpu_song.gpu.sample_once", side_effect=sample):
        try:
            sampler.start()
            first_worker = sampler._thread
            assert first_worker is not None
            assert first_sampling.wait(1.0)
            assert sampler.usage == 10.0
            sampler.start()
            assert sampler._thread is first_worker
            sampler.stop()
            assert not first_worker.is_alive()
            assert sampler._thread is None
            assert sampler.usage == 20.0
            sampler.stop()

            sampler.start()
            restarted_worker = sampler._thread
            assert restarted_worker is not None
            assert restarted_worker is not first_worker
            assert restarted_sampling.wait(1.0)
            assert sampler.usage == 30.0
            sampler.stop()
            assert not restarted_worker.is_alive()
            assert sampler._thread is None
            assert sampler.usage == 40.0
        finally:
            sampler.stop()
