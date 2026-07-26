"""Tests for powermetrics parsing and sampling."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from gpu_song.gpu import GPUSampleError, parse_powermetrics_output, sample_once

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
