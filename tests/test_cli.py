"""Tests for CLI argument handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpu_song.cli import build_parser, choose_drums, choose_track, main
from gpu_song.gpu import GPUSampleError


def test_parser_track_and_drums_flags() -> None:
    args = build_parser().parse_args(["--track", "warm", "--drums", "--drum-level", "0.4"])
    assert args.track == "warm"
    assert args.drums is True
    assert args.drum_level == pytest.approx(0.4)


def test_parser_no_drums() -> None:
    args = build_parser().parse_args(["--no-drums"])
    assert args.drums is False


def test_choose_track_explicit() -> None:
    assert choose_track("pulse") == "pulse"


def test_choose_track_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown track"):
        choose_track("nope")


def test_choose_drums_explicit() -> None:
    assert choose_drums(True) is True
    assert choose_drums(False) is False


def test_main_rejects_bad_volume() -> None:
    assert main(["--volume", "0", "--track", "hum", "--no-drums"]) == 2


def test_main_rejects_bad_drum_level() -> None:
    assert main(["--drum-level", "2", "--track", "hum", "--no-drums"]) == 2


def test_parser_smooth_flag() -> None:
    args = build_parser().parse_args(["--smooth", "0"])
    assert args.smooth == 0.0
    assert build_parser().parse_args([]).smooth == pytest.approx(500.0)


def test_main_rejects_negative_smooth() -> None:
    assert main(["--smooth", "-1", "--track", "hum", "--no-drums"]) == 2


def test_main_gpu_sample_error() -> None:
    fake_sampler = MagicMock()
    fake_sampler.start.side_effect = GPUSampleError("powermetrics requires root")
    with patch("gpu_song.cli.GPUSampler", return_value=fake_sampler):
        code = main(["--track", "hum", "--no-drums"])
    assert code == 1


def test_main_runs_until_interrupt() -> None:
    fake_sampler = MagicMock()
    fake_sampler.usage = 12.5
    fake_sampler.error = None
    fake_synth = MagicMock()
    fake_synth.drum_bpm = 96.0
    fake_synth.current_usage = 12.5

    with (
        patch("gpu_song.cli.GPUSampler", return_value=fake_sampler),
        patch("gpu_song.cli.DroneSynth", return_value=fake_synth),
        patch("gpu_song.cli.time.sleep", side_effect=KeyboardInterrupt),
    ):
        code = main(["--track", "warm", "--drums"])

    assert code == 0
    fake_sampler.start.assert_called_once()
    fake_synth.start.assert_called_once()
    fake_synth.set_usage.assert_called()
    fake_synth.stop.assert_called_once()
    fake_sampler.stop.assert_called_once()
