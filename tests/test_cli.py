"""Tests for CLI argument handling."""

from __future__ import annotations

from tempfile import TemporaryFile
from unittest.mock import MagicMock, patch

import pytest

from gpu_song.audio import DroneSynth
from gpu_song.cli import build_parser, choose_drums, choose_track, main
from gpu_song.gpu import GPUSampleError, GPUSampler


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


def test_main_rejects_negative_smooth() -> None:
    assert main(["--smooth", "-1", "--track", "hum", "--no-drums"]) == 2


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
        code = main(
            ["--track", "warm", "--drums", "--glide", "0", "--smooth", "0"]
        )

    assert code == 0


@pytest.mark.parametrize(
    ("option", "value"),
    [
        (option, value)
        for option in ("octaves", "glide", "smooth", "volume", "drum-level")
        for value in ("nan", "inf", "-inf")
    ]
    + [("glide", "-1"), ("octaves", "1e308")],
)
def test_main_rejects_invalid_numeric_values(option: str, value: str) -> None:
    with patch("gpu_song.cli.GPUSampler", side_effect=AssertionError("runtime started")):
        assert main([f"--{option}={value}", "--track", "hum", "--no-drums"]) == 2


@pytest.mark.parametrize(
    ("stage", "failure", "code"),
    [
        ("sampler", GPUSampleError("sampling failed"), 1),
        ("sampler", KeyboardInterrupt(), 130),
        ("audio", RuntimeError("start failed"), 1),
        ("audio", KeyboardInterrupt(), 130),
    ],
)
def test_startup_failure_cleans_up(
    stage: str, failure: BaseException, code: int
) -> None:
    sampler = GPUSampler()
    synth = DroneSynth(drums=False)
    with TemporaryFile() as resource:
        stream = MagicMock()
        stream.close.side_effect = resource.close
        with (
            patch("gpu_song.gpu.sample_once", return_value=12.5) as sample,
            patch("gpu_song.audio.sd.OutputStream", return_value=stream),
            patch("gpu_song.cli.GPUSampler", return_value=sampler),
            patch("gpu_song.cli.DroneSynth", return_value=synth),
        ):
            if stage == "sampler":
                sample.side_effect = failure
            else:
                stream.start.side_effect = failure
            try:
                assert main(["--track", "hum", "--no-drums"]) == code
                assert sampler._thread is None
                if stage == "audio":
                    assert resource.closed
            finally:
                sampler.stop()


def test_audio_stop_failure_still_stops_sampler() -> None:
    sampler = GPUSampler()
    synth = DroneSynth(drums=False)
    with TemporaryFile() as resource:
        stream = MagicMock()
        stream.stop.side_effect = RuntimeError("stop failed")
        stream.close.side_effect = resource.close
        with (
            patch("gpu_song.gpu.sample_once", return_value=12.5),
            patch("gpu_song.audio.sd.OutputStream", return_value=stream),
            patch("gpu_song.cli.GPUSampler", return_value=sampler),
            patch("gpu_song.cli.DroneSynth", return_value=synth),
            patch("gpu_song.cli.time.sleep", side_effect=KeyboardInterrupt),
        ):
            try:
                with pytest.raises(RuntimeError):
                    main(["--track", "hum", "--no-drums"])
                assert resource.closed
                assert sampler._thread is None
            finally:
                sampler.stop()
