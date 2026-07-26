"""Tests for note mapping and timbre renderers."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_song.audio import (
    DroneSynth,
    UsageMapper,
    _RENDERERS,
    build_scale_freqs,
    midi_to_hz,
    midi_to_name,
    parse_note,
)


def test_parse_note_a2() -> None:
    assert parse_note("A2") == 45
    assert parse_note("a2") == 45


def test_parse_note_sharp() -> None:
    assert parse_note("C#3") == 49  # C3=48, C#3=49 (middle C = C4 = 60)


def test_parse_note_invalid() -> None:
    with pytest.raises(ValueError):
        parse_note("H2")
    with pytest.raises(ValueError):
        parse_note("A")


def test_midi_roundtrip_name() -> None:
    assert midi_to_name(45) == "A2"
    assert midi_to_hz(69) == pytest.approx(440.0)


def test_scale_freqs_ascending() -> None:
    freqs = build_scale_freqs(root_midi=45, octaves=2.0)
    assert len(freqs) >= 2
    assert freqs == sorted(freqs)
    assert freqs[0] == pytest.approx(midi_to_hz(45))


def test_usage_mapper_endpoints() -> None:
    mapper = UsageMapper(root_midi=45, octaves=2.0)
    _, note_lo, hz_lo = mapper.describe(0)
    _, note_hi, hz_hi = mapper.describe(100)
    assert hz_lo < hz_hi
    assert note_lo == "A2"
    assert note_hi == "A4"


def test_usage_mapper_clamps() -> None:
    mapper = UsageMapper(root_midi=45, octaves=2.0)
    assert mapper.target_hz(-10) == mapper.target_hz(0)
    assert mapper.target_hz(200) == mapper.target_hz(100)


@pytest.mark.parametrize("track_id", sorted(_RENDERERS))
def test_renderers_finite(track_id: str) -> None:
    phases = np.linspace(0, 4 * np.pi, 256)
    t = np.linspace(0, 0.01, 256)
    wave = _RENDERERS[track_id](phases, t)
    assert wave.shape == phases.shape
    assert np.isfinite(wave).all()


def test_drone_synth_track_and_drums_flags() -> None:
    synth = DroneSynth(track="glass", drums=True)
    assert synth.track_id == "glass"
    assert synth.drums_enabled is True
    synth.set_track("buzz")
    synth.set_drums(False)
    assert synth.track_id == "buzz"
    assert synth.drums_enabled is False


def test_drone_synth_rejects_unknown_track() -> None:
    with pytest.raises(ValueError, match="Unknown track"):
        DroneSynth(track="nope")
