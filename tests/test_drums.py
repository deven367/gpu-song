"""Tests for the procedural drum machine."""

from __future__ import annotations

import numpy as np

from gpu_song.drums import DrumMachine


def test_render_produces_signal() -> None:
    drums = DrumMachine(sample_rate=22050, enabled=True)
    buf = drums.render(4096, usage_pct=40.0)
    assert buf.shape == (4096,)
    assert float(np.abs(buf).max()) > 0.01
    assert np.isfinite(buf).all()


def test_bpm_rises_with_usage() -> None:
    drums = DrumMachine(sample_rate=22050)
    drums.render(1024, 0.0)
    bpm_lo = drums.bpm
    for _ in range(50):
        drums.render(1024, 100.0)
    assert drums.bpm > bpm_lo
    assert drums.bpm <= drums.bpm_max + 1.0


def test_disabled_is_silent_for_new_hits() -> None:
    drums = DrumMachine(sample_rate=22050, enabled=False)
    buf = drums.render(4096, usage_pct=80.0)
    assert float(np.abs(buf).max()) == 0.0


def test_voice_tails_carry_across_buffers() -> None:
    drums = DrumMachine(sample_rate=22050, enabled=True, level=1.0)
    # Tiny first buffer forces kick tail to spill into the next buffer.
    first = drums.render(32, usage_pct=10.0)
    second = drums.render(2048, usage_pct=10.0)
    assert float(np.abs(first).max()) > 0.0
    assert float(np.abs(second).max()) > 0.0


def test_stream_stays_active() -> None:
    drums = DrumMachine(sample_rate=22050)
    chunks = [drums.render(1024, 70.0) for _ in range(40)]
    combined = np.concatenate(chunks)
    assert float(np.abs(combined).max()) > 0.01
    assert float(np.mean(np.abs(combined) > 1e-4)) > 0.1
