"""Tests for the procedural drum machine."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("sample_rate", "chunks", "first_pos"),
    [
        (150, [25, 1, 24, 350, 37], 0.0),
        (153, [26, 1, 24, 357, 31], 0.5),
    ],
)
def test_step_boundaries_match_across_buffer_splits(
    sample_rate: int, chunks: list[int], first_pos: float
) -> None:
    single = DrumMachine(sample_rate=sample_rate, bpm_min=90.0, bpm_max=90.0)
    split = DrumMachine(sample_rate=sample_rate, bpm_min=90.0, bpm_max=90.0)
    parts = []
    for index, frames in enumerate(chunks):
        part = split.render(frames, 100.0)
        if index == 0:
            assert split._pos_in_step == first_pos
            assert split._pending_trigger
            assert split._step == 1
        parts.append(part)
    np.testing.assert_allclose(
        np.concatenate(parts), single.render(sum(chunks), 100.0),
        rtol=0.0, atol=1e-14,
    )
    assert split._step == single._step == 1
    assert split._pending_trigger == single._pending_trigger
    assert split._pos_in_step == pytest.approx(single._pos_in_step)


def test_tempo_shortening_advances_before_triggering_overdue_step() -> None:
    drums = DrumMachine(sample_rate=150, bpm_max=900.0)
    drums._step = 15
    drums._pos_in_step = 24.75
    drums._pending_trigger = False
    np.testing.assert_array_equal(drums.render(1, 100.0), [0.0])
    assert drums._step == 0
    assert drums._pending_trigger
    actual = drums.render(2, 0.0)
    expected = (drums._kick[:2] + drums._hat[:2] * 0.7) * drums.level
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert drums._step == 0
    assert not drums._pending_trigger
