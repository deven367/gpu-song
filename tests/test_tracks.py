"""Tests for track preset registry."""

from __future__ import annotations

import pytest

from gpu_song.tracks import (
    DEFAULT_TRACK,
    TRACK_BY_ID,
    TRACKS,
    resolve_track,
    track_choices,
)


def test_default_track_exists() -> None:
    assert DEFAULT_TRACK in TRACK_BY_ID


def test_resolve_track_case_insensitive() -> None:
    assert resolve_track("HuM").id == "hum"


def test_resolve_track_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown track"):
        resolve_track("polka")


def test_track_choices_cover_all() -> None:
    choices = track_choices()
    ids = [track_id for _, track_id in choices]
    assert ids == [t.id for t in TRACKS]
    assert all("—" in label for label, _ in choices)
