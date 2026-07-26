"""Named track presets that shape the drone timbre."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPreset:
    id: str
    label: str
    description: str


TRACKS: tuple[TrackPreset, ...] = (
    TrackPreset(
        id="hum",
        label="Hum",
        description="Pure sine — the original soft drone",
    ),
    TrackPreset(
        id="warm",
        label="Warm pad",
        description="Two slightly detuned sines for a chorused hum",
    ),
    TrackPreset(
        id="buzz",
        label="Buzz",
        description="Soft saw via harmonics — edgier under load",
    ),
    TrackPreset(
        id="pulse",
        label="Pulse",
        description="Rounded square — hollow, rhythmic feel",
    ),
    TrackPreset(
        id="glass",
        label="Glass",
        description="Bright partials with a light shimmer",
    ),
)

TRACK_BY_ID: dict[str, TrackPreset] = {t.id: t for t in TRACKS}
DEFAULT_TRACK = "hum"


def track_choices() -> list[tuple[str, str]]:
    """Return (display label, track id) pairs for a dropdown."""
    return [(f"{t.label} — {t.description}", t.id) for t in TRACKS]


def resolve_track(track_id: str) -> TrackPreset:
    key = track_id.strip().lower()
    if key not in TRACK_BY_ID:
        known = ", ".join(t.id for t in TRACKS)
        raise ValueError(f"Unknown track {track_id!r}. Choose one of: {known}")
    return TRACK_BY_ID[key]
