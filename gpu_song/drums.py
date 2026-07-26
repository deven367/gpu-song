"""Procedural drum machine — tempo and density follow GPU usage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# 16th-note patterns over one bar (16 steps). 1 = hit.
_KICK_IDLE = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
_KICK_BUSY = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0)
_SNARE = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
_HAT_IDLE = (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
_HAT_BUSY = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

_KICK_LEN = 0.22
_SNARE_LEN = 0.16
_HAT_LEN = 0.06


def _synth_kick(n: int, sample_rate: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate
    freq = 140.0 * np.exp(-t * 22.0) + 38.0
    phase = np.cumsum(2.0 * math.pi * freq / sample_rate)
    amp = np.exp(-t * 14.0)
    click = np.exp(-t * 180.0) * 0.35
    return (np.sin(phase) * amp + click).astype(np.float64)


def _synth_snare(n: int, sample_rate: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate
    noise = np.random.default_rng(0).standard_normal(n)
    noise = np.concatenate([[0.0], np.diff(noise)])
    body = np.sin(2.0 * math.pi * 180.0 * t) * np.exp(-t * 28.0) * 0.35
    snap = noise * np.exp(-t * 22.0) * 0.55
    return (body + snap).astype(np.float64)


def _synth_hat(n: int, sample_rate: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate
    noise = np.random.default_rng(1).standard_normal(n)
    bright = np.concatenate([[0.0], np.diff(noise)])
    return (bright * np.exp(-t * 55.0) * 0.28).astype(np.float64)


@dataclass
class DrumMachine:
    """16-step drum sequencer mixed under the drone."""

    sample_rate: int = 44100
    enabled: bool = True
    level: float = 0.55
    bpm_min: float = 72.0
    bpm_max: float = 148.0

    _step: int = field(default=0, init=False)
    _pos_in_step: float = field(default=0.0, init=False)
    _pending_trigger: bool = field(default=True, init=False)
    _bpm: float = field(default=90.0, init=False)
    _kick: np.ndarray = field(init=False, repr=False)
    _snare: np.ndarray = field(init=False, repr=False)
    _hat: np.ndarray = field(init=False, repr=False)
    # Active one-shots carried across buffers: (samples_already_played, wave)
    _voices: list[tuple[int, np.ndarray]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        sr = self.sample_rate
        self._kick = _synth_kick(max(1, int(_KICK_LEN * sr)), sr)
        self._snare = _synth_snare(max(1, int(_SNARE_LEN * sr)), sr)
        self._hat = _synth_hat(max(1, int(_HAT_LEN * sr)), sr)

    @property
    def bpm(self) -> float:
        return self._bpm

    def render(self, frames: int, usage_pct: float) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float64)
        usage = max(0.0, min(100.0, usage_pct)) / 100.0

        # Finish any one-shots that spilled past the previous buffer first.
        carried = self._voices
        self._voices = []
        self._mix_carried(out, carried)

        if not self.enabled:
            return out * self.level

        target_bpm = self.bpm_min + (self.bpm_max - self.bpm_min) * usage
        self._bpm += (target_bpm - self._bpm) * 0.08
        samples_per_step = max(1.0, (self.sample_rate * 60.0) / (self._bpm * 4.0))

        triggers: list[tuple[int, int]] = []
        pos = self._pos_in_step
        step = self._step
        pending = self._pending_trigger
        for i in range(frames):
            if pending:
                triggers.append((i, step))
                pending = False
            pos += 1.0
            if pos >= samples_per_step:
                pos -= samples_per_step
                step = (step + 1) % 16
                pending = True
        self._pos_in_step = pos
        self._step = step
        self._pending_trigger = pending

        for offset, step_i in triggers:
            self._queue_step(step_i, usage, offset, frames, out)

        return out * self.level

    def _queue_step(
        self,
        step: int,
        usage: float,
        offset: int,
        frames: int,
        out: np.ndarray,
    ) -> None:
        kick_pat = _KICK_BUSY if usage > 0.55 else _KICK_IDLE
        hat_pat = _HAT_BUSY if usage > 0.35 else _HAT_IDLE

        if kick_pat[step]:
            self._place(self._kick, offset, frames, out)
        if _SNARE[step]:
            self._place(self._snare, offset, frames, out)
        elif usage > 0.75 and step in (6, 10, 14):
            self._place(self._snare * 0.35, offset, frames, out)
        if hat_pat[step]:
            hat_gain = 0.7 + 0.3 * usage
            self._place(self._hat * hat_gain, offset, frames, out)

    def _place(
        self,
        wave: np.ndarray,
        offset: int,
        frames: int,
        out: np.ndarray,
    ) -> None:
        n = min(frames - offset, wave.shape[0])
        if n > 0:
            out[offset : offset + n] += wave[:n]
        if n < wave.shape[0]:
            self._voices.append((n, wave))

    def _mix_carried(
        self,
        out: np.ndarray,
        voices: list[tuple[int, np.ndarray]],
    ) -> None:
        frames = out.shape[0]
        for played, wave in voices:
            remaining = wave.shape[0] - played
            if remaining <= 0:
                continue
            n = min(frames, remaining)
            out[:n] += wave[played : played + n]
            played += n
            if played < wave.shape[0]:
                self._voices.append((played, wave))
