"""Musical scale mapping and continuous sine drone synthesis."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

# A minor pentatonic MIDI degrees relative to root: 1, b3, 4, 5, b7
_PENTATONIC_INTERVALS = (0, 3, 5, 7, 10)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_name(midi: int) -> str:
    name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return f"{name}{octave}"


def parse_note(note: str) -> int:
    """Parse a note name like 'A2' or 'C#3' into a MIDI number."""
    text = note.strip().upper().replace("♯", "#")
    if len(text) < 2:
        raise ValueError(f"Invalid note: {note!r}")

    if text[1] == "#":
        letter, rest = text[:2], text[2:]
    else:
        letter, rest = text[0], text[1:]

    if letter not in NOTE_NAMES or not rest or not rest.lstrip("-").isdigit():
        raise ValueError(f"Invalid note: {note!r}")

    octave = int(rest)
    midi = NOTE_NAMES.index(letter) + (octave + 1) * 12
    if not 0 <= midi <= 127:
        raise ValueError(f"Note out of MIDI range: {note!r}")
    return midi


def build_scale_freqs(
    root_midi: int = 45,  # A2
    octaves: float = 2.0,
) -> list[float]:
    """Build ascending minor-pentatonic frequencies from root across octaves."""
    max_midi = root_midi + int(12 * octaves)
    freqs: list[float] = []
    degree = 0
    while True:
        octave_offset = (degree // len(_PENTATONIC_INTERVALS)) * 12
        interval = _PENTATONIC_INTERVALS[degree % len(_PENTATONIC_INTERVALS)]
        midi = root_midi + octave_offset + interval
        if midi > max_midi:
            break
        freqs.append(midi_to_hz(midi))
        degree += 1
    if not freqs:
        freqs.append(midi_to_hz(root_midi))
    return freqs


@dataclass
class UsageMapper:
    """Map GPU usage percent to a target frequency on a musical scale."""

    root_midi: int = 45
    octaves: float = 2.0
    freqs: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.freqs = build_scale_freqs(self.root_midi, self.octaves)

    def target_hz(self, usage_pct: float) -> float:
        usage = max(0.0, min(100.0, usage_pct))
        if len(self.freqs) == 1:
            return self.freqs[0]
        index = int(round((usage / 100.0) * (len(self.freqs) - 1)))
        return self.freqs[index]

    def describe(self, usage_pct: float) -> tuple[float, str, float]:
        hz = self.target_hz(usage_pct)
        midi = int(round(hz_to_midi(hz)))
        return usage_pct, midi_to_name(midi), hz


@dataclass
class DroneSynth:
    """Streaming sine drone with portamento and usage-linked volume."""

    sample_rate: int = 44100
    volume: float = 0.25
    volume_floor: float = 0.35  # fraction of volume at 0% usage
    glide_ms: float = 120.0
    blocksize: int = 1024

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _target_hz: float = field(default=110.0, init=False)
    _current_hz: float = field(default=110.0, init=False)
    _usage: float = field(default=0.0, init=False)
    _phase: float = field(default=0.0, init=False)
    _stream: sd.OutputStream | None = field(default=None, init=False)

    def set_usage(self, usage_pct: float, target_hz: float) -> None:
        with self._lock:
            self._usage = max(0.0, min(100.0, usage_pct))
            self._target_hz = max(20.0, target_hz)

    @property
    def current_hz(self) -> float:
        with self._lock:
            return self._current_hz

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        _time,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            # Underruns etc. — keep going; status is logged by PortAudio.
            pass

        with self._lock:
            target_hz = self._target_hz
            current_hz = self._current_hz
            usage = self._usage
            phase = self._phase

        # Exponential approach toward target frequency each sample.
        # alpha ≈ 1 - exp(-dt / tau); tau = glide_ms.
        tau = max(1e-3, self.glide_ms / 1000.0)
        dt = 1.0 / self.sample_rate
        alpha = 1.0 - math.exp(-dt / tau)
        decay = 1.0 - alpha

        amp = self.volume * (
            self.volume_floor + (1.0 - self.volume_floor) * (usage / 100.0)
        )

        steps = np.arange(1, frames + 1, dtype=np.float64)
        hz = target_hz + (current_hz - target_hz) * (decay**steps)
        phase_inc = 2.0 * math.pi * hz / self.sample_rate
        phases = phase + np.cumsum(phase_inc)
        samples = (np.sin(phases) * amp).astype(np.float32)
        outdata[:, 0] = samples

        with self._lock:
            self._current_hz = float(hz[-1])
            self._phase = float(phases[-1] % (2.0 * math.pi))
