# AGENTS.md

Project guidance for AI agents working on **gpu-song**.

`CLAUDE.md` is a symlink to this file — keep one source of truth.

## Project

Python CLI that hums a continuous musical drone pitched from Apple Silicon GPU usage.

- Package: `gpu_song/`
- Console script: `gpu-song` → `gpu_song.cli:entry` (defined in `pyproject.toml`)
- No top-level `main.py` — invoke via `uv run gpu-song` or `python -m gpu_song`

## Conventions

- Prefer editing within `gpu_song/`; keep the package small and focused
- After meaningful sessions, append a short entry under [Session learnings](#session-learnings)
- Do not edit plan files the user attaches unless asked
- Do not commit unless the user asks
- Run tests with `uv run pytest` (dev group: `uv sync --group dev`)

## Session learnings

### 2026-07-26 — v1 drone + packaging cleanup

- Target machine is Apple M2 Pro (arm64). There is no `nvidia-smi`; GPU usage comes from `powermetrics --samplers gpu_power`.
- `powermetrics` must run as root. Without sudo it prints `powermetrics must be invoked as the superuser`. Surface a clear `sudo uv run gpu-song` message on failure.
- Parse `GPU HW active residency: X.XX%` first; fall back to `100 - GPU idle residency`.
- Sound design: continuous sine drone, **A minor pentatonic** (~A2–A4), portamento glide (~120 ms), volume soft-linked to usage with a floor.
- Audio callback must stay vectorized (`numpy`) — a per-sample Python loop underruns easily with `sounddevice`.
- Sampler runs on a background thread; the audio callback only reads the latest usage under a lock.
- Console entry is `gpu_song.cli:entry` (raises `SystemExit`) so exit codes work. Top-level `main.py` was removed on purpose — do not reintroduce it.
- `CLAUDE.md` → `AGENTS.md` symlink; put durable notes here, not only in chat.

### 2026-07-26 — track presets + dropdown (`feat/audio-tracks`)

- Branch for richer audio: `feat/audio-tracks`.
- Track timbres live in `gpu_song/tracks.py`; renderers in `audio.py` (`hum`, `warm`, `buzz`, `pulse`, `glass`). Keep callbacks vectorized; soft-clip additive waves with `tanh`.
- Interactive track picker uses `questionary.select` (terminal dropdown). Skip it when `--track` is set or stdin is not a TTY (default `hum`).
- Console entry remains `gpu_song.cli:entry`; no top-level `main.py`.

### 2026-07-26 — procedural drums

- Drums live in `gpu_song/drums.py` (`DrumMachine`): synthesized kick/snare/hat, 16-step patterns, BPM ~72–148 from GPU usage, denser kicks/hats when busy.
- Mixed under the drone inside `DroneSynth` callback; carry one-shot tails across buffers carefully (mix carried voices before queueing new hits).
- CLI: `--drums` / `--no-drums` (`BooleanOptionalAction`), `--drum-level`, plus a confirm prompt when unset on a TTY.
- Status line shows current drum BPM when drums are on.

### 2026-07-26 — pytest suite

- Dev dep group: `pytest` via `[dependency-groups] dev` in `pyproject.toml`.
- Tests live in `tests/`: `test_gpu.py` (powermetrics parse + mocked `sample_once`), `test_audio.py` (notes/scale/renderers), `test_drums.py`, `test_tracks.py`, `test_cli.py` (flags + mocked main loop).
- Avoid real `powermetrics` / audio devices in unit tests — mock `subprocess.run`, `GPUSampler`, and `DroneSynth` at the CLI boundary.
