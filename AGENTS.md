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
