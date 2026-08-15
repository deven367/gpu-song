# AGENTS.md

Project guidance for AI agents working on **gpu-song**.

`CLAUDE.md` is a symlink to this file — keep one source of truth.

## Project

Python CLI that hums a continuous musical drone pitched from Apple Silicon GPU usage.

- Package: `gpu_song/`; console script `gpu-song` → `gpu_song.cli:entry` (raises `SystemExit` so exit codes work)
- Invoke via `uv run gpu-song` or `python -m gpu_song`. **No top-level `main.py`** — it was removed on purpose; do not reintroduce it
- Target: macOS on Apple Silicon (e.g. M2 Pro). There is no `nvidia-smi`; GPU usage comes from `powermetrics`
- Sound: minor pentatonic (root `A2`, default span 2 octaves), portamento glide (~120 ms), volume soft-linked to usage with a floor

## File map

- `gpu.py` — `GPUSampler`: background thread polls `powermetrics --samplers gpu_power` every `--interval` ms; latest usage (0–100) kept under a lock
- `audio.py` — `UsageMapper` (usage → scale frequency) and `DroneSynth` (streaming callback: portamento, usage-linked volume, usage smoothing, drum mix); renderers for the 5 timbres
- `tracks.py` — frozen `TrackPreset` registry: `hum`, `warm`, `buzz`, `pulse`, `glass`
- `drums.py` — `DrumMachine`: synthesized kick/snare/hat, 16-step patterns, BPM ~72–148 from usage, denser hits when busy
- `cli.py` — argparse flags, `questionary` prompts (track dropdown, drums confirm; both skipped when non-TTY or explicit flag → defaults `hum` + drums on), main loop, status line (usage %, note, Hz, drum BPM)
- `tests/` — `test_gpu.py`, `test_audio.py`, `test_drums.py`, `test_tracks.py`, `test_cli.py`

## Invariants

- The audio callback must stay vectorized (`numpy`) — a per-sample Python loop underruns easily with `sounddevice`. Soft-clip additive waves with `tanh`
- Sampler runs on a background thread; the callback only reads the latest state under the synth's lock
- `import sounddevice` in `audio.py` is at module level, so **tests also need PortAudio** (macOS/Windows only — why CI uses `macos-latest`)
- Drum one-shots carry tails across buffers: mix carried voices before queueing new hits
- Usage smoothing: `DroneSynth` eases `_usage` toward `_target_usage` each buffer (`--smooth` ms, default 500, `0` = passthrough); the first sample snaps (no artificial attack). With `scale` passed in, the pitch target derives from the *smoothed* usage; the `set_usage` hz argument is only a fallback when `scale` is `None`. The status line shows `synth.current_usage` (what is heard), not `sampler.usage`

## GPU sampling

- `powermetrics` must run as root; without sudo it prints `powermetrics must be invoked as the superuser`. Surface a clear `sudo uv run gpu-song` message on failure
- Parse `GPU HW active residency: X.XX%` first; fall back to `100 - GPU idle residency`

## Conventions

- Prefer editing within `gpu_song/`; keep the package small and focused
- Commands: `make sync` (deps, default goal), `make test` (pytest + coverage gate `--cov-fail-under=75`); dev deps live in `[dependency-groups] dev`
- Unit tests never touch real `powermetrics` or audio devices — mock `subprocess.run`, `GPUSampler`, and `DroneSynth` at the CLI boundary; drive callback math via `synth._callback(out, frames, None, None)` with a fake buffer
- Do not edit plan files the user attaches unless asked
- Do not commit unless the user asks
- After meaningful sessions, update the relevant sections above so this file stays current

## Tooling & CI

- `Makefile`: `sync` (`uv sync --group dev`) and `test` (`uv run pytest`, depends on `sync`)
- `.github/workflows/ci.yml`: `make sync` + `make test` on pushes to `main` and on PRs, on `macos-latest`

## Releases

- Bump `version` in `pyproject.toml` and the tag together: `vX.Y.Z` — minor for features, patch for fixes — tagging the current `main`
- Create with `gh release create vX.Y.Z --title "gpu-song vX.Y.Z" --notes-file <notes>`; write notes by hand (features, requirements, install & run) rather than `--generate-notes`
- Latest: `v0.1.0` (2026-08-15) — drone, 5 tracks, procedural drums, `--smooth` transitions, CI
