# gpu-song

Hum a continuous musical drone whose pitch follows Apple Silicon GPU usage.

Higher GPU load → higher note on a minor pentatonic scale, with a short
portamento glide so it feels like humming rather than jumping.

## Requirements

- macOS on Apple Silicon
- Python 3.12+
- Root access for `powermetrics` (GPU residency sampling)

## Install

```bash
uv sync
```

## Run

`powermetrics` must run as root:

```bash
sudo uv run gpu-song
```

Or as a module:

```bash
sudo uv run python -m gpu_song
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--interval MS` | `500` | Sample interval for powermetrics |
| `--root NOTE` | `A2` | Root of the minor pentatonic (e.g. `C3`) |
| `--octaves N` | `2.0` | Scale span in octaves |
| `--volume V` | `0.25` | Peak volume `0–1` |
| `--glide MS` | `120` | Portamento time constant |

Example:

```bash
sudo uv run gpu-song --root C3 --octaves 2.5 --volume 0.2
```

Status line while running:

```text
GPU  42.3% → E4 (329.6 Hz)
```

Ctrl+C stops cleanly.

## How it works

1. Background thread samples GPU HW active residency via `powermetrics`
2. Usage percent maps onto a minor pentatonic spanning `--octaves` from `--root`
3. A streaming sine drone glides toward that pitch and softens volume when idle
