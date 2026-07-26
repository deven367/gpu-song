"""CLI for gpu-song."""

from __future__ import annotations

import argparse
import sys
import time

from gpu_song import __version__
from gpu_song.audio import DroneSynth, UsageMapper, parse_note
from gpu_song.gpu import GPUSampleError, GPUSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-song",
        description=(
            "Hum a continuous musical drone whose pitch follows "
            "Apple Silicon GPU usage."
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        metavar="MS",
        help="powermetrics sample interval in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="A2",
        help="root note of the minor pentatonic scale (default: A2)",
    )
    parser.add_argument(
        "--octaves",
        type=float,
        default=2.0,
        help="scale span in octaves (default: 2.0)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.25,
        help="peak volume 0–1 (default: 0.25)",
    )
    parser.add_argument(
        "--glide",
        type=float,
        default=120.0,
        metavar="MS",
        help="portamento time constant in milliseconds (default: 120)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 0.0 < args.volume <= 1.0:
        print("error: --volume must be in (0, 1]", file=sys.stderr)
        return 2
    if args.interval < 100:
        print("error: --interval must be >= 100 ms", file=sys.stderr)
        return 2
    if args.octaves <= 0:
        print("error: --octaves must be > 0", file=sys.stderr)
        return 2

    try:
        root_midi = parse_note(args.root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mapper = UsageMapper(root_midi=root_midi, octaves=args.octaves)
    sampler = GPUSampler(interval_ms=args.interval)
    synth = DroneSynth(volume=args.volume, glide_ms=args.glide)

    print("gpu-song: sampling Apple Silicon GPU via powermetrics…")
    try:
        sampler.start()
    except GPUSampleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        synth.start()
    except Exception as exc:  # noqa: BLE001
        sampler.stop()
        print(f"error: could not open audio device: {exc}", file=sys.stderr)
        return 1

    print("Humming — Ctrl+C to stop.")
    try:
        while True:
            usage = sampler.usage
            err = sampler.error
            _, note, hz = mapper.describe(usage)
            synth.set_usage(usage, hz)
            status = f"\rGPU {usage:5.1f}% → {note} ({hz:6.1f} Hz)"
            if err:
                status += f"  [{err}]"
            print(status, end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        synth.stop()
        sampler.stop()

    return 0


def entry() -> None:
    """Console-script entry that preserves process exit codes."""
    raise SystemExit(main())


if __name__ == "__main__":
    entry()
