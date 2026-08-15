"""CLI for gpu-song."""

from __future__ import annotations

import argparse
import sys
import time

import questionary
from questionary import Style

from gpu_song import __version__
from gpu_song.audio import DroneSynth, UsageMapper, parse_note
from gpu_song.gpu import GPUSampleError, GPUSampler
from gpu_song.tracks import (
    DEFAULT_TRACK,
    TRACK_BY_ID,
    TRACKS,
    resolve_track,
    track_choices,
)

_DROPDOWN_STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:cyan"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:cyan"),
    ]
)


def build_parser() -> argparse.ArgumentParser:
    track_ids = ", ".join(t.id for t in TRACKS)
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
        "--smooth",
        type=float,
        default=500.0,
        metavar="MS",
        help=(
            "usage smoothing time constant in milliseconds; "
            "0 disables smoothing (default: 500)"
        ),
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            f"track timbre ({track_ids}). "
            "If omitted, shows an interactive dropdown when stdin is a TTY."
        ),
    )
    parser.add_argument(
        "--drums",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="layer procedural drums under the drone (default: ask / on)",
    )
    parser.add_argument(
        "--drum-level",
        type=float,
        default=0.55,
        metavar="V",
        help="drum mix level 0–1 (default: 0.55)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def choose_track(explicit: str | None) -> str:
    """Resolve track from --track, interactive dropdown, or default."""
    if explicit is not None:
        return resolve_track(explicit).id

    if not sys.stdin.isatty():
        print(
            f"gpu-song: no TTY for track dropdown; using default track "
            f"{DEFAULT_TRACK!r}",
            file=sys.stderr,
        )
        return DEFAULT_TRACK

    choices = [
        questionary.Choice(title=label, value=track_id)
        for label, track_id in track_choices()
    ]
    selected = questionary.select(
        "Select a track timbre:",
        choices=choices,
        default=DEFAULT_TRACK,
        style=_DROPDOWN_STYLE,
        use_shortcuts=True,
    ).ask()

    if selected is None:
        raise KeyboardInterrupt
    return selected


def choose_drums(explicit: bool | None) -> bool:
    """Resolve drums on/off from flag, prompt, or default-on."""
    if explicit is not None:
        return explicit
    if not sys.stdin.isatty():
        return True
    answer = questionary.confirm(
        "Add drums? (tempo rises with GPU load)",
        default=True,
        style=_DROPDOWN_STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return bool(answer)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 0.0 < args.volume <= 1.0:
        print("error: --volume must be in (0, 1]", file=sys.stderr)
        return 2
    if not 0.0 <= args.drum_level <= 1.0:
        print("error: --drum-level must be in [0, 1]", file=sys.stderr)
        return 2
    if args.interval < 100:
        print("error: --interval must be >= 100 ms", file=sys.stderr)
        return 2
    if args.octaves <= 0:
        print("error: --octaves must be > 0", file=sys.stderr)
        return 2
    if args.smooth < 0:
        print("error: --smooth must be >= 0 ms", file=sys.stderr)
        return 2

    try:
        root_midi = parse_note(args.root)
        track_id = choose_track(args.track)
        drums_on = choose_drums(args.drums)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    preset = TRACK_BY_ID[track_id]
    mapper = UsageMapper(root_midi=root_midi, octaves=args.octaves)
    sampler = GPUSampler(interval_ms=args.interval)
    synth = DroneSynth(
        volume=args.volume,
        glide_ms=args.glide,
        smooth_ms=args.smooth,
        scale=mapper.freqs,
        track=track_id,
        drums=drums_on,
        drum_level=args.drum_level,
    )

    print(f"Track: {preset.label} ({preset.id})")
    print(f"Drums: {'on' if drums_on else 'off'}")
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
            raw_usage = sampler.usage
            err = sampler.error
            # Feed in the raw sample; the status line shows what is heard.
            synth.set_usage(raw_usage, mapper.target_hz(raw_usage))
            heard = synth.current_usage
            _, note, hz = mapper.describe(heard)
            if drums_on:
                status = (
                    f"\r[{preset.id}] GPU {heard:5.1f}% → {note} "
                    f"({hz:6.1f} Hz)  drums {synth.drum_bpm:5.1f} BPM"
                )
            else:
                status = (
                    f"\r[{preset.id}] GPU {heard:5.1f}% → {note} ({hz:6.1f} Hz)"
                )
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
