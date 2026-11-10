from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .persistence import consolidate
from .pipeline import ProcessingConfig, RunOptions, run_pipeline


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed a flat folder of MP4 videos with CLIP")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--videos-per-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--duplicate-threshold", type=float, default=0.97)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _consolidate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate video embedding shards")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "consolidate":
            args = _consolidate_parser().parse_args(arguments[1:])
            count = consolidate(args.output_dir, args.destination)
            print(json.dumps({"consolidated": count, "destination": str(args.destination)}))
            return

        args = _run_parser().parse_args(arguments)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        config = ProcessingConfig(
            samples=args.samples,
            duplicate_threshold=args.duplicate_threshold,
            precision=args.precision,
        )
        options = RunOptions(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            videos_per_batch=args.videos_per_batch,
            workers=args.workers,
            device=args.device,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
        manifest = run_pipeline(config, options)
        print(json.dumps(manifest["stats"], sort_keys=True))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
